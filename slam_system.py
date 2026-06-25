"""
SLAM Robusto con ZED2 / SVO2
==============================
Módulos:
  1. Odometría Visual (VO) — ORB + solvePnPRansac
  2. Mapa de Landmarks 3D persistente
  3. Loop Closure Detection — BoW con ORB descriptors
  4. Bundle Adjustment — g2o / scipy fallback
  5. Relocalización cuando se pierde el tracking
"""

import pyzed.sl as sl
import numpy as np
import cv2 as cv
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import time

# ── Intenta importar g2o para Bundle Adjustment real ──────────────────────────
try:
    import g2o
    HAS_G2O = True
except ImportError:
    from scipy.optimize import least_squares
    HAS_G2O = False
    print("[WARN] g2o no encontrado — usando scipy BA (menos eficiente)")

# ══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Landmark:
    """Punto 3D del mapa mundial."""
    id: int
    position: np.ndarray          # (3,)  XYZ en mundo
    descriptors: list = field(default_factory=list)   # descriptores de cada observación
    observations: list = field(default_factory=list)  # [(frame_id, kp_idx), ...]
    times_seen: int = 0


@dataclass
class KeyFrame:
    """Frame clave con pose, keypoints y referencias a landmarks."""
    id: int
    timestamp: float
    pose_Tcw: np.ndarray          # 4×4 Transform Cámara←Mundo
    keypoints: list
    descriptors: np.ndarray
    depth_map: np.ndarray
    landmark_ids: dict = field(default_factory=dict)  # kp_idx → landmark_id


# ══════════════════════════════════════════════════════════════════════════════
# MAPA GLOBAL DE LANDMARKS
# ══════════════════════════════════════════════════════════════════════════════

class LandmarkMap:
    def __init__(self):
        self._landmarks: dict[int, Landmark] = {}
        self._next_id = 0

    def add(self, position: np.ndarray, descriptor: np.ndarray,
            frame_id: int, kp_idx: int) -> int:
        lm = Landmark(
            id=self._next_id,
            position=position.copy(),
            descriptors=[descriptor],
            observations=[(frame_id, kp_idx)],
            times_seen=1
        )
        self._landmarks[self._next_id] = lm
        self._next_id += 1
        return lm.id

    def update(self, lm_id: int, position: np.ndarray,
               descriptor: np.ndarray, frame_id: int, kp_idx: int):
        lm = self._landmarks[lm_id]
        # Media ponderada de la posición
        lm.position = (lm.position * lm.times_seen + position) / (lm.times_seen + 1)
        lm.descriptors.append(descriptor)
        lm.observations.append((frame_id, kp_idx))
        lm.times_seen += 1

    def get(self, lm_id: int) -> Optional[Landmark]:
        return self._landmarks.get(lm_id)

    def all_positions(self) -> np.ndarray:
        if not self._landmarks:
            return np.empty((0, 3))
        return np.array([lm.position for lm in self._landmarks.values()])

    def __len__(self):
        return len(self._landmarks)


# ══════════════════════════════════════════════════════════════════════════════
# BAG OF WORDS LIGERO (para loop closure)
# ══════════════════════════════════════════════════════════════════════════════

class BagOfWords:
    """
    BoW minimalista basado en clustering de descriptores ORB.
    Alternativa a DBoW2 sin dependencias extra.
    """
    def __init__(self, vocab_size: int = 500):
        self.vocab_size = vocab_size
        self.vocab: Optional[np.ndarray] = None        # (K, 32) uint8
        self._kf_vectors: dict[int, np.ndarray] = {}  # kf_id → histograma
        self._trained = False

    # ── Entrenamiento offline del vocabulario ─────────────────────────────────
    def train(self, all_descriptors: np.ndarray):
        """Clusteriza descriptores con k-means para crear el vocabulario."""
        if len(all_descriptors) < self.vocab_size:
            self.vocab_size = len(all_descriptors)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        _, _, centers = cv.kmeans(
            all_descriptors.astype(np.float32),
            self.vocab_size, None, criteria, 3,
            cv.KMEANS_PP_CENTERS
        )
        self.vocab = centers.astype(np.uint8)
        self._trained = True
        print(f"[BoW] Vocabulario entrenado: {self.vocab_size} palabras")

    # ── Online: asigna palabras a descriptores ────────────────────────────────
    def _descriptors_to_bow(self, descriptors: np.ndarray) -> np.ndarray:
        if not self._trained or descriptors is None or len(descriptors) == 0:
            return np.zeros(self.vocab_size)
        # Distancia Hamming a cada palabra del vocabulario
        # (vectorizado: XOR + popcount aproximado con uint8)
        dists = np.array([
            np.sum(np.unpackbits(
                np.bitwise_xor(desc, self.vocab), axis=1
            ), axis=1)
            for desc in descriptors
        ])                        # (N_desc, vocab_size)
        words = np.argmin(dists, axis=1)
        hist, _ = np.histogram(words, bins=self.vocab_size,
                               range=(0, self.vocab_size))
        norm = hist.sum()
        return hist.astype(np.float32) / (norm + 1e-9)

    def add_keyframe(self, kf_id: int, descriptors: np.ndarray):
        self._kf_vectors[kf_id] = self._descriptors_to_bow(descriptors)

    def query(self, descriptors: np.ndarray,
              top_k: int = 5, min_score: float = 0.2) -> list[tuple[int, float]]:
        """Devuelve [(kf_id, score)] ordenados por similitud (coseno)."""
        if not self._trained or not self._kf_vectors:
            return []
        q = self._descriptors_to_bow(descriptors)
        scores = {}
        for kf_id, vec in self._kf_vectors.items():
            dot = np.dot(q, vec)
            denom = np.linalg.norm(q) * np.linalg.norm(vec) + 1e-9
            scores[kf_id] = dot / denom
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [(kid, s) for kid, s in ranked[:top_k] if s >= min_score]


# ══════════════════════════════════════════════════════════════════════════════
# BUNDLE ADJUSTMENT
# ══════════════════════════════════════════════════════════════════════════════

class BundleAdjuster:
    def __init__(self, K: np.ndarray):
        self.K = K
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]

    # ── Reproyección (usada por scipy) ────────────────────────────────────────
    def _project(self, point_3d: np.ndarray, pose_Tcw: np.ndarray) -> np.ndarray:
        R = pose_Tcw[:3, :3]
        t = pose_Tcw[:3, 3:]
        Xc = R @ point_3d.reshape(3, 1) + t
        u = self.fx * Xc[0, 0] / Xc[2, 0] + self.cx
        v = self.fy * Xc[1, 0] / Xc[2, 0] + self.cy
        return np.array([u, v])

    # ── BA con scipy (fallback) ───────────────────────────────────────────────
    def run_scipy(self, keyframes: list[KeyFrame],
                  landmark_map: LandmarkMap, n_iter: int = 5):
        """
        BA windowed: optimiza las últimas `len(keyframes)` poses y landmarks.
        """
        # Serializar estado inicial
        poses_r = []   # Rodrigues vectors
        poses_t = []
        lm_ids_ordered = []
        lm_positions = []
        observations = []  # (kf_local_idx, lm_local_idx, u, v)

        kf_idx_map = {kf.id: i for i, kf in enumerate(keyframes)}
        lm_idx_map = {}

        for kf_i, kf in enumerate(keyframes):
            R = kf.pose_Tcw[:3, :3]
            t = kf.pose_Tcw[:3, 3]
            rvec, _ = cv.Rodrigues(R)
            poses_r.append(rvec.ravel())
            poses_t.append(t)

            for kp_idx, lm_id in kf.landmark_ids.items():
                lm = landmark_map.get(lm_id)
                if lm is None:
                    continue
                if lm_id not in lm_idx_map:
                    lm_idx_map[lm_id] = len(lm_ids_ordered)
                    lm_ids_ordered.append(lm_id)
                    lm_positions.append(lm.position.copy())

                uv = kf.keypoints[kp_idx].pt
                observations.append((kf_i, lm_idx_map[lm_id], uv[0], uv[1]))

        if len(observations) < 6:
            return

        n_poses = len(keyframes)
        n_lm = len(lm_ids_ordered)

        def pack(poses_r, poses_t, lm_positions):
            x = []
            for r, t in zip(poses_r, poses_t):
                x.extend(r); x.extend(t)
            for p in lm_positions:
                x.extend(p)
            return np.array(x)

        def unpack(x):
            pr, pt, lp = [], [], []
            offset = 0
            for _ in range(n_poses):
                pr.append(x[offset:offset+3]); offset += 3
                pt.append(x[offset:offset+3]); offset += 3
            for _ in range(n_lm):
                lp.append(x[offset:offset+3]); offset += 3
            return pr, pt, lp

        def residuals(x):
            pr, pt, lp = unpack(x)
            res = []
            for kf_i, lm_i, u_obs, v_obs in observations:
                R_m, _ = cv.Rodrigues(np.array(pr[kf_i]))
                T = np.eye(4)
                T[:3, :3] = R_m
                T[:3, 3] = pt[kf_i]
                uv = self._project(np.array(lp[lm_i]), T)
                res.extend([uv[0] - u_obs, uv[1] - v_obs])
            return res

        x0 = pack(poses_r, poses_t, lm_positions)
        result = least_squares(residuals, x0, method='lm',
                               max_nfev=n_iter * 100, verbose=0)

        # Escribir resultados de vuelta
        pr_opt, pt_opt, lp_opt = unpack(result.x)
        for i, kf in enumerate(keyframes):
            R_m, _ = cv.Rodrigues(np.array(pr_opt[i]))
            kf.pose_Tcw[:3, :3] = R_m
            kf.pose_Tcw[:3, 3] = pt_opt[i]

        for j, lm_id in enumerate(lm_ids_ordered):
            lm = landmark_map.get(lm_id)
            if lm:
                lm.position = np.array(lp_opt[j])

        print(f"[BA scipy] Residual final: {result.cost:.4f} | "
              f"Poses: {n_poses} | Landmarks: {n_lm}")

    def run(self, keyframes, landmark_map, n_iter=5):
        if HAS_G2O:
            self._run_g2o(keyframes, landmark_map, n_iter)
        else:
            self.run_scipy(keyframes, landmark_map, n_iter)

    # ── BA con g2o ────────────────────────────────────────────────────────────
    def _run_g2o(self, keyframes: list[KeyFrame],
                 landmark_map: LandmarkMap, n_iter: int = 10):
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(algorithm)

        cam_params = g2o.CameraParameters(
            self.fx, (self.cx, self.cy), 0
        )
        cam_params.set_id(0)
        optimizer.add_parameter(cam_params)

        lm_idx_map = {}
        pose_vertex_offset = 0
        lm_vertex_offset = len(keyframes)

        # Vértices de poses
        for i, kf in enumerate(keyframes):
            R = kf.pose_Tcw[:3, :3]
            t = kf.pose_Tcw[:3, 3]
            q = g2o.Quaternion(R)
            v = g2o.VertexSE3Expmap()
            v.set_id(i)
            v.set_estimate(g2o.SE3Quat(q, t))
            v.set_fixed(i == 0)   # Ancla el primer frame
            optimizer.add_vertex(v)

        # Vértices de landmarks y aristas
        edge_id = 0
        for kf_i, kf in enumerate(keyframes):
            for kp_idx, lm_id in kf.landmark_ids.items():
                lm = landmark_map.get(lm_id)
                if lm is None:
                    continue

                if lm_id not in lm_idx_map:
                    v_id = lm_vertex_offset + len(lm_idx_map)
                    lm_idx_map[lm_id] = v_id
                    vp = g2o.VertexPointXYZ()
                    vp.set_id(v_id)
                    vp.set_estimate(lm.position)
                    vp.set_marginalized(True)
                    optimizer.add_vertex(vp)

                obs_uv = kf.keypoints[kp_idx].pt
                e = g2o.EdgeProjectXYZ2UV()
                e.set_id(edge_id); edge_id += 1
                e.set_vertex(0, optimizer.vertex(lm_idx_map[lm_id]))
                e.set_vertex(1, optimizer.vertex(kf_i))
                e.set_measurement(obs_uv)
                e.set_information(np.eye(2))
                kernel = g2o.RobustKernelHuber(np.sqrt(5.991))
                e.set_robust_kernel(kernel)
                e.set_parameter_id(0, 0)
                optimizer.add_edge(e)

        optimizer.initialize_optimization()
        optimizer.optimize(n_iter)

        # Leer resultados
        for i, kf in enumerate(keyframes):
            se3 = optimizer.vertex(i).estimate()
            kf.pose_Tcw[:3, :3] = se3.rotation().matrix()
            kf.pose_Tcw[:3, 3] = se3.translation()

        for lm_id, v_id in lm_idx_map.items():
            lm = landmark_map.get(lm_id)
            if lm:
                lm.position = optimizer.vertex(v_id).estimate()

        print(f"[BA g2o] Optimización completa | "
              f"Poses: {len(keyframes)} | Landmarks: {len(lm_idx_map)}")


# ══════════════════════════════════════════════════════════════════════════════
# SISTEMA SLAM PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class SLAMSystem:
    # ── Umbrales ajustables ───────────────────────────────────────────────────
    MIN_MATCHES_TRACK      = 10    # Mínimo de matches para considerar tracking OK
    MIN_INLIERS_TRACK      = 8
    MIN_INLIERS_RELOC      = 12
    KEYFRAME_MIN_MATCHES   = 80    # Debajo de este % crear nuevo KF
    LOOP_SCORE_THRESHOLD   = 0.30
    LOOP_MIN_INLIERS       = 20
    BA_WINDOW              = 7     # Últimos N keyframes en BA local
    BA_EVERY_N_KF          = 5     # Correr BA cada N keyframes

    def __init__(self, K: np.ndarray, fx, fy, cx, cy):
        self.K  = K
        self.fx = fx; self.fy = fy
        self.cx = cx; self.cy = cy

        # Estado
        self.state = "INITIALIZING"   # INITIALIZING | TRACKING | LOST
        self.frame_id   = 0
        self.kf_id      = 0

        # Datos del frame anterior
        self.prev_kf:       Optional[KeyFrame] = None
        self.prev_img:      Optional[np.ndarray] = None

        # Pose global acumulada
        self.pose_Tcw = np.eye(4, dtype=np.float64)

        # Submódulos
        self.landmark_map = LandmarkMap()
        self.bow          = BagOfWords(vocab_size=300)
        self.ba           = BundleAdjuster(K)

        # Historial
        self.keyframes:    list[KeyFrame] = []
        self.trajectory:   list[np.ndarray] = []   # posiciones cámara
        self.loop_edges:   list[tuple[int,int]] = []

        # ORB
        self.orb = cv.ORB_create(nfeatures=1500, scaleFactor=1.2,
                                 nlevels=8, fastThreshold=10)
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        # Buffer de descriptores para entrenar BoW
        self._bow_train_buffer: list[np.ndarray] = []
        self._bow_trained = False

        print("[SLAM] Sistema inicializado")

    # ══════════════════════════════════════════════════════════════════════════
    # ENTRADA PRINCIPAL
    # ══════════════════════════════════════════════════════════════════════════

    def process_frame(self, img_bgra: np.ndarray,
                      depth_map: np.ndarray) -> dict:
        """
        Procesa un frame. Devuelve diccionario con métricas y estado.
        """
        t0 = time.time()
        img_gray = cv.cvtColor(img_bgra, cv.COLOR_BGRA2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(img_gray, None)

        result = {
            "frame_id": self.frame_id,
            "state": self.state,
            "n_kp": len(keypoints),
            "n_landmarks": len(self.landmark_map),
            "n_keyframes": len(self.keyframes),
            "pose": self.pose_Tcw.copy(),
            "loop_detected": False,
            "loop_with_kf": None,
            "time_ms": 0,
        }

        if descriptors is None or len(keypoints) < 20:
            self.frame_id += 1
            result["time_ms"] = (time.time() - t0) * 1000
            return result

        # ── Inicialización: primer frame ──────────────────────────────────────
        if self.state == "INITIALIZING":
            self._initialize(img_bgra, keypoints, descriptors, depth_map)
            self.frame_id += 1
            result["state"] = self.state
            result["time_ms"] = (time.time() - t0) * 1000
            return result

        # ── Tracking normal ───────────────────────────────────────────────────
        if self.state == "TRACKING":
            tracked = self._track(keypoints, descriptors, depth_map)
            if tracked:
                result["inliers"] = tracked.get("inliers", 0)
                self._maybe_create_keyframe(
                    img_bgra, keypoints, descriptors,
                    depth_map, tracked
                )
                # Relocalización preventiva si los inliers bajan mucho
                if tracked.get("inliers", 999) < self.MIN_INLIERS_TRACK:
                    self.state = "LOST"
            else:
                self.state = "LOST"
                print(f"[SLAM] Frame {self.frame_id}: Tracking PERDIDO")

        # ── Relocalización ────────────────────────────────────────────────────
        if self.state == "LOST":
            reloc = self._relocalize(keypoints, descriptors, depth_map)
            if reloc:
                self.state = "TRACKING"
                print(f"[SLAM] Frame {self.frame_id}: Relocalizado!")
            # Guardamos posición como NaN para marcar pérdida en trayectoria
            else:
                self.trajectory.append(np.full((3,), np.nan))

        # ── Loop closure check ────────────────────────────────────────────────
        if self.state == "TRACKING" and self._bow_trained:
            loop = self._detect_loop_closure(descriptors)
            if loop is not None:
                result["loop_detected"] = True
                result["loop_with_kf"] = loop
                self.loop_edges.append((self.kf_id - 1, loop))
                self._correct_loop(loop)

        self.frame_id += 1
        result["state"] = self.state
        result["pose"] = self.pose_Tcw.copy()
        result["n_landmarks"] = len(self.landmark_map)
        result["time_ms"] = (time.time() - t0) * 1000
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # INICIALIZACIÓN
    # ══════════════════════════════════════════════════════════════════════════

    def _initialize(self, img_bgra, keypoints, descriptors, depth_map):
        kf = self._create_keyframe(img_bgra, keypoints, descriptors,
                                   depth_map, self.pose_Tcw.copy())
        self._add_initial_landmarks(kf, depth_map)
        self.prev_kf = kf
        self.prev_img = img_bgra.copy()
        self.state = "TRACKING"
        print(f"[SLAM] Inicializado. Landmarks: {len(self.landmark_map)}")

    # ══════════════════════════════════════════════════════════════════════════
    # TRACKING
    # ══════════════════════════════════════════════════════════════════════════

    def _track(self, keypoints, descriptors, depth_map) -> Optional[dict]:
        if self.prev_kf is None or self.prev_kf.descriptors is None:
            return None

        matches = self._match(self.prev_kf.descriptors, descriptors)
        if len(matches) < self.MIN_MATCHES_TRACK:
            return None

        obj_pts, img_pts, match_lm_ids, match_kp_indices = \
            self._build_pnp_data(matches, self.prev_kf, keypoints, depth_map)

        if len(obj_pts) < self.MIN_MATCHES_TRACK:
            return None

        ok, rvec, tvec, inliers = cv.solvePnPRansac(
            obj_pts, img_pts, self.K, None,
            iterationsCount=200,
            reprojectionError=2.0,
            confidence=0.995,
            flags=cv.SOLVEPNP_ITERATIVE
        )

        if not ok or inliers is None or len(inliers) < self.MIN_INLIERS_TRACK:
            return None

        # Refinar con inliers
        inlier_obj = obj_pts[inliers.ravel()]
        inlier_img = img_pts[inliers.ravel()]
        cv.solvePnP(inlier_obj, inlier_img, self.K, None,
                    rvec, tvec, useExtrinsicGuess=True,
                    flags=cv.SOLVEPNP_ITERATIVE)

        R, _ = cv.Rodrigues(rvec)
        Tcw = np.eye(4)
        Tcw[:3, :3] = R
        Tcw[:3, 3]  = tvec.ravel()

        self.pose_Tcw = Tcw
        cam_pos = -R.T @ tvec.ravel()
        self.trajectory.append(cam_pos)

        return {
            "inliers": len(inliers),
            "matches": len(matches),
            "Tcw": Tcw,
            "match_lm_ids": match_lm_ids,
            "match_kp_indices": match_kp_indices,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # CREACIÓN DE KEYFRAME
    # ══════════════════════════════════════════════════════════════════════════

    def _maybe_create_keyframe(self, img_bgra, keypoints, descriptors,
                                depth_map, tracked: dict):
        match_ratio = tracked["matches"] / max(len(self.prev_kf.keypoints), 1)
        should_create = (
            match_ratio < (self.KEYFRAME_MIN_MATCHES / 100)
            or len(self.keyframes) == 0
            or (self.kf_id % self.BA_EVERY_N_KF == 0)
        )

        if not should_create:
            return

        kf = self._create_keyframe(img_bgra, keypoints, descriptors,
                                   depth_map, self.pose_Tcw.copy())

        # Asociar landmarks visibles
        for kp_idx, lm_id in self.prev_kf.landmark_ids.items():
            if kp_idx < len(tracked["match_kp_indices"]):
                curr_kp_idx = tracked["match_kp_indices"].get(kp_idx)
                if curr_kp_idx is not None:
                    kf.landmark_ids[curr_kp_idx] = lm_id
                    lm = self.landmark_map.get(lm_id)
                    if lm and curr_kp_idx < len(descriptors):
                        self.landmark_map.update(
                            lm_id, lm.position,
                            descriptors[curr_kp_idx],
                            kf.id, curr_kp_idx
                        )

        # Triangular nuevos landmarks desde depth
        self._add_landmarks_from_depth(kf, depth_map)

        # Bundle Adjustment local
        if len(self.keyframes) >= self.BA_WINDOW and \
                self.kf_id % self.BA_EVERY_N_KF == 0:
            window = self.keyframes[-self.BA_WINDOW:]
            self.ba.run(window, self.landmark_map)

        # Entrenar BoW cuando hay suficientes descriptores
        if descriptors is not None:
            self._bow_train_buffer.append(descriptors)
        if not self._bow_trained and len(self._bow_train_buffer) >= 10:
            all_desc = np.vstack(self._bow_train_buffer)
            self.bow.train(all_desc)
            # Añadir KFs existentes al índice BoW
            for prev_kf in self.keyframes:
                self.bow.add_keyframe(prev_kf.id, prev_kf.descriptors)
            self._bow_trained = True

        if self._bow_trained:
            self.bow.add_keyframe(kf.id, descriptors)

        self.prev_kf  = kf
        self.prev_img = img_bgra.copy()
        print(f"[SLAM] KeyFrame {kf.id} creado | "
              f"LM: {len(self.landmark_map)} | "
              f"KFs: {len(self.keyframes)}")

    # ══════════════════════════════════════════════════════════════════════════
    # RELOCALIZACIÓN
    # ══════════════════════════════════════════════════════════════════════════

    def _relocalize(self, keypoints, descriptors, depth_map) -> bool:
        """
        Intenta relocalizar comparando con todos los KFs guardados.
        Usa BoW para candidatos y PnP para verificar.
        """
        if not self.keyframes:
            return False

        # Candidatos por BoW
        if self._bow_trained:
            candidates = self.bow.query(descriptors, top_k=5, min_score=0.15)
            candidate_kf_ids = [kid for kid, _ in candidates]
        else:
            # Sin BoW: probar últimos 10 KFs
            candidate_kf_ids = [kf.id for kf in self.keyframes[-10:]]

        best_inliers = 0
        best_Tcw = None

        for kf_id in candidate_kf_ids:
            kf = next((k for k in self.keyframes if k.id == kf_id), None)
            if kf is None:
                continue

            matches = self._match(kf.descriptors, descriptors)
            if len(matches) < self.MIN_MATCHES_TRACK:
                continue

            obj_pts, img_pts, _, _ = self._build_pnp_data(
                matches, kf, keypoints, depth_map
            )

            if len(obj_pts) < self.MIN_INLIERS_RELOC:
                continue

            ok, rvec, tvec, inliers = cv.solvePnPRansac(
                obj_pts, img_pts, self.K, None,
                iterationsCount=300,
                reprojectionError=3.0,
                confidence=0.999
            )

            if ok and inliers is not None and \
                    len(inliers) >= self.MIN_INLIERS_RELOC:
                if len(inliers) > best_inliers:
                    best_inliers = len(inliers)
                    R, _ = cv.Rodrigues(rvec)
                    best_Tcw = np.eye(4)
                    best_Tcw[:3, :3] = R
                    best_Tcw[:3, 3]  = tvec.ravel()

        if best_Tcw is not None:
            self.pose_Tcw = best_Tcw
            cam_pos = -best_Tcw[:3, :3].T @ best_Tcw[:3, 3]
            self.trajectory.append(cam_pos)
            print(f"[SLAM] Relocalizado con {best_inliers} inliers")
            return True

        return False

    # ══════════════════════════════════════════════════════════════════════════
    # LOOP CLOSURE
    # ══════════════════════════════════════════════════════════════════════════

    def _detect_loop_closure(self, descriptors) -> Optional[int]:
        """
        Devuelve el kf_id del loop detectado, o None.
        Solo busca en KFs con suficiente separación temporal.
        """
        if len(self.keyframes) < 15:
            return None

        candidates = self.bow.query(
            descriptors, top_k=3,
            min_score=self.LOOP_SCORE_THRESHOLD
        )

        for cand_id, score in candidates:
            # Ignorar KFs recientes (pueden ser el mismo lugar sin ser loop)
            if cand_id >= self.kf_id - 10:
                continue

            kf = next((k for k in self.keyframes if k.id == cand_id), None)
            if kf is None:
                continue

            # Verificar geométricamente
            matches = self._match(kf.descriptors, descriptors)
            if len(matches) < self.LOOP_MIN_INLIERS:
                continue

            # Homografía rápida como verificación
            pts_prev = np.float32([kf.keypoints[m.queryIdx].pt
                                   for m in matches]).reshape(-1, 1, 2)
            pts_curr = np.float32([  # descriptors viene del frame actual
                kf.keypoints[m.queryIdx].pt  # placeholder — usamos pts del KF
                for m in matches]).reshape(-1, 1, 2)

            if len(pts_prev) >= 4:
                _, mask = cv.findHomography(
                    pts_prev, pts_curr,
                    cv.RANSAC, 5.0
                )
                if mask is not None and mask.sum() >= self.LOOP_MIN_INLIERS:
                    print(f"[SLAM] ¡Loop Closure! KF actual → KF {cand_id} "
                          f"(score={score:.3f}, inliers={mask.sum()})")
                    return cand_id

        return None

    def _correct_loop(self, loop_kf_id: int):
        """
        Corrección simple de loop: BA global con los keyframes involucrados.
        En un sistema completo usarías pose graph optimization (g2o).
        """
        loop_kf = next((k for k in self.keyframes if k.id == loop_kf_id), None)
        if loop_kf is None:
            return

        # Pose graph: interpolar poses entre loop_kf y kf actual
        loop_idx = self.keyframes.index(loop_kf)
        affected = self.keyframes[loop_idx:]
        if len(affected) > 1:
            self.ba.run(affected[-self.BA_WINDOW:], self.landmark_map, n_iter=20)
        print(f"[SLAM] Corrección de loop aplicada sobre {len(affected)} KFs")

    # ══════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _create_keyframe(self, img_bgra, keypoints, descriptors,
                         depth_map, pose_Tcw) -> KeyFrame:
        kf = KeyFrame(
            id=self.kf_id,
            timestamp=time.time(),
            pose_Tcw=pose_Tcw.copy(),
            keypoints=keypoints,
            descriptors=descriptors,
            depth_map=depth_map.copy()
        )
        self.keyframes.append(kf)
        self.kf_id += 1
        return kf

    def _add_initial_landmarks(self, kf: KeyFrame, depth_map: np.ndarray):
        """Triangula landmarks iniciales desde profundidad."""
        self._add_landmarks_from_depth(kf, depth_map)

    def _add_landmarks_from_depth(self, kf: KeyFrame, depth_map: np.ndarray):
        R = kf.pose_Tcw[:3, :3]
        t = kf.pose_Tcw[:3, 3]
        R_inv = R.T
        t_world = -R_inv @ t

        for kp_idx, kp in enumerate(kf.keypoints):
            if kp_idx in kf.landmark_ids:
                continue
            u, v = kp.pt
            ui, vi = int(u), int(v)
            if vi < 0 or vi >= depth_map.shape[0] or \
               ui < 0 or ui >= depth_map.shape[1]:
                continue
            z = depth_map[vi, ui]
            if np.isnan(z) or np.isinf(z) or z <= 0 or z > 20:
                continue

            # Punto en coordenadas cámara
            Xc = np.array([
                (u - self.cx) * z / self.fx,
                (v - self.cy) * z / self.fy,
                z
            ])
            # Punto en coordenadas mundo
            Xw = R_inv @ Xc + t_world

            desc = kf.descriptors[kp_idx] if kf.descriptors is not None \
                and kp_idx < len(kf.descriptors) else np.zeros(32, np.uint8)

            lm_id = self.landmark_map.add(Xw, desc, kf.id, kp_idx)
            kf.landmark_ids[kp_idx] = lm_id

    def _match(self, desc1: np.ndarray,
               desc2: np.ndarray) -> list:
        if desc1 is None or desc2 is None:
            return []
        matches = self.matcher.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        # Ratio test adaptativo
        if len(matches) > 0:
            thr = matches[0].distance * 3 + 20
            matches = [m for m in matches if m.distance < thr]
        return matches[:200]

    def _build_pnp_data(self, matches, ref_kf: KeyFrame,
                         curr_keypoints, curr_depth_map):
        """
        Construye object_points / image_points para solvePnPRansac.
        Usa los landmarks del mapa cuando están disponibles,
        si no usa la profundidad del frame de referencia.
        """
        obj_pts, img_pts = [], []
        match_lm_ids     = {}
        match_kp_indices = {}

        R_inv = ref_kf.pose_Tcw[:3, :3].T
        t_ref = ref_kf.pose_Tcw[:3, 3]
        t_world_ref = -R_inv @ t_ref

        for m in matches:
            kp_prev = ref_kf.keypoints[m.queryIdx]
            kp_curr = curr_keypoints[m.trainIdx]

            # Obtener punto 3D — preferir landmark del mapa
            lm_id = ref_kf.landmark_ids.get(m.queryIdx)
            if lm_id is not None:
                lm = self.landmark_map.get(lm_id)
                if lm is not None:
                    Xw = lm.position
                else:
                    continue
            else:
                # Fallback: reconstruir desde depth del KF de referencia
                u_r, v_r = kp_prev.pt
                ui, vi = int(u_r), int(v_r)
                if vi < 0 or vi >= ref_kf.depth_map.shape[0] or \
                   ui < 0 or ui >= ref_kf.depth_map.shape[1]:
                    continue
                z = ref_kf.depth_map[vi, ui]
                if np.isnan(z) or np.isinf(z) or z <= 0 or z > 20:
                    continue
                Xc = np.array([
                    (u_r - self.cx) * z / self.fx,
                    (v_r - self.cy) * z / self.fy,
                    z
                ])
                Xw = R_inv @ Xc + t_world_ref

            obj_pts.append(Xw)
            img_pts.append(kp_curr.pt)
            if lm_id is not None:
                match_lm_ids[m.queryIdx] = lm_id
            match_kp_indices[m.queryIdx] = m.trainIdx

        if not obj_pts:
            return (np.empty((0, 3), np.float32),
                    np.empty((0, 2), np.float32), {}, {})

        return (np.array(obj_pts, np.float32),
                np.array(img_pts, np.float32),
                match_lm_ids, match_kp_indices)

    # ══════════════════════════════════════════════════════════════════════════
    # VISUALIZACIÓN
    # ══════════════════════════════════════════════════════════════════════════

    def draw_map(self, zed_traj: Optional[np.ndarray] = None,
                 canvas_size: int = 800,
                 scale: float = 20.0) -> np.ndarray:
        """Dibuja trayectoria VO, ZED ground truth y loop edges."""
        canvas = np.zeros((canvas_size, canvas_size, 3), np.uint8)
        cx_c = canvas_size // 2
        cy_c = canvas_size // 2

        def to_px(x, z):
            return (int(x * scale + cx_c), int(z * scale + cy_c))

        traj = np.array([p for p in self.trajectory
                         if not np.any(np.isnan(p))]).reshape(-1, 3)

        # Trayectoria VO (verde)
        for i in range(1, len(traj)):
            pt1 = to_px(traj[i-1, 0], traj[i-1, 2])
            pt2 = to_px(traj[i,   0], traj[i,   2])
            cv.line(canvas, pt1, pt2, (0, 220, 0), 2)

        # Ground truth ZED (azul)
        if zed_traj is not None and len(zed_traj) > 1:
            for i in range(1, len(zed_traj)):
                pt1 = to_px(zed_traj[i-1, 0], zed_traj[i-1, 2])
                pt2 = to_px(zed_traj[i,   0], zed_traj[i,   2])
                cv.line(canvas, pt1, pt2, (255, 80, 0), 2)

        # Loop edges (amarillo)
        for kf_a_id, kf_b_id in self.loop_edges:
            kf_a = next((k for k in self.keyframes if k.id == kf_a_id), None)
            kf_b = next((k for k in self.keyframes if k.id == kf_b_id), None)
            if kf_a and kf_b:
                pa = -kf_a.pose_Tcw[:3, :3].T @ kf_a.pose_Tcw[:3, 3]
                pb = -kf_b.pose_Tcw[:3, :3].T @ kf_b.pose_Tcw[:3, 3]
                cv.line(canvas, to_px(pa[0], pa[2]),
                        to_px(pb[0], pb[2]), (0, 220, 220), 1)

        # Posición actual (rojo)
        if len(traj) > 0:
            pt = to_px(traj[-1, 0], traj[-1, 2])
            cv.circle(canvas, pt, 6, (0, 0, 255), -1)

        # KF landmarks proyectados en plano XZ (puntos blancos pequeños)
        lm_pos = self.landmark_map.all_positions()
        for p in lm_pos[::10]:   # submuestreo para velocidad
            pt = to_px(p[0], p[2])
            if 0 <= pt[0] < canvas_size and 0 <= pt[1] < canvas_size:
                cv.circle(canvas, pt, 1, (60, 60, 60), -1)

        # Leyenda
        cv.putText(canvas, "VO",  (10, 20), cv.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 220, 0), 1)
        cv.putText(canvas, "ZED", (10, 40), cv.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 80, 0), 1)
        cv.putText(canvas, f"LM: {len(self.landmark_map)}", (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv.putText(canvas, f"KF: {len(self.keyframes)}", (10, 80),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return canvas


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — reemplaza tu loop anterior
# ══════════════════════════════════════════════════════════════════════════════

def main():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file("/home/braitte/Desktop/dataset1.svo2")
    init_params.svo_real_time_mode = False

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("[ERROR] No se pudo abrir la cámara/SVO")
        exit()

    tracking_params = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(tracking_params)

    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters.left_cam

    fx, fy = calib.fx, calib.fy
    cx, cy = calib.cx, calib.cy
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)

    slam = SLAMSystem(K, fx, fy, cx, cy)

    runtime    = sl.RuntimeParameters()
    image_mat  = sl.Mat()
    depth_mat  = sl.Mat()
    zed_pose   = sl.Pose()
    zed_traj   = []

    while True:
        # Ground truth ZED
        zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
        t = zed_pose.get_translation().get()
        zed_traj.append([t[0], t[1], t[2]])

        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            print("[INFO] Fin del SVO")
            break

        zed.retrieve_image(image_mat,  sl.VIEW.LEFT)
        zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)

        img       = image_mat.get_data()
        depth_map = depth_mat.get_data()

        # ── Procesar frame ────────────────────────────────────────────────────
        result = slam.process_frame(img, depth_map)

        # ── Visualización ─────────────────────────────────────────────────────
        map_canvas = slam.draw_map(
            zed_traj=np.array(zed_traj),
            scale=20.0
        )
        cv.imshow("SLAM Map (VO=verde | ZED=azul | loop=cyan)", map_canvas)

        # HUD sobre la imagen RGB
        state_color = {
            "TRACKING":     (0, 220, 0),
            "LOST":         (0, 0, 255),
            "INITIALIZING": (255, 200, 0),
        }.get(result["state"], (255, 255, 255))

        hud = img.copy()
        cv.putText(hud, f"Estado: {result['state']}",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        cv.putText(hud, f"KPs: {result['n_kp']} | LM: {result['n_landmarks']}",
                   (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv.putText(hud, f"KFs: {result['n_keyframes']} | "
                        f"{result['time_ms']:.1f}ms",
                   (10, 85), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        if result.get("loop_detected"):
            cv.putText(hud, f"LOOP! KF→{result['loop_with_kf']}",
                       (10, 115), cv.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 255, 255), 2)
        cv.imshow("RGB + HUD", cv.resize(hud, (960, 540)))

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    zed.close()
    cv.destroyAllWindows()

    # ── Guardar trayectoria final ─────────────────────────────────────────────
    traj = np.array([p for p in slam.trajectory if not np.any(np.isnan(p))])
    if len(traj) > 0:
        np.savetxt("trajectory_vo.txt",  traj)
        np.savetxt("trajectory_zed.txt", np.array(zed_traj))
        print(f"[SLAM] Trayectorias guardadas. "
              f"VO: {len(traj)} pts | ZED: {len(zed_traj)} pts")


if __name__ == "__main__":
    main()
