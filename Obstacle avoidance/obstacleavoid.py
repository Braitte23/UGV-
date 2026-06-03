import pyzed.sl as sl
import numpy as np
import cv2 as cv


# ==============================
# FUNCIÓN DE OCUPACIÓN
# ==============================
def riesgo(roi, prev_roi, threshold=3.0, alpha=0.1, beta = 0.5):
    roi_clean = roi[np.isfinite(roi)] / 1000.0  # mm → metros

    if roi_clean.size == 0:
        return 0, np.mean(prev_roi) if prev_roi is not None else 0.0
    
    d = np.mean(roi_clean)

    if prev_roi is None:
        return alpha * (d < threshold), d
    
    d_prev = prev_roi
    dd = d  - d_prev #derivada discreta

    R = alpha * (d < threshold) +  beta * (-dd)

    return R, d
    #q10 = np.percentile(roi_clean, 10)
    #p = np.mean(roi_clean <= threshold)

    #return int((q10 <= threshold) and (p > alpha))


# ==============================
# INICIALIZACIÓN ZED
# ==============================
zed = sl.Camera()

init_params = sl.InitParameters()
init_params.set_from_svo_file("/home/braitte/Desktop/record.svo2")
init_params.svo_real_time_mode = False

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Error al abrir el SVO")
    exit()

runtime = sl.RuntimeParameters()
image = sl.Mat()
depth = sl.Mat()

#==============================
# MEMORIA GLOBAL 
#==============================

prev_left = None
prev_center = None
prev_right = None


# ==============================
# LOOP PRINCIPAL
# ==============================
while True:

    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:

        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        img_np = image.get_data()
        depth_np = depth.get_data()

        # ==============================
        # ROI GLOBAL
        # ==============================
        H, W = depth_np.shape

        v1 = int(0.3 * H)
        v2 = int(0.6 * H)
        u1 = int(0.3 * W)
        u2 = int(0.7 * W)

        # ==============================
        # DIVISIÓN EN 3 REGIONES
        # ==============================
        w = u2 - u1

        uC1 = u1 + w // 3
        uC2 = u1 + 2 * w // 3

        roi_left   = depth_np[v1:v2, u1:uC1]
        roi_center = depth_np[v1:v2, uC1:uC2]
        roi_right  = depth_np[v1:v2, uC2:u2]

        # ==============================
        # ESTADOS BINARIOS
        # ==============================
        R_L, dL = riesgo(roi_left, prev_left)
        R_C, dC = riesgo(roi_center, prev_center)
        R_R, dR= riesgo(roi_right, prev_right)

        prev_left = dL
        prev_center = dC
        prev_right = dR

        #Decision por arg min

        riesgos = np.array([R_L, R_C, R_R])
        idx = np.argmin(riesgos)

        # ==============================
        # POLÍTICA DE CONTROL
        # ==============================
        if idx == 0:
            accion = "left"
        elif idx == 1:
            accion = "forward"
        else:
            accion = "right"
        # ==============================
        # VISUALIZACIÓN
        # ==============================
        img_cv = cv.cvtColor(img_np, cv.COLOR_RGBA2BGR)

        texto = f"L:{R_L} C:{R_C} R:{R_R} -> {accion}"

        cv.putText(
            img_cv,
            texto,
            (50, 50),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv.imshow("ZED", img_cv)

        print(R_L, R_C, R_R, accion)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

zed.close()
