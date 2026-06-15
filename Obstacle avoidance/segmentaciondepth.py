import pyzed.sl as sl
import numpy as np
import cv2 as cv
import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation


# ==============================
# MODELO SEGMENTACIÓN (AI REAL)
# ==============================
model_name = "nvidia/segformer-b0-finetuned-ade-512-512"

processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForSemanticSegmentation.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()


def segmentador(rgb):
    # SegFormer espera RGB
    rgb_in = cv.cvtColor(rgb, cv.COLOR_BGR2RGB)

    inputs = processor(images=rgb_in, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # (1, classes, h, w)
    mask = torch.argmax(logits, dim=1)[0].cpu().numpy()

    return mask


# ==============================
# OCUPACIÓN
# ==============================
def ocupado(roi, threshold=3.0, alpha=0.1):
    roi_clean = roi[np.isfinite(roi)] / 1000.0  # mm ? m

    if roi_clean.size == 0:
        return 0

    q10 = np.percentile(roi_clean, 10)
    p = np.mean(roi_clean <= threshold)

    return int((q10 <= threshold) and (p > alpha))


# ==============================
# OBSTÁCULOS (ADE20K simple)
# ==============================
def obstacle_mask(mask):
    return mask != 0


# ==============================
# ZED INIT
# ==============================
zed = sl.Camera()

init_params = sl.InitParameters()
init_params.set_from_svo_file("/home/braitte/Desktop/record.svo2")
init_params.svo_real_time_mode = False

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Error al abrir SVO")
    exit()

runtime = sl.RuntimeParameters()
image = sl.Mat()
depth = sl.Mat()


# ==============================
# LOOP PRINCIPAL
# ==============================
while True:

    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:

        # =========================
        # CAPTURA
        # =========================
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        img_np = image.get_data()
        depth_np = depth.get_data()

        img_cv = cv.cvtColor(img_np, cv.COLOR_RGBA2BGR)

        # =========================
        # SEGMENTACIÓN (LOW RES)
        # =========================
        mask = segmentador(img_cv)
        mask_np = obstacle_mask(mask).astype(np.uint8)

        # =========================
        # FIX CRÍTICO: RESIZE A RGB ORIGINAL
        # =========================
        mask_np = cv.resize(
            mask_np,
            (img_cv.shape[1], img_cv.shape[0]),
            interpolation=cv.INTER_NEAREST
        ).astype(bool)

        # =========================
        # VISUAL SEGMENTACIÓN
        # =========================
        overlay = img_cv.copy()
        overlay[mask_np] = (0, 0, 255)

        img_seg = cv.addWeighted(img_cv, 0.7, overlay, 0.3, 0)

        mask_vis = (mask_np.astype(np.uint8) * 255)

        # =========================
        # DEPTH FILTRADO
        # =========================
        depth_filtrado = np.where(mask_np, depth_np, np.nan)

        # =========================
        # ROI
        # =========================
        H, W = depth_np.shape

        v1 = int(0.3 * H)
        v2 = int(0.6 * H)
        u1 = int(0.3 * W)
        u2 = int(0.7 * W)

        w = u2 - u1
        uC1 = u1 + w // 3
        uC2 = u1 + 2 * w // 3

        roi_left   = depth_filtrado[v1:v2, u1:uC1]
        roi_center = depth_filtrado[v1:v2, uC1:uC2]
        roi_right  = depth_filtrado[v1:v2, uC2:u2]

        # =========================
        # OCUPACIÓN
        # =========================
        sL = ocupado(roi_left)
        sC = ocupado(roi_center)
        sR = ocupado(roi_right)

        # =========================
        # CONTROL SIMPLE
        # =========================
        if not sL and not sC and not sR:
            accion = "forward"

        elif sC:
            if not sL:
                accion = "left"
            elif not sR:
                accion = "right"
            else:
                accion = "back"

        elif sL and not sR:
            accion = "right"

        elif sR and not sL:
            accion = "left"

        else:
            accion = "forward"

        # =========================
        # DEBUG VISUAL
        # =========================
        texto = f"L:{sL} C:{sC} R:{sR} -> {accion}"

        cv.putText(
            img_seg,
            texto,
            (50, 50),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # =========================
        # SHOW WINDOWS
        # =========================
        cv.imshow("RGB + SEGMENTACION AI", img_seg)
        cv.imshow("MASK", mask_vis)

        print(sL, sC, sR, accion)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

zed.close()
cv.destroyAllWindows()
