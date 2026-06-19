import pyzed.sl as sl
import numpy as np
import cv2 as cv
import serial 

esp32 = serial.Serial("/dev/ttyUSB1", 115200)
ultimo_comando = None

# ==============================
# FUNCIÓN DE OCUPACIÓN
# ==============================
def ocupado(roi, threshold=3.0, alpha=0.1):
    roi_clean = roi[np.isfinite(roi)] / 1000.0  # mm → metros

    if roi_clean.size == 0:
        return 0

    q10 = np.percentile(roi_clean, 10)
    p = np.mean(roi_clean <= threshold)

    return int((q10 <= threshold) and (p > alpha))


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
        sL = ocupado(roi_left)
        sC = ocupado(roi_center)
        sR = ocupado(roi_right)

        # ==============================
        # POLÍTICA DE CONTROL
        # ==============================

        # Todo libre → seguir recto
        if not sL and not sC and not sR:
            comando = 'S'
            accion = "straight"

        # Obstáculo al frente
        elif sC:

            # izquierda libre
            if not sL and sR:
                comando = 'L'
                accion = "left"

            # derecha libre
            elif not sR and sL:
                comando = 'R'
                accion = "right"

            # ambas libres: escoger izquierda por defecto
            elif not sL and not sR:
                comando = 'L'
                accion = "left"

            # encerrado
            else:
                comando = 'S'
                accion = "blocked"

        # obstáculo sólo a la izquierda
        elif sL and not sR:
            comando = 'R'
            accion = "right"

        # obstáculo sólo a la derecha
        elif sR and not sL:
            comando = 'L'
            accion = "left"

        # obstáculos laterales pero centro libre
        else:
            comando = 'S'
            accion = "straight"

        # ==============================
        # VISUALIZACIÓN
        # ==============================
        img_cv = cv.cvtColor(img_np, cv.COLOR_RGBA2BGR)

        # ROI izquierda
        cv.rectangle(
            img_cv,
            (u1, v1),
            (uC1, v2),
            (0, 255, 0),
            2
        )

        # ROI centro
        cv.rectangle(
            img_cv,
            (uC1, v1),
            (uC2, v2),
            (255, 0, 0),
            2
        )

        # ROI derecha
        cv.rectangle(
            img_cv,
            (uC2, v1),
            (u2, v2),
            (0, 0, 255),
            2
        )

        texto = f"L:{sL} C:{sC} R:{sR} -> {accion}"

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

        if comando != ultimo_comando:
            esp32.write(comando.encode())
            ultimo_comando = comando

        print(
            f"L={sL} C={sC} R={sR} -> {accion} ({comando})"
        )

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

zed.close()