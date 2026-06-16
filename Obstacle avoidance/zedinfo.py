import pyzed.sl as sl
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

zed = sl.Camera()

init_params = sl.InitParameters()
init_params.set_from_svo_file("/home/braitte/Desktop/record.svo2")
init_params.svo_real_time_mode = False

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    exit()

runtime = sl.RuntimeParameters()

image = sl.Mat()
depth = sl.Mat()

#creamos orb una sola vez
orb = cv.ORB_create(nfeatures=1000)


while True:

    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:

        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        img = image.get_data()
        print(f" this is the image shape {img.shape}")
        depth_map = depth.get_data()
        print(f" this is the depth shape {depth_map.shape}")

        img_bw = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)

        keypoints, descriptors = orb.detectAndCompute(
            img_bw,
            None
        )

        print(f"depth min: {np.nanmin(depth_map)}, max:{np.nanmax(depth_map)}, mean: {np.nanmean(depth_map)}, std: {np.nanstd(depth_map)}" )

        print(f"Number of ORB: {len(keypoints)}")

        img_features = cv.drawKeypoints(
            img,
            keypoints,
            None,
            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        img_features = cv.resize(
            img_features,(1000,650))
        
        cv.imshow(
            "ORB Features",
            img_features
        )


        cv.imshow("RGB", img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break


zed.close()
cv.destroyAllWindows()



"""
ORB
↓
Feature Matching
↓
Tomar depth del Frame_t
↓
Generar puntos 3D
↓
solvePnPRansac
↓
Pose relativa

"""