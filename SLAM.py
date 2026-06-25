import pyzed.sl as sl
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

zed = sl.Camera()

init_params = sl.InitParameters()
init_params.set_from_svo_file("/home/braitte/Desktop/dataset1.svo2")
init_params.svo_real_time_mode = False

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    exit()
tracking_params = sl.PositionalTrackingParameters()
zed.enable_positional_tracking(tracking_params)


cam_info = zed.get_camera_information()
calib = cam_info.camera_configuration.calibration_parameters.left_cam

fx = calib.fx
fy = calib.fy
cx = calib.cx
cy = calib.cy

runtime = sl.RuntimeParameters()

image = sl.Mat()
depth = sl.Mat()

# Creamos ORB una sola vez
orb = cv.ORB_create(nfeatures=1000)

matcher = cv.BFMatcher(
    cv.NORM_HAMMING,
    crossCheck=True
)

prev_frame = None
prev_keypoints = None
prev_descriptors = None
prev_depth = None

R_total = np.eye(3, dtype=np.float32)
t_total = np.zeros((3,1), dtype=np.float32)

trajectory = []

zed_pose = sl.Pose()
zed_translation = []

while True:

    zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)

    t = zed_pose.get_translation().get()
    zed_translation.append([t[0], t[1], t[2]])

    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:

        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        img = image.get_data()
        print(f"this is the image shape {img.shape}")

        depth_map = depth.get_data()
        print(f"this is the depth shape {depth_map.shape}")

        img_bw = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)

        keypoints, descriptors = orb.detectAndCompute(
            img_bw,
            None
        )

        if prev_descriptors is None:

            prev_img = img.copy()
            prev_keypoints = keypoints
            prev_descriptors = descriptors
            prev_depth = depth_map.copy()

            continue

        matches = matcher.match(
            prev_descriptors,
            descriptors
        )

        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:100]

        points_prev = []
        points_curr = []

        object_points = []
        image_points = []

        for m in matches:

            kp_prev = prev_keypoints[m.queryIdx]
            kp_curr = keypoints[m.trainIdx]

            u_prev, v_prev = kp_prev.pt
            u_curr, v_curr = kp_curr.pt

            u_prev_int = int(u_prev)
            v_prev_int = int(v_prev)

            z = prev_depth[v_prev_int, u_prev_int]

            if np.isnan(z) or np.isinf(z) or z <= 0:
                continue

            points_prev.append([u_prev, v_prev])
            points_curr.append([u_curr, v_curr])

            X = (u_prev - cx) * z / fx
            Y = (v_prev - cy) * z / fy
            Z = z

            object_points.append([X, Y, Z])
            image_points.append([u_curr, v_curr])

        K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)

        object_points = np.array(
            object_points,
            dtype=np.float32
        )

        image_points = np.array(
            image_points,
            dtype=np.float32
        )

        if len(object_points) > 10:

            print("object_points:", len(object_points))
            print("image_points:", len(image_points))

            succes,rvec,tvec,inliers = cv.solvePnPRansac(
                object_points,
                image_points,
                K, 
                None
            )
#rvec = rotacion
#tvec = traslacion
            if succes:

                R,_ = cv.Rodrigues(rvec)
                t_total = t_total + R_total @ tvec
                trajectory.append(t_total.copy())
                R_total = R @ R_total
                print("Pose estimada")
                print("Global position:")
                print(t_total.ravel())

                print("rvec:\n", rvec)
                print("tvec:\n", tvec)
                print("inliers:", len(inliers))


                vo = np.array(trajectory).reshape(-1,3)
                zed_traj = np.array(zed_translation).reshape(-1, 3)

                if len(vo) > 5 and len(zed_traj) > 5:
                    x = vo[:, 0]
                    z = vo[:, 2]   # usamos X-Z (plano horizontal típico)

                    canvas = np.zeros((700, 700, 3), dtype=np.uint8)

                    scale = 20  # ajustable

                    for i in range(1, len(x)):
                        pt1 = (int(x[i-1]*scale + 300), int(z[i-1]*scale + 300))
                        pt2 = (int(x[i]*scale + 300), int(z[i]*scale + 300))
                        cv.line(canvas, pt1, pt2, (0, 255, 0), 2)

                    for i in range(1, len(zed_traj)):
                        pt1 = (int(zed_traj[i-1,0]*scale + 300), int(zed_traj[i-1,2]*scale + 300))
                        pt2 = (int(zed_traj[i,0]*scale + 300), int(zed_traj[i,2]*scale + 300))
                        cv.line(canvas, pt1, pt2, (255,0,0), 2)

                    cv.imshow("VO vs ZED", canvas)

                    

        m = matches[0]

        print(m.queryIdx)
        print(m.trainIdx)
        print(m.distance)

        kp = prev_keypoints[m.queryIdx]
        print(kp.pt)
        print("This is the length:", kp)

        print(f"this is the {len(matches)}")

        print(
            f"depth min: {np.nanmin(depth_map)}, "
            f"max: {np.nanmax(depth_map)}, "
            f"mean: {np.nanmean(depth_map)}, "
            f"std: {np.nanstd(depth_map)}"
        )

        print(f"Number of ORB: {len(keypoints)}")

        img_features = cv.drawKeypoints(
            img,
            keypoints,
            None,
            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        img_matches = cv.drawMatches(
            prev_img,
            prev_keypoints,
            img,
            keypoints,
            matches[:100],
            None
        )

        img_features = cv.resize(
            img_features,
            (1000, 650)
        )

        img_matches = cv.resize(
            img_matches,
            (920, 720)
        )

        cv.imshow(
            "ORB Features",
            img_features
        )

        cv.imshow(
            "Matches",
            img_matches
        )

        cv.imshow(
            "RGB",
            img
        )

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

    prev_img = img.copy()
    prev_keypoints = keypoints
    prev_descriptors = descriptors
    prev_depth = depth_map.copy()

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