import cv2
import plyfile
import numpy as np
import matplotlib.pyplot as plt

from kitti_reader import DatasetReaderKITTI
from feature_tracking import FeatureTracker
from utils import drawFrameFeatures, updateTrajectoryDrawing, savePly

if __name__ == "__main__":
    tracker = FeatureTracker()
    detector = cv2.GFTTDetector_create() #cv2.SIFT_create()
    dataset_reader = DatasetReaderKITTI("/home/corcasta/Downloads/data_odometry_gray/dataset/sequences/00/")

    K = dataset_reader.readCameraMatrix()

    prev_points = tuple(np.empty(0))
    prev_frame_BGR = dataset_reader.readFrame(0)
    kitti_positions, track_positions = [], []
    camera_rot, camera_pos = np.eye(3), np.zeros((3,1))

    plt.show()

    # Process next frames
    for frame_no in range(1, 2200):
        curr_frame_BGR = dataset_reader.readFrame(frame_no)
        prev_frame = cv2.cvtColor(prev_frame_BGR, cv2.COLOR_BGR2GRAY)
        curr_frame = cv2.cvtColor(curr_frame_BGR, cv2.COLOR_BGR2GRAY)

        # Feature detection & filtering
        #if len(prev_points) <= 9: 
        if frame_no == 1 or frame_no % 1 == 0:
            prev_points = detector.detect(prev_frame)
            prev_points = cv2.KeyPoint_convert(prev_points)#sorted(prev_points, key = lambda p: p.response, reverse=True))
            
            # This if statement is kind of a damper, it adds the previous tracked points
            # to track them again, but since the current frame is ahead of where the prev points
            # are supposed to be it works as a break since lk is trying to predict the best motion of them
            # they are a little bit behind on where they are supposed to be so it makes it more stable
            # ans less prone to noise
            if frame_no > 1 and frame_no % 5 == 0:
                prev_points = np.concatenate((prev_points, prev_points_np), axis=0)
                print(f"DEBUG: {len(prev_points)}")
            
        # Feature tracking (optical flow)
        prev_points, curr_points, _ = tracker.trackFeatures(prev_frame, curr_frame, prev_points, removeOutliers=True)
        print (f"{len(curr_points)} features left after feature tracking.")

        
        # Essential matrix, pose estimation
        E, mask = cv2.findEssentialMat(curr_points, prev_points, K, cv2.RANSAC, 0.99, 1.0, None)
        prev_points = np.array([pt for (idx, pt) in enumerate(prev_points) if mask[idx] == 1])#.astype(int)
        curr_points = np.array([pt for (idx, pt) in enumerate(curr_points) if mask[idx] == 1])#.astype(int)
        _, R, T, _ = cv2.recoverPose(E, curr_points, prev_points, K)
        print(f"prev_points: {len(prev_points)} features left after pose estimation.")
        print(f"curr_points: {len(curr_points)} features left after pose estimation.")
        
        # Read groundtruth translation T and absolute scale for computing trajectory
        kitti_pos, kitti_scale = dataset_reader.readGroundtuthPosition(frame_no)
        if kitti_scale <= 0.1:
            continue

        camera_pos = camera_pos + kitti_scale * camera_rot.dot(T)
        camera_rot = R.dot(camera_rot)

        kitti_positions.append(kitti_pos)
        track_positions.append(camera_pos)
        updateTrajectoryDrawing(np.array(track_positions), np.array(kitti_positions))
        drawFrameFeatures(curr_frame, prev_points.astype(int), curr_points.astype(int), frame_no)

        if cv2.waitKey(1) == ord('q'):
            break
            
        prev_points_np = curr_points
        prev_frame_BGR =  curr_frame_BGR

    cv2.destroyAllWindows()

