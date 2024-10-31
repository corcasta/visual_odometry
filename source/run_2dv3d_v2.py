import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from kitti_reader import DatasetReaderKITTI   

def magic(frame_old, frame, K, trans_of2w=np.eye(4), scale=1):
    # Feature detector and optical flow parameters
    detector = cv2.ORB_create()
    #@lk_params = dict(winSize=(21, 21), maxLevel=3,
    #@                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    # Detect features in the first frame
    kp0 = detector.detect(frame_old, None)
    old_pts = np.array([kp.pt for kp in kp0], dtype=np.float32).reshape(-1, 2)
    old_pts = np.squeeze(old_pts)

    # Track points from frame 1 to frame 2
    new_pts, st, err = cv2.calcOpticalFlowPyrLK(prevImg=frame_old, nextImg=frame, prevPts=old_pts, nextPts=None)
    old_pts = old_pts[st[:,0] == 1,:]   # Shape (N, 2)
    new_pts = new_pts[st[:,0] == 1,:]   # Shape (N, 2)
    #print(f"old_pts: {type(old_pts)} \t len: {len(old_pts)}")
    #print(f"new_pts: {type(new_pts)} \t len: {len(new_pts)}")
    
    # Estimate the essential matrix and get initial pose
    E, mask = cv2.findEssentialMat(new_pts, old_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    # Rotation and translation will be from new frame to old frame
    _, R, t, mask = cv2.recoverPose(E, new_pts, old_pts, K)
    #old_pts = old_pts[mask[:,0] == 255,:]   # Extracting only inliers used for recovering Pose
    #new_pts = new_pts[mask[:,0] == 255,:]   # Extracting only inliers used for recovering Pose
    trans_nf2of = np.concatenate((np.concatenate((R, t), axis=1), [[0,0,0,1]]), axis=0)
    # We need the inv because new_pts are w.r.t the old frame, this is what we need for triangulation
    trans_of2nf = np.linalg.inv(trans_nf2of)

    # Triangulate initial 3D points
    projMtx0 = np.concatenate((K, [[0],[0],[1]]), axis=1) @ np.eye(4)   # Shape (3,4)
    projMtx1 = np.concatenate((K, [[0],[0],[1]]), axis=1) @ trans_of2nf # Shape (3,4)
    
    # This points are returned in world coord system (this world frame is based on the first camera/first frame)
    pts4D_hom = cv2.triangulatePoints(projMtx0, projMtx1, old_pts.T, new_pts.T) # Shape (4, N)
    trans_nf2w = trans_of2w @ trans_nf2of
    pts4D_hom_w = trans_nf2w @ pts4D_hom
    
    # This points are returned in world coord system
    pts3D = pts4D_hom_w[:3, :] / pts4D_hom_w[-1, :] # Shape (3, N)
    #print(pts3D[:,0])
    
    # Shape (N, 2), # Shape (N, 3), # Shape (4, 4)
    return new_pts, pts3D.T, trans_nf2w, trans_nf2of

if __name__ == "__main__":
    dataset_reader = DatasetReaderKITTI("/home/corcasta/Downloads/data_odometry_gray/dataset/sequences/00/")

    trans_list = []
    kitti_positions, track_positions, scales = [], [], []
    trans_init = np.eye(4)

    kitti_pos, kitti_scale = dataset_reader.readGroundtuthPosition(0)
    track_positions.append(kitti_scale*trans_init[:3,-1])
    kitti_positions.append(kitti_pos)
    scales.append(kitti_scale)
    trans_list.append(trans_init)

    # Load camera intrinsic matrix (e.g., KITTI sequence 00)
    K = dataset_reader.readCameraMatrix()

    # Initialize with the first two frames
    frame_old = cv2.cvtColor(dataset_reader.readFrame(0), cv2.IMREAD_GRAYSCALE)
    frame = cv2.cvtColor(dataset_reader.readFrame(1), cv2.IMREAD_GRAYSCALE)

    # Shape (N, 2), # Shape (N, 3), # Shape (4, 4)
    old_pts, old_pts3D, trans_of2w, trans_local = magic(frame_old, frame, K)

    # Initialize pose (identity for the first frame)
    frame_old = np.copy(frame)
    kitti_pos, kitti_scale = dataset_reader.readGroundtuthPosition(1)
    kitti_positions.append(kitti_pos)
    scales.append(kitti_scale)
    trans_list.append(trans_of2w)

    # Process each new frame
    for frame_no in range(2, 500):
        print(frame_no)
        frame = cv2.cvtColor(dataset_reader.readFrame(frame_no), cv2.IMREAD_GRAYSCALE)
        
        # Track features from previous frame to current frame
        new_pts, st, err = cv2.calcOpticalFlowPyrLK(prevImg=frame_old, nextImg=frame, prevPts=old_pts, nextPts=None)
        old_pts = old_pts[st[:,0] == 1,:]   # Shape (N, 2)
        new_pts = new_pts[st[:,0] == 1,:]   # Shape (N, 2)
        old_pts3D = old_pts3D[st[:,0] == 1,:] # Shape (N, 3)
        
        # Use PnP to estimate pose
        # rotation from 3dpoints coord frame to camera frame
        _, rvec, tvec, inliers = cv2.solvePnPRansac(old_pts3D, new_pts, K, None) 
        R, _ = cv2.Rodrigues(rvec)
        trans_w2nf = np.concatenate((np.concatenate((R, tvec), axis=1), [[0,0,0,1]]), axis=0)
        trans_nf2w = np.linalg.inv(trans_w2nf)
        
        # Append current position to track_positions
        kitti_pos, kitti_scale = dataset_reader.readGroundtuthPosition(frame_no)
        kitti_positions.append(kitti_pos)
        scales.append(kitti_scale)
        trans_list.append(trans_nf2w)
        
        # Occasionally triangulate new points if enough parallax is detected
        if frame_no % 1 == 0:
            # Shape (N, 2), # Shape (N, 3), # Shape (4, 4)
            old_pts, old_pts3D, trans_of2w, trans_local = magic(frame_old, frame, K, trans_of2w)
            frame_old = np.copy(frame)
    
    # This is done just to plot the results and scaled the results properly        
    track_scaled_positions = []
    trans_scaled_co2w = trans_list[0]
    for i, trans_cn2w in enumerate(trans_list[1:], 1):
        trans_prev_w2co = np.linalg.inv(trans_list[i-1])
        trans_cn2co = trans_prev_w2co @ trans_cn2w
        trans_cn2co[:3,-1] *= scales[i]
        trans_scaled_co2w = trans_scaled_co2w @ trans_cn2co
        track_scaled_positions.append(trans_scaled_co2w[:3,-1])
        

    track_scaled_positions = np.asarray(track_scaled_positions)
    kitti_positions = np.asarray(kitti_positions)
    plt.plot(track_scaled_positions[:,0], track_scaled_positions[:,2])
    plt.plot(kitti_positions[:,0], kitti_positions[:,2])
    plt.pause(30)