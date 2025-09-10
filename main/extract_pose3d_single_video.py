import cv2
import mediapipe as mp
import numpy as np
import os

# ---------------- CONFIG ----------------
VIDEO_PATH = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d04_mBR0_ch01.mp4" 
SAVE_PATH = r"C:\Users\kutay\OneDrive\Desktop\example_pose3d.npz"
FPS = 10
CHUNK_DURATION = 4
# ----------------------------------------

# Init MediaPipe Pose (3D)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=2,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(VIDEO_PATH)
orig_fps = cap.get(cv2.CAP_PROP_FPS)
interval = int(round(orig_fps / FPS))

frame_idx = 0
chunk = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % interval == 0:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark
            # Center by hip midpoint
            hip_center = np.array([
                (landmarks[23].x + landmarks[24].x) / 2,  # x
                (landmarks[23].y + landmarks[24].y) / 2,  # y
                (landmarks[23].z + landmarks[24].z) / 2   # z
            ])

            keypoints = [[lm.x - hip_center[0],
                        lm.y - hip_center[1],
                        lm.z - hip_center[2]] for lm in landmarks]

        else:
            keypoints = [[np.nan, np.nan, np.nan]] * 33

        chunk.append(keypoints)

        # Stop after 4 seconds worth of frames
        if len(chunk) == FPS * CHUNK_DURATION:
            break

    frame_idx += 1

cap.release()
pose.close()

# Convert to NumPy array first
chunk = np.array(chunk)  # Shape: (40, 33, 3)

# Normalize per frame (optional)
for i in range(chunk.shape[0]):
    frame = chunk[i]
    valid = ~np.isnan(frame).any(axis=1)
    if np.any(valid):
        max_dist = np.max(np.linalg.norm(frame[valid], axis=1))
        if max_dist > 0:
            chunk[i] = frame / max_dist


np.savez(SAVE_PATH, keypoints=chunk)

print(f"✅ Saved 3D pose to: {SAVE_PATH}")
print(f"✔️ Shape: {chunk.shape} (frames, joints, xyz)")
