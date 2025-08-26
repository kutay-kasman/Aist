import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

# ---------------- CONFIG ----------------
VIDEO_PATH = r"C:\Users\kutay\OneDrive\Desktop\Computer Vision\AistDataset\gBR_sFM_c01_d04_mBR0_ch01.mp4"
GENRE_LABEL = "break"
FPS = 10
CHUNK_DURATION = 4
BASE_DIR = r"C:\Users\kutay\OneDrive\Desktop\Computer Vision\AistDataset"
KEYPOINT_DIM = 2
# ----------------------------------------

# Extract video name without extension
video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]

# Final directories:
pose_dir = os.path.join(BASE_DIR, GENRE_LABEL, video_name, "pose")
angle_dir = os.path.join(BASE_DIR, GENRE_LABEL, video_name, "angles")

os.makedirs(pose_dir, exist_ok=True)
os.makedirs(angle_dir, exist_ok=True)

# 8 joint angle definitions (triplets)
ANGLE_KEYPOINTS = {
    "Elbow (L)":  [11, 13, 15],
    "Elbow (R)":  [12, 14, 16],
    "Knee (L)":   [23, 25, 27],
    "Knee (R)":   [24, 26, 28],
    "Shoulder (L)": [13, 11, 23],
    "Shoulder (R)": [14, 12, 24],
    "Hip (L)":    [11, 23, 25],
    "Hip (R)":    [12, 24, 26],
}

def compute_angle(a, b, c):
    if np.any(np.isnan([a, b, c])):
        return np.nan
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

# ---------- STEP 1: Load and Process Video ----------
cap = cv2.VideoCapture(VIDEO_PATH)
orig_fps = cap.get(cv2.CAP_PROP_FPS)
interval = int(round(orig_fps / FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

pose_model = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

all_chunks = []
frame_idx = 0
chunk = []

print(f"🎞 Video FPS: {orig_fps}, Total Frames: {total_frames}")
print("🔄 Extracting pose keypoints...")
pbar = tqdm(total=total_frames)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % interval == 0:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose_model.process(img_rgb)

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark
            keypoints = [[p.x, p.y] for p in lm]
        else:
            keypoints = [[np.nan, np.nan]] * 33  # no detection → fill with NaNs

        chunk.append(keypoints)

        if len(chunk) == FPS * CHUNK_DURATION:
            all_chunks.append(np.array(chunk))  # shape: (40, 33, 2)
            chunk = []

    frame_idx += 1
    pbar.update(1)

cap.release()
pbar.close()
pose_model.close()

print(f"✅ Got {len(all_chunks)} valid chunks of 4s")

# ---------- STEP 2: Save Poses & Angles ----------
for i, pose_chunk in enumerate(all_chunks):
    chunk_id = f"{GENRE_LABEL}_{i:03d}"

    # Save pose keypoints
    np.savez(os.path.join(pose_dir, f"{chunk_id}_pose.npz"), keypoints=pose_chunk)

    # Compute angles
    angles = []
    for frame in pose_chunk:
        frame_angles = []
        for (a, b, c) in ANGLE_KEYPOINTS.values():
            angle = compute_angle(frame[a], frame[b], frame[c])
            frame_angles.append(angle)
        angles.append(frame_angles)
    angles = np.array(angles)  # shape: (40, 8)

    # Save angles with label
    np.savez(os.path.join(angle_dir, f"{chunk_id}_angles.npz"), angles=angles, label=GENRE_LABEL)

print(f"🎉 All .npz files saved under: {os.path.join(BASE_DIR, GENRE_LABEL, video_name)}")
