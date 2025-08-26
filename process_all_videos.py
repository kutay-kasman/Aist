import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

# ---------------- CONFIG ----------------
ROOT_DIR = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split"  # Ana klasör
FPS = 10
CHUNK_DURATION = 4
KEYPOINT_DIM = 2
# ----------------------------------------

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

def process_video(video_path, genre_label, save_pose_dir, save_angle_dir):
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(round(orig_fps / FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pose_model = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    frame_idx = 0
    chunk = []
    all_chunks = []

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
                keypoints = [[np.nan, np.nan]] * 33

            chunk.append(keypoints)

            if len(chunk) == FPS * CHUNK_DURATION:
                all_chunks.append(np.array(chunk))
                chunk = []

        frame_idx += 1

    cap.release()
    pose_model.close()

    for i, pose_chunk in enumerate(all_chunks):
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        chunk_name = f"{base_name}_{i:03d}"

        # Save pose
        pose_out = os.path.join(save_pose_dir, f"{chunk_name}_pose.npz")
        np.savez(pose_out, keypoints=pose_chunk)

        # Calculate angles
        angles = []
        for frame in pose_chunk:
            frame_angles = []
            for (a, b, c) in ANGLE_KEYPOINTS.values():
                angle = compute_angle(frame[a], frame[b], frame[c])
                frame_angles.append(angle)
            angles.append(frame_angles)
        angles = np.array(angles)

        angle_out = os.path.join(save_angle_dir, f"{chunk_name}_angles.npz")
        np.savez(angle_out, angles=angles, label=genre_label)

def main():
    for genre_folder in os.listdir(ROOT_DIR):
        genre_path = os.path.join(ROOT_DIR, genre_folder)
        if not os.path.isdir(genre_path):
            continue

        for video_file in os.listdir(genre_path):
            if not video_file.lower().endswith(".mp4"):
                continue

            video_path = os.path.join(genre_path, video_file)
            video_name = os.path.splitext(video_file)[0]

            # 📁 Save folders: ROOT/GENRE/VIDEO_NAME/pose + /angles
            out_base = os.path.join(genre_path, video_name)
            pose_dir = os.path.join(out_base, "pose")
            angle_dir = os.path.join(out_base, "angles")
            os.makedirs(pose_dir, exist_ok=True)
            os.makedirs(angle_dir, exist_ok=True)

            print(f"▶️ Processing {video_file}...")
            process_video(video_path, genre_label=genre_folder.lower(), save_pose_dir=pose_dir, save_angle_dir=angle_dir)

    print("✅ All videos processed.")

if __name__ == "__main__":
    main()
