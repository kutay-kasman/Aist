import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

# ---------------- CONFIG ----------------
ROOT_DIR = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split"  # Ana klasör
FPS = 10
CHUNK_DURATION = 4
KEYPOINT_DIM = 3
# ----------------------------------------
def process_video(video_path, genre_label, save_pose_dir):
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
                keypoints = [[p.x, p.y, p.z] for p in lm]

                # ✅ Center on hips (23, 24)
                if not any(np.isnan(keypoints[23])) and not any(np.isnan(keypoints[24])):
                    hip_center = np.mean([keypoints[23], keypoints[24]], axis=0)
                    keypoints = [[x - hip_center[0], y - hip_center[1], z - hip_center[2]] for (x, y, z) in keypoints]

                    # ✅ Optional: Normalize size (e.g., based on shoulder distance or max joint distance)
                    joint_dists = [np.linalg.norm([x, y, z]) for (x, y, z) in keypoints if not any(np.isnan([x, y, z]))]
                    max_dist = max(joint_dists) if joint_dists else 1.0
                    if max_dist > 1e-3:
                        keypoints = [[x / max_dist, y / max_dist, z / max_dist] for (x, y, z) in keypoints]

            else:
                keypoints = [[np.nan, np.nan, np.nan]] * 33  

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

        pose_out = os.path.join(save_pose_dir, f"{chunk_name}_3d_pose.npz")
        np.savez(pose_out, keypoints=pose_chunk)

def main():
    for genre_folder in os.listdir(ROOT_DIR):
        genre_path = os.path.join(ROOT_DIR, genre_folder)
        if not os.path.isdir(genre_path):
            continue

        video_files = [f for f in os.listdir(genre_path) if f.lower().endswith(".mp4")]
        for video_file in tqdm(video_files, desc=f"{genre_folder}", ncols=100):
            video_path = os.path.join(genre_path, video_file)
            video_name = os.path.splitext(video_file)[0]

            # 📁 Save folders: ROOT/GENRE/VIDEO_NAME/pose
            out_base = os.path.join(genre_path, video_name)
            pose_dir = os.path.join(out_base, "3d_pose")
            os.makedirs(pose_dir, exist_ok=True)

            process_video(video_path, genre_label=genre_folder.lower(), save_pose_dir=pose_dir)

    print("✅ All videos processed.")


if __name__ == "__main__":
    main()
