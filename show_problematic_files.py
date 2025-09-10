import os
import numpy as np
import pandas as pd

# 17 keypoint isimleri (MediaPipe formatında varsayıyoruz)
KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

def count_problematic_keypoints(seq):
    if seq.shape[1] < 51:
        return 0, []

    poses = seq[:, :51].reshape(-1, 17, 3)
    nan_mask = np.isnan(poses)
    inf_mask = np.isinf(poses)
    bad_mask = nan_mask | inf_mask

    bad_keypoints_per_frame = bad_mask.any(axis=2)  # (T, 17)
    bad_keypoints = bad_keypoints_per_frame.any(axis=0)  # (17,)
    count = bad_keypoints.sum()

    problem_names = [KEYPOINT_NAMES[i] for i in range(17) if bad_keypoints[i]]
    return count, problem_names

def analyze_npz_folder(folder_path, save_csv=False, split_name=""):
    data = []
    for filename in os.listdir(folder_path):
        if not filename.endswith(".npz"):
            continue

        path = os.path.join(folder_path, filename)
        try:
            npz = np.load(path)
            seq = npz["seq"]

            count, problem_names = count_problematic_keypoints(seq)

            if count > 0:
                data.append({
                    "File": filename,
                    "Problematic Keypoints": count,
                    "Keypoints": ", ".join(problem_names)
                })

        except Exception as e:
            print(f"❌ Error in {filename}: {e}")

    df = pd.DataFrame(data)
    if df.empty:
        print(f"✅ No problematic files found in {split_name or folder_path}.")
    else:
        df = df.sort_values(by="Problematic Keypoints", ascending=False)
        print(f"\n📊 Problematic files in {split_name or folder_path}:\n")
        print(df.to_string(index=False))

        if save_csv:
            output_path = os.path.join(
                r"C:\Users\kutay\OneDrive\Desktop",
                f"problematic_{split_name or 'output'}.csv"
            )
            df.to_csv(output_path, index=False)
            print(f"💾 Saved CSV to: {output_path}")

if __name__ == "__main__":
    directory_val = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_split_pickles_npz\val"
    directory_train = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_split_pickles_npz\train"
    directory_test = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_split_pickles_npz\test"

    analyze_npz_folder(directory_train, save_csv=True, split_name="train")
    analyze_npz_folder(directory_val, save_csv=True, split_name="val")
    analyze_npz_folder(directory_test, save_csv=True, split_name="test")
