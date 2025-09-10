import os
import pickle
import numpy as np

# --- CONFIG ---
pkl_path = r"C:\Users\kutay\OneDrive\Desktop\aist_plusplus_final\keypoints3d\gBR_sBM_cAll_d04_mBR0_ch01.pkl"
output_data_path = "data/dance_dataset/train_data.npy"
output_label_path = "data/dance_dataset/train_label.pkl"
label_index = 0  # label for this sample (e.g., 0 for "break")

# --- LOAD AND CONVERT ---
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

key = "keypoints3d_optim" if "keypoints3d_optim" in data else "keypoints3d"
keypoints = np.asarray(data[key], dtype=np.float32)  # shape: (T, 17, 3)

# Convert to (C, T, V, M) = (3, T, 17, 1)
T, V, C = keypoints.shape
M = 1
converted = keypoints.transpose(2, 0, 1).reshape(1, C, T, V, M)  # add batch dim

# Save .npy
os.makedirs("data/dance_dataset", exist_ok=True)
np.save(output_data_path, converted)

# Save label.pkl (list of tuples: (sample_name, label))
sample_name = os.path.splitext(os.path.basename(pkl_path))[0]
label_data = [(sample_name, label_index)]
with open(output_label_path, "wb") as f:
    pickle.dump(label_data, f)

print(f"Saved:\n- {output_data_path}\n- {output_label_path}")
