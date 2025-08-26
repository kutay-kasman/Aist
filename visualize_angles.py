import numpy as np
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
NPZ_PATH = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\KR\gKR_sFM_c01_d28_mKR3_ch07\angles\gKR_sFM_c01_d28_mKR3_ch07_006_angles.npz"
# ----------------------------

# ---------- LOAD DATA ----------
data = np.load(NPZ_PATH)
angles = data["angles"]  # shape: (40, 8)
label = data["label"].tolist() if "label" in data else "unknown"

# Angle names (same order as extraction)
angle_names = [
    "Elbow (L)", "Elbow (R)",
    "Knee (L)", "Knee (R)",
    "Shoulder (L)", "Shoulder (R)",
    "Hip (L)", "Hip (R)"
]

# ---------- PLOT ----------
plt.figure(figsize=(14, 10))
for i in range(8):
    plt.subplot(4, 2, i + 1)
    plt.plot(angles[:, i])
    plt.title(f"{angle_names[i]}")
    plt.xlabel("Frame")
    plt.ylabel("Angle (°)")
    plt.ylim(0, 180)
    plt.grid(True)

plt.suptitle(f"Angle Trajectories — Label: {label}", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
