import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ---------- CONFIG ----------
NPZ_PATH = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\KR\gKR_sFM_c01_d28_mKR3_ch07\pose\gKR_sFM_c01_d28_mKR3_ch07_002_pose.npz"
SHOW_BONES = True
# ----------------------------

EDGES = [
    (0,1),(1,2),(2,3),(3,7), (0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (17,19),(12,14),(14,16),(16,18),(16,20),(16,22),
    (18,20),(11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28),(27,29),(28,30),(29,31),(30,32)
]

# ---------- LOAD DATA ----------
data = np.load(NPZ_PATH)
pose_data = data["keypoints"]  # shape: (40, 33, 2)
num_frames = pose_data.shape[0]

# ---------- SETUP PLOT ----------
fig, ax = plt.subplots(figsize=(6, 8))
plt.subplots_adjust(bottom=0.15)  # space for slider

scatter = ax.scatter([], [], c='blue', s=20)
lines = [ax.plot([], [], c='gray')[0] for _ in EDGES]

ax.set_xlim(0, 1)
ax.set_ylim(-1, 0)
ax.set_aspect('equal')
ax.axis('off')
title = ax.set_title("")

# ---------- UPDATE FUNCTION ----------
def update(frame_idx):
    frame = pose_data[frame_idx]
    x = frame[:, 0]
    y = -frame[:, 1]  # flip y

    scatter.set_offsets(np.c_[x, y])
    title.set_text(f"Frame {frame_idx}")

    for line, (i, j) in zip(lines, EDGES):
        if not (np.any(np.isnan(frame[i])) or np.any(np.isnan(frame[j]))):
            line.set_data([x[i], x[j]], [y[i], y[j]])
        else:
            line.set_data([], [])

    fig.canvas.draw_idle()

# ---------- SLIDER ----------
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
frame_slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)

frame_slider.on_changed(lambda val: update(int(val)))

# ---------- INITIAL DRAW ----------
update(0)
plt.show()
