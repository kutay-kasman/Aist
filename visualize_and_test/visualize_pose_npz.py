import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting

# ---------- CONFIG ----------
NPZ_PATH = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d04_mBR0_ch01\3d_pose\gBR_sFM_c01_d04_mBR0_ch01_002_3d_pose.npz"
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
pose_data = data["keypoints"]  # shape: (40, 33, 3)
num_frames = pose_data.shape[0]

# ---------- SETUP 3D PLOT ----------
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.15)  # space for slider

scat = ax.scatter([], [], [], c='blue', s=20)
lines = [ax.plot([], [], [], c='gray')[0] for _ in EDGES]
title = ax.set_title("")

# Fix 3D limits (adjust if needed)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-2, 1)
ax.set_zlim(-1.5, 1.5)
ax.view_init(elev=30, azim=70)  # rotate view
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z') 

# ---------- UPDATE FUNCTION ----------
def update(frame_idx):
    frame = pose_data[frame_idx]
    x = frame[:, 0]
    y = -frame[:, 1]  # flip y for upright view
    z = frame[:, 2]

    scat._offsets3d = (x, y, z)
    title.set_text(f"Frame {frame_idx}")

    for line, (i, j) in zip(lines, EDGES):
        if not (np.any(np.isnan(frame[i])) or np.any(np.isnan(frame[j]))):
            line.set_data([x[i], x[j]], [y[i], y[j]])
            line.set_3d_properties([z[i], z[j]])
        else:
            line.set_data([], [])
            line.set_3d_properties([])

    fig.canvas.draw_idle()

# ---------- SLIDER ----------
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
frame_slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)
frame_slider.on_changed(lambda val: update(int(val)))

# ---------- INITIAL DRAW ----------
update(0)
plt.show()
