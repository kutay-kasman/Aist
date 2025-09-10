import os, pickle, json
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# ---------- Config ----------
TRIM_RATIO = 0.10  # trim top-10% worst joints when refitting
USE_WEIGHTS = True # use mediapipe visibility as weights
SAVE_DIR = "./kp_cache"

COCO_EDGES = [(0,1),(1,3),(0,2),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
              (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
COCO_TO_MEDIAPIPE = [0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28]  # 17 joints

# ---------- IO ----------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def save_npy(path, arr):
    ensure_dir(os.path.dirname(path))
    np.save(path, arr)

def load_aist_keypoints(pkl_path):
    with open(pkl_path, "rb") as f: data = pickle.load(f)
    return np.asarray(data["keypoints3d"], dtype=np.float32)   # (T,17,3)

def extract_mediapipe_keypoints3d(video_path, max_frames=None):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2)
    cap = cv2.VideoCapture(video_path)
    pts, vis = [], []
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break
        if max_frames and len(pts) >= max_frames: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if not res.pose_world_landmarks:
            continue
        lm = res.pose_world_landmarks.landmark
        pts.append([[p.x, p.y, p.z] for p in lm])
        vis.append([p.visibility for p in lm])
    cap.release(); pose.close()
    pts = np.asarray(pts, dtype=np.float32)      # (T,33,3)
    vis = np.asarray(vis, dtype=np.float32)      # (T,33)
    return pts, vis

# ---------- helpers ----------
def pelvis_center_17(k):  # left/right hip: 11,12
    return 0.5*(k[11] + k[12])

def center_by_pelvis(seq17):  # (T,17,3)
    c = 0.5*(seq17[:,11] + seq17[:,12])  # (T,3)
    return seq17 - c[:,None,:]

def scale_by_shoulder(seq17):
    # shoulder dist per frame (5,6)
    d = np.linalg.norm(seq17[:,5] - seq17[:,6], axis=1) + 1e-8  # (T,)
    return seq17 / d[:,None,None]

def maybe_flip_z(mp_seq):
    # Try z and -z, keep the one with smaller distance to AIST after rough norm
    return np.stack([ mp_seq, mp_seq * np.array([1,1,-1],dtype=np.float32) ], axis=0)

def umeyama_similarity(src, dst, w=None, with_scale=True):
    """
    Weighted Umeyama (similarity) src->dst. src/dst: (N,3), w: (N,) or None
    returns (s, R(3x3), t(3,))
    """
    src = src.astype(np.float64); dst = dst.astype(np.float64)
    if w is None: w = np.ones(len(src), dtype=np.float64)
    w = w / (w.sum() + 1e-12)

    mu_s = (w[:,None]*src).sum(axis=0)
    mu_d = (w[:,None]*dst).sum(axis=0)
    X = src - mu_s; Y = dst - mu_d
    C = (w[:,None,None] * (Y[:,:,None] @ X[:,None,:])).sum(axis=0)
    U,S,Vt = np.linalg.svd(C)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = U @ Vt
    if with_scale:
        var = (w * (X*X).sum(axis=1)).sum()
        s = (S.sum()) / (var + 1e-12)
    else:
        s = 1.0
    t = mu_d - s*(R @ mu_s)
    return float(s), R, t

def apply_similarity(seq, s, R, t):
    # seq: (...,3)
    shp = seq.shape
    return (s * (seq.reshape(-1,3) @ R.T) + t).reshape(shp)

def mean_frame_error(A,B):  # (T,17,3)
    return np.linalg.norm(A-B, axis=2).mean(axis=1)

# ---------- robust fit over ALL frames ----------
def fit_similarity_all_frames(aist17, mp17, mp_vis17=None, trim_ratio=TRIM_RATIO):
    """
    aist17/mp17: (T,17,3) normalized (centered+scaled). Optional mp_vis17: (T,17) in [0,1]
    1) stack all frames -> big (T*17,3)
    2) weighted Umeyama
    3) trim top k% largest residuals, refit
    """
    T = len(aist17)
    A = aist17.reshape(-1,3)
    M = mp17.reshape(-1,3)
    if mp_vis17 is None:
        w = np.ones((T*17,), dtype=np.float64)
    else:
        w = mp_vis17.reshape(-1).astype(np.float64)
        w = np.clip(w, 0.05, 1.0)  # avoid zeros

    # 1st fit
    s, R, t = umeyama_similarity(M, A, w=w, with_scale=True)

    # residuals
    M1 = apply_similarity(M, s, R, t)
    res = np.linalg.norm(M1 - A, axis=1)
    # Trim worst k%
    keep = np.ones_like(res, dtype=bool)
    if trim_ratio > 0:
        k = int(len(res) * trim_ratio)
        idx = np.argpartition(res, -k)[-k:]  # worst
        keep[idx] = False

    # Refit on inliers
    s2, R2, t2 = umeyama_similarity(M[keep], A[keep], w=w[keep], with_scale=True)
    return s2, R2, t2

# ---------- main ----------
if __name__ == "__main__":
    # paths (edit)
    pkl_path   = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_split_pickles\test\gBR_sBM_cAll_d06_mBR2_ch01.pkl"
    video_path = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d04_mBR0_ch01.mp4"

    ensure_dir(SAVE_DIR)
    aist = load_aist_keypoints(pkl_path)                  # (T1,17,3)

    mp33_path = os.path.join(SAVE_DIR, "mp33.npy")
    mpvis_path= os.path.join(SAVE_DIR, "mp_vis.npy")

    if os.path.exists(mp33_path) and os.path.exists(mpvis_path):
        mp33 = np.load(mp33_path); mp_vis = np.load(mpvis_path)
    else:
        mp33, mp_vis = extract_mediapipe_keypoints3d(video_path)  # (T2,33,3),(T2,33)
        save_npy(mp33_path, mp33); save_npy(mpvis_path, mp_vis)

    T = min(len(aist), len(mp33))
    aist = aist[:T]
    mp17 = mp33[:T, COCO_TO_MEDIAPIPE, :]
    mpv17 = mp_vis[:T, COCO_TO_MEDIAPIPE]

    # Z-flip auto-check (rough)
    cand = maybe_flip_z(mp17)  # (2,T,17,3)
    errs = []
    for c in range(2):
        a_c = center_by_pelvis(aist)
        m_c = center_by_pelvis(cand[c])
        a_n = scale_by_shoulder(a_c)
        m_n = scale_by_shoulder(m_c)
        e = mean_frame_error(a_n, m_n).mean()
        errs.append(e)
    mp17 = cand[int(np.argmin(errs))]

    # Normalize both (center pelvis + shoulder scale)
    a_c = center_by_pelvis(aist); a_n = scale_by_shoulder(a_c)
    m_c = center_by_pelvis(mp17);  m_n = scale_by_shoulder(m_c)

    # Fit over ALL frames (weighted + trimmed)
    s, R, t = fit_similarity_all_frames(a_n, m_n, mpv17, trim_ratio=TRIM_RATIO)
    print(f"[Similarity] scale={s:.4f}\nR=\n{R}\nt={t}")

    # Apply
    m_aligned = apply_similarity(m_n, s, R, t)

    # Errors
    before = mean_frame_error(a_n, m_n)
    after  = mean_frame_error(a_n, m_aligned)
    print(f"Mean distance BEFORE: {before.mean():.4f}")
    print(f"Mean distance AFTER : {after.mean():.4f}")

    # Plot error curves
    plt.figure(figsize=(12,5))
    plt.plot(before, label="Before")
    plt.plot(after,  label="After")
    plt.title("AIST++ vs MediaPipe mean 3D distance per frame")
    plt.xlabel("Frame"); plt.ylabel("Mean distance"); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.show()

    # Save transform (use in inference)
    tf_path = os.path.join(SAVE_DIR, "similarity_transform.json")
    with open(tf_path, "w") as f:
        json.dump({"s": float(s), "R": R.tolist(), "t": t.tolist()}, f, indent=2)
    print(f"Saved transform → {tf_path}")

    # Example: apply at inference
    #   1) take a mediapipe chunk (T,33,3) → select 17 joints
    #   2) center+scale like above
    #   3) apply_similarity with stored (s,R,t)
    #   4) flatten to (T, 17*3) for the LSTM
