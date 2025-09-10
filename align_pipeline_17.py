import os, json, argparse, pickle
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# ---------------- Config ----------------
SAVE_DIR = "./kp_cache"
TRIM_RATIO = 0.10
USE_WEIGHTS = True
TRY_TIMESHIFT = True
MAX_SHIFT = 30

# MediaPipe 33 -> COCO 17 map
COCO_TO_MEDIAPIPE = [
    0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16,
    23, 24, 25, 26, 27, 28
]

def ensuredir(p): os.makedirs(p, exist_ok=True)

# ---------------- Loaders ----------------
def load_aist_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    arr = np.asarray(data["keypoints3d"], np.float32)  # (T,17,3)
    assert arr.shape[1:] == (17,3), "AIST++ pkl must have (T,17,3)"
    return arr

# ---------------- MediaPipe Extraction ----------------
def extract_mediapipe_17(video_path, out_prefix):
    mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2)
    cap = cv2.VideoCapture(video_path)
    pts17_list, vis17_list, ts = [], [], []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break
        t = i / fps; i += 1
        res = mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.pose_world_landmarks:
            lm = res.pose_world_landmarks.landmark
            pts33 = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            vis33 = np.array([p.visibility for p in lm], dtype=np.float32)
            pts17 = pts33[COCO_TO_MEDIAPIPE]
            vis17 = vis33[COCO_TO_MEDIAPIPE]
            pts17_list.append(pts17)
            vis17_list.append(vis17)
            ts.append(t)
    cap.release(); mp_pose.close()

    pts17 = np.asarray(pts17_list, np.float32)   # (T,17,3)
    vis17 = np.asarray(vis17_list, np.float32)   # (T,17)
    ts    = np.asarray(ts, np.float32)

    ensuredir(SAVE_DIR)
    np.save(os.path.join(SAVE_DIR, f"{out_prefix}_mp17.npy"), pts17)
    np.save(os.path.join(SAVE_DIR, f"{out_prefix}_mp17_vis.npy"), vis17)
    np.save(os.path.join(SAVE_DIR, f"{out_prefix}_ts.npy"), ts)
    print(f"✅ Saved {pts17.shape} to {out_prefix}_mp17.npy")
    return pts17, vis17, ts

# ---------------- Normalization ----------------
def center_by_pelvis(seq17):
    c = 0.5 * (seq17[:,11] + seq17[:,12])
    return seq17 - c[:,None,:]

def scale_by_shoulder(seq17):
    d = np.linalg.norm(seq17[:,5] - seq17[:,6], axis=1) + 1e-8
    return seq17 / d[:,None,None]

def maybe_zflip(seq):
    return np.stack([seq, seq * np.array([1,1,-1], np.float32)], axis=0)

# ---------------- Umeyama Similarity ----------------
def umeyama_similarity(src, dst, w=None, with_scale=True):
    src = src.astype(np.float64); dst = dst.astype(np.float64)
    if w is None: w = np.ones(len(src), np.float64)
    w = w / (w.sum() + 1e-12)

    mu_s = (w[:,None]*src).sum(0)
    mu_d = (w[:,None]*dst).sum(0)
    X = src - mu_s; Y = dst - mu_d

    C = (w[:,None,None] * (Y[:,:,None] @ X[:,None,:])).sum(0)
    U,S,Vt = np.linalg.svd(C)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1]*=-1
        R = U @ Vt

    if with_scale:
        var = (w * (X*X).sum(1)).sum()
        s = (S.sum()) / (var + 1e-12)
    else:
        s = 1.0
    t = mu_d - s*(R @ mu_s)
    return float(s), R, t

def apply_similarity(seq, s, R, t):
    shp = seq.shape
    return (s * (seq.reshape(-1,3) @ R.T) + t).reshape(shp)

# ---------------- Time-shift ----------------
def best_time_shift(a_sig, m_sig, max_shift=30):
    best = -1e9; arg = 0
    for sh in range(-max_shift, max_shift+1):
        if sh < 0: a, m = a_sig[-sh:], m_sig[:len(m_sig)+sh]
        else:      a, m = a_sig[:len(a_sig)-sh], m_sig[sh:]
        if len(a) < 5: continue
        c = np.corrcoef(a, m)[0,1]
        if c > best: best, arg = c, sh
    return arg

# ---------------- Fit Transform ----------------
def fit_transform_full(aist, mp17, vis17, trim_ratio=TRIM_RATIO, use_weights=USE_WEIGHTS, try_timeshift=TRY_TIMESHIFT):
    T = min(len(aist), len(mp17))
    aist = aist[:T]; mp17 = mp17[:T]; vis17 = vis17[:T]

    if try_timeshift:
        a_sig = np.linalg.norm(aist[:,5]-aist[:,6], axis=1)
        m_sig = np.linalg.norm(mp17[:,5]-mp17[:,6], axis=1)
        sh = best_time_shift(a_sig, m_sig, MAX_SHIFT)
        if sh < 0:
            aist = aist[-sh:]; mp17 = mp17[:len(aist)]; vis17 = vis17[:len(aist)]
        elif sh > 0:
            mp17 = mp17[sh:]; vis17 = vis17[sh:]; aist = aist[:len(mp17)]

    a_n = scale_by_shoulder(center_by_pelvis(aist))
    m_cands = maybe_zflip(scale_by_shoulder(center_by_pelvis(mp17)))
    pre_err = [np.linalg.norm(a_n - m, axis=2).mean() for m in m_cands]
    m_n = m_cands[int(np.argmin(pre_err))]

    A = a_n.reshape(-1,3)
    M = m_n.reshape(-1,3)
    W = vis17.reshape(-1)
    if not use_weights: W = None
    else: W = np.clip(W, 0.05, 1.0)

    s,R,t = umeyama_similarity(M, A, w=W, with_scale=True)
    M1 = apply_similarity(M, s, R, t)
    res = np.linalg.norm(M1 - A, axis=1)

    if trim_ratio > 0:
        k = int(len(res) * trim_ratio)
        keep = np.ones_like(res, bool)
        keep[np.argpartition(res, -k)[-k:]] = False
        s,R,t = umeyama_similarity(M[keep], A[keep], w=None if W is None else W[keep], with_scale=True)

    before = np.mean(np.linalg.norm((m_n - a_n).reshape(-1,3), axis=1))
    after  = np.mean(np.linalg.norm((apply_similarity(M, s, R, t) - A), axis=1))
    return (s,R,t), a_n, m_n, before, after

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp_e = sub.add_parser("extract")
    sp_e.add_argument("--video", required=True)
    sp_e.add_argument("--prefix", required=True)

    sp_f = sub.add_parser("fit")
    sp_f.add_argument("--pkl", required=True)
    sp_f.add_argument("--prefix", required=True)
    sp_f.add_argument("--trim", type=float, default=TRIM_RATIO)
    sp_f.add_argument("--no-weights", action="store_true")
    sp_f.add_argument("--no-timeshift", action="store_true")

    sp_a = sub.add_parser("apply")
    sp_a.add_argument("--prefix", required=True)
    sp_a.add_argument("--mp17", required=True)
    sp_a.add_argument("--out", required=True)

    args = ap.parse_args()
    ensuredir(SAVE_DIR)

    if args.cmd == "extract":
        extract_mediapipe_17(args.video, args.prefix)

    elif args.cmd == "fit":
        aist = load_aist_pkl(args.pkl)
        mp17 = np.load(os.path.join(SAVE_DIR, f"{args.prefix}_mp17.npy"))
        vis17= np.load(os.path.join(SAVE_DIR, f"{args.prefix}_mp17_vis.npy"))
        (s,R,t), a_n, m_n, before, after = fit_transform_full(
            aist, mp17, vis17,
            trim_ratio=args.trim,
            use_weights=not args.no_weights,
            try_timeshift=not args.no_timeshift
        )
        tf_path = os.path.join(SAVE_DIR, f"{args.prefix}_similarity.json")
        with open(tf_path, "w") as f:
            json.dump({"s": float(s), "R": R.tolist(), "t": t.tolist()}, f, indent=2)
        print(f"[Similarity] scale={s:.6f}\nR=\n{R}\nt={t}")
        print(f"Mean distance BEFORE: {before:.4f}")
        print(f"Mean distance AFTER : {after:.4f}")
        print(f"Saved → {tf_path}")

    elif args.cmd == "apply":
        with open(os.path.join(SAVE_DIR, f"{args.prefix}_similarity.json")) as f:
            d = json.load(f)
        s = d["s"]; R = np.array(d["R"]); t = np.array(d["t"])
        mp17 = np.load(args.mp17)
        mp17 = scale_by_shoulder(center_by_pelvis(mp17))
        aligned = apply_similarity(mp17, s, R, t)
        np.save(args.out, aligned)
        print(f"Aligned seq saved → {args.out}")

if __name__ == "__main__":
    main()
