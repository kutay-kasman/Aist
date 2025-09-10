"""
Local runner for 3D pose pipeline (Colab→Windows)
- Preprocess PKL → NPZ (chunk=40, normalize)
- LSTM train/val/test
- Save/Load model
- Predict from .mp4

Gereken kütüphaneler: numpy, tqdm, torch, torchvision, mediapipe, opencv-python
"""

import os, pickle
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
from collections import defaultdict
import matplotlib.pyplot as plt
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # TEMP: allows duplicate OpenMP runtimes

# ============== CONFIG ==============
# 1) Senin PKL klasörlerin (train/val/test alt klasörleri bu dizinin içinde olmalı)
INPUT_BASE = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_split_pickles"

# 2) NPZ çıktıları
OUTPUT_BASE = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_split_pickles_npz"

# 3) Eğitilmiş model dosyası (kaydetme & yükleme için aynı yol)
MODEL_PATH = r"C:\Users\kutay\OneDrive\Desktop\Computer Vision\AistDataset\dance_lstm_3d.pth"

# 4) (İsteğe bağlı) Tahmin için bir video
VIDEO_PATH = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\HO\gHO_sFM_c01_d19_mHO2_ch07.mp4"

CHUNK_SIZE = 30
SPLITS = ["train", "val", "test"]

# 5) Label haritası
GENRE_TO_IDX = {'BR':0,'HO':1,'JB':2,'JS':3,'KR':4,'LH':5,'LO':6,'MH':7,'PO':8,'WA':9}
IDX_TO_GENRE = {v:k for k,v in GENRE_TO_IDX.items()}

# 6) Donanım
import torch, torch.nn as nn, torch.nn.functional as F
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ====================================


# --------- Yardımcılar ----------
def ensure_dirs():
    for split in SPLITS:
        os.makedirs(os.path.join(OUTPUT_BASE, split), exist_ok=True)

def unit_vector(v):
    return v / np.linalg.norm(v) if np.linalg.norm(v) != 0 else v

def angle_between(v1, v2):
    v1_u, v2_u = unit_vector(v1), unit_vector(v2)
    dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return np.arccos(dot)

def compute_angles(frame):  # frame: (17, 3)
    angles = []
    kp = frame
    try:
        angles.append(angle_between(kp[11] - kp[13], kp[15] - kp[13]))  # Left Knee
        angles.append(angle_between(kp[12] - kp[14], kp[16] - kp[14]))  # Right Knee
        angles.append(angle_between(kp[5] - kp[7], kp[9] - kp[7]))      # Left Elbow
        angles.append(angle_between(kp[6] - kp[8], kp[10] - kp[8]))     # Right Elbow
        angles.append(angle_between(kp[11] - kp[5], kp[13] - kp[11]))   # Left Hip
        angles.append(angle_between(kp[12] - kp[6], kp[14] - kp[12]))   # Right Hip
        angles.append(angle_between(kp[5] - kp[6], kp[7] - kp[5]))      # Left Shoulder
        angles.append(angle_between(kp[6] - kp[5], kp[8] - kp[6]))      # Right Shoulder
    except:
        angles = [0.0] * 8
    return np.array(angles, dtype=np.float32)

def normalize_3d(chunk):
    # chunk: (T=40, J=17, C=3)
    center = chunk[:, 0:1, :]   # pelvis/root joint (0)
    chunk = chunk - center
    scale = np.mean(np.linalg.norm(chunk, axis=-1))  # ortalama uzaklık
    if scale > 0:
        chunk = chunk / scale
    return chunk

def scan_and_convert_pkls():
    """PKL dosyalarından 60-frame’lik chunk'lar çıkarır, poz + açı + ivme + açı delta'sı ile .npz olarak kaydeder"""
    ensure_dirs()

    for split in SPLITS:
        in_path = os.path.join(INPUT_BASE, split)
        out_path = os.path.join(OUTPUT_BASE, split)
        if not os.path.isdir(in_path):
            print(f"⚠️ Split klasörü yok: {in_path}")
            continue

        for fname in tqdm(sorted(os.listdir(in_path)), desc=f"[{split.upper()}]"):
            if not fname.endswith(".pkl"):
                continue

            try:
                genre = fname[1:3].upper()
                if genre not in GENRE_TO_IDX:
                    print(f"⚠️ Etiket eşleşmedi: {fname} -> {genre}")
                    continue
            except:
                print(f"⚠️ Etiket okunamadı: {fname}")
                continue

            with open(os.path.join(in_path, fname), "rb") as f:
                data = pickle.load(f)
            keypoints = data["keypoints3d"]  # (frames, 17, 3)

            for i in range(0, len(keypoints) - CHUNK_SIZE + 1, CHUNK_SIZE // 2):  # Overlapping chunks
                chunk = keypoints[i:i+CHUNK_SIZE]
                if chunk.shape != (CHUNK_SIZE, 17, 3):
                    continue

                chunk = normalize_3d(chunk)

                # ➕ 1. Pose flatten: (60, 17, 3) → (60, 51)
                poses = chunk.reshape(CHUNK_SIZE, -1)

                # ➕ 2. Angle: compute_angles(frame) returns (8,)
                angles = np.array([compute_angles(f) for f in chunk])  # (60, 8)

                # ➕ Velocity (instead of acceleration)
                velocity = np.zeros_like(poses)
                velocity[1:] = poses[1:] - poses[:-1]

                # ➕ Angle delta (keep as is)
                angle_deltas = np.zeros_like(angles)
                angle_deltas[1:] = angles[1:] - angles[:-1]

                # ✅ Final fusion: (60, 118)
                fused_seq = np.concatenate([poses, angles, velocity, angle_deltas], axis=1)

                out_name = f"{os.path.splitext(fname)[0]}_chunk{i//CHUNK_SIZE:03d}.npz"
                save_path = os.path.join(out_path, out_name)
                if os.path.exists(save_path):
                    continue
                np.savez_compressed(save_path, seq=fused_seq, label=genre)

    print("✅ PKL→NPZ dönüşümü bitti (pose + angle + accel + delta).")



def validate_npz(data_dir=OUTPUT_BASE, splits=SPLITS, min_warn=-20, max_warn=20):
    issues = 0
    for split in splits:
        split_path = os.path.join(data_dir, split)
        if not os.path.isdir(split_path):
            print(f"⚠️ Yok: {split_path}")
            continue
        print(f"\n🔍 Checking split: {split}")
        for fname in sorted(os.listdir(split_path)):
            if not fname.endswith(".npz"): 
                continue
            p = os.path.join(split_path, fname)
            try:
                d = np.load(p)
                seq = d["seq"]
                if np.isnan(seq).any():
                    print(f"❌ NaN: {fname}"); issues += 1
                if np.isinf(seq).any():
                    print(f"❌ Inf: {fname}"); issues += 1
                mx, mn = np.max(seq), np.min(seq)
                if mx > max_warn or mn < min_warn:
                    print(f"⚠️ Range: {fname}  min={mn:.2f}, max={mx:.2f}")
            except Exception as e:
                print(f"❌ Yükleme hatası {fname}: {e}"); issues += 1
    if issues == 0:
        print("\n✅ NPZ dosyaları temiz görünüyor.")
    else:
        print(f"\n❗ Potansiyel sorunlu {issues} dosya var.")


def delete_nan_or_extreme(data_dir=OUTPUT_BASE, splits=SPLITS, min_thresh=-50, max_thresh=100):
    deleted = 0
    for split in splits:
        split_path = os.path.join(data_dir, split)
        if not os.path.isdir(split_path): 
            continue
        for fname in os.listdir(split_path):
            if not fname.endswith(".npz"): 
                continue
            p = os.path.join(split_path, fname)
            try:
                with np.load(p) as d:   
                    seq = d["seq"]
                    if np.isnan(seq).any() or np.isinf(seq).any():
                        d.close()      
                        os.remove(p); deleted += 1
                        print(f"🗑️ {fname} (NaN/Inf)")
                        continue
                    mn, mx = float(np.min(seq)), float(np.max(seq))
                    if mn < min_thresh or mx > max_thresh:
                        d.close()
                        os.remove(p); deleted += 1
                        print(f"🗑️ {fname} (min={mn:.2f}, max={mx:.2f})")
            except Exception as e:
                try:
                    os.remove(p)
                    deleted += 1
                    print(f"🗑️ {fname} (load err: {e})")
                except PermissionError:
                    print(f"⚠️ {fname} silinemedi (hala kilitli). Sonradan manuel sil.")
    print(f"✅ Silinen dosya: {deleted}")



# ---------- Dataset ----------
from torch.utils.data import Dataset, DataLoader

class Pose3DDataset(Dataset):
    def __init__(self, folder_path):
        self.samples = []
        if not os.path.isdir(folder_path):
            print(f"⚠️ Dataset klasörü yok: {folder_path}")
            return
        for file in os.listdir(folder_path):
            if not file.endswith(".npz"):
                continue
            d = np.load(os.path.join(folder_path, file))
            seq = d["seq"] 
            label = GENRE_TO_IDX[str(d["label"])]
            self.samples.append((
                torch.tensor(seq, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long)
            ))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

# ---------- Model ----------
class DanceLSTM(nn.Module):
    def __init__(self, input_size=118, hidden_size=64, num_layers=3, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x): 
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # last timestep


# ---------- Train / Eval ----------
def run_training():
    data_dir = OUTPUT_BASE
    train_set = Pose3DDataset(os.path.join(data_dir, "train"))
    val_set   = Pose3DDataset(os.path.join(data_dir, "val"))
    EPOCHS = 50
    PATIENCE = 5

    if len(train_set) == 0 or len(val_set) == 0:
        print("❌ Train/Val set boş. Önce PKL→NPZ yapıldığına emin ol.")
        return

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=8)

    model = DanceLSTM().to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = OneCycleLR(optim, max_lr=1e-3, total_steps=len(train_loader) * EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    wait_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for X, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            X, y = X.to(DEVICE), y.to(DEVICE)
            optim.zero_grad()
            X = X.view(X.size(0), X.size(1), -1)
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optim.step()
            scheduler.step()

            total_loss += loss.item() * X.size(0)
            total_samples += X.size(0)

        epoch_loss = total_loss / total_samples

        # --- Validation ---
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                preds = model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = 100.0 * correct / max(1, total)

        print(f"✅ Epoch {epoch}: Train Loss={epoch_loss:.4f}  Val Acc={acc:.2f}%")

        # --- Early Stopping & Save Best Model ---
        if acc > best_acc:
            best_acc = acc
            wait_counter = 0
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"💾 Yeni en iyi model kaydedildi! Val Acc={best_acc:.2f}%")
        else:
            wait_counter += 1
            print(f"⏳ Gelişme yok. Patience: {wait_counter}/{PATIENCE}")
            if wait_counter >= PATIENCE:
                print(f"🛑 Early stopping tetiklendi. En iyi Val Acc: {best_acc:.2f}%")
                break

def run_test_top1():
    test_set = Pose3DDataset(os.path.join(OUTPUT_BASE, "test"))
    if len(test_set) == 0:
        print("❌ Test set boş.")
        return
    test_loader = DataLoader(test_set, batch_size=8)
    model = DanceLSTM().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            preds = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    print(f"🎯 Test Top-1 Acc: {100.0*correct/max(1,total):.2f}%")


def evaluate_topk(model, dataloader, device, topk=(1,3,5)):
    model.eval()
    correct_topk = {k:0 for k in topk}; total=0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)  # (B,10)
            for k in topk:
                _, tk = outputs.topk(k, dim=1)
                match = tk.eq(y.view(-1,1).expand_as(tk))
                correct_topk[k] += match.any(dim=1).float().sum().item()
            total += y.size(0)
    for k in topk:
        print(f"🎯 Top-{k} Acc: {100.0*correct_topk[k]/max(1,total):.2f}%")


def run_test_topk():
    test_set = Pose3DDataset(os.path.join(OUTPUT_BASE, "test"))
    if len(test_set) == 0:
        print("❌ Test set boş.")
        return
    test_loader = DataLoader(test_set, batch_size=8)
    model = DanceLSTM().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    evaluate_topk(model, test_loader, DEVICE, topk=(1,3,5))


# ---------- Video Prediction ----------
import cv2, mediapipe as mp
# --- MP33 -> COCO17 eşlemesi (aynı dosyada kullan) ---
COCO_TO_MEDIAPIPE = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
SAVE_DIR = os.path.join(os.path.dirname(MODEL_PATH), "kp_cache")
os.makedirs(SAVE_DIR, exist_ok=True)

def extract_mediapipe_17(video_path, max_frames=None):
    mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)
    cap = cv2.VideoCapture(video_path)
    pts17_list, vis17_list = [], []
    frame_idx = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break
        if max_frames and frame_idx >= max_frames: break
        res = mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.pose_world_landmarks:
            lm = res.pose_world_landmarks.landmark
            pts33 = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            vis33 = np.array([p.visibility for p in lm], dtype=np.float32)
            pts17 = pts33[COCO_TO_MEDIAPIPE]
            vis17 = vis33[COCO_TO_MEDIAPIPE]
            pts17_list.append(pts17)
            vis17_list.append(vis17)
        frame_idx += 1
    cap.release(); mp_pose.close()
    return np.asarray(pts17_list, np.float32), np.asarray(vis17_list, np.float32)

def _center_by_pelvis(seq17):  # (T,17,3)
    c = 0.5*(seq17[:,11] + seq17[:,12])
    return seq17 - c[:,None,:]

def _scale_by_shoulder(seq17):
    d = np.linalg.norm(seq17[:,5] - seq17[:,6], axis=1) + 1e-8
    return seq17 / d[:,None,None]

def _maybe_zflip(seq):  # try z and -z; choose later
    return np.stack([seq, seq * np.array([1,1,-1], np.float32)], axis=0)

def _umeyama_similarity(src, dst, w=None, with_scale=True):
    src = src.astype(np.float64); dst = dst.astype(np.float64)
    if w is None: w = np.ones(len(src), np.float64)
    w = w / (w.sum() + 1e-12)
    mu_s = (w[:,None]*src).sum(0); mu_d = (w[:,None]*dst).sum(0)
    X = src - mu_s; Y = dst - mu_d
    C = (w[:,None,None] * (Y[:,:,None] @ X[:,None,:])).sum(0)
    U,S,Vt = np.linalg.svd(C); R = U @ Vt
    if np.linalg.det(R) < 0: Vt[-1]*=-1; R = U @ Vt
    s = (S.sum())/((w*(X*X).sum(1)).sum()+1e-12) if with_scale else 1.0
    t = mu_d - s*(R @ mu_s)
    return float(s), R, t

def _apply_similarity(seq, s, R, t):
    shp = seq.shape
    return (s * (seq.reshape(-1,3) @ R.T) + t).reshape(shp)

def fit_similarity_with_pkl(pkl_path, mp17, vis17, trim_ratio=0.10, use_weights=True):
    with open(pkl_path, "rb") as f:
        aist = pickle.load(f)["keypoints3d"]
    aist = np.asarray(aist, np.float32)

    T = min(len(aist), len(mp17))
    aist, mp17, vis17 = aist[:T], mp17[:T], vis17[:T]

    a_n  = _scale_by_shoulder(_center_by_pelvis(aist))
    cand = _maybe_zflip(_scale_by_shoulder(_center_by_pelvis(mp17)))
    pre_err = [np.linalg.norm(a_n - c, axis=2).mean() for c in cand]
    mp_n = cand[int(np.argmin(pre_err))]

    A = a_n.reshape(-1,3); M = mp_n.reshape(-1,3)
    W = np.clip(vis17.reshape(-1),0.05,1.0) if use_weights else None

    s,R,t = _umeyama_similarity(M,A,w=W,with_scale=True)
    np.save("alignment_matrix.npy", {"s": s, "R": R, "t": t})

    res = np.linalg.norm(_apply_similarity(M,s,R,t)-A,axis=1)
    if trim_ratio>0:
        k = int(len(res)*trim_ratio)
        keep = np.ones_like(res,bool)
        keep[np.argpartition(res,-k)[-k:]] = False
        s,R,t = _umeyama_similarity(M[keep],A[keep],w=(None if W is None else W[keep]))
    return s,R,t


def align_mp17_with_pkl(pkl_path, mp17, vis17):
    s,R,t = fit_similarity_with_pkl(pkl_path, mp17, vis17)
    mp17_n = _scale_by_shoulder(_center_by_pelvis(mp17))
    return _apply_similarity(mp17_n, s,R,t)


def extract_mediapipe_3d_keypoints(video_path, max_frames=300):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2,
                        enable_segmentation=False, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    keypoints3d = []
    frames = 0
    while cap.isOpened() and len(keypoints3d) < max_frames:
        ok, frame = cap.read()
        if not ok: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(frame_rgb)
        frames += 1
        if res.pose_world_landmarks:
            lms = res.pose_world_landmarks.landmark
            # 33 eklem (3D)
            frame_kps = np.array([[lm.x, lm.y, lm.z] for lm in lms])  # (33,3)
            keypoints3d.append(frame_kps)
    cap.release(); pose.close()
    keypoints3d = np.array(keypoints3d)  # (T,33,3)
    print(f"📸 Extracted {len(keypoints3d)} frames with keypoints (of {frames})")
    return keypoints3d

def mediapipe_to_coco17(mediapipe_pose):
    # (T,33,3) → (T,17,3) COCO sırası
    coco_indices = [0, 11,13,15, 12,14,16, 23,25,27, 24,26,28, 5,2,7,4]
    return mediapipe_pose[:, coco_indices, :]

def chunk_and_normalize(pose_coco, chunk_size=CHUNK_SIZE):
    chunks = []
    for i in range(0, len(pose_coco) - chunk_size + 1, chunk_size):
        c = pose_coco[i:i+chunk_size]
        if c.shape == (chunk_size,17,3):
            c = normalize_3d(c)
            chunks.append(c.reshape(chunk_size, 17*3))
    return np.array(chunks)  # (N,40,51)

def _fuse_features_from_chunk(chunk_17x3):
    """
    chunk_17x3: (T=CHUNK_SIZE, 17, 3) already aligned
    returns (T, 118) = poses(51) + angles(8) + accel(51) + angle_deltas(8)
    """
    # normalize exactly like training
    c_norm = normalize_3d(chunk_17x3)                          # (T,17,3)
    poses = c_norm.reshape(len(c_norm), -1)                    # (T,51)
    angles = np.array([compute_angles(f) for f in c_norm])     # (T,8)

    accel = np.zeros_like(poses)                               # (T,51)
    accel[1:-1] = poses[2:] - 2*poses[1:-1] + poses[:-2]

    angle_deltas = np.zeros_like(angles)                       # (T,8)
    angle_deltas[1:] = angles[1:] - angles[:-1]

    return np.concatenate([poses, angles, accel, angle_deltas], axis=1)  # (T,118)

def predict_from_single_video_without_pkl(video_path, alignment_path="alignment_matrix.npy", topk=3):
    # 1) MediaPipe 17 keypoint'leri çıkar
    mp17, vis17 = extract_mediapipe_17(video_path)
    if mp17.shape[0] < CHUNK_SIZE:
        print(f"❌ Yeterli frame yok: {mp17.shape[0]} < {CHUNK_SIZE}")
        return

    # 2) Alignment (s,R,t) yükle
    if not os.path.exists(alignment_path):
        print(f"❌ Alignment dosyası yok: {alignment_path}")
        return
    art = np.load(alignment_path, allow_pickle=True).item()
    s, R, t = art["s"], art["R"], art["t"]

    # 3) 3D hizalama
    mp17_aligned = (s * mp17 @ R.T) + t                        # (T,17,3)

    # 4) Chunk’la (eğitimdeki stride ile aynı: CHUNK_SIZE//2) ve 118-D fuse et
    fused_chunks = []
    for i in range(0, len(mp17_aligned) - CHUNK_SIZE + 1, CHUNK_SIZE // 2):
        chunk = mp17_aligned[i:i+CHUNK_SIZE]
        if chunk.shape != (CHUNK_SIZE, 17, 3):
            continue
        fused_seq = _fuse_features_from_chunk(chunk)           # (CHUNK_SIZE,118)
        fused_chunks.append(fused_seq)

    if not fused_chunks:
        print("❌ Geçerli chunk oluşmadı.")
        return

    X = torch.tensor(np.stack(fused_chunks), dtype=torch.float32).to(DEVICE)  # (N,CHUNK,118)

    # 5) Model (118-D) yükle & tahmin
    if not os.path.isfile(MODEL_PATH):
        print(f"❌ Model bulunamadı: {MODEL_PATH}")
        return
    model = DanceLSTM(input_size=118).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        logits = model(X)                         # (N, num_classes)
        probs  = torch.softmax(logits, dim=1)    # (N, num_classes)
        mean_probs = probs.mean(0)               # video-level: (num_classes,)
        tk_probs, tk_idx = torch.topk(mean_probs, k=topk)

    print("\n🎯 Top Predictions (video-level):")
    for rank in range(topk):
        cls = tk_idx[rank].item()
        print(f"{rank+1}. {IDX_TO_GENRE[cls]} ({100*tk_probs[rank].item():.2f}%)")




def save_alignment_matrix(pkl_path, video_path, output_path="alignment_matrix.npy"):
    """
    Belirli bir PKL dosyası ve video için hizalama matrislerini hesaplar ve kaydeder.
    """
    print("🎬 Video'dan MediaPipe 17 keypoint'leri çıkarılıyor...")
    mp17, vis17 = extract_mediapipe_17(video_path)
    if mp17.shape[0] < CHUNK_SIZE:
        print(f"❌ Yeterli frame yok: {mp17.shape[0]} < {CHUNK_SIZE}")
        return

    print("📊 PKL verisi ile hizalama matrisi hesaplanıyor...")
    s, R, t = fit_similarity_with_pkl(pkl_path, mp17, vis17)

    # Matrisi kaydet
    np.save(output_path, {"s": s, "R": R, "t": t})
    print(f"✅ Hizalama matrisi kaydedildi: {output_path}")

def run_test_video_level():
    test_dir = os.path.join(OUTPUT_BASE, "test")
    model = DanceLSTM(input_size=118).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    video_chunks = defaultdict(list)
    video_labels = {}

    for fname in sorted(os.listdir(test_dir)):
        if not fname.endswith(".npz"): continue
        video_id = "_".join(fname.split("_")[:-1])  # chunk öncesi her şey
        path = os.path.join(test_dir, fname)
        video_chunks[video_id].append(path)

        if video_id not in video_labels:
            with np.load(path) as d:
                video_labels[video_id] = GENRE_TO_IDX[str(d["label"])]

    correct = total = 0

    for vid, chunk_paths in tqdm(video_chunks.items(), desc="🎬 Video-Level Test"):
        all_logits = []
        for path in chunk_paths:
            with np.load(path) as d:
                angles = torch.tensor(d["seq"], dtype=torch.float32).to(DEVICE)
                angles = angles.unsqueeze(0)  # (1, 60, 118)
                logits = model(angles)        # (1, num_classes)
                all_logits.append(logits)

        # Chunk'ların ortalamasını al → video-level logits
        mean_logits = torch.stack(all_logits).mean(dim=0)  # (1, num_classes)

        # Olasılıkları hesapla (softmax)
        probs = F.softmax(mean_logits, dim=1)  # (1, num_classes)

        # Top-3 tahminleri al
        top_probs, top_idxs = probs.topk(3, dim=1)  # her biri (1,3)

        # Asıl tahmin (en yüksek olasılık)
        pred = top_idxs[0, 0].item()
        label = video_labels[vid]
        correct += (pred == label)
        total += 1

        # # --- Yazdırma ---
        # print(f"\n🎥 Video: {vid}")
        # print(f"✅ Gerçek Label: {label}")
        # for rank in range(3):
        #     cls = top_idxs[0, rank].item()
        #     prob = top_probs[0, rank].item() * 100
        #     print(f"  {rank+1}. Tahmin: {cls} ({prob:.2f}%)")

    # Final accuracy
    acc = 100.0 * correct / max(1, total)
    print(f"\n🎯 Video-Level Test Acc: {acc:.2f}%")


def predict_from_webcam_live(model, alignment_matrix_path="alignment_matrix.npy"):
    import cv2
    import time

    cap = cv2.VideoCapture(0)  # 0 = default webcam
    mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)
    sRt = np.load(alignment_matrix_path, allow_pickle=True).item()
    s, R, t = sRt["s"], sRt["R"], sRt["t"]

    pose_window = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_pose.process(frame_rgb)

        if res.pose_world_landmarks:
            landmarks = res.pose_world_landmarks.landmark
            kp33 = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
            kp17 = kp33[COCO_TO_MEDIAPIPE]

            # Align and normalize keypoints
            kp17 = (s * kp17 @ R.T) + t
            pose_window.append(kp17)

            if len(pose_window) > CHUNK_SIZE:
                pose_window.pop(0)

            if len(pose_window) == CHUNK_SIZE:
                chunk = np.array(pose_window)  # (60, 17, 3)
                fused = _fuse_features_from_chunk(chunk)  # (60, 118)
                X = torch.tensor(fused, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, 60, 118)

                with torch.no_grad():
                    logits = model(X)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

                top3_idx = probs.argsort()[-3:][::-1]
                pred_str = ", ".join([f"{IDX_TO_GENRE[i]} ({probs[i]*100:.1f}%)" for i in top3_idx])
                print(f"[Prediction] {pred_str}")

                # Draw on frame
                cv2.putText(frame, pred_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

        cv2.imshow("Dance Genre Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    mp_pose.close()


def predict_from_webcam_live_interval(model, alignment_matrix_path="alignment_matrix.npy", interval_seconds=5):
    import cv2
    import time

    cap = cv2.VideoCapture(0)  # webcam
    mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)

    # Load alignment matrix (s, R, t)
    sRt = np.load(alignment_matrix_path, allow_pickle=True).item()
    s, R, t = sRt["s"], sRt["R"], sRt["t"]

    pose_window = []
    last_prediction_time = time.time()

    print("🔴 Press Q to quit...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_pose.process(frame_rgb)

        # Optional: show raw video feed
        cv2.imshow("Dance Genre Prediction", frame)

        if res.pose_world_landmarks:
            landmarks = res.pose_world_landmarks.landmark
            kp33 = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
            kp17 = kp33[COCO_TO_MEDIAPIPE]
            kp17 = (s * kp17 @ R.T) + t  # align

            pose_window.append(kp17)

        current_time = time.time()

        if len(pose_window) >= CHUNK_SIZE and current_time - last_prediction_time >= interval_seconds:
            chunk = np.array(pose_window[:CHUNK_SIZE])  # just first 60
            pose_window = []  # clear buffer for next prediction
            last_prediction_time = current_time

            fused = _fuse_features_from_chunk(chunk)  # (60, 118)
            X = torch.tensor(fused, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(X)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            top3_idx = probs.argsort()[-3:][::-1]
            pred_str = ", ".join([f"{IDX_TO_GENRE[i]} ({probs[i]*100:.1f}%)" for i in top3_idx])
            print(f"\n🎯 [Prediction] {pred_str}")

            # Draw on frame
            cv2.putText(frame, pred_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    mp_pose.close()


def predict_from_webcam_live_confident_only(model, alignment_matrix_path="alignment_matrix.npy", interval_seconds=5, confidence_threshold=0.8):
    import cv2
    import time

    cap = cv2.VideoCapture(0)
    mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)

    sRt = np.load(alignment_matrix_path, allow_pickle=True).item()
    s, R, t = sRt["s"], sRt["R"], sRt["t"]

    pose_window = []
    last_prediction_time = time.time()
    prediction_label = ""  # Label to show on webcam (if confident)

    print("🟢 Webcam running. Will only predict when confidence ≥ 80%. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_pose.process(frame_rgb)

        if res.pose_world_landmarks:
            landmarks = res.pose_world_landmarks.landmark
            kp33 = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
            kp17 = kp33[COCO_TO_MEDIAPIPE]
            kp17 = (s * kp17 @ R.T) + t
            pose_window.append(kp17)

        current_time = time.time()

        # Time to check a prediction
        if len(pose_window) >= CHUNK_SIZE and (current_time - last_prediction_time >= interval_seconds):
            chunk = np.array(pose_window[:CHUNK_SIZE])
            pose_window = []
            last_prediction_time = current_time

            fused = _fuse_features_from_chunk(chunk)
            X = torch.tensor(fused, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(X)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            top1 = np.argmax(probs)
            pred_conf = probs[top1]
            pred_name = IDX_TO_GENRE[top1]

            if pred_conf >= confidence_threshold:
                prediction_label = f"{pred_name} ({pred_conf*100:.1f}%)"
                print(f"✅ Confident prediction: {prediction_label}")
            else:
                prediction_label = ""
                print(f"⚠️ Low confidence ({pred_name}, {pred_conf*100:.1f}%) → Skipping...")

        # Draw prediction on screen if confident
        if prediction_label:
            cv2.putText(frame, prediction_label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow("🕺 Dance Genre Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    mp_pose.close()

def predict_from_single_video_without_pkl_fast(
    video_path,
    alignment_path="alignment_matrix.npy",
    topk=3,
    every_nth=2,
    cache_dir=SAVE_DIR,
):
    import os
    import time
    t0 = time.time()

    cache_path = os.path.join(cache_dir, os.path.basename(video_path) + ".npz")

    # Step 1: Load cached keypoints or extract with MediaPipe
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        mp17 = data["mp17"]
        vis17 = data["vis17"]
        print("📦 Loaded cached keypoints")
    else:
        import cv2, mediapipe as mp
        mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)
        cap = cv2.VideoCapture(video_path)
        pts17_list, vis17_list = [], []
        frame_idx = 0

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % every_nth != 0:
                frame_idx += 1
                continue
            res = mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_world_landmarks:
                pts33 = np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks.landmark], dtype=np.float32)
                vis33 = np.array([p.visibility for p in res.pose_world_landmarks.landmark], dtype=np.float32)
                pts17 = pts33[COCO_TO_MEDIAPIPE]
                vis17 = vis33[COCO_TO_MEDIAPIPE]
                pts17_list.append(pts17)
                vis17_list.append(vis17)
            frame_idx += 1
        cap.release()
        mp_pose.close()

        mp17 = np.asarray(pts17_list, np.float32)
        vis17 = np.asarray(vis17_list, np.float32)
        np.savez_compressed(cache_path, mp17=mp17, vis17=vis17)
        print(f"✅ Extracted {len(mp17)} frames and cached keypoints")

    if mp17.shape[0] < CHUNK_SIZE:
        print(f"❌ Not enough frames: {mp17.shape[0]} < {CHUNK_SIZE}")
        return

    # Step 2: Load alignment
    if not os.path.exists(alignment_path):
        print(f"❌ Missing alignment file: {alignment_path}")
        return
    art = np.load(alignment_path, allow_pickle=True).item()
    s, R, t = art["s"], art["R"], art["t"]

    # Step 3: Align pose
    mp17_aligned = (s * mp17 @ R.T) + t

    # Step 4: Chunk and fuse features
    fused_chunks = []
    for i in range(0, len(mp17_aligned) - CHUNK_SIZE + 1, CHUNK_SIZE // 2):
        chunk = mp17_aligned[i:i + CHUNK_SIZE]
        if chunk.shape != (CHUNK_SIZE, 17, 3):
            continue
        fused_seq = _fuse_features_from_chunk(chunk)
        fused_chunks.append(fused_seq)

    if not fused_chunks:
        print("❌ No valid chunks after preprocessing")
        return

    X = torch.tensor(np.stack(fused_chunks), dtype=torch.float32).to(DEVICE)

    # Step 5: Load model
    if not os.path.isfile(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return
    model = DanceLSTM(input_size=118).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Step 6: Inference
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)
        mean_probs = probs.mean(0)
        tk_probs, tk_idx = torch.topk(mean_probs, k=topk)

    print("\n🎯 Top Predictions (video-level):")
    for rank in range(topk):
        cls = tk_idx[rank].item()
        print(f"{rank + 1}. {IDX_TO_GENRE[cls]} ({100 * tk_probs[rank].item():.2f}%)")

    print(f"⏱️ Done in {time.time() - t0:.2f} seconds.")

# ---------- CLI benzeri basit akış ----------
if __name__ == "__main__": 
    VIDEO_TO_ALIGN = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\KR\gKR_sFM_c01_d28_mKR0_ch01.mp4"
    PKL_TO_ALIGN   = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_split_pickles\train\gKR_sBM_cAll_d28_mKR0_ch01.pkl"
    ALIGrunNMENT_MATRIX_PATH = r"alignment_matrix.npy"
    VIDEO_TO_PREDICT = r"C:\Users\kutay\Videos\Captures\(27) 【 BREAK DANCE 】Power move Collection ( Bboy SNACK , ARIYA , GOODmen , FreeasyClothing ) - YouTube - Google Chrome 2025-09-08 16-04-45.mp4"
    ALIGNMENT_MATRIX_PATH = r"C:\Users\kutay\OneDrive\Desktop\Computer Vision\AistDataset\alignment_matrix.npy"

    # Adım adım kullan:
    # 1) PKL→NPZ üret:
    scan_and_convert_pkls()
    validate_npz()
    delete_nan_or_extreme()
    validate_npz()
    
    # 2) Eğitim:
    run_training()

    # 3 Test:
    run_test_top1()
    run_test_topk()
    run_test_video_level()

    save_alignment_matrix(PKL_TO_ALIGN, VIDEO_TO_ALIGN, ALIGNMENT_MATRIX_PATH)
    
    # 4) Videodan tahmin:
    # predict_from_single_video_without_pkl(VIDEO_PATH, ALIGN_PATH, topk=3)
    predict_from_single_video_without_pkl(VIDEO_TO_PREDICT, ALIGNMENT_MATRIX_PATH, topk=10)
    predict_from_single_video_without_pkl_fast(VIDEO_TO_PREDICT, ALIGNMENT_MATRIX_PATH, topk=3)

    print("Bu dosyayı düzenleyip, __main__ altındaki çağrıları sırayla aç/kapat.")


# if __name__ == "__main__":
#     ALIGNMENT_MATRIX_PATH = r"C:\Users\kutay\OneDrive\Desktop\Computer Vision\AistDataset\alignment_matrix.npy"
#     model = DanceLSTM(input_size=118).to(DEVICE)
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#     model.eval()

#     # predict_from_webcam_live(model, alignment_matrix_path=ALIGNMENT_MATRIX_PATH)
#     predict_from_webcam_live_confident_only(model, alignment_matrix_path=ALIGNMENT_MATRIX_PATH, interval_seconds=10)
