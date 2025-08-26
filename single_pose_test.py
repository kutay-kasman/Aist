# single_pose_infer.py
# Amaç: Video okumadan, doğrudan pose .npz (keypoints) dosyasından tahmin yapmak.
# Not: Etiket sırası dataset_loader.py ile AYNI tutuldu.

import os
import numpy as np
import torch

from train_lstm import DanceLSTM  # eğitimde kullandığın sınıfla birebir aynı olmalı

# ------------------ CONFIG ------------------
TEST_POSE_NPZ = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\PO\gPO_sFM_c01_d10_mPO3_ch04\pose\gPO_sFM_c01_d10_mPO3_ch04_002_pose.npz"
MODEL_PATH = "lstm_dance_model.pth"
TARGET_FPS = 10
CHUNK_DURATION = 4
SEQ_LEN = TARGET_FPS * CHUNK_DURATION  # 40
DEVICE = "cpu"
TOPK = 10
# --------------------------------------------

# ---- LABELS: dataset_loader.py ile AYNI SIRA ----
# genre_to_idx = {'BR':0,'HO':1,'JB':2,'JS':3,'KR':4,'LH':5,'LO':6,'MH':7,'PO':8,'WA':9}
LABELS = ["BR","HO","JB","JS","KR","LH","LO","MH","PO","WA"]
NUM_CLASSES = len(LABELS)

# ---- Eğitimde kullandığınla AYNI keypoint üçlüleri ----
# MediaPipe Pose (33 nokta) indeksleri
ANGLE_KEYPOINTS = [
    [11,13,15],  # Elbow (L)
    [12,14,16],  # Elbow (R)
    [23,25,27],  # Knee  (L)
    [24,26,28],  # Knee  (R)
    [13,11,23],  # Shoulder (L)
    [14,12,24],  # Shoulder (R)
    [11,23,25],  # Hip (L)
    [12,24,26],  # Hip (R)
]

def compute_angle(a, b, c):
    """Eğitimdeki compute_angle ile birebir aynı (0..180 derece)."""
    if (np.isnan(a).any() or np.isnan(b).any() or np.isnan(c).any()):
        return np.nan
    a, b, c = np.asarray(a), np.asarray(b), np.asarray(c)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine = np.dot(ba, bc) / denom
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def angles_from_keypoints_seq(keypoints_seq):
    """
    keypoints_seq: (T, 33, 2) np.array
    Çıktı: (40, 8) np.array  (pad/truncate uygulanır, NaN->0 yapılır)
    """
    T = keypoints_seq.shape[0]

    # T != 40 ise pad/truncate
    if T < SEQ_LEN:
        pad = np.full((SEQ_LEN - T, 33, 2), np.nan, dtype=np.float32)
        keypoints_seq = np.concatenate([keypoints_seq, pad], axis=0)
    elif T > SEQ_LEN:
        keypoints_seq = keypoints_seq[:SEQ_LEN]

    angles_seq = []
    for frm in keypoints_seq:  # (33,2)
        frame_angles = []
        for (a, b, c) in ANGLE_KEYPOINTS:
            frame_angles.append(compute_angle(frm[a], frm[b], frm[c]))
        angles_seq.append(frame_angles)

    angles_seq = np.asarray(angles_seq, dtype=np.float32)  # (40,8)
    angles_seq = np.nan_to_num(angles_seq, nan=0.0)        # eğitimle aynı NaN politikası
    return angles_seq

def load_pose_npz(npz_path):
    """
    .npz dosyasından keypoints'i okur.
    Eğitim pipeline'ında np.savez(..., keypoints=pose_chunk) olarak kaydedilmişti.
    """
    data = np.load(npz_path)
    if "keypoints" not in data:
        raise KeyError(f"'keypoints' anahtarı {npz_path} içinde yok. Dosyayı kontrol et.")
    kps = data["keypoints"]  # (T,33,2)
    kps = np.asarray(kps, dtype=np.float32)
    return kps

def predict_from_pose_npz(npz_path, topk=3):
    # Model: eğitimdeki mimariyle birebir
    model = DanceLSTM(input_size=8, hidden_size=64, num_layers=2, num_classes=NUM_CLASSES)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval().to(DEVICE)

    # .npz -> keypoints -> angles
    keypoints_seq = load_pose_npz(npz_path)          # (T,33,2)
    angles_seq = angles_from_keypoints_seq(keypoints_seq)  # (40,8)

    with torch.no_grad():
        x = torch.tensor(angles_seq, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1,40,8)
        logits = model(x)  # (1,C)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_idx = probs.argsort()[::-1][:topk]
    return [(LABELS[i], float(probs[i])) for i in top_idx]

if __name__ == "__main__":
    preds = predict_from_pose_npz(TEST_POSE_NPZ, topk=TOPK)
    print("🎯 Tahmin Dağılımı (pose .npz üzerinden):")
    for lbl, pr in preds:
        print(f"→ {lbl}: {pr*100:.2f}%")
