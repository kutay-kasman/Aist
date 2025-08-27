# single_video_infer.py
# JSON YOK. Etiket sırası dataset_loader.py ile AYNI.
# Örnekleme, açı seti, NaN politikası eğitimdekiyle 1:1 aynı.

import os
import cv2
import json
import numpy as np
import torch
import mediapipe as mp

from train_lstm import DanceLSTM  # eğitimdekiyle birebir aynı olmalı

# ------------------ CONFIG ------------------
TEST_VIDEO = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split_excerpt\JS\gJS_sFM_c01_d01_mJS3_ch04.mp4"
MODEL_PATH = r"main\lstm_dance_model.pth"
TARGET_FPS = 10
CHUNK_DURATION = 4
SEQ_LEN = TARGET_FPS * CHUNK_DURATION  # 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
    # Eğitimdeki compute_angle ile birebir aynı (0..180 derece)
    if (np.isnan(a).any() or np.isnan(b).any() or np.isnan(c).any()):
        return np.nan
    a, b, c = np.asarray(a), np.asarray(b), np.asarray(c)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine = np.dot(ba, bc) / denom
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def extract_all_chunks_angles(video_path):
    """ Videodan eğitimdekiyle aynı şekilde 10 FPS örnekleyip
        40-frame pencereler (N x 40 x 8) döndürür.
    """
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if not orig_fps or orig_fps <= 0:
        orig_fps = TARGET_FPS  # fallback
    interval = int(round(orig_fps / TARGET_FPS))
    if interval < 1:
        interval = 1

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    frame_idx = 0
    chunk_keypoints = []
    all_chunks = []  # list of (40,8)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(img_rgb)

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                kps = np.array([[p.x, p.y] for p in lm], dtype=np.float32)  # (33,2)
            else:
                kps = np.full((33, 2), np.nan, dtype=np.float32)

            chunk_keypoints.append(kps)

            if len(chunk_keypoints) == SEQ_LEN:
                # 40 frame → 8 açı
                angles_seq = []
                for frm in chunk_keypoints:
                    frame_angles = []
                    for (a, b, c) in ANGLE_KEYPOINTS:
                        frame_angles.append(compute_angle(frm[a], frm[b], frm[c]))
                    angles_seq.append(frame_angles)
                angles_seq = np.array(angles_seq, dtype=np.float32)  # (40,8)

                # Eğitimdeki NaN politikasıyla uyumlu: NaN → 0
                angles_seq = np.nan_to_num(angles_seq, nan=0.0)

                all_chunks.append(angles_seq)
                chunk_keypoints = []

        frame_idx += 1

    # # Kalan parçayı pad'leyip ekle (opsiyonel ama stabilite için iyi)
    # if 0 < len(chunk_keypoints) < SEQ_LEN:
    #     pad_needed = SEQ_LEN - len(chunk_keypoints)
    #     chunk_keypoints += [np.full((33,2), np.nan, dtype=np.float32)] * pad_needed

    #     angles_seq = []
    #     for frm in chunk_keypoints:
    #         frame_angles = []
    #         for (a, b, c) in ANGLE_KEYPOINTS:
    #             frame_angles.append(compute_angle(frm[a], frm[b], frm[c]))
    #         angles_seq.append(frame_angles)
    #     angles_seq = np.array(angles_seq, dtype=np.float32)
    #     angles_seq = np.nan_to_num(angles_seq, nan=0.0)
    #     all_chunks.append(angles_seq)

    cap.release()
    pose.close()
    return all_chunks  # list of (40,8)

def predict_video(video_path, topk=3):
    # Model mimarisi eğitimde neyse aynı olmalı (input_size=8!)
    model = DanceLSTM(input_size=8, hidden_size=64, num_layers=2, num_classes=NUM_CLASSES)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval().to(DEVICE)

    chunks = extract_all_chunks_angles(video_path)  # N x (40,8)
    if len(chunks) == 0:
        return [("NO_DATA", 0.0)]

    with torch.no_grad():
        probs_accum = np.zeros(NUM_CLASSES, dtype=np.float32)
        for seq in chunks:
            x = torch.tensor(seq, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1,40,8)
            logits = model(x)  # (1,C)
            p = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            probs_accum += p
        probs_mean = probs_accum / len(chunks)

    top_idx = probs_mean.argsort()[::-1][:topk]
    return [(LABELS[i], float(probs_mean[i])) for i in top_idx]

if __name__ == "__main__":
    preds = predict_video(TEST_VIDEO, topk=TOPK)
    print("🎯 Tahmin Dağılımı:")
    for lbl, pr in preds:
        print(f"→ {lbl}: {pr*100:.2f}%")
