import os
import pickle
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --------- COCO-17 edges (iskelet çizimi) ----------
COCO_EDGES = [
    (0,1),(1,3),      # right eye-ear
    (0,2),(2,4),      # left eye-ear
    (5,6),(5,7),(7,9),
    (6,8),(8,10),
    (5,11),(6,12),
    (11,12),
    (11,13),(13,15),
    (12,14),(14,16),
]

# MediaPipe(33) -> COCO(17) indeks eşlemesi
COCO_TO_MEDIAPIPE = [
    0,  # Nose
    2,  # Left Eye
    5,  # Right Eye
    7,  # Left Ear
    8,  # Right Ear
    11, # Left Shoulder
    12, # Right Shoulder
    13, # Left Elbow
    14, # Right Elbow
    15, # Left Wrist
    16, # Right Wrist
    23, # Left Hip
    24, # Right Hip
    25, # Left Knee
    26, # Right Knee
    27, # Left Ankle
    28, # Right Ankle
]

# ---------- IO ----------
def load_aist_keypoints(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    # Beklenen: dict['keypoints3d'] -> (T,17,3)
    return np.asarray(data["keypoints3d"], dtype=np.float32)

def extract_mediapipe_keypoints3d(video_path, max_frames=300):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2)
    cap = cv2.VideoCapture(video_path)
    out = []
    while cap.isOpened() and len(out) < max_frames:
        ok, frame = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_world_landmarks:
            lm = res.pose_world_landmarks.landmark
            pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)  # (33,3)
            out.append(pts)
    cap.release(); pose.close()
    return np.asarray(out, dtype=np.float32)  # (T,33,3)

# ---------- normalize yardımcıları ----------
def pelvis_center_17(kps17):
    # COCO: left_hip=11, right_hip=12
    return 0.5*(kps17[11] + kps17[12])

def center_by_pelvis(kps17):
    c = pelvis_center_17(kps17)
    return kps17 - c[None, :]

def scale_by_shoulder(kps17):
    # left_shoulder=5, right_shoulder=6
    d = np.linalg.norm(kps17[5] - kps17[6]) + 1e-8
    return kps17 / d

# ---------- Umeyama: similarity (R, t, s) ----------
def umeyama_alignment(src, dst, with_scale=True):
    """
    src -> dst  (N,3)
    Returns s, R, t  (scale, rotation(3x3), translation(3,))
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape and src.shape[1] == 3

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    X = src - mu_src
    Y = dst - mu_dst

    C = (Y.T @ X) / src.shape[0]  # covariance
    U, S, Vt = np.linalg.svd(C)
    R = U @ Vt
    # Det(R) negatif ise yansıma düzelt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    if with_scale:
        var_src = (X**2).sum() / src.shape[0]
        s = (S.sum()) / (var_src + 1e-12)
    else:
        s = 1.0

    t = mu_dst - s * (R @ mu_src)
    return float(s), R.astype(np.float64), t.astype(np.float64)

def apply_similarity(pts, s, R, t):
    return (s * (pts @ R.T) + t)

# ---------- çizim ----------
def draw_skeleton_3d(ax, keypoints, color, label, lw=2):
    ax.scatter(keypoints[:,0], keypoints[:,1], keypoints[:,2], c=color, s=20)
    for i,j in COCO_EDGES:
        xs = [keypoints[i,0], keypoints[j,0]]
        ys = [keypoints[i,1], keypoints[j,1]]
        zs = [keypoints[i,2], keypoints[j,2]]
        ax.plot(xs, ys, zs, color=color, linewidth=lw)
    if label: ax.plot([],[],[], color=color, label=label)

def plot_pair(aist17, mp17, title):
    fig = plt.figure(figsize=(9,8))
    ax = fig.add_subplot(111, projection='3d')
    draw_skeleton_3d(ax, aist17, 'blue', 'AIST++ (PKL)')
    draw_skeleton_3d(ax, mp17,   'red',  'MediaPipe (aligned)')
    ax.legend(); ax.set_title(title)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout(); plt.show()

# # ---------- ana akış ----------
# if __name__ == "__main__":
#     # ---- dosya yolları: kendine göre değiştir ----
#     pkl_path   = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_split_pickles\test\gBR_sBM_cAll_d06_mBR2_ch01.pkl"
#     video_path = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d04_mBR0_ch01.mp4"

#     aist = load_aist_keypoints(pkl_path)             # (T1,17,3)  (mutlak/ölçekli)
#     mp33 = extract_mediapipe_keypoints3d(video_path) # (T2,33,3)  (normalize/bağıl)

#     print("Aist")
#     print(aist)
#     print("MediaPipe")
#     print(mp33)
#     # ortak frame sayısı
#     T = min(len(aist), len(mp33))
#     aist = aist[:T]
#     mp17 = mp33[:T, COCO_TO_MEDIAPIPE, :]            # (T,17,3)

#     print("AIST frames:", len(aist), "MediaPipe frames:", len(mp17))

#     # --- 1) önce kaba merkezleme/ölçek (ikisi için de aynı yöntem)
#     aist_c = np.stack([center_by_pelvis(f) for f in aist])   # (T,17,3)
#     mp_c   = np.stack([center_by_pelvis(f) for f in mp17])
#     # AIST zaten milimetre benzeri ölçeklerde; MP çok küçük. Umeyama scale zaten çözecek,
#     # ama başlangıç için omuz mesafesine göre ölçekleyelim:
#     aist_n = np.stack([scale_by_shoulder(f) for f in aist_c])
#     mp_n   = np.stack([scale_by_shoulder(f) for f in mp_c])

#     # --- 2) tek bir frame’den similarity öğren (ör: orta frame) VEYA birkaç frame’in ortalamasından
#     idxs = np.linspace(0, T-1, num=min(10,T), dtype=int)  # 10 örnek frame
#     src = np.concatenate([mp_n[i]  for i in idxs], axis=0)   # (10*17,3)
#     dst = np.concatenate([aist_n[i] for i in idxs], axis=0)
#     s, R, t = umeyama_alignment(src, dst, with_scale=True)
#     print(f"[Umeyama] scale={s:.4f}\nR=\n{R}\nt={t}")

#     # --- 3) tüm MP frame’lerini hizala
#     mp_aligned = apply_similarity(mp_n.reshape(-1,3), s, R, t).reshape(T,17,3)

#     # --- 4) önce/sonra mesafe raporu
#     def mean_pair_dist(A,B):  # (T,17,3)
#         return float(np.mean(np.linalg.norm(A-B, axis=2)))
#     before = mean_pair_dist(aist_n, mp_n)
#     after  = mean_pair_dist(aist_n, mp_aligned)
#     print(f"Mean distance BEFORE  alignment: {before:.4f}")
#     print(f"Mean distance AFTER   alignment: {after:.4f}")

#     # --- 5) birkaç frame görselleştir
#     for k in [0, T//3, 2*T//3, T-1]:
#         plot_pair(aist_n[k], mp_aligned[k], title=f"Frame {k} – aligned")

#     # --- 6) zaman içinde hata grafiği
#     per_frame_err_before = np.linalg.norm(aist_n - mp_n, axis=2).mean(axis=1)
#     per_frame_err_after  = np.linalg.norm(aist_n - mp_aligned, axis=2).mean(axis=1)
#     plt.figure(figsize=(10,4))
#     plt.plot(per_frame_err_before, label="Before")
#     plt.plot(per_frame_err_after,  label="After")
#     plt.title("AIST++ vs MediaPipe mean 3D distance per frame")
#     plt.xlabel("Frame"); plt.ylabel("Mean distance"); plt.grid(True); plt.legend()
#     plt.tight_layout(); plt.show()


# Mevcut tüm fonksiyonlar (load_aist_keypoints, extract_mediapipe_keypoints3d, umeyama_alignment, etc.)
# bu dosyanın üst kısmında kalacak.

# ---------- yeni ana akış ----------
def evaluate_alignment_with_pretrained_matrix(video_path, pkl_path, alignment_matrix_path):
    """
    Önceden hesaplanmış bir dönüşüm matrisini kullanarak
    yeni bir videodaki pozları AIST++ verileriyle karşılaştırır.
    """
    
    # 1. Hizalama matrisini yükle
    if not os.path.exists(alignment_matrix_path):
        print(f"❌ Hizalama dosyası bulunamadı: {alignment_matrix_path}")
        return
    data = np.load(alignment_matrix_path, allow_pickle=True).item()
    s, R, t = data["s"], data["R"], data["t"]
    print(f"✅ Hizalama matrisi yüklendi: scale={s:.4f}")

    # 2. Yeni videodan ve PKL'den verileri al
    aist_all = load_aist_keypoints(pkl_path)
    mp33 = extract_mediapipe_keypoints3d(video_path)
    
    T = min(len(aist_all), len(mp33))
    aist_17 = aist_all[:T]
    mp_17 = mp33[:T, COCO_TO_MEDIAPIPE, :]

    print(f"AIST++ ({aist_17.shape[0]} frames) ve MediaPipe ({mp_17.shape[0]} frames) verisi yüklendi.")
    
    # 3. MediaPipe verisine Umeyama dönüşümünü uygula
    # Not: Dönüşüm, merkezlenmiş/ölçeklenmiş veriler üzerinde öğrenildiği için,
    # önce MP verisini aynı şekilde ön işlemden geçirmeliyiz.
    
    # Kaba merkezleme/ölçekleme
    mp_c = np.stack([center_by_pelvis(f) for f in mp_17])
    mp_n = np.stack([scale_by_shoulder(f) for f in mp_c])
    
    # Kaydedilmiş matrisi uygula
    mp_aligned = apply_similarity(mp_n.reshape(-1,3), s, R, t).reshape(T,17,3)
    
    # 4. Hata raporu
    aist_n = np.stack([scale_by_shoulder(center_by_pelvis(f)) for f in aist_17])
    
    per_frame_err_after = np.linalg.norm(aist_n - mp_aligned, axis=2).mean(axis=1)
    mean_err_after = per_frame_err_after.mean()
    print(f"⭐ Uygulanan matris ile ortalama hata: {mean_err_after:.4f}")
    
    # 5. Görselleştirme
    print("Görselleştirme için örnek kareler gösteriliyor...")
    for k in [0, T//3, 2*T//3, T-1]:
        plot_pair(aist_n[k], mp_aligned[k], title=f"Frame {k} - Applied Pre-trained Matrix")
        
    plt.figure(figsize=(10,4))
    plt.plot(per_frame_err_after, label="After Pre-trained Alignment")
    plt.title("Hata Grafiği (Yeni Video)")
    plt.xlabel("Frame"); plt.ylabel("Ortalama Mesafe"); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    # --- Değerlendirme için dosya yolları: kendine göre değiştir ---
    # Not: Bu video ve PKL dosyası, matrisi HESAPLAMAK için kullandığınızdan farklı olmalı.
    EVAL_VIDEO_PATH = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\JB\gJB_sFM_c01_d07_mJB0_ch01.mp4"
    EVAL_PKL_PATH   = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_split_pickles\train\gJB_sBM_cAll_d07_mJB0_ch01.pkl"
    # Kaydedilen matrisin yolu
    ALIGNMENT_MATRIX_PATH = r"C:\Users\kutay\OneDrive\Desktop\Computer Vision\AistDataset\alignment_matrix.npy"

    # Fonksiyonu çağır
    evaluate_alignment_with_pretrained_matrix(EVAL_VIDEO_PATH, EVAL_PKL_PATH, ALIGNMENT_MATRIX_PATH)