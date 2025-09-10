import os
import numpy as np
import torch
from train_lstm import DanceLSTM  # eğitimde kullandığın LSTM modeli

# ------------------ CONFIG ------------------
NPZ_PATH = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split_excerpt\KR\gKR_sBM_c01_d28_mKR3_ch03\angles\gKR_sBM_c01_d28_mKR3_ch03_001_angles.npz"
NPZ_DIR  = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_split_ready\test"  # tüm test için
MODEL_PATH = r"lstm_dance_model4.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOPK = 3
# --------------------------------------------

# ---- LABELS: dataset_loader.py ile AYNI SIRA ----
LABELS = ["BR","HO","JB","JS","KR","LH","LO","MH","PO","WA"]
NUM_CLASSES = len(LABELS)

def predict_npz(npz_path, model, topk=3):
    """ Tek bir npz dosyasından tahmin yap """
    data = np.load(npz_path)
    angles = data["angles"]  # (40,8)
    angles = np.nan_to_num(angles, nan=0.0)  # Eğitimdeki NaN politikası

    x = torch.tensor(angles, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1,40,8)

    with torch.no_grad():
        logits = model(x)  # (1,C)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_idx = probs.argsort()[::-1][:topk]
    return [(LABELS[i], float(probs[i])) for i in top_idx]

def load_model():
    state = torch.load(MODEL_PATH, map_location=DEVICE)

    # hidden_size ve bidirectional'ı checkpoint'ten çıkar
    h0 = state['lstm.weight_ih_l0'].shape[0] // 4          # 4*H
    bidir = any(k.endswith('_reverse') for k in state.keys())

    model = DanceLSTM(input_size=8, hidden_size=h0, num_layers=2,
                      num_classes=NUM_CLASSES, bidirectional=bidir)
    model.load_state_dict(state)   # artık çakışma olmayacak
    model.eval().to(DEVICE)
    return model


def test_single_file():
    """ Tek dosya testi """
    model = load_model()
    preds = predict_npz(NPZ_PATH, model, topk=TOPK)
    print(f"\n🎬 {os.path.basename(NPZ_PATH)}")
    for lbl, pr in preds:
        print(f"→ {lbl}: {pr*100:.2f}%")

def test_all_folder():
    """ Tüm klasörleri gezip accuracy ölç — pose klasörleri atlanır """
    model = load_model()
    total, correct_top1, correct_top3 = 0, 0, 0

    for genre_folder in os.listdir(NPZ_DIR):
        genre_path = os.path.join(NPZ_DIR, genre_folder)
        if not os.path.isdir(genre_path):
            continue

        # os.walk içinde 'pose' klasörlerini komple dışla
        for root, dirs, files in os.walk(genre_path):
            # root altındaki alt klasör listesinde inplace filtreleme:
            dirs[:] = [d for d in dirs if d.lower() != "pose"]

            for file in files:
                if not file.endswith(".npz"):
                    continue

                full_path = os.path.join(root, file)
                total += 1
                try:
                    preds = predict_npz(full_path, model, topk=TOPK)
                    top_labels = [lbl for lbl, _ in preds]

                    print(f"\n🎬 {file} (True: {genre_folder})")
                    for lbl, pr in preds:
                        print(f"→ {lbl}: {pr*100:.2f}%")

                    # Accuracy computation
                    if genre_folder == top_labels[0]:
                        correct_top1 += 1
                    if genre_folder in top_labels:
                        correct_top3 += 1

                except Exception as e:
                    print(f"⚠️ Hata: {file} işlenemedi → {e}")

    # Final result
    if total > 0:
        print("\n📊 Top-1 Accuracy: {:.2f}%".format(100 * correct_top1 / total))
        print("📊 Top-3 Accuracy: {:.2f}%".format(100 * correct_top3 / total))
        print(f"🎥 Total files tested: {total}")
    else:
        print("⚠️ Hiç .npz dosyası bulunamadı!")

if __name__ == "__main__":
    # test_single_file()   # Tek dosya için
    test_all_folder()      # Tüm dataset için
