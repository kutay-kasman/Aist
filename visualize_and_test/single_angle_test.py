# single_angles_infer.py
import numpy as np
import torch
from main.train_lstm import DanceLSTM  # eğitimdekiyle birebir aynı olmalı

MODEL_PATH = "lstm_dance_model.pth"

# dataset_loader.py ile AYNI sıra
LABELS = ["BR","HO","JB","JS","KR","LH","LO","MH","PO","WA"]

def load_angles_npz(path):
    d = np.load(path)
    angles = d["angles"].astype(np.float32)  # (40,8) beklenir
    # güvenlik: pad/truncate
    if angles.shape != (40,8):
        T = angles.shape[0]
        if T < 40:
            pad = np.zeros((40-T,8), dtype=np.float32)
            angles = np.concatenate([angles, pad], axis=0)
        else:
            angles = angles[:40]
    return angles  # NaN’li yoksa değiştirmiyoruz

def predict_angles_npz(path, topk=5, device="cpu"):
    angles = load_angles_npz(path)
    x = torch.tensor(angles, dtype=torch.float32).unsqueeze(0)  # (1,40,8)

    model = DanceLSTM(input_size=8, hidden_size=64, num_layers=2, num_classes=len(LABELS))
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

    idxs = probs.argsort()[::-1][:topk]
    return [(LABELS[i], float(probs[i])) for i in idxs]

if __name__ == "__main__":
    ANGLES_NPZ = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\HO\gHO_sFM_c01_d19_mHO0_ch01\angles\gHO_sFM_c01_d19_mHO0_ch01_004_angles.npz"  # EĞİTİMDE KULLANILAN bir dosya ver
    preds = predict_angles_npz(ANGLES_NPZ, topk=10)
    print("🎯 Tahmin (angles .npz üzerinden):")
    for lbl, p in preds:
        print(f"→ {lbl}: {p*100:.2f}%")
