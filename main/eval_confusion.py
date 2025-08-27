# eval_confusion.py
# Amaç: Tüm .npz test seti için confusion matrix, classification report ve görsel kaydetmek.
# Not: test_npz_files.py içindeki predict_npz, load_model, NPZ_DIR, LABELS kullanılıyor.

import os
import numpy as np
import torch

# test_npz_files.py içinden hazır fonksiyonları ve sabitleri alıyoruz
from test_npz_files import predict_npz, load_model, NPZ_DIR, LABELS

# sklearn ve matplotlib gerek
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = len(LABELS)
IDX = {lbl: i for i, lbl in enumerate(LABELS)}
TOPK = 3  # top-3 ölçümü için

def evaluate_confusion(npz_root_dir, save_prefix="aist_eval"):
    """
    Klasör yapısı beklentisi:
        <root>/<genre>/<video>/angles/*.npz   (pose klasörleri tamamen atlanır)
    """
    y_true, y_pred = [], []
    total, correct_top1, correct_top3 = 0, 0, 0

    # Genre bazlı sayaçlar
    per_class_total = {g: 0 for g in LABELS}
    per_class_top1  = {g: 0 for g in LABELS}
    per_class_top3  = {g: 0 for g in LABELS}

    # Karışıklık çiftleri için ham kayıt
    # errors[(true, pred)] = count
    errors = {}

    model = load_model()  # checkpoint’e uygun mimariyle yükler (sen zaten bunu düzelttin)

    for genre_folder in os.listdir(npz_root_dir):
        genre_path = os.path.join(npz_root_dir, genre_folder)
        if not os.path.isdir(genre_path):
            continue
        if genre_folder not in IDX:
            print(f"⚠️ Etiket listesinde olmayan klasör atlandı: {genre_folder}")
            continue

        for root, dirs, files in os.walk(genre_path):
            # 'pose' klasörünü tamamen dışla
            dirs[:] = [d for d in dirs if d.lower() != "pose"]

            for file in files:
                if not file.endswith(".npz"):
                    continue

                full_path = os.path.join(root, file)
                total += 1
                per_class_total[genre_folder] += 1

                try:
                    preds = predict_npz(full_path, model, topk=TOPK)  # [(label, prob), ...]
                    top_labels = [lbl for lbl, _ in preds]
                    pred_top1 = top_labels[0]

                    # confusion için indeks
                    y_true.append(IDX[genre_folder])
                    y_pred.append(IDX[pred_top1])

                    # top-1 / top-3
                    if pred_top1 == genre_folder:
                        correct_top1 += 1
                        per_class_top1[genre_folder] += 1
                    if genre_folder in top_labels:
                        correct_top3 += 1
                        per_class_top3[genre_folder] += 1

                    # hata çifti kaydı
                    if pred_top1 != genre_folder:
                        key = (genre_folder, pred_top1)
                        errors[key] = errors.get(key, 0) + 1

                except Exception as e:
                    print(f"⚠️ Hata: {file} işlenemedi → {e}")

    if total == 0:
        print("⚠️ Hiç .npz dosyası bulunamadı.")
        return

    # --- Genel metrikler ---
    top1 = 100.0 * correct_top1 / total
    top3 = 100.0 * correct_top3 / total
    print(f"\n📊 GENEL: Top-1 Acc: {top1:.2f}%   |   Top-3 Acc: {top3:.2f}%   |   N={total}")

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    print("\n🔢 Confusion Matrix (counts):")
    print(cm)

    print("\n📄 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=3))

    # Satıra göre normalize (her gerçek sınıfın dağılımı)
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = cm.astype(np.float32) / np.maximum(row_sums, 1)

    # Kaydet: .npy ve .csv
    np.save(f"{save_prefix}_cm_counts.npy", cm)
    np.save(f"{save_prefix}_cm_norm.npy", cm_norm)
    np.savetxt(f"{save_prefix}_cm_counts.csv", cm, fmt="%d", delimiter=",")
    np.savetxt(f"{save_prefix}_cm_norm.csv", cm_norm, fmt="%.4f", delimiter=",")

    # Basit görselleştirme (seaborn kullanmıyoruz)
    fig = plt.figure(figsize=(8.5, 7))
    plt.imshow(cm_norm, interpolation='nearest')
    plt.title("Confusion Matrix (row-normalized)")
    plt.colorbar()
    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, LABELS, rotation=45, ha='right')
    plt.yticks(tick_marks, LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    fig.savefig(f"{save_prefix}_cm.png", dpi=180)
    plt.close(fig)

    # --- En çok karışan sınıf çiftleri ---
    mixed = sorted(
        [(cnt, t, p) for (t, p), cnt in errors.items()],
        key=lambda x: x[0],
        reverse=True
    )
    print("\n🥊 En çok karıştırılan çiftler (ilk 10):")
    for cnt, t, p in mixed[:10]:
        print(f"{t} → {p} : {cnt} örnek")

    # --- Genre bazlı accuracy ---
    print("\n📌 Genre bazlı doğruluklar:")
    for g in LABELS:
        n = per_class_total[g]
        if n == 0:
            print(f"- {g}: veri yok")
            continue
        acc1 = 100.0 * per_class_top1[g] / n
        acc3 = 100.0 * per_class_top3[g] / n
        print(f"- {g}: Top-1={acc1:.2f}%  |  Top-3={acc3:.2f}%  |  N={n}")

def main():
    # Kök klasörü test_npz_files.NPZ_DIR’den alıyoruz
    print(f"📂 Değerlendirilen kök dizin: {NPZ_DIR}")
    evaluate_confusion(NPZ_DIR, save_prefix="aist_eval")

if __name__ == "__main__":
    main()
