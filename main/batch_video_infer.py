import os
from single_video_infer import predict_video

VIDEO_DIR = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split_excerpt"
TOPK = 3  # You can also measure top-3 accuracy later

def is_video(filename):
    return filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))

total = 0
correct_top1 = 0
correct_top3 = 0

for genre_folder in os.listdir(VIDEO_DIR):
    genre_path = os.path.join(VIDEO_DIR, genre_folder)
    if not os.path.isdir(genre_path):
        continue

    for file in os.listdir(genre_path):
        if not is_video(file):
            continue

        full_path = os.path.join(genre_path, file)
        total += 1
        try:
            preds = predict_video(full_path, topk=TOPK)
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
print("\n📊 Top-1 Accuracy: {:.2f}%".format(100 * correct_top1 / total))
print("📊 Top-3 Accuracy: {:.2f}%".format(100 * correct_top3 / total))
print(f"🎥 Total videos tested: {total}")
