import os
import numpy as np

DATA_DIR = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split"
bad_files = []
total_files = 0
deleted_count = 0

for root, dirs, files in os.walk(DATA_DIR):
    if "angles" not in root:
        continue  # Sadece 'angles' klasörlerini kontrol et

    for file in files:
        if not file.endswith('.npz'):
            continue
        total_files += 1
        path = os.path.join(root, file)
        try:
            with np.load(path) as data:
                angles = data['angles']
                if np.isnan(angles).any() or np.isinf(angles).any():
                    print("🟥 NaN/Inf tespit edildi ve silindi:", path)
                    os.remove(path)
                    deleted_count += 1
        except Exception as e:
            print(f"❌ Hata oluştu ve silindi: {path} — Hata: {e}")
            try:
                os.remove(path)  # Dosya açıksa bile yeniden silmeyi dene
                deleted_count += 1
            except Exception as inner_e:
                print(f"🚫 Silinemedi tekrar: {path} — Hata: {inner_e}")
            bad_files.append(path)

print(f"\n🗑️ Silinen bozuk dosya sayısı: {deleted_count}")
print(f"📁 Toplam dosya sayısı (angles klasörlerinde): {total_files}")
