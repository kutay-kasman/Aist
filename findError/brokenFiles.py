import os
import numpy as np

DATA_DIR = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split"
bad_files = []
total_files = 0
for root, dirs, files in os.walk(DATA_DIR):
    if "angles" not in root:
        continue  # Sadece 'angles' klasörlerini kontrol et

    for file in files:
        total_files += 1
        if file.endswith('.npz'):
            path = os.path.join(root, file)
            try:
                data = np.load(path)
                angles = data['angles']
                if np.isnan(angles).any() or np.isinf(angles).any():
                    bad_files.append(path)
            except Exception as e:
                print(f"❌ Error loading {path}: {e}")
                bad_files.append(path)

for f in bad_files:
    print("🟥", f)
print(f"\n🔍 Toplam bozuk ANGLE dosyası: {len(bad_files)}", "total files: ", total_files)
