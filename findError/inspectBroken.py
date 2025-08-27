import os
import numpy as np

# def inspect_npz(path):
#     print(f"\n📂 {path}")
#     try:
#         data = np.load(path)
#         if 'angles' not in data:
#             print("❌ 'angles' anahtarı eksik!")
#             return
#         angles = data['angles']
#         print("✅ 'angles' shape:", angles.shape)

#         if np.isnan(angles).any():
#             print("🟥 İçerikte NaN (Not a Number) var.")
#         if np.isinf(angles).any():
#             print("🟥 İçerikte Inf (sonsuz) var.")
#         if angles.shape != (40, 8):
#             print("🟨 Beklenmeyen boyut!", angles.shape)

#         if 'label' in data:
#             print("🎯 Label:", data['label'])
#         else:
#             print("⚠️ Label eksik.")

#     except Exception as e:
#         print("💥 Hata oluştu:", str(e))

# def bulk_inspect_angles(root_folder):
#     for root, dirs, files in os.walk(root_folder):
#         if 'pose' in root:  # ❗️ pose klasörlerini atla
#             continue
#         for file in files:
#             if file.endswith('.npz'):
#                 path = os.path.join(root, file)
#                 inspect_npz(path)

# # Başlat
# bulk_inspect_angles(r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split")

# Pick one suspicious file
path = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\WA\gWA_sFM_c01_d25_mWA5_ch06\angles\gWA_sFM_c01_d25_mWA5_ch06_003_angles.npz"

try:
    data = np.load(path)
    print("Keys:", data.files)

    if "angles" in data:
        print("angles shape:", data["angles"].shape)
        print("angles sample:\n", data["angles"][:5])  # first 5 frames
    else:
        print("🚨 'angles' key missing!")

    if "label" in data:
        print("label:", data["label"])
    else:
        print("🚨 'label' key missing!")

except Exception as e:
    print(f"❌ Error reading {path}:\n{e}")
