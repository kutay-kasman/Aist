import numpy as np

# Yazdırma sınırlarını kaldır
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# Dosyayı yükle
npz_path = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\WA\gWA_sFM_c01_d25_mWA5_ch06\pose\gWA_sFM_c01_d25_mWA5_ch06_003_pose.npz"
data = np.load(npz_path)

# Anahtarları göster
print("Keys:", data.files)

# Tüm keypoint verisini yazdır
if 'keypoints' in data:
    print("Keypoints shape:", data['keypoints'].shape)
    print("Keypoints data:\n", data['keypoints'])
else:
    print("❌ 'keypoints' anahtarı bulunamadı.")
