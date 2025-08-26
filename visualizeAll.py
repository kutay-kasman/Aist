import numpy as np
import matplotlib.pyplot as plt
import os

# 🔴 Broken .npz files — Listeyi kendi dosyalarınla doldur
broken_files = [
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d04_mBR1_ch02\angles\gBR_sFM_c01_d04_mBR1_ch02_004_angles.npz",
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d04_mBR2_ch03\angles\gBR_sFM_c01_d04_mBR2_ch03_006_angles.npz",
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d04_mBR1_ch02\angles\gBR_sFM_c01_d04_mBR1_ch02_004_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d04_mBR2_ch03\angles\gBR_sFM_c01_d04_mBR2_ch03_006_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d04_mBR2_ch03\angles\gBR_sFM_c01_d04_mBR2_ch03_008_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d04_mBR3_ch04\angles\gBR_sFM_c01_d04_mBR3_ch04_004_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d04_mBR3_ch04\angles\gBR_sFM_c01_d04_mBR3_ch04_005_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d04_mBR4_ch05\angles\gBR_sFM_c01_d04_mBR4_ch05_004_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d04_mBR4_ch05\angles\gBR_sFM_c01_d04_mBR4_ch05_005_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d04_mBR4_ch05\angles\gBR_sFM_c01_d04_mBR4_ch05_007_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d04_mBR4_ch07\angles\gBR_sFM_c01_d04_mBR4_ch07_001_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d04_mBR4_ch07\angles\gBR_sFM_c01_d04_mBR4_ch07_007_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d04_mBR5_ch06\angles\gBR_sFM_c01_d04_mBR5_ch06_005_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d05_mBR4_ch13\angles\gBR_sFM_c01_d05_mBR4_ch13_001_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d05_mBR4_ch13\angles\gBR_sFM_c01_d05_mBR4_ch13_003_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d05_mBR4_ch13\angles\gBR_sFM_c01_d05_mBR4_ch13_005_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d05_mBR5_ch14\angles\gBR_sFM_c01_d05_mBR5_ch14_000_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d05_mBR5_ch14\angles\gBR_sFM_c01_d05_mBR5_ch14_002_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d05_mBR5_ch14\angles\gBR_sFM_c01_d05_mBR5_ch14_004_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d05_mBR5_ch14\angles\gBR_sFM_c01_d05_mBR5_ch14_005_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d06_mBR2_ch16\angles\gBR_sFM_c01_d06_mBR2_ch16_008_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d06_mBR4_ch20\angles\gBR_sFM_c01_d06_mBR4_ch20_003_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d06_mBR4_ch20\angles\gBR_sFM_c01_d06_mBR4_ch20_004_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d06_mBR4_ch20\angles\gBR_sFM_c01_d06_mBR4_ch20_005_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d06_mBR4_ch20\angles\gBR_sFM_c01_d06_mBR4_ch20_006_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d06_mBR5_ch19\angles\gBR_sFM_c01_d06_mBR5_ch19_003_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d06_mBR5_ch21\angles\gBR_sFM_c01_d06_mBR5_ch21_002_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d06_mBR5_ch21\angles\gBR_sFM_c01_d06_mBR5_ch21_003_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d06_mBR5_ch21\angles\gBR_sFM_c01_d06_mBR5_ch21_004_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\BR\gBR_sFM_c01_d06_mBR5_ch21\angles\gBR_sFM_c01_d06_mBR5_ch21_005_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\HO\gHO_sFM_c01_d19_mHO3_ch04\angles\gHO_sFM_c01_d19_mHO3_ch04_000_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\HO\gHO_sFM_c01_d19_mHO3_ch04\angles\gHO_sFM_c01_d19_mHO3_ch04_003_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\HO\gHO_sFM_c01_d20_mHO1_ch09\angles\gHO_sFM_c01_d20_mHO1_ch09_004_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\HO\gHO_sFM_c01_d20_mHO3_ch11\angles\gHO_sFM_c01_d20_mHO3_ch11_001_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\HO\gHO_sFM_c01_d20_mHO3_ch14\angles\gHO_sFM_c01_d20_mHO3_ch14_004_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\HO\gHO_sFM_c01_d20_mHO5_ch13\angles\gHO_sFM_c01_d20_mHO5_ch13_000_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\HO\gHO_sFM_c01_d21_mHO3_ch18\angles\gHO_sFM_c01_d21_mHO3_ch18_006_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\HO\gHO_sFM_c01_d21_mHO3_ch21\angles\gHO_sFM_c01_d21_mHO3_ch21_004_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\JB\gJB_sFM_c01_d07_mJB0_ch01\angles\gJB_sFM_c01_d07_mJB0_ch01_003_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\JB\gJB_sFM_c01_d07_mJB2_ch03\angles\gJB_sFM_c01_d07_mJB2_ch03_006_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\JB\gJB_sFM_c01_d08_mJB0_ch08\angles\gJB_sFM_c01_d08_mJB0_ch08_000_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\JB\gJB_sFM_c01_d09_mJB0_ch15\angles\gJB_sFM_c01_d09_mJB0_ch15_000_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\JB\gJB_sFM_c01_d09_mJB1_ch16\angles\gJB_sFM_c01_d09_mJB1_ch16_005_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\JB\gJB_sFM_c01_d09_mJB2_ch17\angles\gJB_sFM_c01_d09_mJB2_ch17_005_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\JB\gJB_sFM_c01_d09_mJB3_ch18\angles\gJB_sFM_c01_d09_mJB3_ch18_001_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\JB\gJB_sFM_c01_d09_mJB3_ch18\angles\gJB_sFM_c01_d09_mJB3_ch18_004_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\JB\gJB_sFM_c01_d09_mJB4_ch19\angles\gJB_sFM_c01_d09_mJB4_ch19_002_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\JB\gJB_sFM_c01_d09_mJB4_ch19\angles\gJB_sFM_c01_d09_mJB4_ch19_003_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\JB\gJB_sFM_c01_d09_mJB4_ch19\angles\gJB_sFM_c01_d09_mJB4_ch19_004_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\JS\gJS_sFM_c01_d01_mJS1_ch02\angles\gJS_sFM_c01_d01_mJS1_ch02_001_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\JS\gJS_sFM_c01_d01_mJS1_ch07\angles\gJS_sFM_c01_d01_mJS1_ch07_001_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\JS\gJS_sFM_c01_d01_mJS2_ch03\angles\gJS_sFM_c01_d01_mJS2_ch03_000_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\JS\gJS_sFM_c01_d02_mJS0_ch08\angles\gJS_sFM_c01_d02_mJS0_ch08_000_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\JS\gJS_sFM_c01_d02_mJS0_ch08\angles\gJS_sFM_c01_d02_mJS0_ch08_003_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\JS\gJS_sFM_c01_d02_mJS0_ch08\angles\gJS_sFM_c01_d02_mJS0_ch08_004_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\JS\gJS_sFM_c01_d02_mJS2_ch03\angles\gJS_sFM_c01_d02_mJS2_ch03_005_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\JS\gJS_sFM_c01_d03_mJS0_ch01\angles\gJS_sFM_c01_d03_mJS0_ch01_001_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\JS\gJS_sFM_c01_d03_mJS0_ch01\angles\gJS_sFM_c01_d03_mJS0_ch01_002_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\JS\gJS_sFM_c01_d03_mJS2_ch03\angles\gJS_sFM_c01_d03_mJS2_ch03_000_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\KR\gKR_sFM_c01_d28_mKR3_ch04\angles\gKR_sFM_c01_d28_mKR3_ch04_001_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\KR\gKR_sFM_c01_d28_mKR3_ch07\angles\gKR_sFM_c01_d28_mKR3_ch07_003_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\KR\gKR_sFM_c01_d30_mKR0_ch15\angles\gKR_sFM_c01_d30_mKR0_ch15_005_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\LO\gLO_sFM_c01_d14_mLO1_ch09\angles\gLO_sFM_c01_d14_mLO1_ch09_008_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\LO\gLO_sFM_c01_d14_mLO2_ch10\angles\gLO_sFM_c01_d14_mLO2_ch10_000_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\LO\gLO_sFM_c01_d14_mLO2_ch10\angles\gLO_sFM_c01_d14_mLO2_ch10_003_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\MH\gMH_sFM_c01_d22_mMH1_ch02\angles\gMH_sFM_c01_d22_mMH1_ch02_007_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\MH\gMH_sFM_c01_d22_mMH1_ch07\angles\gMH_sFM_c01_d22_mMH1_ch07_004_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\MH\gMH_sFM_c01_d22_mMH3_ch04\angles\gMH_sFM_c01_d22_mMH3_ch04_000_angles.npz"
    r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split\WA\gWA_sFM_c01_d25_mWA5_ch06\angles\gWA_sFM_c01_d25_mWA5_ch06_003_angles.npz"
]

angle_names = [
    "Elbow (L)", "Elbow (R)",
    "Knee (L)", "Knee (R)",
    "Shoulder (L)", "Shoulder (R)",
    "Hip (L)", "Hip (R)"
]

for file in broken_files:
    print(f"🔍 Showing angles for: {file}")
    try:
        data = np.load(file)
        angles = data["angles"]
        label = data["label"].tolist() if "label" in data else "unknown"

        plt.figure(figsize=(14, 10))
        for i in range(8):
            plt.subplot(4, 2, i + 1)
            plt.plot(angles[:, i])
            plt.title(f"{angle_names[i]}")
            plt.xlabel("Frame")
            plt.ylabel("Angle (°)")
            plt.ylim(0, 180)
            plt.grid(True)

        plt.suptitle(f"{os.path.basename(file)} — Label: {label}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    except Exception as e:
        print(f"❌ Could not load {file}: {e}")
