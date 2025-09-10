import os
import random
import shutil

# === CONFIG ===
TRAIN_VAL_ROOT = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split"
TEST_ROOT = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split_excerpt"
DEST_ROOT = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_split_ready"
SPLIT_RATIO = {'train': 0.8, 'val': 0.2}  # train/val ratio for TRAIN_VAL_ROOT
SEED = 42
random.seed(SEED)
# ===============

def copy_angles(src_video_path, dest_base, genre, video_name):
    src_angles = os.path.join(src_video_path, 'angles')
    if not os.path.exists(src_angles):
        return
    dest_dir = os.path.join(dest_base, genre, video_name, 'angles')
    os.makedirs(dest_dir, exist_ok=True)
    for file in os.listdir(src_angles):
        if file.endswith('.npz'):
            shutil.copy2(os.path.join(src_angles, file), os.path.join(dest_dir, file))

def split_train_val():
    genres = [g for g in os.listdir(TRAIN_VAL_ROOT) if os.path.isdir(os.path.join(TRAIN_VAL_ROOT, g))]
    for genre in genres:
        genre_path = os.path.join(TRAIN_VAL_ROOT, genre)
        videos = [v for v in os.listdir(genre_path) if os.path.isdir(os.path.join(genre_path, v))]
        random.shuffle(videos)
        
        n_total = len(videos)
        n_train = int(n_total * SPLIT_RATIO['train'])
        train_videos = videos[:n_train]
        val_videos = videos[n_train:]

        for vid in train_videos:
            video_path = os.path.join(genre_path, vid)
            copy_angles(video_path, os.path.join(DEST_ROOT, 'train'), genre, vid)

        for vid in val_videos:
            video_path = os.path.join(genre_path, vid)
            copy_angles(video_path, os.path.join(DEST_ROOT, 'val'), genre, vid)

        print(f"✅ {genre}: {n_total} video → train: {len(train_videos)}, val: {len(val_videos)}")

def copy_test_set():
    genres = [g for g in os.listdir(TEST_ROOT) if os.path.isdir(os.path.join(TEST_ROOT, g))]
    for genre in genres:
        genre_path = os.path.join(TEST_ROOT, genre)
        videos = [v for v in os.listdir(genre_path) if os.path.isdir(os.path.join(genre_path, v))]

        for vid in videos:
            video_path = os.path.join(genre_path, vid)
            copy_angles(video_path, os.path.join(DEST_ROOT, 'test'), genre, vid)
        
        print(f"✅ {genre}: test: {len(videos)} video copied.")

if __name__ == "__main__":
    split_train_val()
    copy_test_set()
    print("🎉 All angles split and copied successfully.")
