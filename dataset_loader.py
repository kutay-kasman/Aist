# dataset_loader.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset

genre_to_idx = {
    'BR': 0, 'HO': 1, 'JB': 2, 'JS': 3, 'KR': 4,
    'LH': 5, 'LO': 6, 'MH': 7, 'PO': 8, 'WA': 9
}

class AngleDataset(Dataset):
    def __init__(self, root_folder):
        self.samples = []
        for genre in os.listdir(root_folder):
            genre_path = os.path.join(root_folder, genre)
            if not os.path.isdir(genre_path):
                continue
            for video_folder in os.listdir(genre_path):
                angles_folder = os.path.join(genre_path, video_folder, 'angles')
                if os.path.exists(angles_folder):
                    for file in os.listdir(angles_folder):
                        if file.endswith('.npz'):
                            self.samples.append(os.path.join(angles_folder, file))
    

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        data = np.load(path)
        angles = data['angles']  # (40, 8)
        label_str = str(data['label']).upper()
        label = genre_to_idx[label_str]
        return torch.tensor(angles, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
