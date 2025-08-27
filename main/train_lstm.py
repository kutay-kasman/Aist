import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_loader import AngleDataset, genre_to_idx
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ------------------ CONFIG ------------------
DATA_DIR = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split"
BATCH_SIZE = 8  
EPOCHS = 50
LEARNING_RATE = 0.001
NUM_CLASSES = len(genre_to_idx)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --------------------------------------------
# ----- LSTM Model -----
class DanceLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, num_classes=10, bidirectional=True):
        super().__init__()
        self.bi = bidirectional
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=bidirectional)
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):  # x: (B, 40, 8)
        out, _ = self.lstm(x)          # (B, 40, H*(1|2))
        last = out[:, -1, :]           # (B, H*(1|2))
        return self.fc(last)           # (B, C)


# ----- Train Function -----
def train():
    print(f"📦 Loading dataset from: {DATA_DIR}")
    dataset = AngleDataset(DATA_DIR)
    print(f"📊 Toplam sample: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DanceLSTM().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"📚 Epoch {epoch}/{EPOCHS}", unit="batch")
        for angles, labels in pbar:
            angles = angles.to(DEVICE)
            labels = labels.to(DEVICE)

            # Skip invalid data
            if torch.isnan(angles).any() or torch.isinf(angles).any():
                pbar.write("⚠️ Skipping batch: NaN or Inf in angles")
                continue
            if (labels >= NUM_CLASSES).any():
                pbar.write("⚠️ Skipping batch: Invalid label out of range")
                continue

            outputs = model(angles)
            loss = criterion(outputs, labels)

            if torch.isnan(loss):
                pbar.write("⚠️ Skipping batch: NaN loss")
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch [{epoch}/{EPOCHS}] - Avg Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)
        for param_group in optimizer.param_groups:
            print(f"🔄 Learning rate: {param_group['lr']}")

    torch.save(model.state_dict(), "lstm_dance_model4.pth")
    print("💾 Model saved as lstm_dance_model4.pth")

if __name__ == "__main__":
    train()
