import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_loader import AngleDataset, genre_to_idx
from tqdm import tqdm

# ------------------ CONFIG ------------------
DATA_DIR = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_genre_split"
BATCH_SIZE = 8  
EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = len(genre_to_idx)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --------------------------------------------
# ----- LSTM Model -----
class DanceLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, num_classes=NUM_CLASSES):
        super(DanceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):  # x: (B, 40, 8)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # only last time step
        return self.fc(out)

# ----- Train Function -----
def train():
    print(f"📦 Loading dataset from: {DATA_DIR}")
    dataset = AngleDataset(DATA_DIR)
    print(f"📊 Toplam sample: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DanceLSTM().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

    torch.save(model.state_dict(), "lstm_dance_model.pth")
    print("💾 Model saved as lstm_dance_model.pth")

if __name__ == "__main__":
    train()
