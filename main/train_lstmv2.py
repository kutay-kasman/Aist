import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_loader import AngleDataset, genre_to_idx
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ------------------ CONFIG ------------------
TRAIN_DIR = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_split_ready\train"
VAL_DIR = r"C:\Users\kutay\OneDrive\Desktop\AISTpp_split_ready\val"
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.001
NUM_CLASSES = len(genre_to_idx)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "lstm_dance_model_new.pth"
# --------------------------------------------

# --------- LSTM MODEL -----------------------
class DanceLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, num_classes=NUM_CLASSES, bidirectional=True):
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
# --------------------------------------------

# --------- TRAIN FUNCTION -------------------
def train():
    print(f"📦 Loading datasets...")
    train_dataset = AngleDataset(TRAIN_DIR)
    val_dataset = AngleDataset(VAL_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"📊 Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    model = DanceLSTM().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"📚 Epoch {epoch}/{EPOCHS}", unit="batch")
        for angles, labels in pbar:
            angles = angles.to(DEVICE)
            labels = labels.to(DEVICE)

            # Skip corrupted data
            if torch.isnan(angles).any() or torch.isinf(angles).any():
                pbar.write("⚠️ Skipping: NaN or Inf in input")
                continue
            if (labels >= NUM_CLASSES).any():
                pbar.write("⚠️ Skipping: Invalid label")
                continue

            outputs = model(angles)
            loss = criterion(outputs, labels)

            if torch.isnan(loss):
                pbar.write("⚠️ Skipping: NaN loss")
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for angles, labels in val_loader:
                angles = angles.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(angles)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct / len(val_dataset)

        print(f"\n✅ Epoch {epoch} Summary:")
        print(f"   🏋️ Train Loss: {avg_train_loss:.4f}")
        print(f"   🧪 Val Loss:   {avg_val_loss:.4f} | Accuracy: {val_acc:.2f}%")

        scheduler.step(avg_val_loss)
        for param_group in optimizer.param_groups:
            print(f"   🔄 LR: {param_group['lr']}")

    # -------- SAVE MODEL --------
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 Model saved to {MODEL_SAVE_PATH}")
# --------------------------------------------

if __name__ == "__main__":
    train()
