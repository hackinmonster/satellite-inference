import os
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import CLASSES, LandCoverNet

IMG_SIZE = 64
SAMPLES_PER_CLASS = 250


def make_tile(label, size=IMG_SIZE):
    """Generate a fake satellite tile. Colors are exaggerated on purpose so
    a small CNN can actually learn something without a real dataset."""
    rng = np.random.default_rng()
    img = np.zeros((size, size, 3), dtype=np.uint8)

    if label == "water":
        img[:, :, 2] = rng.integers(90, 180, (size, size))
        img[:, :, 1] = rng.integers(40, 90, (size, size))
        img[:, :, 0] = rng.integers(10, 40, (size, size))
        # a few whitecap-looking specks
        for _ in range(20):
            y, x = rng.integers(0, size, 2)
            img[y : y + 2, x : x + 2] = [200, 210, 230]
    elif label == "forest":
        img[:, :, 1] = rng.integers(70, 140, (size, size))
        img[:, :, 0] = rng.integers(20, 60, (size, size))
        img[:, :, 2] = rng.integers(10, 40, (size, size))
    elif label == "urban":
        gray = rng.integers(70, 160, (size, size))
        img[:, :, 0] = gray
        img[:, :, 1] = gray - rng.integers(0, 15, (size, size))
        img[:, :, 2] = gray - rng.integers(0, 20, (size, size))
        # crude "buildings"
        for _ in range(8):
            y, x = rng.integers(0, size - 8, 2)
            img[y : y + 8, x : x + 6] = rng.integers(40, 80, 3)
    elif label == "agriculture":
        img[:, :, 1] = rng.integers(90, 170, (size, size))
        img[:, :, 0] = rng.integers(60, 120, (size, size))
        img[:, :, 2] = rng.integers(20, 50, (size, size))
        # field strips
        for i in range(0, size, 8):
            img[i : i + 3] = img[i : i + 3] * 0.7
    elif label == "barren":
        img[:, :, 0] = rng.integers(140, 200, (size, size))
        img[:, :, 1] = rng.integers(110, 160, (size, size))
        img[:, :, 2] = rng.integers(70, 110, (size, size))
    elif label == "cloud":
        base = rng.integers(180, 240, (size, size))
        img[:, :, 0] = base
        img[:, :, 1] = base
        img[:, :, 2] = np.clip(base + 10, 0, 255)

    return Image.fromarray(img)


class TileDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img, label = self.items[idx]
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor, label


def build_data():
    items = []
    for i, name in enumerate(CLASSES):
        for _ in range(SAMPLES_PER_CLASS):
            items.append((make_tile(name), i))
    random.shuffle(items)
    split = int(0.8 * len(items))
    return TileDataset(items[:split]), TileDataset(items[split:])


def main():
    os.makedirs("weights", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    for name in CLASSES:
        make_tile(name).save(f"samples/{name}.png")

    train_ds, val_ds = build_data()
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}")

    model = LandCoverNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(8):
        model.train()
        total = 0
        correct = 0
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)
        train_acc = correct / total

        model.eval()
        vtotal = 0
        vcorrect = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                vcorrect += (pred == y).sum().item()
                vtotal += x.size(0)
        print(
            f"epoch {epoch+1}  loss={running/total:.3f}  "
            f"train_acc={train_acc:.3f}  val_acc={vcorrect/vtotal:.3f}"
        )

    torch.save(model.state_dict(), "weights/landcover.pt")
    print("saved weights/landcover.pt")


if __name__ == "__main__":
    main()
