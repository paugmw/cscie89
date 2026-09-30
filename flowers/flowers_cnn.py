import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
DATA_DIR = Path("flower_photos")   # folder created by extracting flower_photos.tgz
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4
SEED = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)

# ----------------------------------------------------------------------------
# Data: load every image once (resized), split 80 / 10 / 10
# ----------------------------------------------------------------------------
class_names = sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir())
samples = [(p, label) for label, name in enumerate(class_names)
           for p in sorted((DATA_DIR / name).glob("*.jpg"))]

preload = T.Resize(IMAGE_SIZE + 16)   # shorter side -> 144, keeps aspect ratio
images = [preload(Image.open(p).convert("RGB")) for p, _ in samples]
labels = [label for _, label in samples]

perm = torch.randperm(len(images)).tolist()
n_train, n_valid = int(0.8 * len(perm)), int(0.1 * len(perm))
train_idx = perm[:n_train]
valid_idx = perm[n_train:n_train + n_valid]
test_idx = perm[n_train + n_valid:]

normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_tf = T.Compose([
    T.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToImage(), T.ToDtype(torch.float32, scale=True),
    normalize,
])
eval_tf = T.Compose([
    T.CenterCrop(IMAGE_SIZE),
    T.ToImage(), T.ToDtype(torch.float32, scale=True),
    normalize,
])


class FlowerDataset(Dataset):
    def __init__(self, indices, transform):
        self.indices, self.transform = indices, transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        return self.transform(images[j]), labels[j]


train_loader = DataLoader(FlowerDataset(train_idx, train_tf), batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=4, persistent_workers=True)
valid_loader = DataLoader(FlowerDataset(valid_idx, eval_tf), batch_size=64, num_workers=2)
test_loader = DataLoader(FlowerDataset(test_idx, eval_tf), batch_size=64, num_workers=2)

# ----------------------------------------------------------------------------
# Model: still 5 convolutional layers
# ----------------------------------------------------------------------------
def conv_block(in_ch, out_ch, drop):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout2d(p=drop),
    )


class BasicCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32, 0.05),     # 128 -> 64
            conv_block(32, 64, 0.05),    # 64 -> 32
            conv_block(64, 128, 0.1),    # 32 -> 16
            conv_block(128, 256, 0.1),   # 16 -> 8
            conv_block(256, 256, 0.1),   # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),     # 256 x 4 x 4 -> 256 x 1 x 1
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = BasicCNN(len(class_names)).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LEARNING_RATE, epochs=EPOCHS, steps_per_epoch=len(train_loader))

# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def evaluate(loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            correct += (model(X).argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return correct / total


best_acc = 0.0
for epoch in range(EPOCHS):
    start = time.time()
    model.train()
    total_loss = correct = total = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()                      # OneCycle steps every batch
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    valid_acc = evaluate(valid_loader)
    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(model.state_dict(), "best_flowers_cnn.pt")
    print(f"Epoch {epoch + 1:2d}/{EPOCHS}  loss {total_loss / total:.4f}  "
          f"train acc {correct / total:.3f}  valid acc {valid_acc:.3f}  "
          f"({time.time() - start:.0f}s)", flush=True)

model.load_state_dict(torch.load("best_flowers_cnn.pt"))
print(f"Best valid acc: {best_acc:.3f}")
print(f"Test acc:       {evaluate(test_loader):.3f}")
