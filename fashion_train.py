"""Step 4: Build a CNN image classifier, train it with accuracy as the target metric,
plot train/validation accuracy per epoch, and return the trained model."""

import copy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from hardware_check import get_device


class FashionCNN(nn.Module):
    """Small convolutional network for 1x28x28 grayscale images and 10 classes.

    Two conv blocks (conv -> batchnorm -> relu, twice, then max-pool) extract visual
    features; the classifier head maps them to 10 class scores (logits).
    """

    def __init__(self, n_classes: int = 10, dropout: float = 0.3):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(1, 32),   # -> 32 x 14 x 14
            conv_block(32, 64),  # -> 64 x 7 x 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, X):
        return self.classifier(self.features(X))


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """Run one pass over `loader`. Trains if an optimizer is given, otherwise only evaluates.

    Returns (mean loss, accuracy).
    """
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y_batch)
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total += len(y_batch)
    return total_loss / total, correct / total


def train_model(train_loader, valid_loader, n_epochs: int = 15, lr: float = 1e-3,
                device=None, seed: int = 42):
    """Train a FashionCNN and return (model, history).

    - Loss: cross-entropy (the standard loss for multi-class classification).
    - Optimizer: Adam, with the learning rate halved when validation accuracy stops improving.
    - The returned model holds the weights from the epoch with the best validation accuracy.
    """
    device = device or get_device()
    torch.manual_seed(seed)
    model = FashionCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2)

    history = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}
    best_acc, best_state = 0.0, None
    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
        valid_loss, valid_acc = run_epoch(model, valid_loader, loss_fn, device)
        scheduler.step(valid_acc)
        for key, value in zip(history, (train_loss, train_acc, valid_loss, valid_acc)):
            history[key].append(value)
        if valid_acc > best_acc:
            best_acc, best_state = valid_acc, copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch:2d}/{n_epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"valid loss {valid_loss:.4f} acc {valid_acc:.4f}")

    model.load_state_dict(best_state)
    best_epoch = history["valid_acc"].index(best_acc)
    print(f"\nFinal (last epoch)  -> train loss {history['train_loss'][-1]:.4f}, "
          f"train acc {history['train_acc'][-1]:.4f}, "
          f"valid loss {history['valid_loss'][-1]:.4f}, valid acc {history['valid_acc'][-1]:.4f}")
    print(f"Best model (epoch {best_epoch + 1}) -> valid loss {history['valid_loss'][best_epoch]:.4f}, "
          f"valid acc {best_acc:.4f}  (these weights are returned)")
    return model, history


def plot_accuracy(history, save_path: str | None = "fashion_accuracy.png"):
    """Plot train and validation accuracy for each epoch."""
    epochs = range(1, len(history["train_acc"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], "o-", label="Train accuracy")
    plt.plot(epochs, history["valid_acc"], "s-", label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("FashionMNIST CNN: accuracy per epoch")
    plt.xticks(list(epochs))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()


if __name__ == "__main__":
    from fashion_dataloaders import get_dataloaders

    train_loader, valid_loader, test_loader, class_names = get_dataloaders()
    model, history = train_model(train_loader, valid_loader)
    plot_accuracy(history)
    torch.save(model.state_dict(), "fashion_cnn.pt")
