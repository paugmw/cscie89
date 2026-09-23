"""Step 5: Evaluate a trained model on the test set (loss, accuracy, per-class accuracy)."""

import torch
import torch.nn as nn

from hardware_check import get_device


def evaluate_model(model, test_loader, class_names=None, device=None):
    """Return (test_loss, test_accuracy) and print accuracy per class."""
    device = device or next(model.parameters()).device
    model.to(device).eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    n_classes = len(class_names) if class_names else 10
    correct_per_class = torch.zeros(n_classes)
    total_per_class = torch.zeros(n_classes)
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            total_loss += loss_fn(logits, y_batch).item()
            preds = logits.argmax(dim=1)
            for c in range(n_classes):
                mask = y_batch == c
                total_per_class[c] += mask.sum().item()
                correct_per_class[c] += (preds[mask] == c).sum().item()

    n = total_per_class.sum().item()
    test_loss = total_loss / n
    test_acc = correct_per_class.sum().item() / n

    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")
    if class_names:
        print("Accuracy per class:")
        for name, correct, total in zip(class_names, correct_per_class, total_per_class):
            print(f"  {name:<12} {correct / total:.4f}")
    return test_loss, test_acc


if __name__ == "__main__":
    from fashion_dataloaders import get_dataloaders
    from fashion_train import FashionCNN

    _, _, test_loader, class_names = get_dataloaders()
    device = get_device()
    model = FashionCNN().to(device)
    model.load_state_dict(torch.load("fashion_cnn.pt", map_location=device))
    evaluate_model(model, test_loader, class_names)
