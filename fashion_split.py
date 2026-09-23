"""Step 1: Load FashionMNIST from torchvision and split it into train / validation / test."""

import torch
import torchvision
import torchvision.transforms.v2 as T
from torch.utils.data import random_split

# Converts a PIL image to a float32 tensor of shape [1, 28, 28] with values in [0, 1].
TO_TENSOR = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])


def load_fashion_mnist_splits(root: str = "datasets", valid_size: int = 5_000, seed: int = 42):
    """Return (train_data, valid_data, test_data).

    FashionMNIST ships with 60,000 training and 10,000 test images. The official test set
    is kept untouched for the final evaluation, and the 60,000 training images are split
    into 55,000 for training and 5,000 for validation (used to monitor each epoch).
    """
    train_and_valid = torchvision.datasets.FashionMNIST(
        root=root, train=True, download=True, transform=TO_TENSOR)
    test_data = torchvision.datasets.FashionMNIST(
        root=root, train=False, download=True, transform=TO_TENSOR)

    train_size = len(train_and_valid) - valid_size
    generator = torch.Generator().manual_seed(seed)  # reproducible split
    train_data, valid_data = random_split(
        train_and_valid, [train_size, valid_size], generator=generator)

    return train_data, valid_data, test_data


if __name__ == "__main__":
    train_data, valid_data, test_data = load_fashion_mnist_splits()
    print(f"train: {len(train_data)}, valid: {len(valid_data)}, test: {len(test_data)}")
    X, y = train_data[0]
    print(f"sample image shape: {tuple(X.shape)}, dtype: {X.dtype}, label: {y} "
          f"({test_data.classes[y]})")
