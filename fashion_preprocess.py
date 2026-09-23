"""Step 2: Preprocess the images.

Two preprocessing steps are applied because they usually help the model:

1. Standardization: pixels are shifted/scaled with the *training set* mean and std, so the
   inputs are centered around 0 with unit variance. This makes optimization faster and more
   stable than raw [0, 1] pixels. The statistics come from the training split only, so no
   information leaks from validation/test.
2. Data augmentation (training set only): a random horizontal flip. A mirrored shirt or shoe
   is still the same class, so this gives the model more varied examples and reduces
   overfitting. Validation and test images are never augmented.
"""

import torch
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset, Subset


def compute_mean_std(train_subset: Subset):
    """Compute pixel mean and std over the training images only (values in [0, 1])."""
    pixels = train_subset.dataset.data[train_subset.indices].float() / 255.0
    return pixels.mean().item(), pixels.std().item()


def get_transforms(mean: float, std: float, augment: bool = True):
    """Return (train_transform, eval_transform) to apply on top of the [0, 1] tensors."""
    normalize = T.Normalize(mean=[mean], std=[std])
    eval_transform = normalize
    train_transform = T.Compose([T.RandomHorizontalFlip(), normalize]) if augment else normalize
    return train_transform, eval_transform


class TransformedDataset(Dataset):
    """Wraps a dataset and applies an extra transform to each image (labels unchanged)."""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        X, y = self.dataset[idx]
        return self.transform(X), y


def preprocess_splits(train_data, valid_data, test_data, augment: bool = True):
    """Apply standardization (and augmentation on train) to the three splits."""
    mean, std = compute_mean_std(train_data)
    train_transform, eval_transform = get_transforms(mean, std, augment)
    return (TransformedDataset(train_data, train_transform),
            TransformedDataset(valid_data, eval_transform),
            TransformedDataset(test_data, eval_transform),
            (mean, std))


if __name__ == "__main__":
    from fashion_split import load_fashion_mnist_splits

    train, valid, test, (mean, std) = preprocess_splits(*load_fashion_mnist_splits())
    print(f"training mean: {mean:.4f}, std: {std:.4f}")
    X_valid = torch.stack([valid[i][0] for i in range(1000)])
    print(f"validation sample after standardization: mean={X_valid.mean():.3f}, "
          f"std={X_valid.std():.3f}")
