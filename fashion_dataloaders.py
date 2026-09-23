"""Step 3: Build DataLoaders so the model is trained and evaluated in mini-batches."""

import torch
from torch.utils.data import DataLoader

from fashion_preprocess import preprocess_splits
from fashion_split import load_fashion_mnist_splits


def get_dataloaders(batch_size: int = 128, augment: bool = True, num_workers: int = 0,
                    root: str = "datasets", seed: int = 42):
    """Return (train_loader, valid_loader, test_loader, class_names).

    The training loader shuffles each epoch; validation/test keep a fixed order.
    Increase num_workers (e.g. 2-4) to load batches in parallel if data loading is slow.
    """
    train_data, valid_data, test_data = load_fashion_mnist_splits(root=root, seed=seed)
    class_names = test_data.classes
    train_data, valid_data, test_data, _ = preprocess_splits(
        train_data, valid_data, test_data, augment=augment)

    pin_memory = torch.cuda.is_available()  # speeds up CPU -> GPU copies
    generator = torch.Generator().manual_seed(seed)  # reproducible shuffling
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              generator=generator)
    valid_loader = DataLoader(valid_data, batch_size=batch_size,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader, class_names


if __name__ == "__main__":
    train_loader, valid_loader, test_loader, class_names = get_dataloaders()
    X_batch, y_batch = next(iter(train_loader))
    print(f"batches -> train: {len(train_loader)}, valid: {len(valid_loader)}, "
          f"test: {len(test_loader)}")
    print(f"X batch: {tuple(X_batch.shape)} {X_batch.dtype}, y batch: {tuple(y_batch.shape)} "
          f"{y_batch.dtype}")
    print(f"classes: {class_names}")
