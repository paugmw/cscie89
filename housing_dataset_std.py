"""Load California housing, split into train/valid/test, standardize X, return torch tensors."""

import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def housing_dataset_std(random_state: int = 42, device: torch.device | str = "cpu"):
    """Return (X_train, y_train, X_valid, y_valid, X_test, y_test) as float32 tensors.

    - X tensors have shape (n_samples, 8), standardized with the training-set mean/std
      (the scaler is fit on train only, so no information leaks from valid/test).
    - y tensors have shape (n_samples, 1) so they match a model output of shape (N, 1)
      when used with losses like nn.MSELoss.
    """
    housing = fetch_california_housing()

    # 1) Split: test = 25% of all data, then valid = 25% of the remaining training data.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        housing.data, housing.target, random_state=random_state)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, random_state=random_state)

    # 2) Standardize X using statistics from the training set only.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    # 3) Convert to torch tensors (float32, as expected by torch layers) on the target device.
    def to_x(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    def to_y(a):
        return torch.tensor(a, dtype=torch.float32, device=device).reshape(-1, 1)

    return (to_x(X_train), to_y(y_train),
            to_x(X_valid), to_y(y_valid),
            to_x(X_test), to_y(y_test))


if __name__ == "__main__":
    X_train, y_train, X_valid, y_valid, X_test, y_test = housing_dataset_std()
    for name, t in [("X_train", X_train), ("y_train", y_train),
                    ("X_valid", X_valid), ("y_valid", y_valid),
                    ("X_test", X_test), ("y_test", y_test)]:
        print(f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}")
    print(f"X_train mean per feature ~0: {X_train.mean(dim=0).abs().max().item():.2e}")
    print(f"X_train std per feature ~1:  {X_train.std(dim=0).mean().item():.4f}")
