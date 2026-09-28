"""Step 3: predict next week's IPC return from sentiment up to week t-1.

Everything is evaluated walk-forward: for each week t the model is trained only on
weeks before t, then predicts week t. This is the only honest way to measure it;
an in-sample fit on 100 weekly points will always look great and mean nothing.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_features(matrix, returns, lags=1, use_volume=False, use_changes=False, use_past_returns=False):
    """X row for week t uses only information published before week t's Monday decision.

    Keep the feature count small: with ~100 weekly observations, every extra column
    (more lags, news volume, changes) makes overfitting more likely. The defaults give
    one feature per subject.

    Returns (X, y, X_next): the training rows, their targets, and the feature row for
    the upcoming week, whose return is not known yet (used for the live decision).
    """
    next_week = returns.index[-1] + pd.Timedelta(days=7)
    index = returns.index.append(pd.DatetimeIndex([next_week]))
    returns = returns.reindex(index)
    base = matrix.reindex(index.union(matrix.index)).ffill().reindex(index).fillna(0.0)
    sent_cols = [c for c in base.columns if c.endswith("_sent")]
    cols = sent_cols + ([c for c in base.columns if c.endswith("_n")] if use_volume else [])
    base = base[cols].copy()
    vol_cols = [c for c in cols if c.endswith("_n")]
    base[vol_cols] = np.log1p(base[vol_cols])

    parts = [base.shift(k).add_suffix(f"_lag{k}") for k in range(1, lags + 1)]
    for c in sent_cols if use_changes else []:  # change in sentiment over the last two weeks
        parts.append((base[c].shift(1) - base[c].shift(2)).rename(f"{c}_chg"))
    if use_past_returns:
        parts.append(returns.shift(1).rename("ret_lag1"))

    X = pd.concat(parts, axis=1).dropna()
    y = returns.reindex(X.index)
    has_y = y.notna()
    return X[has_y], y[has_y], X.iloc[[-1]]


def make_ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30)))


class TorchMLP:
    """Small feed-forward net (same idea as the course notebook). Needs `torch`."""

    def __init__(self, hidden=16, epochs=300, lr=1e-3, weight_decay=1e-2, dropout=0.3, seed=0):
        self.hidden, self.epochs, self.lr = hidden, epochs, lr
        self.weight_decay, self.dropout, self.seed = weight_decay, dropout, seed

    def fit(self, X, y):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        self.scaler = StandardScaler().fit(X)
        self.y_mu, self.y_sd = float(np.mean(y)), float(np.std(y) or 1.0)
        Xt = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
        yt = torch.tensor((np.asarray(y) - self.y_mu) / self.y_sd, dtype=torch.float32).unsqueeze(1)
        self.net = nn.Sequential(
            nn.Linear(Xt.shape[1], self.hidden), nn.ReLU(), nn.Dropout(self.dropout), nn.Linear(self.hidden, 1)
        )
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = nn.functional.mse_loss(self.net(Xt), yt)
            loss.backward()
            opt.step()
        return self

    def predict(self, X):
        import torch

        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.tensor(self.scaler.transform(X), dtype=torch.float32)).numpy().ravel()
        return out * self.y_sd + self.y_mu


MODELS = {"ridge": make_ridge, "mlp": TorchMLP}


def walk_forward(X, y, model="ridge", min_train=52, refit_every=1):
    """Out-of-sample predictions for every week after the first `min_train` weeks."""
    preds = pd.Series(np.nan, index=y.index, name="pred")
    fitted = None
    for i in range(min_train, len(y)):
        if fitted is None or (i - min_train) % refit_every == 0:
            fitted = MODELS[model]().fit(X.iloc[:i].values, y.iloc[:i].values)
        preds.iloc[i] = fitted.predict(X.iloc[[i]].values)[0]
    return preds.dropna()
