"""Training loop: masked MSE on standardised targets, Adam, early stopping on validation loss."""
from __future__ import annotations

import copy

import numpy as np
import torch

from .model import MLP, GRUNet, Normalizer, WorldModel


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)


def train_model(train: tuple, val: tuple, epochs: int = 300, lr: float = 1e-3, batch: int = 512,
                patience: int = 20, seed: int = 0, hidden: int = 128, layers: int = 2,
                log=None, arch: str = "mlp", history: int = 1) -> tuple[WorldModel, list[dict]]:
    """train / val = (X, Y, M), X already prepared for the network ([N, n] MLP, [N, k, n] GRU).
    Returns the model at its best validation loss, and the history."""
    torch.manual_seed(seed)
    X, Y, M = train
    xn, yn = Normalizer.fit(X.reshape(-1, X.shape[-1])), Normalizer.fit(Y)   # GRU: per-window statistics
    Xt, Yt, Mt = (torch.from_numpy(a) for a in (xn.transform(X), yn.transform(Y), M.astype(np.float32)))
    Xv, Yv, Mv = (torch.from_numpy(a) for a in (xn.transform(val[0]), yn.transform(val[1]),
                                                val[2].astype(np.float32)))
    mlp = GRUNet(X.shape[-1], Y.shape[1], hidden) if arch == "gru" else MLP(X.shape[1], Y.shape[1], hidden, layers)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(epochs):
        mlp.train()
        perm, tot = torch.randperm(len(Xt), generator=gen), 0.0
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            loss = masked_mse(mlp(Xt[idx]), Yt[idx], Mt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        mlp.eval()
        with torch.no_grad():
            vl = masked_mse(mlp(Xv), Yv, Mv).item()
        hist.append({"epoch": ep, "train_loss": tot / len(Xt), "val_loss": vl})
        if log:
            log(hist[-1])
        if vl < best - 1e-6:
            best, best_state, bad = vl, copy.deepcopy(mlp.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    mlp.load_state_dict(best_state)
    return WorldModel(mlp, xn, yn, {"best_val_loss": best, "epochs_run": len(hist),
                                    "arch": arch, "history": history}), hist
