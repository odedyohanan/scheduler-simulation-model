"""The learned world model: a plain MLP that maps [state, policy, known] -> change in state.

WorldModel wraps the network with its input/output normalisers, so callers work in encoded-state
units (worldmodel.features) and never see standardised values. A GRU or other model can slot in
behind the same predict_delta / predict_next interface.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, a: np.ndarray) -> "Normalizer":
        std = a.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)       # constant columns: leave scale alone
        return cls(a.mean(axis=0).astype(np.float32), std.astype(np.float32))

    def transform(self, a: np.ndarray) -> np.ndarray:
        return ((a - self.mean) / self.std).astype(np.float32)

    def inverse(self, a: np.ndarray) -> np.ndarray:
        return (a * self.std + self.mean).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, layers: int = 2):
        super().__init__()
        dims, mods = [in_dim] + [hidden] * layers, []
        for a, b in zip(dims[:-1], dims[1:]):
            mods += [nn.Linear(a, b), nn.ReLU()]
        mods.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*mods)
        self.in_dim, self.out_dim, self.hidden, self.layers = in_dim, out_dim, hidden, layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WorldModel:
    def __init__(self, mlp: MLP, x_norm: Normalizer, y_norm: Normalizer, meta: dict | None = None):
        self.mlp, self.x_norm, self.y_norm, self.meta = mlp, x_norm, y_norm, meta or {}

    @torch.no_grad()
    def predict_delta(self, X: np.ndarray) -> np.ndarray:
        self.mlp.eval()
        out = self.mlp(torch.from_numpy(self.x_norm.transform(np.atleast_2d(X)))).numpy()
        return self.y_norm.inverse(out)

    def predict_next(self, X: np.ndarray, S: np.ndarray) -> np.ndarray:
        return np.atleast_2d(S) + self.predict_delta(X)

    def save(self, path) -> None:
        m = self.mlp
        torch.save({"state_dict": m.state_dict(),
                    "arch": dict(in_dim=m.in_dim, out_dim=m.out_dim, hidden=m.hidden, layers=m.layers),
                    "x_norm": (self.x_norm.mean, self.x_norm.std),
                    "y_norm": (self.y_norm.mean, self.y_norm.std),
                    "meta": self.meta}, path)

    @classmethod
    def load(cls, path) -> "WorldModel":
        ck = torch.load(path, weights_only=False)
        mlp = MLP(**ck["arch"])
        mlp.load_state_dict(ck["state_dict"])
        return cls(mlp, Normalizer(*ck["x_norm"]), Normalizer(*ck["y_norm"]), ck.get("meta"))
