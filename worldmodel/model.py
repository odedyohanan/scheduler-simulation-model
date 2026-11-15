"""The learned world models: map [state, policy, known] (now, or the last k windows) -> change in state.

  MLP     the last window, or the last k windows concatenated (oldest first)
  GRUNet  the last k windows fed in order; the final hidden state predicts the change

WorldModel wraps the network with its input/output normalisers, so callers work in encoded-state
units (worldmodel.features) and never see standardised values. `prepare` turns a history array
[N, k, n_inputs] (worldmodel.history) into whatever shape the network takes.
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


class GRUNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)
        self.in_dim, self.out_dim, self.hidden = in_dim, out_dim, hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)              # h: [1, B, hidden], the state after the newest window
        return self.head(h[-1])

    def params(self) -> dict:
        return dict(in_dim=self.in_dim, out_dim=self.out_dim, hidden=self.hidden)


def _mlp_params(m: MLP) -> dict:
    return dict(in_dim=m.in_dim, out_dim=m.out_dim, hidden=m.hidden, layers=m.layers)


class WorldModel:
    def __init__(self, net: nn.Module, x_norm: Normalizer, y_norm: Normalizer, meta: dict | None = None):
        self.net, self.x_norm, self.y_norm, self.meta = net, x_norm, y_norm, meta or {}

    @property
    def arch(self) -> str:
        return self.meta.get("arch", "mlp")

    @property
    def history(self) -> int:
        return int(self.meta.get("history", 1))

    def prepare(self, H: np.ndarray) -> np.ndarray:
        """History [N, >=k, n_in] (newest row last) -> network input. Only the last k rows are used."""
        H = H[:, -self.history:]
        if self.arch == "gru":
            return H
        return H[:, -1] if self.history == 1 else H.reshape(len(H), -1)

    @torch.no_grad()
    def predict_delta(self, X: np.ndarray) -> np.ndarray:
        """X is a prepared input: [N, n] for an MLP, [N, k, n] for a GRU (a single sample may drop N)."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == (2 if self.arch == "gru" else 1):
            X = X[None]
        self.net.eval()
        out = self.net(torch.from_numpy(self.x_norm.transform(X))).numpy()
        return self.y_norm.inverse(out)

    def predict_next(self, X: np.ndarray, S: np.ndarray) -> np.ndarray:
        return np.atleast_2d(S) + self.predict_delta(X)

    def save(self, path) -> None:
        m = self.net
        torch.save({"state_dict": m.state_dict(),
                    "arch": m.params() if isinstance(m, GRUNet) else _mlp_params(m),
                    "x_norm": (self.x_norm.mean, self.x_norm.std),
                    "y_norm": (self.y_norm.mean, self.y_norm.std),
                    "meta": self.meta}, path)

    @classmethod
    def load(cls, path) -> "WorldModel":
        ck = torch.load(path, weights_only=False)
        meta = ck.get("meta") or {}
        net = GRUNet(**ck["arch"]) if meta.get("arch") == "gru" else MLP(**ck["arch"])
        net.load_state_dict(ck["state_dict"])
        return cls(net, Normalizer(*ck["x_norm"]), Normalizer(*ck["y_norm"]), meta)
