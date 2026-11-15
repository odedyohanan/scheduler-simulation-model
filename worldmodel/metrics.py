"""Per-target error metrics in real units."""
from __future__ import annotations

import numpy as np


def per_target(true: dict[str, np.ndarray], pred: dict[str, np.ndarray],
               mask: dict[str, np.ndarray]) -> dict[str, dict]:
    """MAE and R² per target, over entries that are unmasked and finite in both arrays."""
    out = {}
    for k, p in pred.items():
        t, p = np.asarray(true[k], float), np.asarray(p, float)
        ok = (np.asarray(mask[k]) > 0) & np.isfinite(t) & np.isfinite(p)
        t, p = t[ok], p[ok]
        if not len(t):
            out[k] = {"mae": float("nan"), "r2": float("nan"), "n": 0}
            continue
        ss_tot = ((t - t.mean()) ** 2).sum()
        r2 = 1.0 - ((t - p) ** 2).sum() / ss_tot if ss_tot > 0 else float("nan")
        out[k] = {"mae": float(np.abs(t - p).mean()), "r2": float(r2), "n": int(len(t))}
    return out
