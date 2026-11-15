"""The range of each state number seen in training, used to keep rollout predictions physical.

Bounds are in encoded space (log1p for latency/queue fields), over both the input states and the
next (outcome) states of the given records.
"""
from __future__ import annotations

import numpy as np

from .features import encode_state


def state_bounds(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    S = np.stack([encode_state(r["input"]["telemetry"]) for r in records]
                 + [encode_state(r["outcome"]) for r in records])
    return np.nanmin(S, 0).astype(np.float32), np.nanmax(S, 0).astype(np.float32)
