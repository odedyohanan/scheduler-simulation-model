"""Multi-step rollouts: the model feeds on its own predictions.

Open loop: each step uses the policy that was actually applied in the recorded run (and the run's known
parameters), so the only difference from the real trajectory is the model's own error, which compounds.
"""
from __future__ import annotations

import numpy as np

from .features import clip_state, encode_input, encode_state
from .model import WorldModel


def rollout(model: WorldModel, run_records: list[dict], start: int,
            horizon: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Predicted and true encoded states for steps 1..h (h = min(horizon, records left))."""
    steps = run_records[start:start + horizon]
    s = encode_state(steps[0]["input"]["telemetry"])
    pred, true = [], []
    for r in steps:
        x = encode_input(s, r["applied"], r["input"]["known"])
        s = clip_state(model.predict_next(x, s)[0])
        pred.append(s)
        true.append(encode_state(r["outcome"]))
    return np.stack(pred), np.stack(true)
