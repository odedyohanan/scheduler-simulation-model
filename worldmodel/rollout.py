"""Multi-step rollouts: the model feeds on its own predictions.

Open loop: each step uses the policy that was actually applied in the recorded run (and the run's known
parameters), so the only difference from the real trajectory is the model's own error, which compounds.

Models with history (worldmodel.history) keep a sliding window of their last k inputs: rows before the
start point are the real recorded inputs (padded with the run's first input), later rows hold the
model's own predicted state with the recorded policy.

If the model has bounds (worldmodel.bounds), every predicted state is clamped into the training range.
"""
from __future__ import annotations

import numpy as np

from .features import clip_state, encode_input, encode_state
from .model import WorldModel


def _real_input(r: dict) -> np.ndarray:
    return encode_input(encode_state(r["input"]["telemetry"]), r["applied"], r["input"]["known"])


def rollout(model: WorldModel, run_records: list[dict], start: int,
            horizon: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Predicted and true encoded states for steps 1..h (h = min(horizon, records left))."""
    k, bounds = model.history, model.bounds
    past = [_real_input(run_records[max(start - j, 0)]) for j in range(k - 1, 0, -1)]   # oldest first
    s = encode_state(run_records[start]["input"]["telemetry"])
    pred, true = [], []
    for r in run_records[start:start + horizon]:
        x = encode_input(s, r["applied"], r["input"]["known"])
        hist = np.stack(past + [x])[None]                       # [1, k, n_in]
        s = clip_state(model.predict_next(model.prepare(hist)[0], s)[0])
        if bounds is not None:
            s = np.clip(s, bounds[0], bounds[1])
        past = (past + [x])[1:] if k > 1 else []
        pred.append(s)
        true.append(encode_state(r["outcome"]))
    return np.stack(pred), np.stack(true)
