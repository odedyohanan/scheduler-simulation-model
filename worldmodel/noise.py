"""Noise floor: how unpredictable is the next window, even knowing the simulator's full hidden state?

Replay a generated run to a window, then simulate that one window K times with different random
seeds and the same policy. The spread of the outcomes is error no model can remove. It is reported as
the mean absolute deviation from the mean outcome, which is directly comparable to a model's MAE
(the best possible predictor, the true mean, would score exactly this).
"""
from __future__ import annotations

import copy

import numpy as np

from .features import KEY_TARGETS, decode_state, encode_state, state_mask
from .generate import replay_to


def noise_floor(run_id: int, windows: list[int], k: int = 20, duration_s: float = 60.0) -> dict[str, float]:
    per_window = {t: [] for t in KEY_TARGETS}
    for w in windows:
        sim, policy = replay_to(run_id, w, duration_s=duration_s)
        outs = []
        for seed in range(k):
            s = copy.deepcopy(sim)
            s.rng = np.random.default_rng(10_000_000 + seed)
            tel = s.run_window(policy)
            d = decode_state(encode_state(tel))
            if state_mask(tel).min() == 0:          # no completions: p99 undefined
                d["a.p99_ms"] = np.nan
            outs.append(d)
        for t in KEY_TARGETS:
            v = np.array([o[t] for o in outs], float)
            v = v[np.isfinite(v)]
            if len(v) >= 2:
                per_window[t].append(float(np.abs(v - v.mean()).mean()))
    return {t: float(np.mean(v)) if v else float("nan") for t, v in per_window.items()}
