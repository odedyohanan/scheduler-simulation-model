"""Noise floor: how unpredictable is the next window, even knowing the simulator's full hidden state?

Replay a generated run to a window, then simulate that one window K times with different random
seeds and the same policy. The deep-copied simulator keeps its hidden state (individual queued requests,
burst schedule, migration timers), so the spread is the randomness left *given everything*. It is
reported as the mean absolute deviation from the mean outcome, comparable to a model's MAE.

It is a lower bound: a model that sees only telemetry can't know the hidden state, so its best
possible error is higher. It is also measured on a sample of states, not on every test transition.
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
