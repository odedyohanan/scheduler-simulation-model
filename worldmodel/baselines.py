"""Baselines the MLP has to beat.

persistence  "nothing changes": the next window looks like this one.
fluid        the hand-written physics model the MPC teacher used (RK4 fluid model + Erlang C),
             run for one 500 ms window. It only predicts some of the key targets.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from sim.config import Spec
from sim.policy import policy_from_dict
from sim.predictor import FluidPredictor


def persistence_next(S: np.ndarray) -> np.ndarray:
    return S


def spec_from_known(known: dict) -> Spec:
    """Only what a real scheduler knows; the hidden load process stays at defaults (the fluid model is
    driven by the observed arrival rate instead)."""
    s = Spec()
    return replace(s, a=replace(s.a, service_ms=known["a_service_ms"], interference=known["a_interference"]),
                   thermal=replace(s.thermal, ambient=known["thermal_ambient"]))


def fluid_next(rec: dict) -> dict[str, float]:
    tel, spec = rec["input"]["telemetry"], spec_from_known(rec["input"]["known"])
    pred = FluidPredictor(spec, horizon_s=spec.policy_period_s).predict(
        policy_from_dict(rec["applied"]),
        {"q_a": tel["a"]["queue_work_ms"] / 1000.0, "buf_b": tel["b"]["buffer"], "temps": tel["gpu_temp"]},
        lam_forecast=tel["a"]["rate"])
    p99 = pred["a_p99_ms"]
    return {"a.p99_ms": p99 if math.isfinite(p99) else float("nan"),
            "a.queue_work_ms": pred["q_a_end"] * 1000.0,
            "b.rate": pred["b_batches_per_s"],
            "c.work_core_ms": pred["c_core_ms_per_s"] * spec.policy_period_s,
            "gpu_temp_0": pred["gpu_temp_end"][0],
            "gpu_temp_1": pred["gpu_temp_end"][1]}
