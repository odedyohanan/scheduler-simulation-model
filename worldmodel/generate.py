"""Transition data: the adaptive controller drives the simulator, sometimes a perturbed policy is applied.

One record per 500 ms window, same shape as priority_scheduler_model's teacher_v2 records:
  input.telemetry       the window that just ended (what the controller saw)
  input.current_policy  the policy that ran during that window
  input.known           parameters a real scheduler would know
  applied               the policy applied for the next window (perturbed when `explored`)
  outcome               the next window's telemetry  <- what the world model predicts
"""
from __future__ import annotations

import json

import numpy as np

from sim.config import Spec
from sim.controllers import AdaptiveController
from sim.dataset import perturb_policy, policy_dict
from sim.policy import Policy
from sim.randomize import known_params, random_spec
from sim.simulator import Simulator

SCENARIO_BASE = 20000   # scenario stream; teacher data used 10000+, evaluation seeds 900000+
PERTURB_BASE = 30000


def scenario_spec(run_id: int) -> Spec:
    return random_spec(np.random.default_rng(SCENARIO_BASE + run_id))


def _plain(x):
    """numpy scalars -> python, so records are JSON-identical before and after saving."""
    return json.loads(json.dumps(x, default=lambda o: o.item()))


def _episode(run_id: int, duration_s: float, explore: float, stop_at: int | None):
    spec = scenario_spec(run_id)
    sim = Simulator(spec, run_id, duration_s)
    ctrl = AdaptiveController(spec)
    prng = np.random.default_rng(PERTURB_BASE + run_id)
    known = known_params(spec)
    current = ctrl.initial_policy()
    tel = _plain(sim.run_window(current))
    records = []
    while not sim.done():
        p = ctrl.decide(tel)
        explored = bool(prng.random() < explore)
        if explored:
            p = perturb_policy(p, prng, spec)
        if stop_at is not None and len(records) == stop_at:
            return sim, p
        out = _plain(sim.run_window(p))
        records.append({"run": run_id, "t": tel["t"], "driver": "adaptive",
                        "input": {"telemetry": tel, "current_policy": policy_dict(current), "known": known},
                        "explored": explored, "applied": policy_dict(p), "outcome": out})
        current, tel = p, out
    if stop_at is not None:
        raise ValueError(f"run has only {len(records)} records, asked for {stop_at}")
    return records


def run_episode(run_id: int, duration_s: float = 60.0, explore: float = 0.3) -> list[dict]:
    return _episode(run_id, duration_s, explore, None)


def replay_to(run_id: int, window: int, duration_s: float = 60.0,
              explore: float = 0.3) -> tuple[Simulator, Policy]:
    """Simulator state just before record `window`'s outcome is simulated, and the policy about to run."""
    return _episode(run_id, duration_s, explore, window)
