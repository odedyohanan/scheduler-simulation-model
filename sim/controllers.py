"""Controllers that need no solver (copied from priority-scheduler sched/controllers.py).

StaticController    always the same policy.
AdaptiveController  the hand-written feedback baseline; drives data generation here.
The CP-SAT MPC teacher is deliberately not copied (no OR-Tools in this project).
"""
from __future__ import annotations

import time

import numpy as np

from .config import Spec
from .policy import Policy, WorkloadPolicy

def example_policy() -> Policy:
    """The policy sketched in the conversation (plus GPUs for B)."""
    return Policy(version=1, valid_for_ms=500, workloads=[
        WorkloadPolicy([0, 1, 2, 3], 40, "critical"),
        WorkloadPolicy([4, 5, 6, 7], 30, "batch", gpus=[0, 1]),
        WorkloadPolicy([], 30, "best_effort"),
    ])


def isolated_policy() -> Policy:
    """A sensible hand-made static policy: A alone on its own L3 domain."""
    return Policy(version=1, valid_for_ms=500, workloads=[
        WorkloadPolicy([0, 1, 2, 3], 40, "critical"),
        WorkloadPolicy([4, 5, 6, 7], 50, "batch", gpus=[0, 1]),
        WorkloadPolicy([4, 5, 6, 7], 10, "best_effort"),
    ])


class StaticController:
    def __init__(self, policy: Policy, name: str):
        self.policy, self.name = policy, name

    def initial_policy(self) -> Policy:
        return self.policy

    def decide(self, telemetry: dict) -> Policy:
        return self.policy



C_CPUSETS = [[], [4, 5, 6, 7], [6, 7], [7]]       # C's cpuset by A-pressure level (empty = any CPU)


class AdaptiveController:
    """
    Non-ML feedback baseline (docs/adaptive_baseline.md). Every window:
      1. A pressure level L in 0..3: up one level at once on trouble (p99 > r_hi*SLO or more than one
         SLO of queued work), down one level after `hold` consecutive calm windows.
      2. C's cpuset shrinks with L (C always wants every core; this limits interference with A).
      3. Shares proportional to estimated needs: A from rate, service time, interference and backlog
         (× headroom h0 + 0.5 L); B from its observed batch rate, boosted when its buffer runs low;
         C the rest. Nearest split on the students' 5 % grid, changed only if it moves >= 10 points.
    Both GPUs belong to B; A and B may run on every CPU.
    Uses only telemetry the learned policies also see, plus known_params and fixed machine constants.
    """
    name = "adaptive"

    def __init__(self, spec: Spec, h0: float = 1.5, r_hi: float = 0.9, hold: int = 2):
        from .randomize import known_params
        from .shares import SHARE_COMBOS, SHARE_STEP
        self.spec, self.known = spec, known_params(spec)
        self.h0, self.r_hi, self.hold = h0, r_hi, hold
        self.combos = np.array(SHARE_COMBOS, dtype=float) * SHARE_STEP
        self.level, self.calm = 0, 0
        self.shares = (60, 30, 10)
        self.version = 1
        self.log: list[dict] = []

    def _policy(self) -> Policy:
        a, b, c = self.shares
        return Policy(version=self.version, valid_for_ms=int(self.spec.policy_period_s * 1000), workloads=[
            WorkloadPolicy([], a, "critical"),
            WorkloadPolicy([], b, "batch", gpus=list(range(self.spec.machine.n_gpus))),
            WorkloadPolicy(list(C_CPUSETS[self.level]), c, "best_effort"),
        ])

    def initial_policy(self) -> Policy:
        return self._policy()

    def decide(self, tel: dict) -> Policy:
        t0 = time.perf_counter()
        k, a, b = self.known, tel["a"], tel["b"]
        slo = k["a_slo_p99_ms"]
        trouble = a["p99_ms"] > self.r_hi * slo or a["queue_work_ms"] > slo
        calm = a["p99_ms"] < 0.5 * slo and a["queue_work_ms"] < 0.2 * slo
        if trouble:
            self.level, self.calm = min(self.level + 1, len(C_CPUSETS) - 1), 0
        elif calm:
            self.calm += 1
            if self.calm >= self.hold:
                self.level, self.calm = max(self.level - 1, 0), 0
        else:
            self.calm = 0
        n_cpu = self.spec.machine.n_cpus
        need_a = (a["rate"] * k["a_service_ms"] / 1000 * (1 + k["a_interference"])
                  + a["queue_work_ms"] / 1000 / self.spec.policy_period_s) * (self.h0 + 0.5 * self.level)
        fill = b["buffer"] / self.spec.b.buffer_cap
        boost = 1.5 if fill < 0.25 else 1.0 if fill < 0.75 else 0.8
        need_b = max(b["rate"] * k["b_cpu_ms_per_batch"] / 1000 * boost, 0.5)
        need_c = max(n_cpu - need_a - need_b, 0.0)
        target = 100 * np.array([need_a, need_b, need_c]) / (need_a + need_b + need_c)
        best = self.combos[np.argmin(np.abs(self.combos - target).sum(axis=1))]
        if np.abs(best - np.array(self.shares)).sum() >= 10:
            self.shares = tuple(int(x) for x in best)
        self.version += 1
        self.log.append({"t": tel["t"], "level": self.level, "shares": self.shares,
                         "overdue": a.get("queue_overdue", 0),          # evaluation only, not used above
                         "decide_ms": (time.perf_counter() - t0) * 1000})
        return self._policy()
