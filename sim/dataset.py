"""Turning teacher runs into training examples.

One record = one decision:

  input   what the controller could see when deciding
            telemetry       last window's telemetry (exactly what the simulator emitted)
            current_policy  the policy that was running during that window
            known           parameters a real scheduler would know (service time, SLO, ...)
  target  the teacher's chosen policy (the label the student must learn)
  teacher the teacher's internal numbers (estimates, predictions, solver status)
  applied the policy actually applied (differs from target on exploration steps)
  outcome the telemetry of the window that followed (for evaluation / RL later)

Exploration: if the student only sees states the teacher itself reaches, it never
learns how to recover from its own mistakes (distribution shift). So with
probability `explore` a randomly perturbed policy is applied instead of the
teacher's. The label is still the teacher's answer for that state.
"""
from __future__ import annotations

import copy
import json

import numpy as np

from .config import A, Spec
from .policy import Policy


def policy_dict(p: Policy) -> dict:
    return json.loads(p.to_json())


def perturb_policy(p: Policy, rng: np.random.Generator, spec: Spec) -> Policy:
    """A plausible but different policy: shift shares, flip CPUs, move a GPU."""
    q = copy.deepcopy(p)
    q.predicted = None
    n_cpus, n_gpus = spec.machine.n_cpus, spec.machine.n_gpus
    kind = rng.integers(3)
    if kind == 0:                                   # shift 10-30 % share between two workloads
        i, j = rng.choice(3, size=2, replace=False)
        d = int(min(rng.choice([10, 20, 30]), q.workloads[i].cpu_share - 10))
        if d > 0:
            q.workloads[i].cpu_share -= d
            q.workloads[j].cpu_share += d
    elif kind == 1:                                 # add or remove 1-2 CPUs for one workload
        w = int(rng.integers(3))
        cpus = set(q.workloads[w].preferred_cpus or range(n_cpus))
        for k in rng.choice(n_cpus, size=int(rng.integers(1, 3)), replace=False):
            cpus.symmetric_difference_update({int(k)})
        if cpus:
            q.workloads[w].preferred_cpus = [] if len(cpus) == n_cpus else sorted(cpus)
    else:                                           # give a GPU to the other GPU user
        g = int(rng.integers(n_gpus))
        owner = q.gpu_owner(n_gpus)[g]
        for wp in q.workloads:
            if g in wp.gpus:
                wp.gpus.remove(g)
        new_owner = {None: 1, 1: 2, 2: 1}[owner] if owner != A else 1
        q.workloads[new_owner].gpus = sorted(q.workloads[new_owner].gpus + [g])
    # Every CPU must stay usable by someone.
    covered = q.eligibility(n_cpus).any(axis=0)
    if not covered.all():
        return copy.deepcopy(p)
    return q
