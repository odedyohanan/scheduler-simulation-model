"""CPU-weight split grid (copied from priority-scheduler sched/student/features.py)."""
from __future__ import annotations

# CPU-weight splits, in units of SHARE_STEP %, each workload at least MIN_SHARE %. 5 % steps: 153 splits.
SHARE_STEP = 5
MIN_SHARE = 10


def share_combos(step: int, min_share: int = MIN_SHARE) -> list[tuple[int, int, int]]:
    n, lo = 100 // step, min_share // step
    return [(a, b, n - a - b) for a in range(lo, n) for b in range(lo, n - a) if n - a - b >= lo]


SHARE_COMBOS = share_combos(SHARE_STEP)
