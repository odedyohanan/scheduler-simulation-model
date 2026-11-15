"""CPU capacity allocation: who gets how much of each CPU.

Shared by the simulator (per 1 ms step) and the RK4 predictor (per derivative
evaluation), so both use exactly the same sharing rules.

Rule: weighted, work-conserving fair sharing ("progressive filling").
Each round, every CPU with spare capacity offers it to the workloads that may
run there and still want more, split by cpu_share. A workload takes what it
needs and leaves the rest for the next round.
"""
import numpy as np


def allocate(elig: np.ndarray, demand: np.ndarray, weights: np.ndarray,
             cpu_cap: float = 1.0, rounds: int = 6):
    """
    elig    (W x K) bool    workload w may run on CPU k
    demand  (W,)   cores    how much CPU each workload wants right now
    weights (W,)            cpu_share
    Returns (got (W,) cores, usage (W x K) cores).
    """
    n_w, n_k = elig.shape
    spare = np.full(n_k, cpu_cap)
    usage = np.zeros((n_w, n_k))
    got = np.zeros(n_w)
    for _ in range(rounds):
        want = demand - got
        active = want > 1e-9
        if not active.any() or spare.max() <= 1e-12:
            break
        w_mat = elig * (active * weights)[:, None]          # (W x K)
        col = w_mat.sum(axis=0)
        col[col == 0] = 1.0
        offer = w_mat / col * spare                         # what each CPU offers each workload
        total = offer.sum(axis=1)
        take = np.minimum(want, total)
        frac = np.divide(take, total, out=np.zeros(n_w), where=total > 0)
        used = offer * frac[:, None]
        usage += used
        got += take
        spare -= used.sum(axis=0)
    return got, usage
