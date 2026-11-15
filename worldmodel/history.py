"""Model inputs with history: for every record, its own input plus the k-1 windows before it.

H[i] has shape [k, n_inputs], rows oldest -> newest; the newest row is record i's own input
(its telemetry, the policy applied next, known params). History never crosses runs; at the start of a
run the missing rows are copies of the run's first input.
"""
from __future__ import annotations

import numpy as np

from .data import group_by_run, run_key
from .features import build_arrays


def build_history_arrays(records: list[dict], k: int):
    """Returns (H [N,k,n_in], S, Y, M, ordered_records), grouped by run and sorted by time."""
    ordered = [r for recs in group_by_run(records).values() for r in recs]
    X, S, Y, M = build_arrays(ordered)
    H = np.empty((len(ordered), k, X.shape[1]), dtype=np.float32)
    first = 0
    for i, r in enumerate(ordered):
        if i and run_key(ordered[i - 1]) != run_key(r):
            first = i
        for j in range(k):                       # j = 0 is the oldest row
            H[i, j] = X[max(i - (k - 1 - j), first)]
    return H, S, Y, M, ordered
