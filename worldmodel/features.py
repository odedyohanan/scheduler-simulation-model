"""Turn telemetry, policies and known parameters into fixed-length float vectors.

Only telemetry fields present in both the old (teacher_v2) and new formats are used.
Heavy-tailed fields are log1p-compressed; everything else stays in its own units. Standardisation
happens later (model.Normalizer), fitted on training data.

The model predicts the next window's *whole* encoded state, as a delta from the current one, so a
prediction can be fed back in (multi-step rollouts).
"""
from __future__ import annotations

import numpy as np

N_CPUS, N_GPUS, WL = 8, 2, ("A", "B", "C")

_SCALARS = [("a", "rate"), ("a", "p50_ms"), ("a", "p99_ms"), ("a", "queue_len"), ("a", "queue_work_ms"),
            ("a", "cpu_cores"), ("b", "rate"), ("b", "buffer"), ("b", "cpu_cores"),
            ("c", "work_core_ms"), ("c", "cpu_cores")]
LOG_FIELDS = {"a.p50_ms", "a.p99_ms", "a.queue_len", "a.queue_work_ms"}

STATE_NAMES: list[str] = ([f"{w}.{k}" for w, k in _SCALARS]
                          + [f"usage_{w}_{k}" for w in range(3) for k in range(N_CPUS)]
                          + [f"gpu_temp_{g}" for g in range(N_GPUS)]
                          + [f"gpu_power_{g}" for g in range(N_GPUS)])
_LOG_MASK = np.array([n in LOG_FIELDS for n in STATE_NAMES])
_P50, _P99 = STATE_NAMES.index("a.p50_ms"), STATE_NAMES.index("a.p99_ms")

KEY_TARGETS = ["a.p99_ms", "a.queue_work_ms", "a.cpu_cores", "b.rate", "b.buffer", "c.work_core_ms",
               "gpu_temp_0", "gpu_temp_1"]

POLICY_NAMES: list[str] = ([f"elig_{w}_{k}" for w in range(3) for k in range(N_CPUS)]
                           + [f"share_{w}" for w in WL]
                           + [f"gpu{g}_{o}" for g in range(N_GPUS) for o in ("none", "B", "C")])
KNOWN_NAMES = ["a_service_ms", "a_interference", "thermal_ambient"]
INPUT_NAMES = STATE_NAMES + POLICY_NAMES + KNOWN_NAMES


def encode_state(tel: dict) -> np.ndarray:
    v = [float(tel[w][k]) for w, k in _SCALARS]
    v += [float(x) for row in tel["cpu_usage_by_workload"] for x in row]
    v += [float(x) for x in tel["gpu_temp"]] + [float(x) for x in tel["gpu_power"]]
    v = np.array(v, dtype=np.float64)
    v[_LOG_MASK] = np.log1p(np.maximum(v[_LOG_MASK], 0.0))
    return v.astype(np.float32)


def decode_state(vec: np.ndarray) -> dict[str, float]:
    """Real units. Every state quantity is non-negative, so predictions are clipped at 0."""
    v = np.asarray(vec, dtype=np.float64).copy()
    v[_LOG_MASK] = np.expm1(v[_LOG_MASK])
    v = np.maximum(v, 0.0)
    return dict(zip(STATE_NAMES, v.tolist()))


def clip_state(vec: np.ndarray) -> np.ndarray:
    """Encoded state with physically impossible (negative) values clipped."""
    return np.maximum(np.asarray(vec, dtype=np.float32), 0.0)


def encode_policy(p: dict) -> np.ndarray:
    v = []
    for w in WL:
        cpus = p["workloads"][w]["preferred_cpus"]
        row = np.ones(N_CPUS) if not cpus else np.isin(np.arange(N_CPUS), cpus).astype(float)
        v += row.tolist()
    v += [p["workloads"][w]["cpu_share"] / 100.0 for w in WL]
    owner = ["none"] * N_GPUS
    for w in WL:
        for g in p["workloads"][w]["gpus"]:
            owner[g] = w
    for g in range(N_GPUS):
        v += [float(owner[g] == o) for o in ("none", "B", "C")]
    return np.array(v, dtype=np.float32)


def encode_known(known: dict) -> np.ndarray:
    return np.array([known[k] for k in KNOWN_NAMES], dtype=np.float32)


def encode_input(state: np.ndarray, policy: dict, known: dict) -> np.ndarray:
    return np.concatenate([state, encode_policy(policy), encode_known(known)]).astype(np.float32)


def state_mask(tel: dict) -> np.ndarray:
    """1 = use this target. p50/p99 are meaningless (reported as 0) when no A request completed."""
    m = np.ones(len(STATE_NAMES), dtype=np.float32)
    a = tel["a"]
    no_completions = a["completions"] == 0 if "completions" in a else a["p99_ms"] == 0.0
    if no_completions:
        m[[_P50, _P99]] = 0.0
    return m


def build_arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """X = [state, applied policy, known], S = state, Y = next state - state, M = target mask."""
    S = np.stack([encode_state(r["input"]["telemetry"]) for r in records])
    X = np.stack([encode_input(s, r["applied"], r["input"]["known"]) for s, r in zip(S, records)])
    Y = np.stack([encode_state(r["outcome"]) for r in records]) - S
    M = np.stack([state_mask(r["outcome"]) for r in records])
    return X, S, Y.astype(np.float32), M
