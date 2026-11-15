"""Domain randomisation: a different machine/workload mix for every dataset run.

A student trained on one fixed setting only learns that setting. Varying the
load shape, burstiness, service time, interference and ambient temperature
makes it learn the *relationship* between telemetry and good policies.

What stays fixed: hardware layout, SLO, and the objective weights, i.e. what
"good" means. Every record stores the parameters that a real scheduler would
know (service time, interference, ambient), so the student can use them as
input.
"""
from dataclasses import asdict, replace

import numpy as np

from .config import Spec


def random_spec(rng: np.random.Generator) -> Spec:
    s = Spec()
    a = replace(s.a,
                base_rate=float(rng.uniform(1200, 2800)),
                wave_amp=float(rng.uniform(0, 1500)),
                wave_period=float(rng.uniform(20, 80)),
                burst_every=float(rng.uniform(8, 40)),
                burst_len=float(rng.uniform(1, 6)),
                burst_extra=float(rng.uniform(1000, 3500)),
                service_ms=float(rng.uniform(0.7, 1.3)),
                interference=float(rng.uniform(0.1, 0.4)))
    th = replace(s.thermal, ambient=float(rng.uniform(20, 40)))
    return replace(s, a=a, thermal=th)


def known_params(spec: Spec) -> dict:
    """What a real scheduler would know in advance (from profiling/config), not the hidden load."""
    return {
        "a_service_ms": round(spec.a.service_ms, 4),
        "a_interference": round(spec.a.interference, 4),
        "a_slo_p99_ms": spec.a.slo_p99_ms,
        "b_cpu_ms_per_batch": spec.b.cpu_ms_per_batch,
        "b_gpu_ms_per_batch": spec.b.gpu_ms_per_batch,
        "b_value_per_batch": spec.b.value_per_batch,
        "c_gpu_core_equiv": spec.c.gpu_core_equiv,
        "c_value_per_core_ms": spec.c.value_per_core_ms,
        "thermal_ambient": round(spec.thermal.ambient, 2),
    }


def hidden_params(spec: Spec) -> dict:
    """Ground truth of the load process: stored for analysis only, never a model input."""
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in asdict(spec.a).items()}
