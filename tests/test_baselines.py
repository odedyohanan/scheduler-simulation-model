import json
from pathlib import Path

import numpy as np

from worldmodel.baselines import fluid_next, persistence_next, spec_from_known
from worldmodel.metrics import per_target

FIX = Path(__file__).parent / "fixtures"


def _teacher():
    return [r for r in map(json.loads, open(FIX / "teacher_v2_head.jsonl")) if not r.get("header")]


def test_persistence_is_identity():
    S = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.testing.assert_array_equal(persistence_next(S), S)


def test_spec_from_known():
    k = _teacher()[0]["input"]["known"]
    s = spec_from_known(k)
    assert s.a.service_ms == k["a_service_ms"] and s.thermal.ambient == k["thermal_ambient"]


def test_fluid_steady_temps():
    r = json.loads(json.dumps(_teacher()[0]))
    amb = r["input"]["known"]["thermal_ambient"]
    for w in r["applied"]["workloads"].values():
        w["gpus"] = []
    r["input"]["telemetry"]["gpu_temp"] = [amb, amb]
    p = fluid_next(r)
    assert abs(p["gpu_temp_0"] - amb) < 0.5 and abs(p["gpu_temp_1"] - amb) < 0.5


def test_fluid_runs_on_old_format():
    p = fluid_next(_teacher()[1])
    assert set(p) == {"a.p99_ms", "a.queue_work_ms", "b.rate", "c.work_core_ms", "gpu_temp_0", "gpu_temp_1"}
    assert all(np.isfinite(v) or np.isnan(v) for v in p.values())


def test_metrics_perfect():
    t = {"x": np.array([1.0, 2.0, 3.0])}
    m = per_target(t, t, {"x": np.ones(3)})
    assert m["x"]["mae"] == 0 and m["x"]["r2"] == 1 and m["x"]["n"] == 3


def test_metrics_skip_nan_and_mask():
    t = {"x": np.array([1.0, 2.0, 3.0, 4.0])}
    p = {"x": np.array([1.0, np.nan, 100.0, 4.0])}
    m = per_target(t, p, {"x": np.array([1.0, 1.0, 0.0, 1.0])})
    assert m["x"]["n"] == 2 and m["x"]["mae"] == 0


def test_metrics_median_robust_to_outlier():
    t = {"x": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
    p = {"x": np.array([1.0, 2.0, 3.0, 4.0, 1e7])}
    m = per_target(t, p, {"x": np.ones(5)})
    assert m["x"]["medae"] == 0 and m["x"]["mae"] > 1e6
