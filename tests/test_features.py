import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from worldmodel import features as F
from worldmodel.data import group_by_run, load_records, split_of

FIX = Path(__file__).parent / "fixtures"


def _read(name):
    return [json.loads(l) for l in open(FIX / name)]


@pytest.fixture
def adaptive():
    return _read("adaptive_head.jsonl")


@pytest.fixture
def teacher():
    return [r for r in _read("teacher_v2_head.jsonl") if not r.get("header")]


def test_split_of():
    assert split_of(18) == "val" and split_of(29) == "test" and split_of(7) == "train"


def test_load_skips_header_and_filters_split(tmp_path):
    lines = open(FIX / "teacher_v2_head.jsonl").readlines()
    for run in (7, 8):
        with gzip.open(tmp_path / f"run_{run:05d}.jsonl.gz", "wt") as f:
            for l in lines:
                r = json.loads(l)
                r["run"] = run
                f.write(json.dumps(r) + "\n")
    assert len(load_records([str(tmp_path)])) == 8
    val = load_records([str(tmp_path)], split="val")
    assert len(val) == 4 and {r["run"] for r in val} == {8}
    g = group_by_run(load_records([str(tmp_path)]))
    assert [r["t"] for r in g[7]] == sorted(r["t"] for r in g[7])


def test_old_format_encodes(adaptive, teacher):
    Xa, Sa, Ya, Ma = F.build_arrays(adaptive)
    Xt, St, Yt, Mt = F.build_arrays(teacher)
    assert Xa.shape[1] == Xt.shape[1] and Ya.shape[1] == Yt.shape[1] == len(F.STATE_NAMES)
    assert Xa.dtype == np.float32
    for arr in (Xa, Ya, Xt, Yt):
        assert np.isfinite(arr).all()


def test_roundtrip(adaptive):
    tel = adaptive[2]["outcome"]
    d = F.decode_state(F.encode_state(tel))
    assert d["a.queue_work_ms"] == pytest.approx(tel["a"]["queue_work_ms"], rel=1e-5, abs=1e-5)
    assert d["gpu_temp_1"] == pytest.approx(tel["gpu_temp"][1], rel=1e-5)
    assert d["usage_2_7"] == pytest.approx(tel["cpu_usage_by_workload"][2][7], rel=1e-5, abs=1e-6)


def test_mask_no_completions(teacher):
    tel = copy.deepcopy(teacher[0]["outcome"])
    tel["a"]["p99_ms"] = 0.0
    tel["a"]["p50_ms"] = 0.0
    tel["a"].pop("completions", None)
    m = F.state_mask(tel)
    i50, i99 = F.STATE_NAMES.index("a.p50_ms"), F.STATE_NAMES.index("a.p99_ms")
    assert m[i50] == 0 and m[i99] == 0 and m.sum() == len(m) - 2
    tel2 = copy.deepcopy(teacher[0]["outcome"])
    tel2["a"]["completions"] = 0
    assert F.state_mask(tel2)[i99] == 0


def test_overload_finite(teacher):
    tel = copy.deepcopy(teacher[0]["outcome"])
    tel["a"]["queue_work_ms"] = 1e6
    tel["a"]["queue_len"] = 1e6
    v = F.encode_state(tel)
    assert np.isfinite(v).all()
    assert v[F.STATE_NAMES.index("a.queue_work_ms")] < 20 and v[F.STATE_NAMES.index("a.queue_len")] < 20


def test_empty_cpuset_means_all(adaptive):
    p = copy.deepcopy(adaptive[0]["applied"])
    p["workloads"]["C"]["preferred_cpus"] = []
    v = F.encode_policy(p)
    elig = v[:24].reshape(3, 8)
    assert (elig[2] == 1).all()
    assert len(v) == len(F.POLICY_NAMES)
