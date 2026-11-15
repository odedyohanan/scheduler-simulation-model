import json
from pathlib import Path

import numpy as np
import torch

from worldmodel.data import group_by_run
from worldmodel.features import INPUT_NAMES, STATE_NAMES, build_arrays, encode_input
from worldmodel.history import build_history_arrays
from worldmodel.model import MLP, GRUNet, Normalizer, WorldModel

FIX = Path(__file__).parent / "fixtures"
N_IN, N_OUT = len(INPUT_NAMES), len(STATE_NAMES)


def _two_runs():
    a = [json.loads(l) for l in open(FIX / "adaptive_head.jsonl")]
    b = json.loads(json.dumps(a))
    for r in b:
        r["run"] = 12
    return b + a          # deliberately out of order: run 12 first


def test_history_last_row_is_own_input():
    H, S, Y, M, ordered = build_history_arrays(_two_runs(), 3)
    for i, r in enumerate(ordered):
        np.testing.assert_array_equal(H[i, -1], encode_input(S[i], r["applied"], r["input"]["known"]))
    np.testing.assert_array_equal(H[2, -2], H[1, -1])
    np.testing.assert_array_equal(H[2, -3], H[0, -1])


def test_history_padding_and_run_boundary():
    H, S, Y, M, ordered = build_history_arrays(_two_runs(), 4)
    starts = [i for i, r in enumerate(ordered) if i == 0 or ordered[i - 1]["run"] != r["run"]]
    assert len(starts) == 2
    for i in starts:
        assert all((H[i, j] == H[i, -1]).all() for j in range(4))
    j = starts[1]
    assert all((H[j + 1, k] == H[j, -1]).all() for k in range(3))   # padded with its own run's first input


def test_shapes_k1_matches_build_arrays():
    H, S, Y, M, ordered = build_history_arrays(_two_runs(), 1)
    X2, S2, Y2, M2 = build_arrays(ordered)
    assert H.shape == (10, 1, N_IN) and H.dtype == np.float32
    np.testing.assert_array_equal(H[:, 0], X2)
    np.testing.assert_array_equal(Y, Y2)


def test_gru_forward_shape():
    assert GRUNet(N_IN, N_OUT)(torch.zeros(2, 8, N_IN)).shape == (2, N_OUT)


def _wm(arch, k):
    x = np.random.default_rng(0).normal(size=(20, N_IN)).astype(np.float32)
    net = GRUNet(N_IN, N_OUT) if arch == "gru" else MLP(N_IN * (k if k > 1 else 1), N_OUT)
    xn = Normalizer.fit(x if arch == "gru" or k == 1 else np.tile(x, k))
    yn = Normalizer.fit(np.zeros((2, N_OUT), np.float32))
    return WorldModel(net, xn, yn, {"arch": arch, "history": k})


def test_prepare_shapes():
    H = np.zeros((5, 8, N_IN), np.float32)
    assert _wm("mlp", 1).prepare(H[:, -1:]).shape == (5, N_IN)
    assert _wm("mlp", 4).prepare(H[:, -4:]).shape == (5, 4 * N_IN)
    assert _wm("gru", 8).prepare(H).shape == (5, 8, N_IN)
    m = _wm("mlp", 4)
    np.testing.assert_array_equal(m.prepare(H[:, -4:])[:, -N_IN:], H[:, -1])   # newest window last


def test_old_checkpoint_defaults(tmp_path):
    x = np.random.default_rng(1).normal(size=(10, N_IN)).astype(np.float32)
    m = WorldModel(MLP(N_IN, N_OUT), Normalizer.fit(x), Normalizer.fit(np.zeros((2, N_OUT), np.float32)), {})
    m.save(tmp_path / "old.pt")
    m2 = WorldModel.load(tmp_path / "old.pt")
    assert m2.arch == "mlp" and m2.history == 1
    np.testing.assert_allclose(m2.predict_delta(x), m.predict_delta(x))


def test_gru_save_load_same_predictions(tmp_path):
    m = _wm("gru", 8)
    H = np.random.default_rng(2).normal(size=(3, 8, N_IN)).astype(np.float32)
    m.save(tmp_path / "g.pt")
    m2 = WorldModel.load(tmp_path / "g.pt")
    assert m2.arch == "gru" and m2.history == 8
    np.testing.assert_allclose(m2.predict_delta(m2.prepare(H)), m.predict_delta(m.prepare(H)), rtol=1e-6)


def test_same_run_id_two_sources_not_merged():
    a = [json.loads(l) for l in open(FIX / "adaptive_head.jsonl")]
    b = json.loads(json.dumps(a))
    for r in a:
        r["source"] = "a"
    for r in b:
        r["source"] = "b"
        r["input"]["telemetry"]["gpu_temp"] = [t + 5 for t in r["input"]["telemetry"]["gpu_temp"]]
    recs = a + b
    assert len(group_by_run(recs)) == 2
    H, S, Y, M, ordered = build_history_arrays(recs, 4)
    starts = [i for i in range(len(ordered)) if i == 0 or ordered[i - 1]["source"] != ordered[i]["source"]]
    assert starts == [0, 5]
    for i in starts:
        assert all((H[i, j] == H[i, -1]).all() for j in range(4))
    assert all((H[6, j] == H[5, -1]).all() for j in range(3))


def test_group_without_source():
    a = [json.loads(l) for l in open(FIX / "adaptive_head.jsonl")]
    assert list(group_by_run(a)) == [("", 11)]
