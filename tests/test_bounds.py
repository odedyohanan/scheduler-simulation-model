import json
from pathlib import Path

import numpy as np

from worldmodel.bounds import state_bounds
from worldmodel.features import INPUT_NAMES, STATE_NAMES, encode_state
from worldmodel.model import MLP, Normalizer, WorldModel
from worldmodel.rollout import rollout

FIX = Path(__file__).parent / "fixtures"
N = len(STATE_NAMES)


def _recs():
    return [json.loads(l) for l in open(FIX / "adaptive_head.jsonl")]


def _model(bias: float, bounds=None):
    n_in = len(INPUT_NAMES)
    mlp = MLP(n_in, N, hidden=8, layers=1)
    for p in mlp.parameters():
        p.data.zero_()
    x = Normalizer(np.zeros(n_in, np.float32), np.ones(n_in, np.float32))
    y = Normalizer(np.full(N, bias, np.float32), np.ones(N, np.float32))
    meta = {} if bounds is None else {"state_min": bounds[0].tolist(), "state_max": bounds[1].tolist()}
    return WorldModel(mlp, x, y, meta)


def test_state_bounds_covers_inputs_and_outcomes():
    recs = _recs()
    recs[-1]["outcome"]["gpu_temp"] = [999.0, 999.0]
    lo, hi = state_bounds(recs)
    assert lo.shape == hi.shape == (N,) and lo.dtype == np.float32
    for r in recs:
        for s in (encode_state(r["input"]["telemetry"]), encode_state(r["outcome"])):
            assert (s >= lo).all() and (s <= hi).all()
    assert hi[STATE_NAMES.index("gpu_temp_0")] == 999.0


def test_rollout_clamps_to_bounds():
    lo, hi = state_bounds(_recs())
    pred, _ = rollout(_model(1e6, (lo, hi)), _recs(), 0, 4)
    np.testing.assert_array_equal(pred, np.repeat(hi[None], 4, 0))
    pred, _ = rollout(_model(-1e6, (lo, hi)), _recs(), 0, 4)
    np.testing.assert_array_equal(pred, np.repeat(np.maximum(lo, 0)[None], 4, 0))


def test_bounds_roundtrip(tmp_path):
    lo, hi = state_bounds(_recs())
    _model(0.5, (lo, hi)).save(tmp_path / "m.pt")
    b = WorldModel.load(tmp_path / "m.pt").bounds
    assert b[0].dtype == np.float32 and len(b[1]) == N
    np.testing.assert_array_equal(b[0], lo); np.testing.assert_array_equal(b[1], hi)
    _model(0.5).save(tmp_path / "n.pt")
    assert WorldModel.load(tmp_path / "n.pt").bounds is None
