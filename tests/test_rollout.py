import json
from pathlib import Path

import numpy as np

from worldmodel.features import INPUT_NAMES, STATE_NAMES, build_arrays, decode_state
from worldmodel.model import MLP, Normalizer, WorldModel
from worldmodel.rollout import rollout

FIX = Path(__file__).parent / "fixtures"


def _recs():
    return [json.loads(l) for l in open(FIX / "adaptive_head.jsonl")]


def _model(bias=0.0):
    n_in, n_out = len(INPUT_NAMES), len(STATE_NAMES)
    mlp = MLP(n_in, n_out, hidden=8, layers=1)
    x = Normalizer(np.zeros(n_in, np.float32), np.ones(n_in, np.float32))
    y = Normalizer(np.full(n_out, bias, np.float32), np.ones(n_out, np.float32))
    if bias:
        for p in mlp.parameters():
            p.data.zero_()
    return WorldModel(mlp, x, y)


def test_rollout_shapes():
    pred, true = rollout(_model(), _recs(), start=0, horizon=4)
    assert pred.shape == true.shape == (4, len(STATE_NAMES))


def test_rollout_horizon_capped_by_run_length():
    pred, true = rollout(_model(), _recs(), start=2, horizon=10)
    assert len(pred) == len(true) == 3


def test_rollout_step1_equals_onestep():
    m, recs = _model(), _recs()
    X, S, Y, M = build_arrays(recs[:1])
    pred, true = rollout(m, recs, start=0, horizon=2)
    np.testing.assert_allclose(pred[0], np.maximum(m.predict_next(X, S)[0], 0), rtol=1e-5)
    np.testing.assert_allclose(true[0], S[0] + Y[0], rtol=1e-5)


def test_rollout_nonnegative():
    pred, _ = rollout(_model(bias=-50.0), _recs(), start=0, horizon=4)
    for v in pred:
        d = decode_state(v)
        assert d["b.buffer"] >= 0 and d["a.queue_work_ms"] >= 0
    assert (pred >= 0).all()
