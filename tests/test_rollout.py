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


class _Recorder(WorldModel):
    """History-3 MLP stub that records every prepared input it is asked about."""
    def __init__(self):
        n_in, n_out = len(INPUT_NAMES), len(STATE_NAMES)
        super().__init__(MLP(3 * n_in, n_out, hidden=4, layers=1),
                         Normalizer(np.zeros(3 * n_in, np.float32), np.ones(3 * n_in, np.float32)),
                         Normalizer(np.zeros(n_out, np.float32), np.ones(n_out, np.float32)),
                         {"arch": "mlp", "history": 3})
        self.seen = []

    def predict_delta(self, X):
        self.seen.append(np.array(X, dtype=np.float32).reshape(3, -1))
        return super().predict_delta(X)


def test_rollout_history_uses_real_past():
    from worldmodel.features import encode_input, encode_state
    recs, m = _recs(), _Recorder()
    pred, _ = rollout(m, recs, start=2, horizon=2)
    real = [encode_input(encode_state(r["input"]["telemetry"]), r["applied"], r["input"]["known"]) for r in recs]
    np.testing.assert_allclose(m.seen[0], np.stack(real[0:3]), rtol=1e-6)
    n_state = len(STATE_NAMES)
    np.testing.assert_allclose(m.seen[1][-1][:n_state], pred[0], rtol=1e-6)      # own prediction fed back
    np.testing.assert_allclose(m.seen[1][1], real[2], rtol=1e-6)                 # older rows slide


def test_rollout_step1_matches_onestep_gru():
    from worldmodel.history import build_history_arrays
    from worldmodel.model import GRUNet
    n_in, n_out = len(INPUT_NAMES), len(STATE_NAMES)
    m = WorldModel(GRUNet(n_in, n_out, hidden=8), Normalizer(np.zeros(n_in, np.float32), np.ones(n_in, np.float32)),
                   Normalizer(np.zeros(n_out, np.float32), np.ones(n_out, np.float32)), {"arch": "gru", "history": 4})
    recs = _recs()
    H, S, Y, M, ordered = build_history_arrays(recs, 4)
    pred, _ = rollout(m, recs, start=3, horizon=1)
    np.testing.assert_allclose(pred[0], np.maximum(m.predict_next(m.prepare(H[3:4]), S[3:4])[0], 0), rtol=1e-5)
