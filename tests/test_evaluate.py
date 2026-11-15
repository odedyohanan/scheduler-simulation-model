from experiments.evaluate import OVERLOAD_QUEUE_MS, overloaded_mask


def _rec(q):
    return {"input": {"telemetry": {"a": {"queue_work_ms": q}}}}


def test_overload_threshold_is_fixed_not_per_set():
    assert OVERLOAD_QUEUE_MS == 100.0
    calm = [_rec(q) for q in (1, 2, 3, 50)]
    assert not overloaded_mask(calm).any()            # a quantile rule would flag the 50 ms state
    assert overloaded_mask([_rec(99.9), _rec(100.1), _rec(5000)]).tolist() == [False, True, True]


def test_evaluate_alignment_history1_matches_v1():
    import json
    from pathlib import Path

    import numpy as np

    from experiments.evaluate import evaluate_set
    from worldmodel.features import INPUT_NAMES, STATE_NAMES
    from worldmodel.model import MLP, Normalizer, WorldModel
    recs = [json.loads(l) for l in open(Path(__file__).parent / "fixtures" / "adaptive_head.jsonl")]
    n_in, n_out = len(INPUT_NAMES), len(STATE_NAMES)
    m = WorldModel(MLP(n_in, n_out, hidden=4, layers=1), Normalizer(np.zeros(n_in, np.float32), np.ones(n_in, np.float32)),
                   Normalizer(np.zeros(n_out, np.float32), np.ones(n_out, np.float32)), {})
    a = evaluate_set(m, recs)
    b = evaluate_set(m, recs[::-1])
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
