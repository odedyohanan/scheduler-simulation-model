from experiments.evaluate import OVERLOAD_QUEUE_MS, overloaded_mask


def _rec(q):
    return {"input": {"telemetry": {"a": {"queue_work_ms": q}}}}


def test_overload_threshold_is_fixed_not_per_set():
    assert OVERLOAD_QUEUE_MS == 100.0
    calm = [_rec(q) for q in (1, 2, 3, 50)]
    assert not overloaded_mask(calm).any()            # a quantile rule would flag the 50 ms state
    assert overloaded_mask([_rec(99.9), _rec(100.1), _rec(5000)]).tolist() == [False, True, True]
