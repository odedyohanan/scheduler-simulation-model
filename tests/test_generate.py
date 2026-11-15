import numpy as np

from sim.randomize import random_spec
from worldmodel.generate import replay_to, run_episode, scenario_spec


def test_record_shape():
    recs = run_episode(3, duration_s=5.0)
    assert len(recs) == 9
    r = recs[0]
    assert set(r) == {"run", "t", "driver", "input", "explored", "applied", "outcome"}
    assert set(r["input"]) == {"telemetry", "current_policy", "known"}
    for k in range(len(recs) - 1):
        assert recs[k]["outcome"] == recs[k + 1]["input"]["telemetry"]
        assert recs[k]["applied"] == recs[k + 1]["input"]["current_policy"]


def test_reproducible():
    assert run_episode(3, duration_s=5.0) == run_episode(3, duration_s=5.0)


def test_explore_rate():
    recs = run_episode(3, 60.0)
    frac = np.mean([r["explored"] for r in recs])
    assert 0.15 <= frac <= 0.45


def test_scenario_stream():
    assert scenario_spec(5) == random_spec(np.random.default_rng(20005))


def test_replay_to_matches_episode():
    recs = run_episode(4, duration_s=8.0)
    sim, policy = replay_to(4, 10, duration_s=8.0)
    from sim.dataset import policy_dict
    assert policy_dict(policy) == recs[10]["applied"]
    assert sim.run_window(policy) == recs[10]["outcome"]
