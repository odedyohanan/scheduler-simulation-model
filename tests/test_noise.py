import numpy as np

from worldmodel.generate import replay_to, run_episode
from worldmodel.noise import noise_floor


def test_noise_floor_positive():
    nf = noise_floor(9, [10], k=5, duration_s=8.0)
    assert nf["a.p99_ms"] > 0
    assert all(np.isfinite(v) for v in nf.values())


def test_replay_matches_generation():
    sim, policy = replay_to(9, 10, duration_s=8.0)
    assert sim.run_window(policy) == run_episode(9, duration_s=8.0)[10]["outcome"]
