import sys

import numpy as np


def test_no_ortools():
    import sim.controllers  # noqa: F401
    import sim.dataset  # noqa: F401
    import sim.predictor  # noqa: F401
    import sim.simulator  # noqa: F401
    assert "ortools" not in sys.modules


def test_adaptive_closed_loop_runs():
    from sim.controllers import AdaptiveController
    from sim.randomize import random_spec
    from sim.simulator import Simulator
    spec = random_spec(np.random.default_rng(20000))
    sim = Simulator(spec, 0, 5.0)
    ctrl = AdaptiveController(spec)
    p, tels = ctrl.initial_policy(), []
    while not sim.done():
        tel = sim.run_window(p)
        tels.append(tel)
        p = ctrl.decide(tel)
    assert len(tels) == 10
    for t in tels:
        assert {"a", "b", "c", "gpu_temp"} <= set(t)


def test_simulator_deterministic():
    from sim.config import Spec
    from sim.controllers import isolated_policy
    from sim.simulator import Simulator

    def p99s():
        s = Simulator(Spec(), 7, 3.0)
        out = []
        while not s.done():
            out.append(s.run_window(isolated_policy())["a"]["p99_ms"])
        return out
    assert p99s() == p99s()
