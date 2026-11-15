"""Fast deterministic prediction of the next few seconds: RK4 fluid model + queueing theory.

The controller cannot afford a Monte Carlo run per candidate policy, so it uses this
"average behaviour" model instead:

  state y = [Q_a, buf_b, T_gpu0, T_gpu1]
    Q_a     A's backlog, in core-seconds of queued work
    buf_b   B's prepared-batch buffer
    T_g     GPU temperatures

  dQ_a/dt   = λ·s·(1+interference) − cap_a          (queue fills / drains)
  dbuf_b/dt = cap_b / cpu_per_batch − consumption    (CPUs fill, GPUs drain)
  dT_g/dt   = heat·power_g − cool·(T_g − ambient)    (thermal)

where cap_a, cap_b come from the same allocate() the simulator uses.

A fluid model gives averages, not tails. For A's p99 we add an M/M/c (Erlang C)
estimate on top of the fluid backlog.
"""
from __future__ import annotations

import math

import numpy as np

from .alloc import allocate
from .config import A, B, C, Spec, throttle_factor
from .policy import Policy

LN100 = math.log(100.0)


# ---------------------------------------------------------------------------- RK4
def rk4(f, y0: np.ndarray, t0: float, t1: float, h: float):
    """Classic 4th-order Runge-Kutta. Returns final y and the trajectory."""
    y, t = np.array(y0, dtype=float), t0
    traj = [y.copy()]
    while t < t1 - 1e-12:
        h_ = min(h, t1 - t)
        k1 = f(t, y)
        k2 = f(t + h_ / 2, y + h_ / 2 * k1)
        k3 = f(t + h_ / 2, y + h_ / 2 * k2)
        k4 = f(t + h_, y + h_ * k3)
        y = y + h_ / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h_
        traj.append(y.copy())
    return y, np.array(traj)


# ---------------------------------------------------------------------- Erlang C
def erlang_c(c: int, a: float) -> float:
    """Probability an arriving request must wait, M/M/c with offered load a = λ/μ."""
    if a >= c:
        return 1.0
    term, s = 1.0, 1.0
    for k in range(1, c):
        term *= a / k
        s += term
    term *= a / c
    last = term * c / (c - a)
    return last / (s + last)


def p99_response_ms(lam: float, service_ms: float, cores: float) -> float:
    """
    Approximate p99 response time (wait + service) for an M/M/c queue.

    Fractional capacity (e.g. 2.4 cores from fair sharing) is modelled as
    c = ceil(cores) servers, each running at speed cores/c.
    Wait tail:  P(W > t) = C(c, a) · exp(−(cμ − λ) t)   (Erlang C)
    Service tail for exponential service: p99 = mean · ln(100).
    The p99 of the sum is approximated by the sum of the two p99s (pessimistic).
    """
    if cores <= 0:
        return math.inf
    c = math.ceil(cores - 1e-9)
    speed = cores / c
    mu = 1000.0 / service_ms * speed            # requests/s per server
    if lam >= c * mu * 0.999:
        return math.inf
    pw = erlang_c(c, lam / mu)
    wait = 0.0 if pw <= 0.01 else math.log(pw / 0.01) / (c * mu - lam) * 1000.0
    return wait + service_ms / speed * LN100


# ------------------------------------------------------------------- the model
class FluidPredictor:
    def __init__(self, spec: Spec, horizon_s: float = 2.0, h: float = 0.02):
        self.spec, self.horizon, self.h = spec, horizon_s, h

    def a_inflation(self, elig: np.ndarray, others_busy: float) -> float:
        """Interference on A if anyone else may run in A's L3 domain(s)."""
        dom = np.zeros(elig.shape[1], dtype=bool)
        for d in self.spec.machine.l3_domains:
            if elig[A, list(d)].any():
                dom[list(d)] = True
        shared = elig[[B, C]][:, dom].any()
        return 1.0 + self.spec.a.interference * (others_busy if shared else 0.0)

    def a_shared(self, elig: np.ndarray) -> bool:
        return bool(elig[[B, C]][:, elig[A]].any())

    def predict(self, policy: Policy, state: dict, lam_forecast: float,
                lam_scenarios: list | None = None) -> dict:
        """
        state: {"q_a": core-s, "buf_b": batches, "temps": [..]}  (from telemetry)
        lam_forecast: A's expected arrival rate over the horizon (req/s); drives the RK4 run
        lam_scenarios: optional extra arrival rates; A's p99 is evaluated for each one
                       (cheap: queueing formula only, no extra RK4 run)
        """
        sp = self.spec
        elig = policy.eligibility(sp.machine.n_cpus)
        weights = policy.shares()
        owner = policy.gpu_owner(sp.machine.n_gpus)
        infl = self.a_inflation(elig, others_busy=1.0)
        s = sp.a.service_ms / 1000.0
        tau = 0.01  # s, how fast a backlog is "wanted" back

        def deriv(_t, y):
            q, buf = max(y[0], 0.0), max(y[1], 0.0)
            temps = y[2:]
            work_in = lam_forecast * s * infl                  # cores
            d_a = work_in + q / tau
            d_b = (sp.b.buffer_cap - buf) * sp.b.cpu_ms_per_batch / 1000.0 / tau
            got, _ = allocate(elig, np.array([d_a, d_b, 8.0]), weights)
            dq = work_in - got[A]
            prod = got[B] / (sp.b.cpu_ms_per_batch / 1000.0)  # batches/s
            cons, power = 0.0, np.zeros(len(temps))
            for g, o in enumerate(owner):
                f = throttle_factor(temps[g], sp.thermal)
                if o == B:
                    rate = 1000.0 / sp.b.gpu_ms_per_batch * f
                    c_g = rate * buf / (buf + 0.5)            # starves smoothly when buffer empty
                    cons += c_g
                    power[g] = c_g / (1000.0 / sp.b.gpu_ms_per_batch)
                elif o == C:
                    power[g] = sp.c.gpu_power * f
            dbuf = prod - cons
            th = sp.thermal
            dT = th.heat * power - th.cool * (temps - th.ambient)
            return np.concatenate([[dq, dbuf], dT])

        y0 = np.concatenate([[state["q_a"], state["buf_b"]], state["temps"]])
        y_end, traj = rk4(deriv, y0, 0.0, self.horizon, self.h)

        # Average capacities over the horizon, from the trajectory.
        caps = []
        for y in traj:
            q, buf = max(y[0], 0.0), max(y[1], 0.0)
            work_in = lam_forecast * s * infl
            d_a = work_in + q / tau
            d_b = (sp.b.buffer_cap - buf) * sp.b.cpu_ms_per_batch / 1000.0 / tau
            got, _ = allocate(elig, np.array([d_a, d_b, 8.0]), weights)
            caps.append(got)
        caps = np.array(caps).mean(axis=0)

        # A's p99: capacity A *could* get if it needed it (its fair share against busy neighbours)
        a_avail, _ = allocate(elig, np.array([8.0, 8.0, 8.0]), weights)
        delay = sp.sched.delay_ms[policy.workloads[A].latency_class] if self.a_shared(elig) else 0.0
        backlog_ms = max(y_end[0], 0.0) / max(a_avail[A], 1e-6) * 1000.0
        p99 = p99_response_ms(lam_forecast * infl, sp.a.service_ms, a_avail[A]) + delay + backlog_ms

        # Throughputs over the horizon.
        # batches consumed = batches produced − growth of the buffer
        produced = caps[B] / (sp.b.cpu_ms_per_batch / 1000.0) * self.horizon
        b_rate = (produced - (traj[-1][1] - traj[0][1])) / self.horizon
        gpu_c = sum(sp.c.gpu_core_equiv * throttle_factor(y_end[2 + g], sp.thermal)
                    for g, o in enumerate(owner) if o == C)
        c_rate = (caps[C] + gpu_c) * 1000.0                 # core-ms per s
        extra = {}
        if lam_scenarios is not None:
            extra["a_p99_scenarios_ms"] = [
                p99_response_ms(l * infl, sp.a.service_ms, a_avail[A]) + delay + backlog_ms
                for l in lam_scenarios]
        return {
            **extra,
            "a_p99_ms": round(float(p99), 2) if math.isfinite(p99) else math.inf,
            "a_cores_available": round(float(a_avail[A]), 2),
            "b_batches_per_s": round(float(b_rate), 1),
            "c_core_ms_per_s": round(float(c_rate), 1),
            "gpu_temp_end": [round(float(x), 1) for x in y_end[2:]],
            "q_a_end": float(max(y_end[0], 0.0)),
        }

    # ---- helper for the optimizer: how many cores does A need? -------------------
    def a_cores_needed(self, lam: float, interference: bool, delay_class: str | None,
                       others_busy: float = 1.0) -> float:
        """
        Smallest core count (0.05 steps) meeting A's p99 SLO.
          interference  others are busy in A's L3 domain (service time inflated)
          delay_class   A shares its CPUs, so it pays the wake-up delay of this class
        """
        sp = self.spec
        infl = 1.0 + (sp.a.interference * others_busy if interference else 0.0)
        delay = sp.sched.delay_ms[delay_class] if delay_class else 0.0
        target = sp.a.slo_p99_ms * 0.85             # safety margin
        load = lam * infl * sp.a.service_ms / 1000.0  # cores of pure work: can't need less
        c = max(0.05, math.floor(load * 20) / 20)
        while c <= sp.machine.n_cpus:
            if p99_response_ms(lam * infl, sp.a.service_ms, c) + delay <= target:
                return c
            c += 0.05
        return float(sp.machine.n_cpus)
