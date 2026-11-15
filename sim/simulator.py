"""Stochastic time-stepped simulator of the machine: the "real system" for now.

It plays the role eBPF + the kernel will play later:
  - it runs with whatever Policy it is given,
  - it emits telemetry every policy window,
  - the controller never sees its internals (true arrival rate, bursts, ...).

Each step is 1 ms. Workload A requests are tracked individually so tail
latency (p99) is real, not approximated. Randomness: Poisson arrivals,
exponential service times, random burst times. Different seeds = different
Monte Carlo runs.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .alloc import allocate
from .config import A, B, C, Spec, throttle_factor
from .policy import Policy


class Simulator:
    def __init__(self, spec: Spec, seed: int, duration_s: float):
        self.spec = spec
        self.rng = np.random.default_rng(seed)
        self.duration = duration_s
        self.t = 0.0
        m = spec.machine
        self.n_cpus, self.n_gpus = m.n_cpus, m.n_gpus

        # Burst schedule for workload A, hidden from the controller.
        starts, t = [], 0.0
        while True:
            t += self.rng.exponential(spec.a.burst_every)
            if t >= duration_s:
                break
            starts.append(t)
        self.bursts = np.array(starts)

        # A requests: [arrival_t, remaining_ms, extra_delay_ms, counted_late]
        # counted_late: the request already passed its deadline (age > SLO) while queued
        # and was counted as late then; it is not counted again when it completes.
        self.queue: deque = deque()
        self.b_buffer = 0.0
        self.temps = np.full(self.n_gpus, spec.thermal.ambient + 20.0)
        self.migrating_until = np.zeros(3)
        self.prev_elig: np.ndarray | None = None
        self.prev_usage = np.zeros((3, self.n_cpus))
        self.all_latencies: list[np.ndarray] = []

    # ---- ground truth (not visible to the controller) -------------------------
    def rate_a(self, t: float) -> float:
        a = self.spec.a
        r = a.base_rate + a.wave_amp * np.sin(2 * np.pi * t / a.wave_period)
        if len(self.bursts) and np.any((t >= self.bursts) & (t < self.bursts + a.burst_len)):
            r += a.burst_extra
        return max(r, 0.0)

    # ---- one policy window -------------------------------------------------------
    def run_window(self, policy: Policy) -> dict:
        spec, dt = self.spec, self.spec.dt
        dt_ms = dt * 1000.0
        elig = policy.eligibility(self.n_cpus)
        weights = policy.shares()
        gpu_owner = policy.gpu_owner(self.n_gpus)
        a_class = policy.workloads[A].latency_class
        a_delay_mean = spec.sched.delay_ms[a_class]

        # Changing a workload's cpuset costs it some efficiency (cold caches, migrations).
        if self.prev_elig is not None:
            changed = (elig != self.prev_elig).any(axis=1)
            self.migrating_until[changed] = self.t + spec.costs.migration_penalty_s
        self.prev_elig = elig

        # CPUs in the L3 domains A runs in: other workloads there slow A down.
        a_dom = np.zeros(self.n_cpus, dtype=bool)
        for dom in spec.machine.l3_domains:
            if elig[A, list(dom)].any():
                a_dom[list(dom)] = True

        n_steps = int(round(spec.policy_period_s / dt))
        acc = dict(arrivals=0, b_batches=0.0, c_work=0.0,
                   usage=np.zeros((3, self.n_cpus)), gpu_power=np.zeros(self.n_gpus))
        lat_window: list[float] = []
        late_new = 0          # requests that became late this window (completed late, or overdue in queue)

        for _ in range(n_steps):
            # 1. A arrivals. Service time inflated by neighbours' activity in A's L3 domain;
            #    extra wake-up delay if A's CPU is currently shared with someone else.
            n_arr = self.rng.poisson(self.rate_a(self.t) * dt)
            if n_arr:
                others = self.prev_usage[[B, C]].sum(axis=0)
                busy = others[a_dom].sum() / max(a_dom.sum(), 1)
                inflate = 1.0 + spec.a.interference * min(busy, 1.0)
                a_cpus = elig[A]
                shared_frac = (others[a_cpus] > 0.05).mean()
                svc = self.rng.exponential(spec.a.service_ms, n_arr) * inflate
                delay = np.where(self.rng.random(n_arr) < shared_frac,
                                 self.rng.exponential(a_delay_mean, n_arr), 0.0)
                for s, d in zip(svc, delay):
                    self.queue.append([self.t, s, d, False])
                acc["arrivals"] += n_arr

            # 2. Demands (cores wanted this step).
            head = [self.queue[i][1] for i in range(min(len(self.queue), self.n_cpus + 1))]
            d_a = min(sum(min(r, dt_ms) for r in head) / dt_ms, self.n_cpus)
            d_b = min((spec.b.buffer_cap - self.b_buffer) * spec.b.cpu_ms_per_batch / dt_ms,
                      self.n_cpus)
            d_c = float(self.n_cpus)
            got, usage = allocate(elig, np.array([d_a, d_b, d_c]), weights)
            eff = np.where(self.t < self.migrating_until, spec.costs.migration_efficiency, 1.0)

            # 3. Serve A: FIFO, each request uses at most one core.
            cap = got[A] * eff[A] * dt_ms
            unfinished = []
            while cap > 1e-12 and self.queue:
                req = self.queue.popleft()
                give = min(req[1], dt_ms, cap)
                req[1] -= give
                cap -= give
                if req[1] <= 1e-9:
                    lat = (self.t + dt - req[0]) * 1000.0 + req[2]
                    lat_window.append(lat)
                    if lat > spec.a.slo_p99_ms and not req[3]:
                        late_new += 1
                else:
                    unfinished.append(req)
            self.queue.extendleft(reversed(unfinished))

            # 4. B: CPUs fill the batch buffer, GPUs drain it.
            self.b_buffer = min(spec.b.buffer_cap,
                                self.b_buffer + got[B] * eff[B] * dt_ms / spec.b.cpu_ms_per_batch)
            # 5. C: CPU work plus GPU work.
            acc["c_work"] += got[C] * eff[C] * dt_ms

            power = np.zeros(self.n_gpus)
            for g, owner in enumerate(gpu_owner):
                f = throttle_factor(self.temps[g], spec.thermal)
                if owner == B:
                    can = dt_ms / spec.b.gpu_ms_per_batch * f
                    done = min(can, self.b_buffer)
                    self.b_buffer -= done
                    acc["b_batches"] += done
                    power[g] = done / (dt_ms / spec.b.gpu_ms_per_batch)
                elif owner == C:
                    acc["c_work"] += spec.c.gpu_core_equiv * f * dt_ms
                    power[g] = spec.c.gpu_power * f

            # 6. GPU temperatures (Euler is fine at 1 ms).
            th = spec.thermal
            self.temps += dt * (th.heat * power - th.cool * (self.temps - th.ambient))

            acc["usage"] += usage
            acc["gpu_power"] += power
            self.prev_usage = usage
            self.t += dt

        lat = np.array(lat_window)
        self.all_latencies.append(lat)
        # Requests still queued at the window end: their age so far (+ any wake-up delay
        # they will pay) is a lower bound on their latency. Overdue ones are late already.
        slo = spec.a.slo_p99_ms
        ages = np.array([(self.t - r[0]) * 1000.0 + r[2] for r in self.queue])
        overdue = ages > slo
        for r, od in zip(self.queue, overdue):
            if od and not r[3]:
                r[3] = True
                late_new += 1
        ontime = int((lat <= slo).sum())
        win = spec.policy_period_s
        u = acc["usage"] / n_steps
        return {
            "t": round(self.t, 3),
            "window_s": win,
            "a": {
                "arrivals": acc["arrivals"],
                "rate": acc["arrivals"] / win,
                # p50/p99 of requests *completed* in this window. 0.0 when none completed:
                # check `completions` (or p99_valid) before reading 0.0 as "fast".
                "p50_ms": float(np.percentile(lat, 50)) if len(lat) else 0.0,
                "p99_ms": float(np.percentile(lat, 99)) if len(lat) else 0.0,
                "p99_valid": bool(len(lat)),
                "completions": int(len(lat)),
                "completed_ontime": ontime,
                "late_new": int(late_new),
                "queue_overdue": int(overdue.sum()),
                "queue_oldest_age_ms": float(ages.max()) if len(ages) else 0.0,
                "queue_len": len(self.queue),
                "queue_work_ms": float(sum(r[1] for r in self.queue)),
                "cpu_cores": float(u[A].sum()),
            },
            "b": {"batches": acc["b_batches"], "rate": acc["b_batches"] / win,
                  "buffer": self.b_buffer, "cpu_cores": float(u[B].sum())},
            "c": {"work_core_ms": acc["c_work"], "cpu_cores": float(u[C].sum())},
            "cpu_util": u.sum(axis=0).round(3).tolist(),
            "cpu_usage_by_workload": u.round(3).tolist(),
            "gpu_temp": self.temps.round(2).tolist(),
            "gpu_power": (acc["gpu_power"] / n_steps).round(3).tolist(),
        }

    def backlog(self) -> dict:
        """Unfinished A requests right now (call at the end of a run)."""
        ages = np.array([(self.t - r[0]) * 1000.0 + r[2] for r in self.queue])
        return {"count": len(self.queue),
                "overdue": int((ages > self.spec.a.slo_p99_ms).sum()) if len(ages) else 0,
                "oldest_age_ms": float(ages.max()) if len(ages) else 0.0,
                "work_ms": float(sum(r[1] for r in self.queue))}

    def done(self) -> bool:
        return self.t >= self.duration - 1e-9
