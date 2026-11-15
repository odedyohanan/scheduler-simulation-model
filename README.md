# Scheduler simulation model

Can a small neural network learn **how a system behaves** well enough to stand in for its simulator?

In an earlier project ([priority-scheduler](https://github.com/odedyohanan/priority-scheduler)) an ML model
learned to *make* scheduling decisions. Here the question is turned around: given the system's current state
and a scheduling decision, can a model **predict what happens next**? A model like this is called a
*world model*. This repository is a learning project, not a production system. v1 is a plain MLP, and
the code is set up so a GRU or a richer learned world model can slot in later.

## The system being modelled

A simulated machine with 8 CPUs (two cache domains) and 2 GPUs, shared by three workloads:

| Workload | What it models |
|---|---|
| **A** | A latency-critical request service with waves and bursts. Target: p99 latency ≤ 10 ms |
| **B** | GPU training: CPUs prepare batches into a buffer, GPUs consume them; GPUs heat up and throttle |
| **C** | Best-effort batch work that takes any spare CPU (and can use a GPU) |

Every 500 ms a controller picks a **policy**: which CPUs each workload may use, CPU weights, and who owns
each GPU. The simulator (copied from priority-scheduler into `sim/`) then runs that window in 1 ms steps,
with random arrivals, cache interference, migration costs and GPU thermals, and reports telemetry.

## What the model learns

```
 state at window t  ──┐
 (A latency/queue,    │
  B rate/buffer,      ├──►  MLP  ──►  change in state  ──►  state at window t+1
  C work, per-CPU     │      (2 × 128, ReLU)
  usage, GPU temps)   │
 applied policy  ─────┤
 known parameters ────┘
```

- **Inputs (75):** 39 state numbers from the window's telemetry, 33 numbers describing the applied
  policy, and 3 known parameters (A's service time, interference strength, room temperature).
- **Output (39):** the change in every state number. Predicting the *whole* state means the model can feed
  on its own predictions for multi-step rollouts.
- Heavy-tailed values (latency, queue) are log-compressed. Windows where no A request completed (p99
  reported as 0) are left out of the p99 loss.

## Data

- **Training:** 300 simulated minutes, 36k transitions in total (28.5k train, 3.6k validation, 3.6k test). The adaptive controller (a hand-written
  feedback rule) drives, and 30% of the windows apply a randomly perturbed policy instead, so the model
  also sees bad decisions. Whole runs are held out: run id ending in 8 → validation, in 9 → test.
- **Generalisation tests:** transitions recorded in the earlier project, from controllers the MLP never saw:
  the CP-SAT **MPC** teacher (5 950 test transitions) and the imitation **student** (DAgger rounds 1 and 2,
  1 190 each). Our simulator replays those runs bit for bit, so any difference is down to the
  controllers, not a changed simulator.

## Baselines and the noise floor

- **Persistence:** "the next window looks like this one".
- **Fluid model:** the hand-written physics model the MPC teacher used (RK4 fluid model + queueing theory),
  run for one window. It predicts only some of the targets.
- **Noise floor:** the simulator is random. Replaying a test run to a window and re-simulating that
  window 20 times with different seeds measures the randomness left *even when the simulator's full hidden
  state is known*. A model that sees only telemetry can't reach it, so it is a lower bound, not a target.

## Results (v1)

### One step ahead, same controller as training (adaptive test runs)

Absolute error in real units, **mean / median**. Lower is better. A mean far above the median means a few
large misses dominate.

| Target | MLP | Persistence | Fluid model | Noise floor |
|---|---|---|---|---|
| A p99 latency (ms) | **5.1** / **0.50** | 6.4 / 0.67 | 8.7 / 0.78 | 0.4 |
| A queued work (ms) | 74.5 / 1.23 | 31.7 / 1.65 | **21.8** / **1.22** | 1.2 |
| A CPU cores | **0.19** / 0.13 | 0.21 / **0.11** | — | 0.08 |
| B batches/s | **18.5** / 8.7 | 96.8 / **6.0** | 95.0 / 95.6 | 1.2 |
| B buffer (batches) | **1.7** / 0.84 | 3.0 / **0.38** | — | 0.2 |
| C work (core-ms) | 49.9 / 35.4 | 349 / 41.2 | **31.9** / **16.3** | 16.2 |
| GPU 0 temp (°C) | **0.04** / **0.02** | 0.31 / 0.15 | 0.21 / 0.14 | 0.0003 |

### One step ahead, controllers never seen in training

Median absolute error. The mean is dominated by a few wild misses: on MPC data, 5 of 5 950 predictions
produce 64% of the total queue error. Those states lie outside anything seen in training, and the MLP
predicts a queue of about 10⁷ ms where the truth is about 8 000.

| Target | MLP (MPC) | Persistence (MPC) | MLP (student r1) | Persistence (student r1) |
|---|---|---|---|---|
| A p99 latency (ms) | 2.25 | **0.52** | 1.54 | **0.36** |
| A queued work (ms) | 2.96 | **1.74** | 1.68 | **1.64** |
| B batches/s | 80.1 | **6.9** | 80.3 | **5.9** |
| C work (core-ms) | 197 | **46** | 156 | **28** |
| GPU 0 temp (°C) | **0.17** | 0.22 | **0.06** | 0.26 |

Full tables, with means, medians and overloaded states only (A queue > 100 ms):
[`results/onestep/report.md`](results/onestep/report.md).

### Many steps ahead (rollouts)

The MLP feeds on its own predictions for 20 windows (10 s), replaying the policies that were actually
applied.

![Rollout error vs horizon](results/rollout/error_vs_horizon.png)

A good stretch (adaptive test run 9): smooth quantities track the simulator for many seconds.

![Example rollout](results/rollout/example_adaptive_9.png)

Tables and more examples: [`results/rollout/summary.md`](results/rollout/summary.md).

## What this shows

1. **On familiar states the MLP learns the big changes.** On GPU temperatures and A's latency it beats
   both baselines on both mean and median. On B's throughput and buffer it wins on the *mean* (B batches/s
   18.5 vs about 96) but loses on the *median* (8.7 vs 6.0 for "nothing changes"). It catches the rare
   large jumps, for example when a policy takes GPUs away from B, and it does that well. But in an
   ordinary calm window, "nothing changes" is the better guess.
2. **The physics model is hard to beat where the physics is simple.** For C's work and A's queue, the
   fluid model, with a few lines of equations and no training, wins.
3. **A's queue is the MLP's weak spot, even in-distribution.** A few overloaded states with huge queues
   dominate. A small error in log space becomes a very large error in milliseconds.
4. **It does not generalise to other controllers.** On states the MPC teacher or the student produced, the
   MLP is often worse than "nothing changes". The training data came from one controller (plus
   perturbations), and the MPC visits policy and state combinations the MLP never saw. This is the classic
   world-model lesson: **the model is only as good as the decisions in its data.**
5. **Rollouts explode.** Errors compound. On adaptive runs, GPU temperatures and B's rate stay better than
   persistence for the full 10 s. On the other controllers' runs even those drift past persistence
   after 5–7 s. Predicted latency and queue can grow without limit everywhere, because nothing stops the
   model predicting "the queue grows" again and again. Persistence is often the better 10-second forecast.
6. **The errors are well above the noise floor.** The floor is a lower bound: it assumes knowledge of
   hidden state, such as individual queued requests, that telemetry doesn't show. So not all of the gap
   can be closed from telemetry alone. But the gap is large, which points to a model that remembers
   recent windows, i.e. a GRU.

## Next steps

- Train on a mix of controllers (adaptive + MPC + student data) and see how much point 4 improves.
- Keep rollouts physical: bound predictions to what is possible (a queue cannot grow faster than arrivals).
- A **GRU** over several past windows: the simulator has hidden state (individual queued requests, bursts
  in progress) that one window of telemetry doesn't show.
- An ensemble to tell when the model is *uncertain*, for example on unfamiliar controllers.
- Later: a learned world model used for planning, i.e. choosing policies by imagining their outcomes.

## How to run

Python 3 with numpy, torch, matplotlib, pytest. The generalisation test sets come from a checkout of the
earlier project's data in a sibling directory, `../priority_scheduler_model/data`. You can set `OLD_DATA`
to point elsewhere.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest -q
```

```bash
# 1. data: 300 one-minute runs (~3 min at full speed; scripts/gentle.sh runs it at low priority, ~1 h)
.venv/bin/python experiments/generate_data.py --runs 300 --procs 5 --out data/adaptive_v1
# 2. train (a few minutes on CPU)
.venv/bin/python experiments/train.py --data data/adaptive_v1 --out models/mlp_v1.pt
# 3. one-step evaluation against baselines and the noise floor (~3 min)
.venv/bin/python experiments/evaluate.py --model models/mlp_v1.pt --out results/onestep
# 4. multi-step rollouts and plots
.venv/bin/python experiments/rollout.py --model models/mlp_v1.pt --out results/rollout
```

## Layout

```
sim/            the simulator and adaptive controller, copied from priority-scheduler (no CP-SAT)
worldmodel/     generate.py, data.py, features.py, model.py, train.py, baselines.py,
                metrics.py, noise.py, rollout.py
experiments/    command-line scripts for each step above
scripts/        gentle.sh (run a long job at low priority on macOS), stop.sh
results/        reports and plots
tests/          pytest suite
```
