"""Multi-step rollouts: how fast does the MLP's error grow when it feeds on its own predictions?

  .venv/bin/python experiments/rollout.py --model models/mlp_v1.pt --out results/rollout

For every test run, rollouts start every 20 windows and go 20 windows (10 s) ahead, replaying the
policies that were actually applied. "persistence" = the start state held constant.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.evaluate import SETS  # noqa: E402
from worldmodel.data import group_by_run, load_records  # noqa: E402
from worldmodel.features import KEY_TARGETS, decode_state, encode_state, state_mask  # noqa: E402
from worldmodel.model import WorldModel  # noqa: E402
from worldmodel.rollout import rollout  # noqa: E402

UNITS = {"a.p99_ms": "ms", "a.queue_work_ms": "ms", "a.cpu_cores": "cores", "b.rate": "batches/s",
         "b.buffer": "batches", "c.work_core_ms": "core-ms", "gpu_temp_0": "°C", "gpu_temp_1": "°C"}


def _key(states: np.ndarray, recs: list[dict] | None = None) -> dict[str, np.ndarray]:
    d = [decode_state(v) for v in states]
    out = {t: np.array([x[t] for x in d]) for t in KEY_TARGETS}
    if recs is not None:      # true p99 is undefined when no A request completed
        bad = np.array([state_mask(r["outcome"]).min() == 0 for r in recs])
        out["a.p99_ms"][bad] = np.nan
    return out


def error_curves(model, runs: dict, horizon: int, every: int):
    err = {m: {t: [[] for _ in range(horizon)] for t in KEY_TARGETS} for m in ("mlp", "persistence")}
    for recs in runs.values():
        for start in range(0, len(recs) - horizon + 1, every):
            pred, true = rollout(model, recs, start, horizon)
            tk = _key(true, recs[start:start + horizon])
            s0 = encode_state(recs[start]["input"]["telemetry"])
            for name, p in (("mlp", _key(pred)), ("persistence", _key(np.repeat(s0[None], len(true), 0)))):
                for t in KEY_TARGETS:
                    for h, e in enumerate(np.abs(p[t] - tk[t])):
                        if np.isfinite(e):
                            err[name][t][h].append(e)
    return {m: {t: [float(np.mean(v)) if v else float("nan") for v in hs] for t, hs in d.items()}
            for m, d in err.items()}


def plot_curves(curves: dict, path: Path, horizon: int):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    steps = np.arange(1, horizon + 1)
    colors = dict(zip(curves, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    for ax, t in zip(axes.flat, KEY_TARGETS):
        for s, c in curves.items():
            ax.plot(steps, c["mlp"][t], color=colors[s], label=f"{s} mlp")
            ax.plot(steps, c["persistence"][t], color=colors[s], ls="--", alpha=0.6, label=f"{s} persistence")
        ax.set_title(t)
        ax.set_xlabel("windows ahead (×0.5 s)")
        ax.set_ylabel(f"MAE ({UNITS[t]})")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_example(model, recs: list[dict], title: str, path: Path, horizon: int = 40):
    pred, true = rollout(model, recs, 0, horizon)
    p, t = _key(pred), _key(true, recs[:horizon])
    x = np.arange(1, len(pred) + 1) * 0.5
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, k in zip(axes.flat, KEY_TARGETS):
        ax.plot(x, t[k], label="simulator", color="black")
        ax.plot(x, p[k], label="MLP rollout", color="tab:orange")
        ax.set_title(k)
        ax.set_xlabel("seconds")
        ax.set_ylabel(UNITS[k])
    axes.flat[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--every", type=int, default=20)
    a = ap.parse_args()
    model, out = WorldModel.load(a.model), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    curves, md = {}, [f"# Rollouts — `{a.model}`", "",
                      f"MAE of the MLP feeding on its own predictions, {a.horizon} windows ahead "
                      "(persistence in brackets).", ""]
    for name, d in SETS.items():
        runs = group_by_run(load_records([d], split="test"))
        curves[name] = error_curves(model, runs, a.horizon, a.every)
        for i, run in enumerate(list(runs)[:2]):
            plot_example(model, runs[run], f"{name} run {run}: simulator vs MLP rollout from t=0",
                         out / f"example_{name}_{run}.png")
        c = curves[name]
        md += [f"## {name} ({len(runs)} runs)", "", "| target | step 1 | step 5 | step 10 | step 20 |", "|---|---|---|---|---|"]
        for t in KEY_TARGETS:
            cells = [f"{c['mlp'][t][h - 1]:.3g} ({c['persistence'][t][h - 1]:.3g})"
                     for h in (1, 5, 10, 20) if h <= a.horizon]
            md.append(f"| {t} | " + " | ".join(cells) + " |")
        md.append("")
        print(f"{name}: done", flush=True)
    plot_curves(curves, out / "error_vs_horizon.png", a.horizon)
    (out / "curves.json").write_text(json.dumps(curves, indent=1))
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
