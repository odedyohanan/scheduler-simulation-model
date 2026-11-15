"""One-step evaluation: MLP vs persistence vs fluid model, on four test sets, next to the noise floor.

  .venv/bin/python experiments/evaluate.py --model models/mlp_v1.pt --out results/onestep

Test sets use held-out runs only (run id ending in 9):
  adaptive   our own generator (same controller the MLP trained on)
  mpc        priority_scheduler_model teacher_v2: the CP-SAT MPC teacher drove (never seen in training)
  dagger_r1  / dagger_r2: the imitation student drove (never seen in training)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worldmodel.baselines import fluid_next  # noqa: E402
from worldmodel.data import load_records  # noqa: E402
from worldmodel.features import KEY_TARGETS, STATE_NAMES, build_arrays, decode_state  # noqa: E402
from worldmodel.metrics import per_target  # noqa: E402
from worldmodel.model import WorldModel  # noqa: E402
from worldmodel.noise import noise_floor  # noqa: E402

# Data from the earlier project, expected in a sibling checkout; override with OLD_DATA=/path/to/data.
ROOT = Path(__file__).resolve().parents[1]
OLD = os.environ.get("OLD_DATA", str(ROOT.parent / "priority_scheduler_model" / "data"))
SETS = {"adaptive": str(ROOT / "data" / "adaptive_v1"), "mpc": f"{OLD}/teacher_v2",
        "dagger_r1": f"{OLD}/dagger_r1", "dagger_r2": f"{OLD}/dagger_r2"}
_IDX = [STATE_NAMES.index(t) for t in KEY_TARGETS]
# "Overloaded": A has more than 10x its SLO (10 ms) of work queued. One fixed threshold for every test set,
# so the subset means the same thing everywhere (a per-set quantile would not).
OVERLOAD_QUEUE_MS = 100.0


def overloaded_mask(recs: list[dict]) -> np.ndarray:
    return np.array([r["input"]["telemetry"]["a"]["queue_work_ms"] > OVERLOAD_QUEUE_MS for r in recs])


def _decode(arr):
    d = [decode_state(v) for v in arr]
    return {t: np.array([x[t] for x in d]) for t in KEY_TARGETS}


def evaluate_set(model: WorldModel, recs: list[dict]) -> dict:
    X, S, Y, M = build_arrays(recs)
    true = _decode(S + Y)
    mask = {t: M[:, i] for t, i in zip(KEY_TARGETS, _IDX)}
    preds = {"mlp": _decode(model.predict_next(X, S)), "persistence": _decode(S)}
    fl = [fluid_next(r) for r in recs]
    preds["fluid"] = {t: np.array([f.get(t, np.nan) for f in fl]) for t in KEY_TARGETS}
    heavy = overloaded_mask(recs)
    res = {"n": len(recs), "n_overloaded": int(heavy.sum()), "all": {}, "overloaded": {}}
    for name, p in preds.items():
        res["all"][name] = per_target(true, p, mask)
        res["overloaded"][name] = per_target(
            {t: v[heavy] for t, v in true.items()}, {t: v[heavy] for t, v in p.items()},
            {t: v[heavy] for t, v in mask.items()})
    return res


def _table(res: dict, part: str, nf: dict | None) -> list[str]:
    cols = ["mlp", "persistence", "fluid"]
    head = ("| target | " + " | ".join(f"{c} MAE" for c in cols) + " | "
            + " | ".join(f"{c} median" for c in cols) + " | mlp R² |" + (" noise floor* |" if nf else ""))
    lines = [head, "|" + "---|" * (2 * len(cols) + 2 + bool(nf))]
    for t in KEY_TARGETS:
        cells = [res[part][c][t]["mae"] for c in cols] + [res[part][c][t]["medae"] for c in cols]
        row = f"| {t} | " + " | ".join("—" if np.isnan(v) else f"{v:.3g}" for v in cells)
        row += f" | {res[part]['mlp'][t]['r2']:.3f} |"
        if nf:
            row += f" {nf[t]:.3g} |"
        lines.append(row)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--noise-runs", type=int, default=10)
    ap.add_argument("--noise-k", type=int, default=20)
    a = ap.parse_args()
    model = WorldModel.load(a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    test_runs = [r for r in range(300) if r % 10 == 9][:a.noise_runs]
    windows = [10, 40, 70, 100, 115]
    print(f"noise floor: {len(test_runs)} runs x {len(windows)} windows x {a.noise_k} seeds", flush=True)
    per_run = [noise_floor(r, windows, k=a.noise_k) for r in test_runs]
    nf = {t: float(np.nanmean([p[t] for p in per_run])) for t in KEY_TARGETS}

    results, md = {"noise_floor": nf, "sets": {}}, [f"# One-step evaluation — `{a.model}`", "",
        "Mean and median absolute error in real units (ms, batches/s, core-ms per window, °C). Lower is "
        "better. A mean far above the median means a few huge misses dominate.",
        "",
        "*Noise floor: the error left even when the simulator's full hidden state is known (queued requests, "
        "burst schedule), measured on 50 adaptive test states by re-simulating each window with 20 seeds. "
        "A model that sees only telemetry can't reach it; it is a lower bound, not a target.", ""]
    for name, d in SETS.items():
        recs = load_records([d], split="test")
        print(f"{name}: {len(recs)} test transitions", flush=True)
        r = evaluate_set(model, recs)
        results["sets"][name] = r
        md += [f"## {name} ({r['n']} transitions)", ""] + _table(r, "all", nf) + [""]
        md += [f"Overloaded states only (A queue > {OVERLOAD_QUEUE_MS:.0f} ms): {r['n_overloaded']} transitions", ""]
        md += (_table(r, "overloaded", None) if r["n_overloaded"] else ["(none)"]) + [""]
    (out / "results.json").write_text(json.dumps(results, indent=1))
    (out / "report.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
