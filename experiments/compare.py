"""Compare several world models from their saved evaluation results.

  .venv/bin/python experiments/compare.py \
      --models mlp_v1:results/onestep:results/rollout,gru_h8:results/onestep_gru_h8:results/rollout_gru_h8 \
      --out results/compare_step1
Each entry is name:onestep_dir:rollout_dir (outputs of evaluate.py and rollout.py).
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
from worldmodel.features import KEY_TARGETS  # noqa: E402

SETS = ["adaptive", "mpc", "dagger_r1", "dagger_r2"]
PLOT_TARGETS = ["a.p99_ms", "b.rate", "c.work_core_ms", "gpu_temp_0"]
CAP = 1e6


def _fmt(v: float) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:.3g}"


def onestep_table(results: dict[str, dict], set_name: str) -> list[str]:
    """One row per key target; each cell 'mean / median' error; the best median is bolded."""
    names = list(results)
    lines = ["| target | " + " | ".join(names) + " |", "|" + "---|" * (len(names) + 1)]
    for t in KEY_TARGETS:
        cells = [results[n]["sets"][set_name]["all"]["mlp"][t] for n in names]
        meds = [c["medae"] if np.isfinite(c["medae"]) else np.inf for c in cells]
        best = int(np.argmin(meds))
        out = []
        for i, c in enumerate(cells):
            s = f"{_fmt(c['mae'])} / {_fmt(c['medae'])}"
            out.append(f"**{s}**" if i == best else s)
        lines.append(f"| {t} | " + " | ".join(out) + " |")
    return lines


def rollout_table(curves: dict[str, dict], set_name: str, steps=(1, 5, 10, 20)) -> list[str]:
    names = list(curves)
    lines = ["| target | step | " + " | ".join(names) + " | persistence |", "|" + "---|" * (len(names) + 3)]
    for t in PLOT_TARGETS:
        for h in steps:
            vals = [curves[n][set_name]["mlp"][t][h - 1] for n in names]
            pers = curves[names[0]][set_name]["persistence"][t][h - 1]
            lines.append(f"| {t} | {h} | " + " | ".join(_fmt(v) for v in vals) + f" | {_fmt(pers)} |")
    return lines


def plot(curves: dict[str, dict], path: Path):
    sets = ["adaptive", "mpc"]
    fig, axes = plt.subplots(len(sets), len(PLOT_TARGETS), figsize=(16, 7))
    for r, s in enumerate(sets):
        for c, t in enumerate(PLOT_TARGETS):
            ax = axes[r, c]
            for n, cv in curves.items():
                y = np.minimum(np.nan_to_num(np.array(cv[s]["mlp"][t], float), posinf=CAP), CAP)
                ax.plot(np.arange(1, len(y) + 1), y, label=n)
            p = next(iter(curves.values()))[s]["persistence"][t]
            ax.plot(np.arange(1, len(p) + 1), p, "k--", alpha=0.5, label="persistence")
            ax.set_yscale("log")
            ax.set_title(f"{s}: {t}")
            ax.set_xlabel("windows ahead (×0.5 s)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Rollout error (MAE) vs horizon — errors capped at {CAP:.0e} for display")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    results, curves = {}, {}
    for entry in a.models.split(","):
        name, one, roll = entry.split(":")
        results[name] = json.loads((Path(one) / "results.json").read_text())
        curves[name] = json.loads((Path(roll) / "curves.json").read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md = ["# Model comparison", "", "One step ahead: absolute error **mean / median** in real units. "
          "Bold = best median.", ""]
    for s in SETS:
        md += [f"## {s}", ""] + onestep_table(results, s) + [""]
    md += ["# Rollouts", "", "MAE when the model feeds on its own predictions (windows of 0.5 s).", ""]
    for s in SETS:
        md += [f"## {s}", ""] + rollout_table(curves, s) + [""]
    plot(curves, out / "error_vs_horizon.png")
    (out / "report.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
