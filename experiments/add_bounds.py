"""Copy a trained model and add the training range of every state number, for bounded rollouts.

  .venv/bin/python experiments/add_bounds.py --model models/mlp_mix.pt --out models/mlp_mix_bounded.pt
The weights are unchanged; the bounds come from the train split of the model's own training data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worldmodel.bounds import state_bounds  # noqa: E402
from worldmodel.data import load_records  # noqa: E402
from worldmodel.model import WorldModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model = WorldModel.load(a.model)
    dirs = model.meta["args"]["data"].split(",")
    lo, hi = state_bounds(load_records(dirs, split="train"))
    model.meta.update({"state_min": lo.tolist(), "state_max": hi.tolist(), "bounds_from": dirs})
    model.save(a.out)
    print(f"bounds from {len(dirs)} data sets -> {a.out}")


if __name__ == "__main__":
    main()
