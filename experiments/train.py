"""Train the MLP world model.

  .venv/bin/python experiments/train.py --data data/adaptive_v1 --out models/mlp_v1.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worldmodel.data import load_records  # noqa: E402
from worldmodel.features import INPUT_NAMES, STATE_NAMES, build_arrays  # noqa: E402
from worldmodel.train import train_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="comma-separated data directories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dirs = a.data.split(",")
    t0 = time.time()
    tr, va = build_arrays(load_records(dirs, "train")), build_arrays(load_records(dirs, "val"))
    print(f"train {len(tr[0])} / val {len(va[0])} transitions, {len(INPUT_NAMES)} inputs -> "
          f"{len(STATE_NAMES)} outputs ({time.time() - t0:.0f}s to load)", flush=True)

    def log(h):
        if h["epoch"] % 10 == 0:
            print(f"epoch {h['epoch']:4d}  train {h['train_loss']:.4f}  val {h['val_loss']:.4f}", flush=True)

    model, hist = train_model((tr[0], tr[2], tr[3]), (va[0], va[2], va[3]), epochs=a.epochs, lr=a.lr,
                              batch=a.batch, patience=a.patience, seed=a.seed, hidden=a.hidden,
                              layers=a.layers, log=log)
    model.meta.update({"args": vars(a), "state_names": STATE_NAMES, "input_names": INPUT_NAMES,
                       "n_train": len(tr[0]), "n_val": len(va[0])})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(a.out)
    Path(a.out).with_suffix(".history.json").write_text(json.dumps(hist))
    print(f"best val loss {model.meta['best_val_loss']:.4f} after {len(hist)} epochs -> {a.out}")


if __name__ == "__main__":
    main()
