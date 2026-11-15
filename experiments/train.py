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
from worldmodel.features import INPUT_NAMES, STATE_NAMES  # noqa: E402
from worldmodel.history import build_history_arrays  # noqa: E402
from worldmodel.model import WorldModel  # noqa: E402
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
    ap.add_argument("--arch", choices=["mlp", "gru"], default="mlp")
    ap.add_argument("--history", type=int, default=1, help="windows of history the model sees")
    a = ap.parse_args()
    dirs = a.data.split(",")
    t0 = time.time()
    shaper = WorldModel(None, None, None, {"arch": a.arch, "history": a.history})

    def arrays(split):
        H, S, Y, M, _ = build_history_arrays(load_records(dirs, split), a.history)
        return shaper.prepare(H), Y, M

    tr, va = arrays("train"), arrays("val")
    print(f"{a.arch} history {a.history}: train {len(tr[0])} / val {len(va[0])} transitions, input "
          f"{tr[0].shape[1:]} ({len(INPUT_NAMES)} per window) -> {len(STATE_NAMES)} outputs "
          f"({time.time() - t0:.0f}s to load)", flush=True)

    def log(h):
        if h["epoch"] % 10 == 0:
            print(f"epoch {h['epoch']:4d}  train {h['train_loss']:.4f}  val {h['val_loss']:.4f}", flush=True)

    model, hist = train_model(tr, va, epochs=a.epochs, lr=a.lr, batch=a.batch, patience=a.patience,
                              seed=a.seed, hidden=a.hidden, layers=a.layers, log=log,
                              arch=a.arch, history=a.history)
    model.meta.update({"args": vars(a), "state_names": STATE_NAMES, "input_names": INPUT_NAMES,
                       "n_train": len(tr[0]), "n_val": len(va[0])})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(a.out)
    Path(a.out).with_suffix(".history.json").write_text(json.dumps(hist))
    print(f"best val loss {model.meta['best_val_loss']:.4f} after {len(hist)} epochs -> {a.out}")


if __name__ == "__main__":
    main()
