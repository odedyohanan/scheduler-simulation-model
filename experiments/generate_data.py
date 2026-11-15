"""Generate world-model training data: one gzipped JSONL file per run.

  .venv/bin/python experiments/generate_data.py --runs 300 --procs 5 --out data/adaptive_v1
Finished runs are skipped, so a stopped job can be restarted with the same command.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worldmodel.generate import run_episode  # noqa: E402


def _one(job):
    run_id, out, duration, explore = job
    path = Path(out) / f"run_{run_id:05d}.jsonl.gz"
    if path.exists():
        return run_id, 0.0
    t0 = time.time()
    recs = run_episode(run_id, duration, explore)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    tmp.rename(path)
    return run_id, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--explore", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "manifest.json").write_text(json.dumps({**vars(a), "driver": "adaptive"}, indent=1))
    jobs = [(r, a.out, a.duration, a.explore) for r in range(a.start, a.start + a.runs)]
    with Pool(a.procs) as pool:
        for i, (run_id, sec) in enumerate(pool.imap_unordered(_one, jobs), 1):
            print(f"[{i}/{len(jobs)}] run {run_id} {sec:.1f}s", flush=True)


if __name__ == "__main__":
    main()
