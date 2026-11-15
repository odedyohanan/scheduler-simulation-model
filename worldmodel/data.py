"""Load transition records (ours and priority_scheduler_model's) from directories of run_*.jsonl.gz."""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path


def split_of(run: int) -> str:
    """Whole runs go to one split: run id ending in 8 -> val, 9 -> test, otherwise train."""
    return {8: "val", 9: "test"}.get(run % 10, "train")


def load_records(dirs: list[str], split: str | None = None) -> list[dict]:
    out = []
    for d in dirs:
        for path in sorted(Path(d).glob("run_*.jsonl.gz")):
            with gzip.open(path, "rt") as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("header"):          # teacher_v2 files start with a scenario header
                        continue
                    if split is None or split_of(r["run"]) == split:
                        r["source"] = str(d)   # sets reuse run ids, so a run is (source, run)
                        out.append(r)
    return out


def run_key(r: dict) -> tuple[str, int]:
    """A run is identified by its source directory and run id (adaptive and MPC both have run 5)."""
    return (r.get("source", ""), r["run"])


def group_by_run(records: list[dict]) -> dict[tuple[str, int], list[dict]]:
    g: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in records:
        g[run_key(r)].append(r)
    return {k: sorted(v, key=lambda r: r["t"]) for k, v in sorted(g.items())}
