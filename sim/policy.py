"""The policy: what the controller hands to the system, valid for a short window.

Semantics (what the simulator enforces):
  preferred_cpus  The CPUs the workload may run on (a cpuset). Empty = any CPU.
  cpu_share       Weight when workloads compete for the same CPU (like cgroup cpu.weight).
                  Unused capacity is always given to whoever wants it (work-conserving).
  latency_class   Wake-up priority. Affects tail latency when a CPU is shared, not throughput.
  gpus            GPUs owned exclusively by the workload.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .config import NAMES

LATENCY_CLASSES = ["critical", "interactive", "batch", "best_effort"]


@dataclass
class WorkloadPolicy:
    preferred_cpus: list[int] = field(default_factory=list)
    cpu_share: int = 33               # percent
    latency_class: str = "batch"
    gpus: list[int] = field(default_factory=list)


@dataclass
class Policy:
    version: int
    valid_for_ms: int
    workloads: list[WorkloadPolicy]
    # Optional: what the controller expects to happen. Not used by the system.
    predicted: dict | None = None

    def eligibility(self, n_cpus: int) -> np.ndarray:
        """(n_workloads x n_cpus) boolean matrix: may workload w run on CPU k?"""
        e = np.zeros((len(self.workloads), n_cpus), dtype=bool)
        for w, wp in enumerate(self.workloads):
            if wp.preferred_cpus:
                e[w, wp.preferred_cpus] = True
            else:
                e[w, :] = True
        return e

    def shares(self) -> np.ndarray:
        return np.array([max(wp.cpu_share, 1) for wp in self.workloads], dtype=float)

    def gpu_owner(self, n_gpus: int) -> list[int | None]:
        owner: list[int | None] = [None] * n_gpus
        for w, wp in enumerate(self.workloads):
            for g in wp.gpus:
                owner[g] = w
        return owner

    def render(self) -> str:
        """Human-readable form, matching the format sketched in the conversation."""
        lines = [f"Policy version: {self.version}", f"Valid for: {self.valid_for_ms} ms", ""]
        for name, wp in zip(NAMES, self.workloads):
            lines.append(f"Workload {name}:")
            if wp.preferred_cpus:
                lines.append(f"    preferred_cpus: {wp.preferred_cpus}")
            lines.append(f"    cpu_share: {wp.cpu_share}%")
            lines.append(f"    latency_class: {wp.latency_class}")
            if wp.gpus:
                lines.append(f"    gpus: {wp.gpus}")
            lines.append("")
        if self.predicted:
            lines.append("Predicted (by the controller, for this window):")
            for k, v in self.predicted.items():
                lines.append(f"    {k}: {v}")
        return "\n".join(lines).rstrip()

    def to_json(self) -> str:
        """Machine-readable form, e.g. for a cgroup / BPF-map applier."""
        return json.dumps({
            "version": self.version,
            "valid_for_ms": self.valid_for_ms,
            "workloads": {
                name: {
                    "preferred_cpus": wp.preferred_cpus,
                    "cpu_share": wp.cpu_share,
                    "latency_class": wp.latency_class,
                    "gpus": wp.gpus,
                } for name, wp in zip(NAMES, self.workloads)
            },
        })


def policy_from_dict(d: dict, version: int = 0, valid_for_ms: int = 500) -> Policy:
    """Inverse of Policy.to_json() (also accepts a bare {"workloads": ...})."""
    wps = [WorkloadPolicy(preferred_cpus=list(w["preferred_cpus"]), cpu_share=int(w["cpu_share"]),
                          latency_class=w["latency_class"], gpus=list(w["gpus"]))
           for w in (d["workloads"][n] for n in NAMES)]
    return Policy(version=d.get("version", version), valid_for_ms=d.get("valid_for_ms", valid_for_ms),
                  workloads=wps)
