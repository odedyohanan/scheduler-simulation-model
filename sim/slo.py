"""SLO accounting.

v1 (historical, used for every result up to R5): a window violates the SLO when the
p99 of requests *completed in that window* exceeds the SLO. Unfinished requests are
invisible until they complete, a window with no completions reports p99 = 0, and
requests still queued at the end of a run are never counted.

v2 (censoring-aware, opt-in): every request is resolved exactly once, either when it
completes, or as soon as its age passes the SLO while still queued ("late"). A
request's age is a lower bound on its latency, so an overdue queued request is late
no matter when it finishes; requests that are queued but not yet overdue are
undecided and are not used. Per window:

    resolved = completed_ontime + late_new
    violation_v2 = late_new > 0.01 * resolved                 if resolved > 0
                 = queue_overdue > 0                          if resolved == 0

i.e. "more than 1 % of the requests decided in this window were late".

v2 is a separate late-request metric, not an exact equivalent of v1's p99 test.
Even when every request completes within the window, "late share > 1 %" and
"interpolated p99 > SLO" can disagree: p99 is interpolated between neighbouring
completion latencies (so it can exceed the SLO with exactly 1 % or slightly fewer
requests late, or not exceed it depending on interpolation and sample size), and
v2 groups requests by when they are *decided*, not by when they complete. The two
agree closely in practice (SLO v2 − v1 was about +0.1 pp on development
scenarios), but they are reported as distinct metrics. Ages are never mixed into
the completed-latency percentiles.
"""
from __future__ import annotations

SLO_VERSIONS = ("v1", "v2")


def window_violation(tel: dict, slo_ms: float, version: str = "v1") -> bool:
    a = tel["a"]
    if version == "v1":
        return a["p99_ms"] > slo_ms
    resolved = a["completed_ontime"] + a["late_new"]
    if resolved == 0:
        return a["queue_overdue"] > 0
    return a["late_new"] > 0.01 * resolved


def late_fraction(tel: dict) -> float:
    """Late share of the requests decided in this window (0 when none were)."""
    a = tel["a"]
    resolved = a["completed_ontime"] + a["late_new"]
    return a["late_new"] / resolved if resolved else float(a["queue_overdue"] > 0)
