"""System and workload parameters for the toy machine.

Everything the simulator, predictor and optimizer need to agree on lives here,
so the three layers model the *same* machine.

Units: time in seconds unless a name ends in _ms. CPU capacity in cores
(1.0 = one core fully busy).
"""
from dataclasses import dataclass, field

# Workload indices, used everywhere as array positions.
A, B, C = 0, 1, 2
NAMES = ["A", "B", "C"]


@dataclass(frozen=True)
class Machine:
    n_cpus: int = 8
    n_gpus: int = 2
    # CPUs sharing a last-level cache. Workloads in the same domain disturb each other.
    l3_domains: tuple = ((0, 1, 2, 3), (4, 5, 6, 7))


@dataclass(frozen=True)
class WorkloadA:
    """Latency-critical request service (think: an API). CPU only."""
    base_rate: float = 2000.0       # requests/s
    wave_amp: float = 1200.0        # slow sinusoidal load swing
    wave_period: float = 40.0       # s
    burst_every: float = 15.0       # mean seconds between bursts (Poisson)
    burst_len: float = 3.0          # s
    burst_extra: float = 2500.0     # requests/s added during a burst
    service_ms: float = 1.0         # mean CPU time per request (exponential)
    slo_p99_ms: float = 10.0        # target tail latency
    interference: float = 0.25      # service inflation when others are busy in A's L3 domain


@dataclass(frozen=True)
class WorkloadB:
    """GPU training job. CPUs prepare batches, GPUs consume them."""
    cpu_ms_per_batch: float = 4.0   # core-ms of CPU work to prepare one batch
    gpu_ms_per_batch: float = 2.0   # GPU time per batch, one GPU, unthrottled
    buffer_cap: float = 16.0        # prepared batches that can wait for a GPU
    value_per_batch: float = 20.0   # objective weight (a batch is worth 20 core-ms of C's work)


@dataclass(frozen=True)
class WorkloadC:
    """Best-effort ETL. Infinite backlog; can also use a GPU."""
    gpu_core_equiv: float = 3.0     # one GPU does the work of this many cores
    gpu_power: float = 0.6          # relative GPU power draw when C uses it
    value_per_core_ms: float = 1.0  # objective weight per core-ms of work done


@dataclass(frozen=True)
class Thermal:
    """dT/dt = heat * power - cool * (T - ambient). Throttle above t_throttle."""
    ambient: float = 30.0
    heat: float = 4.0               # °C/s at full power
    cool: float = 1.0 / 15.0        # 1/s  (time constant 15 s)
    t_throttle: float = 80.0
    t_max: float = 90.0
    min_speed: float = 0.6          # speed factor at t_max


@dataclass(frozen=True)
class SchedLatency:
    """Extra wake-up delay (ms) a request sees when its CPU is shared, by latency class."""
    delay_ms: dict = field(default_factory=lambda: {
        "critical": 0.1, "interactive": 0.5, "batch": 2.0, "best_effort": 4.0})


@dataclass(frozen=True)
class Costs:
    migration_penalty_s: float = 0.05   # capacity lost after a workload's cpuset changes...
    migration_efficiency: float = 0.7   # ...running at this efficiency meanwhile


@dataclass(frozen=True)
class Spec:
    machine: Machine = field(default_factory=Machine)
    a: WorkloadA = field(default_factory=WorkloadA)
    b: WorkloadB = field(default_factory=WorkloadB)
    c: WorkloadC = field(default_factory=WorkloadC)
    thermal: Thermal = field(default_factory=Thermal)
    sched: SchedLatency = field(default_factory=SchedLatency)
    costs: Costs = field(default_factory=Costs)
    policy_period_s: float = 0.5        # a new policy every 500 ms
    dt: float = 0.001                   # simulator step: 1 ms


def throttle_factor(temp: float, th: Thermal) -> float:
    """GPU speed multiplier: 1.0 below t_throttle, falling linearly to min_speed at t_max."""
    if temp <= th.t_throttle:
        return 1.0
    x = min(1.0, (temp - th.t_throttle) / (th.t_max - th.t_throttle))
    return 1.0 - x * (1.0 - th.min_speed)
