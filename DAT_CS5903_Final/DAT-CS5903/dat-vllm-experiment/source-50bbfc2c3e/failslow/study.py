"""Predeclared whole-run partitions and paired interventions."""

from dataclasses import dataclass

from .data import CLOCK
from .simulator import simulate_run


@dataclass(frozen=True)
class Trial:
    rid: str
    split: str
    seed: int
    case: str
    regime: str = "normal"
    shift: float | None = None
    target: str | None = None
    strength: float = 1.0
    onset: float = 75.0
    end: float = 145.0
    sparse: bool = False
    missing: bool = False
    pair: str | None = None


def design(repetitions, duration, workers, backend, seed=41201):
    onset = duration * 75 / 180
    end = duration * 145 / 180
    shift = duration / 3
    cases = [
        ("prefill", "normal", "p1", False, False),
        ("decode", "normal", "d1", False, False),
        ("transfer", "normal", "l11", False, False),
        ("mixed", "long", "d1", False, False),
        ("missing_path", "normal", "l11", False, True),
    ]
    if workers == 2:
        cases.append(("sparse_worker", "normal", "p2", True, False))
    trials = []
    rg = (
        ["normal", "long", "burst", "cache"]
        if backend == "sim"
        else ["normal", "long", "burst", "long_output"]
    )
    for k, split in enumerate(["fit", "scale", "threshold"]):
        for j, regime in enumerate(rg):
            trials.append(
                Trial(f"{split}-{j}", split, seed + 100 * k + j, "healthy", regime)
            )
    for rep in range(repetitions):
        for j, regime in enumerate(rg):
            trials.append(
                Trial(
                    f"test-h{rep}-{j}",
                    "test",
                    seed + 1000 + 100 * rep + j,
                    "healthy" if j == 0 else "workload",
                    regime,
                    None if j == 0 else shift,
                )
            )
        for j, (label, regime, target, sparse, missing) in enumerate(cases):
            # Different pairs have independent seeds; within each pair the offered trace is identical.
            for k, strength in enumerate([1.5, 3.0]):
                prefix = f"test-{rep}-{label}-{k}"
                common = dict(
                    seed=seed + 5000 + 100 * rep + 2 * j + k,
                    regime=regime,
                    shift=shift if regime != "normal" else None,
                    onset=onset,
                    end=end,
                    sparse=sparse,
                    missing=missing,
                )
                fault = Trial(
                    prefix,
                    "test",
                    case=label,
                    target=target,
                    strength=strength,
                    pair=prefix + "-control",
                    **common,
                )
                control = Trial(
                    prefix + "-control",
                    "control",
                    case="paired_control",
                    pair=prefix,
                    **common,
                )
                trials.extend(
                    [fault, control] if (rep + j + k) % 2 == 0 else [control, fault]
                )
    return trials


def simulate(trial, duration):
    r = simulate_run(
        trial.rid,
        trial.seed,
        duration,
        regime=trial.regime,
        fault_component=trial.target,
        fault_start_s=trial.onset,
        fault_end_s=trial.end,
        slowdown=trial.strength,
        workload_shift_s=trial.shift,
        sparse=trial.sparse,
        missing_transfer=trial.missing,
    )
    view = {
        key: r[key].assign(available_s=r[key][clock]) for key, clock in CLOCK.items()
    }
    return view, r["metadata"]
