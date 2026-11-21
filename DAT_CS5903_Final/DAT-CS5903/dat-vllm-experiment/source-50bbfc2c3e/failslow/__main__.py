import argparse
from dataclasses import replace
from pathlib import Path
from .model import CFG


def main():
    p = argparse.ArgumentParser(description="DAT passive fail-slow diagnosis")
    p.add_argument("command", choices=["suite", "pilot", "overhead", "replay"])
    p.add_argument(
        "--backend", choices=["sim", "numpy", "vllm", "vllm-http"], default="sim"
    )
    p.add_argument(
        "--workers", type=int, choices=[1, 2], default=2, help="Workers per stage"
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument(
        "--with-overhead",
        action="store_true",
        help="Run paired observer on/off trials after a measured suite",
    )
    p.add_argument("--seed", type=int, default=41201)
    p.add_argument(
        "--quick",
        action="store_true",
        help="Short integration suite: 4 s runs, 1 s windows, 40 req/s by default",
    )
    p.add_argument(
        "--rate", type=float, default=None, help="Offered requests/s for measured runs"
    )
    p.add_argument("--ttft-ms", type=float, default=300)
    p.add_argument("--itl-ms", type=float, default=100)
    p.add_argument("--spool", type=Path, help="Temporary local KV transfer filesystem")
    p.add_argument(
        "--baseline",
        type=Path,
        help="Trusted baseline.joblib from a matching healthy calibration",
    )
    p.add_argument("--trace", type=Path, help="A completed run directory")
    a = p.parse_args()
    if a.with_overhead and (a.backend == "sim" or a.command != "suite"):
        p.error("--with-overhead requires a measured suite")
    if a.quick and a.backend == "sim":
        p.error("--quick is a measured-harness integration profile")
    a.duration = a.duration if a.duration is not None else (4.0 if a.quick else 180.0)
    a.rate = a.rate if a.rate is not None else (40.0 if a.quick else 2.6)
    if min(a.duration, a.rate, a.ttft_ms, a.itl_ms) <= 0 or a.reps < 1:
        p.error("Duration, rate, SLOs and repetitions must be positive")
    cfg = replace(
        CFG, duration_s=a.duration, ttft_slo_ms=a.ttft_ms, itl_slo_ms=a.itl_ms
    )
    if a.quick:
        cfg = replace(
            cfg,
            window_s=1.0,
            cadence_s=0.25,
            min_requests=2,
            min_slo_requests=3,
            min_itl_gaps=5,
        )
    from contextlib import ExitStack
    import fcntl

    a.out.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock = stack.enter_context((a.out / ".run.lock").open("a"))
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error("Another experiment is writing this directory")
        execute(a, cfg, p)


def execute(a, cfg, p):
    if a.command == "replay":
        from .online import replay

        if a.trace is None or a.baseline is None:
            p.error("replay needs --trace and --baseline")
        replay(a.trace, a.baseline, a.out)
        return
    if a.command == "suite" and a.duration < cfg.window_s + 9 * cfg.cadence_s:
        p.error("Suite duration must provide at least ten evaluation windows")
    if a.backend == "sim":
        if a.command != "suite" or a.workers != 2:
            p.error(
                "The reference simulator implements suite with two workers per stage"
            )
        from .pipeline import suite

        suite(a, cfg)
        return
    from .collect import Cluster, experiment

    with Cluster(a.backend, a.workers, a.spool, a.out / "server-logs") as cluster:
        experiment(cluster, a, cfg)


if __name__ == "__main__":
    main()
