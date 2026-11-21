"""One driver for synthetic and measured experiments; writes a resumable run directory."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn

from .data import TABLES, load_trace, save_trace, validate, write_json
from .detector import Baseline, Monitor, windows
from .evaluate import evaluate, service_stats
from .study import design, simulate


def suite(args, cfg, cluster=None):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trials = design(args.reps, args.duration, args.workers, args.backend, args.seed)
    plan = dict(
        backend=args.backend,
        workers=args.workers,
        duration_s=args.duration,
        settings=asdict(cfg),
        profile=cluster.profile if cluster else {"engine": "DAT simulator v1"},
        rate=args.rate,
        seed=args.seed,
        trials=[asdict(t) for t in trials],
        software=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            pandas=pd.__version__,
            sklearn=sklearn.__version__,
        ),
    )
    plan_path = out / "study.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError(
            "Different study configuration in output directory; choose a new --out"
        )
    if not plan_path.exists():
        write_json(plan_path, plan)
    split = {s: [] for s in ["fit", "scale", "threshold"]}
    decisions = []
    details = []
    service = []
    baseline = None
    for i, t in enumerate(trials):
        path = out / "runs" / t.rid
        if t.split in ("test", "control") and baseline is None:
            baseline = Baseline.train(split, args.workers, cfg)
            joblib.dump(baseline, out / "baseline.joblib.tmp", compress=3)
            (out / "baseline.joblib.tmp").replace(out / "baseline.joblib")
            digest = hashlib.sha256((out / "baseline.joblib").read_bytes()).hexdigest()
            write_json(
                out / "calibration.json",
                dict(
                    baseline_sha256=digest,
                    run_ids=baseline.run_ids,
                    thresholds={
                        m: (float(x) if np.isfinite(x) else None)
                        for m, x in baseline.thresholds.items()
                    },
                    demand_threshold=baseline.demand_threshold,
                    components=baseline.topology,
                ),
            )
            split.clear()
        if (path / "complete.json").exists() and all(
            (path / (name + ".csv.gz")).is_file() for name in TABLES
        ):
            meta = json.loads((path / "complete.json").read_text())
            view = load_trace(path, t.rid)
        else:
            path.mkdir(parents=True, exist_ok=True)
            (path / "complete.json").unlink(missing_ok=True)
            if cluster is None:
                view, meta = simulate(t, args.duration)
            else:
                from .collect import collect

                view, meta = collect(
                    cluster, t, args.duration, args.rate, path, cfg, baseline=baseline
                )
            validate(view, t.rid)
            save_trace(path, view)
            write_json(path / "complete.json", meta)
        if t.split in split:
            split[t.split].append((t.rid, t.regime == "normal", args.duration, view))
        else:
            monitor = Monitor(baseline)
            local = []
            for tick, raw in windows(
                view, args.duration, baseline.model, baseline.topology, cfg
            ):
                ds, cs = monitor.step(t.rid, tick, raw)
                decisions.extend(ds)
                local.extend(ds)
                details.extend(cs)
            if cluster and (path / "live.jsonl").exists():
                from .online import verify_live

                verify_live(path / "live.jsonl", local)
            service.append(service_stats(view, meta, cfg, args.duration))
        print(f"[{i + 1}/{len(trials)}] {t.rid}", flush=True)
    df = pd.DataFrame(decisions)
    df["ranked"] = df.ranked.map(json.dumps)
    df.to_csv(out / "decisions.csv.gz", index=False)
    pd.DataFrame(details).to_csv(out / "component_scores.csv.gz", index=False)
    if args.backend == "vllm-http":
        check_generation_pairs(out, trials)
    result = evaluate(out, trials, decisions, service, cfg, args.duration)
    print(
        result[
            ["method", "incidents", "top1", "top3", "false_episodes", "healthy_hours"]
        ].to_string(index=False)
    )


def check_generation_pairs(out, trials):
    import gzip

    rows = []

    def outputs(rid):
        with gzip.open(out / "runs" / rid / "generations.jsonl.gz", "rt") as stream:
            return {
                r["request_id"]: (r["prompt_sha256"], r["output_token_ids"])
                for r in map(json.loads, stream)
            }

    for trial in trials:
        if trial.target is None:
            continue
        fault = outputs(trial.rid)
        control = outputs(trial.pair)
        common = fault.keys() & control.keys()
        differences = sum(fault[k] != control[k] for k in common)
        rows.append(
            dict(
                fault=trial.rid,
                control=trial.pair,
                common_completed=len(common),
                only_fault=len(fault) - len(common),
                only_control=len(control) - len(common),
                mismatches=differences,
            )
        )
    pd.DataFrame(rows).to_csv(out / "generation_pairs.csv", index=False)
    if any(r["mismatches"] for r in rows):
        raise RuntimeError(
            "Paired inference outputs differ; inspect generation_pairs.csv"
        )
