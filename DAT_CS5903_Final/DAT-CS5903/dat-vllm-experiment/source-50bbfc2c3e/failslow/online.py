"""A separate live observer tails the collector journal. Replaying is safe after restart."""

import json
import time
import resource
from pathlib import Path

import numpy as np
import joblib
import pandas as pd

from .data import CLOCK, TABLES, frames, load_trace
from .detector import Monitor, raw_window, windows


def tail(baseline, rid, journal, destination):
    # Fresh output on restart: reconstruct all streaks from this run's durable journal.
    monitor = Monitor(baseline)
    buffer = {k: [] for k in TABLES}
    width = baseline.cfg.window_s
    with open(journal) as stream, open(destination, "w", buffering=1) as sink:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line.endswith("\n"):
                stream.seek(offset)
                time.sleep(0.05)
                continue
            kind, row = json.loads(line)
            if kind == "end":
                usage = resource.getrusage(resource.RUSAGE_SELF)
                Path(destination).with_name("observer_stats.json").write_text(
                    json.dumps(
                        dict(
                            observer_cpu_s=usage.ru_utime + usage.ru_stime,
                            observer_peak_rss_kib=usage.ru_maxrss,
                        )
                    )
                )
                return
            if kind != "tick":
                buffer[kind].append(row)
                continue
            t = row
            for k in TABLES:
                buffer[k] = [r for r in buffer[k] if r[CLOCK[k]] > t - width]
            v = frames(
                {
                    k: [
                        r
                        for r in buffer[k]
                        if r[CLOCK[k]] <= t and r["available_s"] <= t
                    ]
                    for k in TABLES
                }
            )
            scored = baseline.model.score(v["observations"])
            ds, _ = monitor.step(
                rid,
                t,
                raw_window(
                    scored,
                    v["arrivals"],
                    v["slo_events"],
                    baseline.topology,
                    baseline.cfg,
                ),
            )
            for d in ds:
                sink.write(pd.Series(d).to_json(double_precision=14) + "\n")


def verify_live(path, reference):
    actual = [json.loads(s) for s in path.read_text().splitlines()]
    if len(actual) != len(reference):
        raise RuntimeError("Live and replay decision counts differ")
    for i, (seen, expected) in enumerate(zip(actual, reference)):
        if seen.keys() != expected.keys():
            raise RuntimeError(f"Decision {i} has different fields")
        for key, value in expected.items():
            if isinstance(value, (float, np.floating)):
                parsed = np.nan if seen[key] is None else seen[key]
                matches = np.isclose(
                    parsed, value, rtol=1.0e-10, atol=1.0e-10, equal_nan=True
                )
            else:
                matches = seen[key] == value
            if not matches:
                raise RuntimeError(f"Live/replay mismatch at decision {i}, field {key}")


def replay(trace, bundle, out):
    b = joblib.load(bundle)
    meta = json.loads((trace / "complete.json").read_text())
    rid = meta["run_id"]
    v = load_trace(trace, rid)
    out.mkdir(parents=True, exist_ok=True)
    mon = Monitor(b)
    d = []
    s = []
    for t, w in windows(v, meta["duration_s"], b.model, b.topology, b.cfg):
        decisions, scores = mon.step(rid, t, w)
        d.extend(decisions)
        s.extend(scores)
    pd.DataFrame(d).to_json(out / "replay.jsonl", orient="records", lines=True)
    pd.DataFrame(s).to_csv(out / "scores.csv", index=False)
    if (trace / "live.jsonl").exists():
        verify_live(trace / "live.jsonl", d)
