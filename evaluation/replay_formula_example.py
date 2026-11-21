"""Reconstruct the report's worked GPU example from archived CSV files on CPU."""
import json
import sys
from math import exp, isclose, log
from pathlib import Path
from statistics import mean

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1] / "DAT_CS5903_Final" / "DAT-CS5903"
sys.path.insert(0, str(PACKAGE))

from failslow.data import load_trace
from failslow.detector import Baseline, Monitor, cut, windows
from failslow.model import Settings

STUDY = PACKAGE / "dat-vllm-experiment" / "study-2x2-r1-t90-q2.6"
RUN = "test-0-decode-1"
COMPONENT = "d1"
METHOD = "conditioned"


def verify_close(observed, reference, label):
    if not isclose(observed, reference, rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError(f"{label}: reconstructed {observed}, archived {reference}")


def main():
    plan = json.loads((STUDY / "study.json").read_text())
    settings = Settings(**plan["settings"])
    duration = plan["duration_s"]
    splits = {name: [] for name in ("fit", "scale", "threshold")}
    for trial in plan["trials"]:
        if trial["split"] in splits:
            run_id = trial["rid"]
            view = load_trace(STUDY / "runs" / run_id, run_id)
            splits[trial["split"]].append((run_id, trial["regime"] == "normal", duration, view))

    baseline = Baseline.train(splits, plan["workers"], settings)
    threshold = baseline.thresholds[METHOD]
    calibration = json.loads((STUDY / "calibration.json").read_text())
    verify_close(threshold, calibration["thresholds"][METHOD], "method threshold")
    view = load_trace(STUDY / "runs" / RUN, RUN)
    scores = baseline.model.score(view["observations"])
    op = scores.set_index("obs_id").loc["r00081-decode-1"]
    excess = max(0, (log(op.duration_ms) - op.q95_log) / (op.q95_log - op.q50_log))
    verify_close(excess, op.conditioned, "operation score")
    print(f"{op.name}: duration {op.duration_ms:.5f} ms, median {exp(op.q50_log):.5f} ms, "
          f"upper bound {exp(op.q95_log):.5f} ms, excess {excess:.5f}")

    ref_median, ref_scale = baseline.reference[METHOD, COMPONENT]
    archived = pd.read_csv(STUDY / "component_scores.csv.gz")
    recorded = archived.loc[(archived.run_id == RUN) & (archived.method == METHOD)
                            & (archived.component == COMPONENT)].set_index("t_s")
    live = {}
    with (STUDY / "runs" / RUN / "live.jsonl").open() as records:
        for line in records:
            row = json.loads(line)
            if row["method"] == METHOD:
                live[row["t_s"]] = row["ranked"]
    print(f"Healthy window median {ref_median:.8f}, scale {ref_scale:.5f}, "
          f"method threshold {threshold:.5f}")
    monitor = Monitor(baseline)
    for tick, raw in windows(view, duration, baseline.model, baseline.topology, settings):
        decisions, _ = monitor.step(RUN, tick, raw)
        ranking = next(row["ranked"] for row in decisions if row["method"] == METHOD)
        if ranking != live[tick]:
            raise ValueError(f"Ranking differs from live decision at {tick} s")
        if tick not in (40, 50):
            continue
        window = cut(scores, "event_s", tick, settings.window_s)
        eligible = window.loc[(window.component == COMPONENT) & window.supported]
        per_request = eligible.groupby("request_id")[METHOD].mean()
        raw_score = mean(per_request)
        z = (raw_score - ref_median) / ref_scale
        verify_close(z, recorded.loc[tick, "score"], f"{COMPONENT} score at {tick} s")
        print(f"{tick:.0f} s: {len(eligible)} operations, {len(per_request)} requests, "
              f"request {op.request_id} mean {per_request[op.request_id]:.5f}, "
              f"window mean {raw_score:.6f}, normalized score {z:.5f}, ranking {ranking}")


if __name__ == "__main__":
    main()
