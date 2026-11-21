"""Verify the archived SageMaker study. Run from any directory with Python 3."""
import csv
import gzip
import hashlib
import json
from collections import Counter
from math import isclose
from pathlib import Path
from statistics import median

STUDY = (Path(__file__).resolve().parents[1] / "DAT_CS5903_Final/DAT-CS5903"
         / "dat-vllm-experiment/study-2x2-r1-t90-q2.6")
METHODS = {"conditioned", "historical", "work_rate", "matched_peer"}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(name):
    with (STUDY / name).open() as f:
        return list(csv.DictReader(f))


def unique(rows, key):
    result = {}
    for row in rows:
        identity = key(row)
        check(identity not in result, f"Duplicate key: {identity}")
        result[identity] = row
    return result


def generations(run):
    with gzip.open(STUDY / "runs" / run / "generations.jsonl.gz", "rt") as f:
        indexed = unique(map(json.loads, f), lambda r: r["request_id"])
    return {key: (r["prompt_sha256"], r["output_token_ids"]) for key, r in indexed.items()}


def verify_generations(trials, complete, faults):
    pairs = unique(read_csv("generation_pairs.csv"), lambda row: row["fault"])
    check(pairs.keys() == faults, "Fault pair inventory")
    total = 0
    for fault, pair in pairs.items():
        control = pair["control"]
        check(trials[fault]["pair"] == control and trials[control]["pair"] == fault, fault)
        check(trials[fault]["seed"] == trials[control]["seed"], fault)
        treated = generations(fault)
        untreated = generations(control)
        check(treated == untreated, f"Unpaired or mismatched outputs: {fault}")
        check(len(treated) == int(pair["common_completed"])
              == complete[fault]["n_completed_requests"]
              == complete[control]["n_completed_requests"], fault)
        check(all(int(pair[k]) == 0 for k in ("only_fault", "only_control", "mismatches")), fault)
        total += len(treated)
    return total


def decision_key(row):
    return row["run_id"], float(row["t_s"]), row["method"]


def verify_decisions(held_out):
    with gzip.open(STUDY / "decisions.csv.gz", "rt") as f:
        offline = unique(csv.DictReader(f), decision_key)
    live_rows = []
    for run in sorted(held_out):
        with (STUDY / "runs" / run / "live.jsonl").open() as f:
            for row in map(json.loads, f):
                check(row["run_id"] == run, f"Unexpected live run ID in {run}")
                live_rows.append(row)
    live = unique(live_rows, decision_key)
    expected = {(run, tick, method) for run in held_out
                for tick in range(30, 91, 10) for method in METHODS}
    check(live.keys() == offline.keys() == expected, "Live/offline decision inventory")
    for key, row in live.items():
        reference = offline[key]
        check(row["ranked"] == json.loads(reference["ranked"])
              and row["explanation"] == reference["explanation"], f"Decision mismatch: {key}")
    return len(live)


def verify_methods(faults, healthy):
    incidents = read_csv("incidents.csv")
    controls = read_csv("healthy_runs.csv")
    summaries = unique(read_csv("summary.csv"), lambda r: r["method"])
    check(summaries.keys() == METHODS, "Method inventory")
    for method, summary in summaries.items():
        cases = unique((r for r in incidents if r["method"] == method), lambda r: r["run_id"])
        no_fault = unique((r for r in controls if r["method"] == method), lambda r: r["run_id"])
        check(cases.keys() == faults and no_fault.keys() == healthy, method)
        hits = sum(json.loads(r["first_ranking"])[:1] == [r["target"]] for r in cases.values())
        abstain = sum(not json.loads(r["first_ranking"]) for r in cases.values())
        wrong = len(cases) - hits - abstain
        episodes = sum(int(r["episodes"]) for r in no_fault.values())
        exposure = sum(float(r["hours"]) for r in no_fault.values())
        delays = [float(r["first_correct_top1_delay_s"]) for r in cases.values()
                  if r["first_correct_top1_delay_s"]]
        observed = (hits, abstain, wrong, episodes, len(delays))
        fields = ("top1", "misses", "wrong_first", "false_episodes", "correct_delays_observed")
        check(observed == tuple(int(summary[k]) for k in fields), method)
        check(isclose(exposure, float(summary["healthy_hours"]))
              and isclose(episodes / exposure, float(summary["episodes_per_hour"]))
              and isclose(median(delays), float(summary["median_correct_top1_delay_s"])), method)
        print(f"{method}: {hits}/{len(cases)} first correct, {abstain} abstentions, "
              f"{episodes} false episodes in {60*exposure:.0f} min; "
              f"median correct delay {median(delays):.1f} s")


def main():
    plan = json.loads((STUDY / "study.json").read_text())
    trials = unique(plan["trials"], lambda r: r["rid"])
    splits = Counter(r["split"] for r in trials.values())
    check(splits == {"fit": 4, "scale": 4, "threshold": 4, "test": 16, "control": 12},
          "Unexpected trial split")
    complete = {p.parent.name: json.loads(p.read_text()) for p in (STUDY / "runs").glob("*/complete.json")}
    check(complete.keys() == trials.keys(), "Trial completion inventory")
    calibration = json.loads((STUDY / "calibration.json").read_text())
    fitted = {rid for rid, trial in trials.items() if trial["split"] in {"fit", "scale", "threshold"}}
    check(set(calibration["run_ids"]) == fitted, "Calibration membership")
    digest = hashlib.sha256((STUDY / "baseline.joblib").read_bytes()).hexdigest()
    check(digest == calibration["baseline_sha256"], "Baseline digest")
    held_out = trials.keys() - fitted
    faults = {rid for rid, trial in trials.items() if trial["target"]}
    check(len(faults) == 12 and faults <= held_out, "Held-out fault inventory")
    for rid, run in complete.items():
        check(run["run_id"] == rid and run["backend"] == "vllm-http" and run["duration_s"] == 90, rid)
        check((run["applied_stage_operations"] > 0) == (rid in faults), rid)
        check(run["accepted"] == run["n_completed_requests"] + run["n_right_censored_requests"], rid)
    common = verify_generations(trials, complete, faults)
    decisions = verify_decisions(held_out)
    verify_methods(faults, held_out - faults)
    print(f"Verified {len(complete)} trials, {common} matched generations and "
          f"{decisions} live/offline rankings and explanations.")
    for field in ("offered", "accepted", "n_completed_requests", "n_right_censored_requests", "verified_loads"):
        print(field, sum(r[field] for r in complete.values()))


if __name__ == "__main__":
    main()
