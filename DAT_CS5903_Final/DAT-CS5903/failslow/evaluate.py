"""Held-out incident, exposure, uncertainty and paired service-impact metrics."""

import json
import pandas as pd
from scipy.stats import chi2
from .model import METHODS


def evaluate(out, trials, decisions, service, cfg, horizon):
    d = pd.DataFrame(decisions)
    metrics = pd.DataFrame(service).set_index("run_id")
    metrics.to_csv(out / "service.csv")
    incident = []
    controls = []
    explanations = []
    for run in trials:
        if run.split not in ("test", "control"):
            continue
        for method in METHODS:
            g = d[(d.run_id == run.rid) & (d.method == method)].sort_values("t_s")
            active = g.ranked.map(bool)
            new = active & ~active.shift(fill_value=False)
            if run.target is None:
                controls.append(
                    dict(
                        method=method,
                        run_id=run.rid,
                        cohort=run.split,
                        episodes=int(new.sum()),
                        hours=(horizon - cfg.window_s) / 3600,
                        windows=len(g),
                        full_windows=int(g.full_coverage.sum()),
                        unknown_windows=int(g.explanation.eq("unknown").sum()),
                    )
                )
            else:
                eligible = g.t_s.between(
                    run.onset, min(horizon, run.end + cfg.window_s)
                )
                scored = g[active & eligible]
                ranking = scored.iloc[0].ranked if len(scored) else []
                delays = g.loc[eligible & new, "t_s"] - run.onset
                correct = (
                    g.loc[
                        eligible
                        & g.ranked.map(lambda x: bool(x) and x[0] == run.target),
                        "t_s",
                    ]
                    - run.onset
                )
                prior = g[g.t_s < run.onset]
                incident.append(
                    dict(
                        method=method,
                        run_id=run.rid,
                        case=run.case,
                        strength=run.strength,
                        target=run.target,
                        first_ranking=json.dumps(ranking),
                        ranked=bool(ranking),
                        top1=bool(ranking and ranking[0] == run.target),
                        top3=run.target in ranking,
                        new_episode_delay_s=delays.min(),
                        first_correct_top1_delay_s=correct.min(),
                        alarm_active_before_onset=bool(
                            len(prior) and prior.iloc[-1].ranked
                        ),
                        slo_breach_during=g.loc[
                            g.t_s.between(run.onset, run.end), "slo_alarm"
                        ].any(),
                        unknown_fraction=g.loc[eligible, "explanation"]
                        .eq("unknown")
                        .mean(),
                    )
                )
            if run.split == "test" and (run.target or run.shift is not None):
                onset = run.onset if run.target else run.shift
                end = run.end if run.target else horizon
                truth = (
                    ("mixed" if run.shift is not None else "degradation")
                    if run.target
                    else "workload"
                )
                stable = g[g.slo_alarm & g.t_s.between(onset + cfg.window_s, end)]
                for row in stable.itertuples():
                    explanations.append(
                        dict(
                            method=method,
                            run_id=run.rid,
                            t_s=row.t_s,
                            truth=truth,
                            explanation=row.explanation,
                            correct=truth == row.explanation,
                        )
                    )
    inc = pd.DataFrame(incident)
    inc.to_csv(out / "incidents.csv", index=False)
    h = pd.DataFrame(controls)
    h.to_csv(out / "healthy_runs.csv", index=False)
    exp = pd.DataFrame(
        explanations,
        columns=["method", "run_id", "t_s", "truth", "explanation", "correct"],
    )
    exp.to_csv(out / "explanations.csv", index=False)
    pairs = []
    for r in trials:
        if r.target:
            row = dict(
                run_id=r.rid, control=r.pair, target=r.target, strength=r.strength
            )
            for name in [
                "ttft_p99_ms",
                "itl_p99_ms",
                "completed_rps",
                "ttft_violation_fraction",
                "itl_violation_fraction",
            ]:
                row[name + "_delta"] = (
                    metrics.loc[r.rid, name] - metrics.loc[r.pair, name]
                )
            pairs.append(row)
    pd.DataFrame(pairs).to_csv(out / "paired_impact.csv", index=False)
    summary = []
    for m in METHODS:
        f = inc[inc.method == m]
        n = h[h.method == m]
        e = exp[exp.method == m]
        k = int(n.episodes.sum())
        hours = n.hours.sum()
        lo = 0 if k == 0 else chi2.ppf(0.025, 2 * k) / 2 / hours
        hi = chi2.ppf(0.975, 2 * (k + 1)) / 2 / hours
        summary.append(
            dict(
                method=m,
                incidents=len(f),
                rankings=int(f.ranked.sum()),
                top1=int(f.top1.sum()),
                top3=int(f.top3.sum()),
                misses=int((~f.ranked).sum()),
                wrong_first=int((f.ranked & ~f.top1).sum()),
                top1_fraction=f.top1.mean(),
                median_new_episode_delay_s=f.new_episode_delay_s.median(),
                median_correct_top1_delay_s=f.first_correct_top1_delay_s.median(),
                correct_delays_observed=int(f.first_correct_top1_delay_s.notna().sum()),
                healthy_hours=hours,
                false_episodes=k,
                episodes_per_hour=k / hours,
                poisson_rate_lo95=lo,
                poisson_rate_hi95=hi,
                healthy_full_coverage=n.full_windows.sum() / n.windows.sum(),
                explanation_windows=len(e),
                explanation_accuracy=e.correct.mean(),
                explanation_coverage=(
                    ~e.explanation.isin(["unknown", "no_diagnosis"])
                ).mean(),
            )
        )
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.csv", index=False)
    return summary


def service_stats(view, meta, cfg, duration):
    row = dict(
        run_id=meta["run_id"],
        arrivals=len(view["arrivals"]),
        completed=meta["n_completed_requests"],
        censored=meta["n_right_censored_requests"],
        completed_rps=meta["n_completed_requests"] / duration,
    )
    for kind, limit in [("ttft", cfg.ttft_slo_ms), ("itl", cfg.itl_slo_ms)]:
        g = view["slo_events"]
        x = g.loc[g.metric == kind + "_ms", "value_ms"]
        row[kind + "_p99_ms"] = x.quantile(0.99)
        row[kind + "_violation_fraction"] = (x > limit).mean()
    return row
