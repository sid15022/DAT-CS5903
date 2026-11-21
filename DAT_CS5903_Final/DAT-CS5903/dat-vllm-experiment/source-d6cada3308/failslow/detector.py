"""A frozen healthy baseline and a time-ordered window monitor shared by replay and live tailing."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import CLOCK
from .model import CFG, METHODS, HealthyModels


def components(workers):
    return {
        "prefill": [f"p{i + 1}" for i in range(workers)],
        "transfer": [
            f"l{i + 1}{j + 1}" for i in range(workers) for j in range(workers)
        ],
        "decode": [f"d{i + 1}" for i in range(workers)],
    }


def ticks(horizon, cfg):
    return np.arange(cfg.window_s, horizon + 1.0e-9, cfg.cadence_s)


def cut(frame, clock, t, width):
    return frame[
        (frame[clock] > t - width) & (frame[clock] <= t) & (frame.available_s <= t)
    ]


def raw_window(obs, arr, slo, topology, cfg):
    """One request gets one vote, even for a long generation."""
    scores = {}
    cols = ["conditioned", "historical", "work_rate", "median_residual"]
    good = obs[obs.supported.astype(bool)]
    if len(good):
        per_req = good.groupby(["component", "request_id"])[cols].mean()
        agg = per_req.groupby(level=0).mean()
        median = per_req.median_residual.groupby(level=0).median()
        supported = good.groupby("component").request_id.nunique()
        counts = good.component.value_counts()
    else:
        supported = counts = {}
    total = obs.component.value_counts()
    for stage, members in topology.items():
        for c in members:
            n, num, den = (
                int(supported.get(c, 0)),
                int(counts.get(c, 0)),
                int(total.get(c, 0)),
            )
            fraction = num / den if den else 0.0
            reason = (
                "no_records"
                if not den
                else "insufficient_requests"
                if n < cfg.min_requests
                else "unsupported_features"
                if fraction < cfg.min_support_fraction
                else "ok"
            )
            values = {m: np.nan for m in cols}
            if reason == "ok":
                values.update(agg.loc[c].to_dict())
                values["median_residual"] = median[c]
            scores[c] = dict(
                stage=stage, n=n, fraction=fraction, reason=reason, **values
            )
        for c in members:
            med = scores[c]["median_residual"]
            peers = [
                scores[p]["median_residual"]
                for p in members
                if p != c and scores[p]["reason"] == "ok"
            ]
            scores[c]["matched_peer"] = (
                max(0.0, med - np.median(peers))
                if peers and np.isfinite(med)
                else np.nan
            )

    demand = np.array(
        [
            len(arr) / cfg.window_s,
            arr.prompt_tokens.median(),
            arr.max_new_tokens.median(),
        ]
    )
    metrics = {}
    validity, breaches = [], []
    for metric, bound, minimum in [
        ("ttft_ms", cfg.ttft_slo_ms, cfg.min_slo_requests),
        ("itl_ms", cfg.itl_slo_ms, cfg.min_itl_gaps),
    ]:
        x = slo.loc[slo.metric == metric, "value_ms"].to_numpy()
        frac = float(np.mean(x > bound)) if len(x) else np.nan
        valid = len(x) >= minimum
        validity.append(valid)
        breaches.append(valid and frac > cfg.allowed_violation_fraction)
        metrics[metric + "_n"] = len(x)
        metrics[metric + "_violation_fraction"] = frac
        metrics[metric + "_p99"] = np.quantile(x, 0.99) if len(x) else np.nan
    metrics.update(slo_alarm=any(breaches), slo_valid=any(breaches) or all(validity))
    return scores, demand, metrics


def windows(view, horizon, model, topology, cfg):
    data = {
        **view,
        "observations": model.score(view["observations"].reset_index(drop=True)),
    }
    for t in ticks(horizon, cfg):
        items = [
            cut(data[k], CLOCK[k], t, cfg.window_s)
            for k in ("observations", "arrivals", "slo_events")
        ]
        yield t, raw_window(*items, topology, cfg)


@dataclass
class Baseline:
    model: HealthyModels
    topology: dict
    cfg: object
    reference: dict
    thresholds: dict
    center: np.ndarray
    scale: np.ndarray
    demand_threshold: float
    run_ids: list

    @classmethod
    def train(cls, splits, workers, cfg=CFG):
        # Input tuples contain run ID, normal-demand flag, horizon and observer view only.
        ids = [r[0] for part in splits.values() for r in part]
        if len(set(ids)) != len(ids) or set(splits) != {"fit", "scale", "threshold"}:
            raise ValueError("Expected three disjoint sets of healthy runs")
        topology = components(workers)
        fit = pd.concat(
            [v["observations"] for _, _, _, v in splits["fit"]], ignore_index=True
        )
        model = HealthyModels(cfg).fit(fit)
        ref = {}

        def collect(part):
            rows, demand = [], []
            for rid, normal, horizon, view in splits[part]:
                for _, (s, d, _) in windows(view, horizon, model, topology, cfg):
                    rows.append(s)
                    if normal and np.isfinite(d).all():
                        demand.append(d)
            return rows, demand

        scale_rows, _ = collect("scale")
        thresh_rows, demand_thresh = collect("threshold")
        all_ids = sum(topology.values(), [])
        for c in all_ids:
            for m in METHODS:
                sample = [s[c][m] for s in scale_rows if np.isfinite(s[c][m])]
                if len(sample) >= 10:
                    med, hi = np.quantile(sample, [0.5, 0.95])
                    ref[m, c] = med, max(hi - med, cfg.component_scale_floor)
        thresholds = {}
        for m in METHODS:
            maxima = []
            for s in thresh_rows:
                values = [
                    (s[c][m] - ref[m, c][0]) / ref[m, c][1]
                    for c in all_ids
                    if (m, c) in ref and np.isfinite(s[c][m])
                ]
                if values:
                    maxima.append(max(values))
            thresholds[m] = (
                max(0.0, float(np.quantile(maxima, cfg.threshold_quantile)))
                if len(maxima) >= 10
                else np.inf
            )
        if not np.isfinite(thresholds["conditioned"]):
            raise ValueError(
                "Insufficient healthy threshold windows; increase traffic or healthy duration"
            )
        normal = []
        for _, flag, horizon, v in splits["fit"]:
            if flag:
                for t in ticks(horizon, cfg):
                    a = cut(v["arrivals"], "arrival_s", t, cfg.window_s)
                    if len(a):
                        normal.append(
                            [
                                len(a) / cfg.window_s,
                                a.prompt_tokens.median(),
                                a.max_new_tokens.median(),
                            ]
                        )
        if not normal or not demand_thresh:
            raise ValueError("Healthy demand reference lacks normal windows")
        center = np.median(normal, axis=0)
        scale = np.maximum(
            np.quantile(abs(np.array(normal) - center), 0.95, axis=0), [0.3, 100, 2]
        )
        dt = max(
            2.0,
            float(
                np.quantile(
                    np.max(abs(np.array(demand_thresh) - center) / scale, axis=1), 0.99
                )
            ),
        )
        return cls(model, topology, cfg, ref, thresholds, center, scale, dt, ids)


class Monitor:
    def __init__(self, baseline):
        self.b = baseline
        self.streak = {}
        self.demand_streak = 0
        self.last = None

    def step(self, rid, t, raw):
        b = self.b
        if self.last is not None and not np.isclose(t - self.last, b.cfg.cadence_s):
            raise ValueError("Monitor requires contiguous ticks within one run")
        self.last = t
        scores, demand, slo = raw
        changed = (
            np.isfinite(demand).all()
            and np.max(abs(demand - b.center) / b.scale) > b.demand_threshold
        )
        self.demand_streak = self.demand_streak + 1 if changed else 0
        shift = self.demand_streak >= b.cfg.persistence
        decisions, details = [], []
        for m in METHODS:
            eligible = {}
            instant = False
            for c, s in scores.items():
                key = (m, c)
                z = np.nan
                if (
                    key in b.reference
                    and np.isfinite(b.thresholds[m])
                    and np.isfinite(s[m])
                ):
                    med, scale = b.reference[key]
                    z = (s[m] - med) / scale
                    eligible[c] = z
                hot = bool(z > b.thresholds[m])
                self.streak[key] = self.streak.get(key, 0) + 1 if hot else 0
                instant |= hot
                reason = (
                    s["reason"]
                    if s["reason"] != "ok" or np.isfinite(z)
                    else "missing_reference_or_peer"
                )
                details.append(
                    dict(
                        run_id=rid,
                        t_s=t,
                        method=m,
                        component=c,
                        score=z,
                        supported_requests=s["n"],
                        support_fraction=s["fraction"],
                        reason=reason,
                    )
                )
            ranked = sorted(
                [c for c in eligible if self.streak[m, c] >= b.cfg.persistence],
                key=lambda c: (-eligible[c], c),
            )[:3]
            full = len(eligible) == len(scores)
            reason = ""
            if ranked:
                category = "mixed" if shift else "degradation"
                if not full:
                    reason = "partial_coverage"
            elif not full or not slo["slo_valid"]:
                category = "unknown"
                reason = (
                    "insufficient_component_coverage"
                    if not full
                    else "insufficient_slo_data"
                )
            elif not slo["slo_alarm"] and not instant:
                category = "no_diagnosis"
            elif shift and not instant:
                category = "workload"
            else:
                category = "unknown"
                reason = "unresolved_slowdown"
            decisions.append(
                dict(
                    run_id=rid,
                    t_s=t,
                    method=m,
                    ranked=ranked,
                    explanation=category,
                    full_coverage=full,
                    demand_changed=shift,
                    unknown_reason=reason,
                    **slo,
                )
            )
        return decisions, details
