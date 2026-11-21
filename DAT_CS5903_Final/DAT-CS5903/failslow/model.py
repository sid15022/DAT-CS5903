"""Healthy quantile and comparator models from the DAT project notebook."""

from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


@dataclass(frozen=True)
class Settings:
    seed: int = 20261009
    duration_s: float = 180.0
    window_s: float = 30.0
    cadence_s: float = 10.0
    persistence: int = 2
    min_requests: int = 5
    min_slo_requests: int = 10
    min_itl_gaps: int = 30
    min_support_fraction: float = 0.80
    scale_floor: float = 0.08
    component_scale_floor: float = 0.025
    threshold_quantile: float = 0.99
    ttft_slo_ms: float = 300.0
    itl_slo_ms: float = 100.0
    allowed_violation_fraction: float = 0.01
    model_rows_per_stage: int = 6000
    trees: int = 60


CFG = Settings()
STAGES = ("prefill", "transfer", "decode")
COMPONENTS = {
    "prefill": ("p1", "p2"),
    "transfer": ("l11", "l12", "l21", "l22"),
    "decode": ("d1", "d2"),
}
FEATURES = {
    "prefill": ["prompt_tokens", "uncached_tokens", "batch_size"],
    "transfer": ["kv_bytes"],
    "decode": ["context_tokens", "batch_size", "generated_so_far"],
}
METHODS = ("conditioned", "historical", "work_rate", "matched_peer")


class HealthyModels:
    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self.models = {}
        self.bounds = {}
        self.kinds = {}
        self.historical = {}
        self.rate_reference = {}
        self.configs = set()

    @staticmethod
    def work_units(frame, stage):
        if stage == "prefill":
            return frame.uncached_tokens.to_numpy().clip(1)
        if stage == "transfer":
            return frame.kv_bytes.to_numpy().clip(1)
        return np.ones(len(frame))  # naive decode tokens/sec baseline

    def fit(self, observations):
        if observations.config_id.isna().any() or observations.config_id.nunique() != 1:
            raise ValueError("Fit one nonnull configuration per healthy baseline")
        self.configs = set(observations.config_id.unique())
        for stage in STAGES:
            g = observations.loc[observations.stage.eq(stage)].copy()
            g = g.dropna(subset=FEATURES[stage] + ["duration_ms"])
            if len(g) < 40:
                continue  # an absent model remains an explicit support gap
            self.kinds[stage] = set(g.measurement_kind.unique())
            if len(self.kinds[stage]) != 1:
                raise ValueError("Do not mix timing semantics in one model")
            raw = g[FEATURES[stage]].to_numpy(float)
            lo = raw.min(axis=0)
            hi = raw.max(axis=0)
            pad = np.maximum((hi - lo) * 0.15, np.maximum(1.0, hi * 0.05))
            pad = np.where(
                hi == lo, 1e-8, pad
            )  # never invent support for unseen batching
            self.bounds[stage] = (np.maximum(0, lo - pad), hi + pad)
            sample = g.sample(
                min(len(g), self.cfg.model_rows_per_stage), random_state=self.cfg.seed
            )
            X = np.log1p(sample[FEATURES[stage]].to_numpy(float))
            y = np.log(sample.duration_ms.to_numpy())
            fitted = []
            for alpha in (0.5, 0.95):
                m = GradientBoostingRegressor(
                    loss="quantile",
                    alpha=alpha,
                    n_estimators=self.cfg.trees,
                    max_depth=3,
                    min_samples_leaf=20,
                    learning_rate=0.06,
                    random_state=self.cfg.seed,
                )
                fitted.append(m.fit(X, y))
            self.models[stage] = fitted
            self.historical[stage] = np.quantile(np.log(g.duration_ms), [0.5, 0.95])
            rates = np.log(g.duration_ms.to_numpy() / self.work_units(g, stage))
            self.rate_reference[stage] = np.quantile(rates, [0.5, 0.95])
        return self

    def score(self, observations):
        out = observations.copy()
        out["supported"] = False
        out["support_reason"] = "missing_stage_model"
        for col in [
            "q50_log",
            "q95_log",
            "conditioned",
            "historical",
            "work_rate",
            "median_residual",
        ]:
            out[col] = np.nan
        for stage in STAGES:
            if stage not in self.models:
                continue
            idx = out.index[out.stage.eq(stage)]
            g = out.loc[idx]
            raw = g[FEATURES[stage]].to_numpy(float)
            finite = np.isfinite(raw).all(axis=1)
            lower, upper = self.bounds[stage]
            in_range = ((raw >= lower) & (raw <= upper)).all(axis=1) & finite
            same = (
                g.config_id.isin(self.configs).to_numpy()
                & g.measurement_kind.isin(self.kinds[stage]).to_numpy()
            )
            good = in_range & same
            out.loc[idx, "support_reason"] = np.where(
                ~same,
                "configuration_or_timing_change",
                np.where(
                    ~finite,
                    "missing_features",
                    np.where(~in_range, "outside_healthy_support", "ok"),
                ),
            )
            out.loc[idx, "supported"] = good
            if not good.any():
                continue
            use = idx[good]
            X = np.log1p(raw[good])
            target = np.log(out.loc[use, "duration_ms"].to_numpy())
            q50, q95 = [m.predict(X) for m in self.models[stage]]
            q95 = np.maximum(q95, q50 + self.cfg.scale_floor)
            scale = np.maximum(q95 - q50, self.cfg.scale_floor)
            out.loc[use, "q50_log"] = q50
            out.loc[use, "q95_log"] = q95
            out.loc[use, "conditioned"] = np.maximum(0, (target - q95) / scale)
            out.loc[use, "median_residual"] = target - q50
            for method, reference, ys in [
                ("historical", self.historical[stage], target),
                (
                    "work_rate",
                    self.rate_reference[stage],
                    np.log(
                        out.loc[use, "duration_ms"].to_numpy()
                        / self.work_units(out.loc[use], stage)
                    ),
                ),
            ]:
                a, b = reference
                out.loc[use, method] = np.maximum(
                    0, (ys - b) / max(b - a, self.cfg.scale_floor)
                )
        return out
