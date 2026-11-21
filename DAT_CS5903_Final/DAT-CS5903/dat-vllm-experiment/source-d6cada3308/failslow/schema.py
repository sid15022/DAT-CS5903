"""Canonical trace schema and route validation."""

import numpy as np
import pandas as pd
from .model import COMPONENTS, STAGES, FEATURES

OBS_COLUMNS = [
    "run_id",
    "obs_id",
    "request_id",
    "stage",
    "component",
    "start_s",
    "event_s",
    "duration_ms",
    "prompt_tokens",
    "uncached_tokens",
    "kv_bytes",
    "context_tokens",
    "batch_size",
    "generated_so_far",
    "config_id",
    "measurement_kind",
]
ARR_COLUMNS = [
    "run_id",
    "request_id",
    "arrival_s",
    "prompt_tokens",
    "max_new_tokens",
    "prefill_id",
    "decode_id",
    "path_id",
    "config_id",
]
SLO_COLUMNS = ["run_id", "request_id", "event_s", "metric", "value_ms"]
FORBIDDEN = {
    "fault",
    "fault_component",
    "true_component",
    "slowdown",
    "scenario",
    "kind",
    "onset_s",
    "end_s",
    "eventual_output_length",
    "final_output_tokens",
}


def validate_tables(observations, arrivals, slo_events):
    """Reject inconsistent canonical records; missing feature values cause abstention later."""
    for frame, columns, label in [
        (observations, OBS_COLUMNS, "observations"),
        (arrivals, ARR_COLUMNS, "arrivals"),
        (slo_events, SLO_COLUMNS, "slo_events"),
    ]:
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{label}: missing columns {sorted(missing)}")
        if FORBIDDEN & set(frame.columns):
            raise ValueError(f"{label}: evaluation-only columns are prohibited")
        for name in ("run_id", "request_id"):
            if frame[name].isna().any():
                raise ValueError(f"{label}: null {name}")
    if observations.duplicated(["run_id", "obs_id"]).any():
        raise ValueError("Duplicate observation IDs")
    if arrivals.duplicated(["run_id", "request_id"]).any():
        raise ValueError("Duplicate request IDs")
    for frame in (observations, arrivals):
        if frame.config_id.isna().any():
            raise ValueError("Null configuration ID")
    for frame, cols in [
        (observations, ["start_s", "event_s", "duration_ms"]),
        (arrivals, ["arrival_s", "prompt_tokens", "max_new_tokens"]),
        (slo_events, ["event_s", "value_ms"]),
    ]:
        for c in cols:
            if (
                not pd.api.types.is_numeric_dtype(frame[c])
                or not np.isfinite(frame[c]).all()
            ):
                raise ValueError(f"{c} must be finite numeric data")
            if (frame[c] < 0).any():
                raise ValueError(f"{c} must be nonnegative")
    if (observations.duration_ms <= 0).any():
        raise ValueError("Nonpositive duration")
    if not np.allclose(
        (observations.event_s - observations.start_s) * 1000,
        observations.duration_ms,
        rtol=1e-5,
        atol=1e-4,
    ):
        raise ValueError("Stage duration and local start/end timestamps disagree")
    if not observations.stage.isin(STAGES).all():
        raise ValueError("Unknown stage")
    for s, g in observations.groupby("stage"):
        if not g.component.isin(COMPONENTS[s]).all():
            raise ValueError("Unknown component ID")
        for c in FEATURES[s]:
            # Null features are allowed, but never silently filled with zero.
            if len(g) and not pd.api.types.is_numeric_dtype(g[c]):
                raise ValueError(f"Nonnumeric {c}")
            v = g[c].dropna()
            if not np.isfinite(v).all() or (v < 0).any():
                raise ValueError(f"Invalid {c}")
    count_columns = [
        "prompt_tokens",
        "uncached_tokens",
        "kv_bytes",
        "context_tokens",
        "batch_size",
        "generated_so_far",
    ]
    for frame, cols in [
        (observations, count_columns),
        (arrivals, ["prompt_tokens", "max_new_tokens"]),
    ]:
        for c in cols:
            if not pd.api.types.is_numeric_dtype(frame[c]):
                raise ValueError(f"Nonnumeric {c}")
            v = frame[c].dropna()
            if (
                not np.isfinite(v).all()
                or (v < 0).any()
                or not np.allclose(v, np.round(v))
            ):
                raise ValueError(f"{c} must contain nonnegative integer counts")
    if (observations.batch_size.dropna() < 1).any():
        raise ValueError("batch_size must be positive")
    if (arrivals[["prompt_tokens", "max_new_tokens"]] < 1).any().any():
        raise ValueError("Invalid request token count")
    if (observations.uncached_tokens > observations.prompt_tokens).any():
        raise ValueError("Uncached tokens exceed prompt")
    if not observations.measurement_kind.isin(["service", "progression"]).all():
        raise ValueError("Unknown measurement kind")
    if not slo_events.metric.isin(["ttft_ms", "itl_ms"]).all():
        raise ValueError("Unsupported SLO metric")
    if (
        slo_events.loc[slo_events.metric.eq("ttft_ms")]
        .duplicated(["run_id", "request_id"])
        .any()
    ):
        raise ValueError("TTFT must have one observation per request")
    if not arrivals.prefill_id.isin(COMPONENTS["prefill"]).all():
        raise ValueError("Invalid prefill route")
    if not arrivals.decode_id.isin(COMPONENTS["decode"]).all():
        raise ValueError("Invalid decode route")
    expected_path = "l" + arrivals.prefill_id.str[-1] + arrivals.decode_id.str[-1]
    if not arrivals.path_id.eq(expected_path).all():
        raise ValueError("Invalid path mapping")
    a = arrivals.set_index(["run_id", "request_id"])
    for frame in (observations, slo_events):
        keys = pd.MultiIndex.from_frame(frame[["run_id", "request_id"]])
        if not keys.isin(a.index).all():
            raise ValueError("Unjoinable request ID")
        av = a.reindex(keys).arrival_s.to_numpy()
        time_values = (
            frame.start_s.to_numpy() if "start_s" in frame else frame.event_s.to_numpy()
        )
        if (time_values + 1e-8 < av).any():
            raise ValueError("Observation predates arrival")
    keys = pd.MultiIndex.from_frame(observations[["run_id", "request_id"]])
    route = a.reindex(keys)
    expected = np.select(
        [observations.stage.eq("prefill"), observations.stage.eq("transfer")],
        [route.prefill_id.to_numpy(), route.path_id.to_numpy()],
        default=route.decode_id.to_numpy(),
    )
    if not observations.component.eq(expected).all():
        raise ValueError("Stage disagrees with route")
    if not observations.config_id.eq(route.config_id.to_numpy()).all():
        raise ValueError("Configuration mismatch within request")
    prompt = observations.prompt_tokens.to_numpy(float)
    if not np.allclose(
        prompt[np.isfinite(prompt)], route.prompt_tokens.to_numpy()[np.isfinite(prompt)]
    ):
        raise ValueError("Observed prompt length disagrees with arrival")
    return True
