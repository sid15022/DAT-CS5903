"""Trace I/O. Observer inputs and experimental labels live in separate files."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import ARR_COLUMNS, OBS_COLUMNS, SLO_COLUMNS, validate_tables

TABLES = {
    "observations": OBS_COLUMNS,
    "arrivals": ARR_COLUMNS,
    "slo_events": SLO_COLUMNS,
}
CLOCK = {"observations": "event_s", "arrivals": "arrival_s", "slo_events": "event_s"}
TEXT = {
    "run_id",
    "obs_id",
    "request_id",
    "stage",
    "component",
    "config_id",
    "measurement_kind",
    "prefill_id",
    "decode_id",
    "path_id",
    "metric",
}


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str) + "\n")
    tmp.replace(path)


def frames(records):
    out = {}
    for name, cols in TABLES.items():
        f = pd.DataFrame(records.get(name, []), columns=[*cols, "available_s"])
        for col in f.columns.difference(list(TEXT)):
            f[col] = pd.to_numeric(f[col], errors="raise").astype("float64")
        for col in f.columns.intersection(list(TEXT)):
            f[col] = f[col].astype("object")
        out[name] = f
    return out


def validate(view, run_id):
    if set(view) != set(TABLES):
        raise ValueError("Pass only the three observer tables")
    for name, columns in TABLES.items():
        table = view[name]
        if set(table) != set(columns + ["available_s"]):
            raise ValueError(f"{name}: unexpected or missing columns")
        if not table.run_id.eq(run_id).all():
            raise ValueError("Mixed run IDs")
        available = table.available_s.to_numpy(float)
        if (
            not np.isfinite(available).all()
            or (available + 1e-8 < table[CLOCK[name]]).any()
        ):
            raise ValueError("Invalid collector availability time")
    validate_tables(**view)


def save_trace(path, view):
    path.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        destination = path / (name + ".csv.gz")
        temporary = destination.with_name(destination.name + ".tmp")
        view[name].to_csv(temporary, index=False, compression="gzip")
        temporary.replace(destination)


def load_trace(path, rid):
    view = {}
    for name, cols in TABLES.items():
        f = pd.read_csv(
            path / (name + ".csv.gz"),
            float_precision="round_trip",
            dtype={c: str for c in cols if c in TEXT},
        )
        for c in f.columns.difference(list(TEXT)):
            f[c] = pd.to_numeric(f[c])
        view[name] = f
    validate(view, rid)
    return view
