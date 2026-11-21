"""Audit saved SageMaker outputs without launching vLLM or importing notebook code."""
import argparse
import csv
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "DAT_CS5903_Final/DAT-CS5903/SageMaker_vLLM.ipynb"


def audit(path):
    cells = json.loads(path.read_text())["cells"]
    reports = []
    for cell in cells:
        for output in cell.get("outputs", []):
            text = "".join(output.get("text", []))
            try:
                obj = json.loads(text)
            except ValueError:
                continue
            if isinstance(obj, dict) and {"profile", "paths"} <= obj.keys():
                reports.append(obj)
    if len(reports) != 1:
        raise ValueError("Expected one saved pilot report")
    rows = reports[0]["paths"]
    if {(r["prefill"], r["decode"]) for r in rows} != {
        (p, d) for p in ("p1", "p2") for d in ("d1", "d2")
    } or len(rows) != 4:
        raise ValueError("Incomplete route coverage")
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["prefill", "decode", "bytes", "loaded_tokens", "copy_ms", "exact_match"])
    for r in rows:
        if not (r["reference_ids"] == r["split_ids"] and len(r["split_ids"]) == 8
                and r["loaded_tokens"] == 128 and r["exact_match"] and r["bytes"] > 0):
            raise ValueError(f"Invalid handoff: {r['prefill']} -> {r['decode']}")
        writer.writerow([r[k] for k in ("prefill","decode","bytes","loaded_tokens","copy_ms","exact_match")])
    return buf.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path, nargs="?", default=DEFAULT)
    print(audit(parser.parse_args().notebook), end="")
