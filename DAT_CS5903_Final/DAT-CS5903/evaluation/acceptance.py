"""Verify a frozen submission from the project root with Python 3."""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "results"
with (root / "synthetic/summary.csv").open() as stream:
    methods = {r["method"]: r for r in csv.DictReader(stream)}
for name, hits, false in [("conditioned", 24, 6), ("matched_peer", 24, 1),
                          ("historical", 12, 3), ("work_rate", 16, 11)]:
    row = methods[name]
    assert tuple(int(row[k]) for k in ("incidents", "top1", "false_episodes")) == (36, hits, false), name
qwen = root / "qwen-vllm-cpu"
runs = [json.loads(p.read_text()) for p in (qwen / "runs").glob("*/complete.json")]
assert sum(r["requests"] for r in runs) == 240
assert sum(r["forward_records"] for r in runs) == 934
rows = [json.loads(line) for line in (qwen / "decisions.jsonl").read_text().splitlines()]
assert len(rows) == 520
for stage in ("prefill", "decode"):
    def outputs(arm):
        return [(r["prompt_sha256"], r["output_ids"]) for r in json.loads(
            (qwen / "runs" / (stage + "-" + arm) / "outputs.json").read_text())]
    assert len(outputs("control")) == 32 and outputs("control") == outputs("fault")
print("PASS: four simulated methods, 240 Qwen requests, 934 forwards, 64 matched outputs, 520 decisions")
