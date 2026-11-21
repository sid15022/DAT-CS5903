"""Render the two evaluation figures from the archived result tables."""
import csv
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "DAT_CS5903_Final" / "DAT-CS5903"
GPU = PROJECT / "dat-vllm-experiment" / "study-2x2-r1-t90-q2.6"
SIM = PROJECT / "results" / "synthetic"
FIGURES = ROOT / "docs" / "figures"
METHODS = {
    "conditioned": "Work-conditioned",
    "historical": "Historical latency",
    "work_rate": "Work rate",
    "matched_peer": "Matched peer",
}
CASES = {
    "prefill": "Prefill", "decode": "Decode", "transfer": "Visible transfer",
    "mixed": "Mixed demand + decode", "sparse_worker": "Sparse worker",
    "missing_path": "Unobserved transfer",
}
TEAL, INK = "#006E70", "#374B70"


def rows(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def save(fig, ax, name):
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#D6DDDF", linewidth=0.7)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False)
    svg = StringIO()
    fig.savefig(svg, format="svg", metadata={"Date": None})
    (FIGURES / name).write_text(svg.getvalue().replace(" \n", "\n"), encoding="utf-8")
    plt.close(fig)


def localization():
    fig, ax = plt.subplots(figsize=(8.9, 3.7), layout="constrained")
    for data, offset, color, label in [
        (GPU, -0.18, TEAL, "GPU"), (SIM, 0.18, INK, "Simulator")
    ]:
        records = {r["method"]: r for r in rows(data / "summary.csv")}
        for index, method in enumerate(METHODS):
            record = records[method]
            hits, total = int(record["top1"]), int(record["incidents"])
            rate = 100 * hits / total
            y = index + offset
            ax.barh(y, rate, height=0.29, color=color, label=label if index == 0 else None)
            ax.text(rate + 1.4, y, f"{hits}/{total}", va="center", color=color, fontsize=10)
    ax.set_yticks(range(len(METHODS)), METHODS.values())
    ax.invert_yaxis()
    ax.set(xlim=(0, 100), xlabel="Incidents with the target ranked first (%)")
    save(fig, ax, "first_rank_accuracy.svg")


def paired_latency():
    fig, ax = plt.subplots(figsize=(8.9, 4.4), layout="constrained")
    cases = {r["run_id"]: r["case"] for r in rows(GPU / "incidents.csv")
             if r["method"] == "conditioned"}
    pairs = {cases[r["run_id"]]: r for r in rows(GPU / "paired_impact.csv")
             if float(r["strength"]) == 3}
    for field, offset, color, marker, label in [
        ("ttft_p99_ms_delta", -0.20, TEAL, "o", "Time to first token"),
        ("itl_p99_ms_delta", 0.20, INK, "s", "Inter-token latency"),
    ]:
        for index, case in enumerate(CASES):
            v = float(pairs[case][field])
            y = index + offset
            ax.plot(v, y, marker=marker, color=color, linestyle="none",
                    label=label if index == 0 else None)
            ax.annotate(f"{v:+.2f}", (v, y), xytext=(7, 0), textcoords="offset points",
                        va="center", color=color, fontsize=9)
    ax.axvline(0, color="#77848B", linewidth=1)
    ax.set_yticks(range(len(CASES)), CASES.values())
    ax.invert_yaxis()
    ax.set(xlim=(-6, 57), xlabel="Difference in p99 latency, incident minus matched control (ms)")
    save(fig, ax, "paired_latency_3x.svg")


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none", "svg.hashsalt": "cs5903-evaluation"})
    localization()
    paired_latency()
