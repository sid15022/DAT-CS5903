"""A supplementary single-engine study on real Qwen weights with vLLM CPU 0.17."""

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .data import frames, validate, save_trace, write_json
from .detector import Baseline, Monitor, windows
from .engines import MODEL, REVISION
from .model import CFG
from .text_workload import TextWorkload


@contextmanager
def singleton_collectives(enabled):
    """Use PyTorch's in-process collective group on a single CPU rank without TCP."""
    if not enabled:
        yield
        return
    import torch.distributed as dist
    from importlib import import_module

    import_module("torch.testing._internal.distributed.multi_threaded_pg")

    if dist.is_initialized() or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Local collectives require one fresh CPU process")
    dist.init_process_group(
        backend="threaded", rank=0, world_size=1, store=dist.HashStore()
    )
    original = dist.new_group

    def one_rank(ranks=None, **kwargs):
        if ranks is not None and list(ranks) != [0]:
            raise RuntimeError("Local validation supports only rank zero")
        return original(ranks, **{**kwargs, "backend": "threaded"})

    dist.new_group = one_rank
    try:
        yield
    finally:
        dist.new_group = original
        if dist.is_initialized():
            dist.destroy_process_group()


class ForwardTimer:
    def __init__(self):
        self.factor = 1.0
        self.spans = []

    def install(self, model):
        def before(_model, args):
            self.start = time.monotonic()

        def after(_model, args, output):
            if self.factor > 1:
                time.sleep((self.factor - 1) * (time.monotonic() - self.start))
            self.spans.append((self.start, time.monotonic()))

        model.register_forward_pre_hook(before)
        model.register_forward_hook(after)


def trial(llm, timer, text, run, seed, n, target, out, cfg_id, min_seconds=0):
    from vllm import SamplingParams

    engine = llm.llm_engine
    trace = {name: [] for name in ("observations", "arrivals", "slo_events")}
    answers = []
    rng = np.random.default_rng(seed)
    origin = time.monotonic()
    i = 0
    onset = end = None
    injected = 0
    while i < n or time.monotonic() - origin < min_seconds:
        rid = f"r{i:04d}"
        ids = text.prompt(seed, rid, int(rng.choice([65, 81, 97])))
        budget = int(rng.choice([3, 4, 5]))
        arrival = time.monotonic() - origin
        if i == 8:
            onset = arrival
        trace["arrivals"].append(
            dict(
                run_id=run,
                request_id=rid,
                arrival_s=arrival,
                available_s=arrival,
                prompt_tokens=len(ids),
                max_new_tokens=budget,
                prefill_id="p1",
                decode_id="d1",
                path_id="l11",
                config_id=cfg_id,
            )
        )
        params = SamplingParams(
            temperature=0, max_tokens=budget, ignore_eos=True, detokenize=False
        )
        engine.add_request(rid, {"prompt_token_ids": ids}, params)
        previous = arrival
        out_ids = []
        while engine.has_unfinished_requests():
            step = len(out_ids)
            stage = "prefill" if step == 0 else "decode"
            component = "p1" if step == 0 else "d1"
            timer.factor = 3.0 if target == component and 8 <= i < 24 else 1.0
            timer.spans.clear()
            result = engine.step()
            if not result and not timer.spans:
                continue
            if len(result) != 1 or len(timer.spans) != 1:
                raise RuntimeError(
                    f"Ambiguous model-forward/token join in {rid}: {len(result)}, {len(timer.spans)}"
                )
            token_ids = result[0].outputs[0].token_ids
            if len(token_ids) != step + 1:
                raise RuntimeError("Unexpected token-step count")
            out_ids.append(int(token_ids[-1]))
            start, stop = timer.spans[0]
            available = time.monotonic() - origin
            injected += int(timer.factor > 1)
            trace["observations"].append(
                dict(
                    run_id=run,
                    request_id=rid,
                    obs_id=f"{rid}-{step}",
                    stage=stage,
                    component=component,
                    start_s=start - origin,
                    event_s=stop - origin,
                    available_s=available,
                    duration_ms=(stop - start) * 1000,
                    prompt_tokens=len(ids),
                    uncached_tokens=len(ids) if step == 0 else None,
                    context_tokens=len(ids) + step,
                    batch_size=1,
                    generated_so_far=step,
                    config_id=cfg_id,
                    measurement_kind="service",
                )
            )
            trace["slo_events"].append(
                dict(
                    run_id=run,
                    request_id=rid,
                    event_s=available,
                    available_s=available,
                    metric="ttft_ms" if step == 0 else "itl_ms",
                    value_ms=(available - previous) * 1000,
                )
            )
            previous = available
        if len(out_ids) != budget:
            raise RuntimeError("Incomplete real generation")
        answers.append(
            dict(
                request_id=rid,
                prompt_tokens=len(ids),
                prompt_sha256=hashlib.sha256(
                    bytes(np.asarray(ids, dtype="int32"))
                ).hexdigest(),
                output_ids=out_ids,
                output_text=text.tok.decode(out_ids, skip_special_tokens=True),
            )
        )
        if i == 23:
            end = time.monotonic() - origin
        i += 1
    duration = time.monotonic() - origin
    timer.factor = 1.0
    view = frames(trace)
    validate(view, run)
    folder = out / "runs" / run
    save_trace(folder, view)
    write_json(folder / "outputs.json", answers)
    meta = dict(
        run_id=run,
        duration_s=duration,
        requests=i,
        forward_records=len(trace["observations"]),
        applied_delay_steps=injected,
        target=target,
        onset_s=onset,
        end_s=end,
        seed=seed,
    )
    write_json(folder / "complete.json", meta)
    print(run, i, round(duration, 2), flush=True)
    return view, meta, answers


def verify_checkpoint(weights, out):
    if weights is None:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(MODEL, "model.safetensors", revision=REVISION))
    else:
        path = weights / "model.safetensors"
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    expected = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
    if digest != expected:
        raise ValueError("Local weights differ from the pinned Qwen revision")
    write_json(
        out / "weights.json",
        dict(
            model=MODEL,
            revision=REVISION,
            filename="model.safetensors",
            bytes=path.stat().st_size,
            sha256=digest,
        ),
    )


def summarize_saved(out):
    study = json.loads((out / "study.json").read_text())
    records = pd.read_json(out / "decisions.jsonl", lines=True)
    rows = []
    for item in study["outcomes"]:
        rid = item["run_id"]
        meta = json.loads((out / "runs" / rid / "complete.json").read_text())
        d = records[(records.run_id == rid) & (records.method == "conditioned")]
        before = d[d.t_s < meta["onset_s"]]
        incident = d[d.t_s.between(meta["onset_s"], meta["end_s"] + study["window_s"])]
        matches = incident[
            incident.ranked.map(lambda x: bool(x) and x[0] == meta["target"])
        ]
        rows.append(
            dict(
                run_id=rid,
                target=meta["target"],
                first_ranking=json.dumps(item["first_ranking"]),
                ranking_before_onset=json.dumps(
                    before.iloc[-1].ranked if len(before) else []
                ),
                first_correct_delay_s=float(matches.iloc[0].t_s - meta["onset_s"])
                if len(matches)
                else None,
                false_episodes=item["false_episodes"],
                requests=item["requests"],
            )
        )
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)


def run(a):
    import joblib
    import torch
    import vllm
    from vllm import LLM

    if vllm.__version__.split("+")[0] != "0.17.0":
        raise RuntimeError("CPU verification targets vLLM 0.17.0")
    from vllm.platforms import current_platform

    if not current_platform.is_cpu():
        raise RuntimeError("Install the CPU vLLM wheel for this validation")
    if (a.out / "runs").exists():
        raise ValueError("Choose a fresh CPU study output directory")
    a.out.mkdir(parents=True, exist_ok=True)
    verify_checkpoint(a.weights, a.out)
    cfg = replace(
        CFG,
        window_s=4.0,
        cadence_s=1.0,
        min_requests=2,
        min_slo_requests=3,
        min_itl_gaps=5,
    )
    build = dict(
        model=MODEL,
        revision=REVISION,
        vllm=vllm.__version__,
        torch=str(torch.__version__),
        device="CPU",
        dtype="float32",
        local_collectives=a.local_collectives,
        timing="synchronous model forward on one rank",
        input_policy="serial closed-loop",
        transfer_visibility="no physical handoff in this study",
    )
    cfg_id = hashlib.sha256(json.dumps(build, sort_keys=True).encode()).hexdigest()[:20]
    kwargs = dict(
        model=str(a.weights) if a.weights else MODEL,
        revision=REVISION,
        tokenizer_revision=REVISION,
        dtype="float32",
        enforce_eager=True,
        max_model_len=384,
        max_num_batched_tokens=384,
        max_num_seqs=1,
        enable_prefix_caching=False,
        async_scheduling=False,
    )
    llm = LLM(**kwargs)
    text = TextWorkload(llm.get_tokenizer())
    timer = ForwardTimer()
    llm.apply_model(timer.install)
    from vllm import SamplingParams

    sample = []
    for prompt in (
        "What is the capital of France? Answer in one short sentence.",
        "What is 7 times 8? Answer with the number.",
    ):
        answer = llm.chat(
            [{"role": "user", "content": prompt}],
            sampling_params=SamplingParams(temperature=0, max_tokens=24),
            use_tqdm=False,
        )[0]
        sample.append(
            dict(
                prompt=prompt,
                output=answer.outputs[0].text,
                token_ids=answer.outputs[0].token_ids,
            )
        )
    write_json(a.out / "examples.json", sample)
    splits = {}
    for i, (s, n) in enumerate([("fit", 48), ("scale", 32), ("threshold", 32)]):
        v, meta, _ = trial(
            llm, timer, text, s, 710 + i, n, None, a.out, cfg_id, min_seconds=20
        )
        splits[s] = [(s, True, meta["duration_s"], v)]
    b = Baseline.train(splits, 1, cfg)
    joblib.dump(b, a.out / "baseline.joblib")
    decisions = []
    outcomes = []
    for group, target, seed in [("prefill", "p1", 918), ("decode", "d1", 1019)]:
        paired = []
        for active in [False, True] if group == "prefill" else [True, False]:
            rid = group + ("-fault" if active else "-control")
            v, meta, answers = trial(
                llm,
                timer,
                text,
                rid,
                seed,
                32,
                target if active else None,
                a.out,
                cfg_id,
            )
            paired.append([(x["prompt_sha256"], x["output_ids"]) for x in answers])
            monitor = Monitor(b)
            local = []
            for tick, raw in windows(v, meta["duration_s"], b.model, b.topology, cfg):
                ds, _ = monitor.step(rid, tick, raw)
                local.extend(ds)
            decisions.extend(local)
            d = [x for x in local if x["method"] == "conditioned"]
            interval = [
                x
                for x in d
                if meta["onset_s"] <= x["t_s"] <= meta["end_s"] + cfg.window_s
            ]
            first = next((x for x in interval if x["ranked"]), None)
            starts = sum(
                bool(x["ranked"]) and (i == 0 or not d[i - 1]["ranked"])
                for i, x in enumerate(d)
            )
            outcomes.append(
                dict(
                    run_id=rid,
                    target=meta["target"],
                    requests=meta["requests"],
                    applied_delay_steps=meta["applied_delay_steps"],
                    first_ranking=first["ranked"] if first else [],
                    first_ranking_delay_s=first["t_s"] - meta["onset_s"]
                    if first
                    else None,
                    false_episodes=None if active else int(starts),
                    duration_s=meta["duration_s"],
                )
            )
        if paired[0] != paired[1]:
            raise RuntimeError("Paired requests or generated tokens differ")
    pd.DataFrame(decisions).to_json(
        a.out / "decisions.jsonl", lines=True, orient="records"
    )
    write_json(
        a.out / "study.json",
        dict(
            **build,
            config_id=cfg_id,
            window_s=4,
            cadence_s=1,
            intervention="3x synchronous model-forward service on requests 8 to 23 inclusive (zero-based)",
            paired_outputs_identical=True,
            outcomes=outcomes,
        ),
    )
    summarize_saved(a.out)
    print(json.dumps(outcomes, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--weights", type=Path, help="Local snapshot of the pinned Qwen revision"
    )
    p.add_argument(
        "--local-collectives",
        action="store_true",
        help="Use process-local collectives for a socket-limited CPU host",
    )
    a = p.parse_args()
    os.environ.update(
        VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_CPU_OMP_THREADS_BIND="nobind"
    )
    os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "1")
    with singleton_collectives(a.local_collectives):
        run(a)


if __name__ == "__main__":
    main()
