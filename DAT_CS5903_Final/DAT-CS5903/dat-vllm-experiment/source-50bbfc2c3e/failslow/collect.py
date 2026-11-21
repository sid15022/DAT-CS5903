"""Single-host disaggregated inference harness with actual KV movement and token delivery."""

import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import shutil
import socket
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd

from .data import TABLES, frames, save_trace, validate, write_json
from .detector import components, ticks
from .engines import cache_key
from .evaluate import service_stats
from .study import Trial
from .workers import delay, worker


def move_kv(source, destination, origin, interval, factor):
    if destination.exists():
        raise RuntimeError("Duplicate KV destination")
    if not source.is_dir():
        raise RuntimeError("Prefill did not export a KV bundle")
    start = time.monotonic()
    files = list(source.iterdir())
    if not files or any(p.suffix not in {".safetensors", ".npz"} for p in files):
        raise RuntimeError("Unexpected KV export")
    destination.mkdir()
    for p in files:
        shutil.copyfile(p, destination / p.name)
    total = sum(p.stat().st_size for p in destination.iterdir())
    if total <= 0:
        raise RuntimeError("Empty transfer")
    applied = delay(start, origin, interval, factor)
    end = time.monotonic()
    shutil.rmtree(source)
    return start, end, total, applied


class Cluster:
    def __init__(self, backend, workers, spool=None, log_dir=None):
        self.servers = {}
        self.text = None
        self.log_dir = log_dir
        self.backend = backend
        self.workers = workers
        self.topology = components(workers)
        self.devices = [None] * (2 * workers)
        self.ctx = mp.get_context("spawn")
        self.procs = {}
        self.conns = {}
        if backend in ("vllm", "vllm-http"):
            if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
                selected = [
                    v.strip()
                    for v in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
                    if v.strip()
                ]
            else:
                try:
                    result = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                        text=True,
                    )
                except (FileNotFoundError, subprocess.CalledProcessError) as exc:
                    raise RuntimeError(
                        "No NVIDIA device inventory; GPU run requires a CUDA host"
                    ) from exc
                selected = result.split()
            if len(selected) < 2 * workers or len(set(selected)) != len(selected):
                raise ValueError(
                    f"Need {2 * workers} distinct visible GPUs for this topology"
                )
            self.devices = selected[: 2 * workers]
        if spool:
            spool.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="dat-kv-", dir=spool)
        self.root = Path(self.tmp.name)
        self.pool = ThreadPoolExecutor(max_workers=2 * workers + workers**2 + 3)
        self.inventory = {}

    def __enter__(self):
        try:
            if self.backend == "vllm-http":
                from .native_server import boot_servers

                boot_servers(self, self.log_dir or self.root / "logs")
            else:
                ids = self.topology["prefill"] + self.topology["decode"]
                for c, dev in zip(ids, self.devices):
                    (self.root / c).mkdir()
                    parent, child = self.ctx.Pipe()
                    proc = self.ctx.Process(
                        target=worker, args=(child, self.backend, dev, self.root / c)
                    )
                    proc.start()
                    child.close()
                    self.conns[c] = parent
                    self.procs[c] = proc
                for c in ids:
                    if not self.conns[c].poll(1200):
                        raise TimeoutError(f"{c}: engine initialization timed out")
                    kind, info = self.conns[c].recv()
                    if kind != "ready":
                        raise RuntimeError(f"{c}: {info}")
                    self.inventory[c] = info
            self.profile = dict(
                host=socket.gethostname(),
                workers=self.inventory,
                topology=self.topology,
                clock="single-host CLOCK_MONOTONIC",
                stage_semantics=(
                    "vllm-server-forward-v1" if self.servers else "file-kv-handoff-v1"
                ),
            )
            self.config_id = hashlib.sha256(
                json.dumps(self.profile, sort_keys=True).encode()
            ).hexdigest()[:20]
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        for server in self.servers.values():
            server.close()
        for conn in self.conns.values():
            try:
                conn.send(None)
            except (EOFError, OSError):
                pass
        for proc in self.procs.values():
            proc.join(3)
            if proc.is_alive():
                proc.terminate()
                proc.join(3)
            if proc.is_alive():
                proc.kill()
                proc.join()
        for conn in self.conns.values():
            conn.close()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.tmp.cleanup()

    def rpc(self, c, job, callback):
        if self.servers:
            return self.servers[c].request(job, callback)
        con = self.conns[c]
        con.send(job)
        while con.poll(180):
            kind, value = con.recv()
            if kind == "done":
                return value
            if kind != "token":
                raise RuntimeError(f"{c}: {value}")
            callback(*value)
        raise TimeoutError(f"{c}: no progress for 180 seconds")

    def pilot(self):
        results = []
        ids = (
            self.text.prompt(0, "r0", 129, pilot=True)
            if self.text
            else self.rpc("p1", {"mode": "prompt"}, lambda *x: None)
        )
        base = dict(
            ids=ids, original=ids, origin=time.monotonic(), interval=(0, 0), factor=1.0
        )
        for p in self.topology["prefill"]:
            for d in self.topology["decode"]:
                rid = f"pilot-{p}-{d}"
                ref = []
                actual = []
                self.rpc(
                    p,
                    dict(base, mode="reference", rid=rid + "-ref", count=8),
                    lambda tok, *_: ref.append(tok),
                )
                self.rpc(
                    p,
                    dict(base, mode="prefill", rid=rid + "-p", count=1),
                    lambda tok, *_: actual.append(tok),
                )
                key = cache_key(ids)
                start, end, size, _ = move_kv(
                    self.root / p / key, self.root / d / key, 0, (0, 0), 1.0
                )
                proof = self.rpc(
                    d,
                    dict(
                        base, ids=ids + actual, mode="decode", rid=rid + "-d", count=7
                    ),
                    lambda tok, *_: actual.append(tok),
                )
                if ref != actual:
                    raise RuntimeError(
                        f"{p}->{d}: split and unsplit greedy token sequences differ"
                    )
                sample = (
                    {
                        "prompt_text": self.text.tok.decode(ids),
                        "generated_text": self.text.tok.decode(
                            actual, skip_special_tokens=True
                        ),
                        "endpoint": "/v1/completions",
                        "forward_records_checked": len(ref) + len(actual),
                    }
                    if self.text
                    else {}
                )
                results.append(
                    dict(
                        **sample,
                        prefill=p,
                        decode=d,
                        bytes=size,
                        copy_ms=(end - start) * 1000,
                        prompt_tokens=len(ids),
                        **proof,
                        reference_ids=ref,
                        split_ids=actual,
                        exact_match=True,
                    )
                )
        return dict(config_id=self.config_id, profile=self.profile, paths=results)


def schedule(trial, duration, rate, n, vocab):
    rng = np.random.default_rng(trial.seed)
    t = 0.0
    rows = []
    while True:
        t += rng.exponential(1 / (2 * rate))
        if t >= duration:
            break
        active = trial.shift is None or t >= trial.shift
        if rng.random() > (1.0 if active and trial.regime == "burst" else 0.5):
            continue
        length = int(
            rng.choice(
                [65, 129, 257],
                p=(
                    [0.1, 0.25, 0.65]
                    if active and trial.regime == "long"
                    else [0.6, 0.3, 0.1]
                ),
            )
        )
        budget = int(
            rng.choice(
                [4, 8, 12],
                p=(
                    [0.1, 0.2, 0.7]
                    if active and trial.regime == "long_output"
                    else [0.5, 0.4, 0.1]
                ),
            )
        )

        def idx():
            return 1 if n == 1 or rng.random() < (0.985 if trial.sparse else 0.5) else 2

        p, d = idx(), idx()
        ids = rng.integers(8, vocab, size=length).tolist()
        rows.append((t, f"r{len(rows):05d}", f"p{p}", f"d{d}", f"l{p}{d}", ids, budget))
    return rows


async def measure(cluster, trial, duration, rate, path, cfg, baseline, observed):
    loop = asyncio.get_running_loop()
    members = sum(cluster.topology.values(), [])
    locks = {c: asyncio.Lock() for c in members}

    def available():
        return time.monotonic() - origin

    trace = {k: [] for k in TABLES}
    ledger = defaultdict(int)
    generations = {}
    workload = schedule(
        trial, duration, rate, cluster.workers, cluster.inventory["p1"]["vocab"]
    )
    if cluster.text is not None:
        workload = [
            (ts, r, p, d, link, cluster.text.prompt(trial.seed, r, len(ids)), budget)
            for ts, r, p, d, link, ids, budget in workload
        ]
    rid = trial.rid
    interval = (trial.onset, trial.end)
    config = cluster.config_id
    journal = path / "events.jsonl"
    fp = journal.open("w", buffering=1) if observed else None
    follower = None
    if baseline is not None and observed:
        from .online import tail

        follower = cluster.ctx.Process(
            target=tail, args=(baseline, rid, journal, path / "live.jsonl")
        )
        follower.start()

    def emit(table, row, when=None):
        when = available() if when is None else when
        if when > duration:
            return
        row.update(available_s=when, run_id=rid)
        trace[table].append(row)
        if fp:
            fp.write(json.dumps([table, row]) + "\n")

    def op(stage, c, r, start, end, fields, **work):
        if observed and not (trial.missing and stage == "transfer"):
            emit(
                "observations",
                dict(
                    request_id=r,
                    obs_id=f"{r}-{stage}-{work.get('generated_so_far', 0)}",
                    component=c,
                    stage=stage,
                    start_s=start - origin,
                    event_s=end - origin,
                    duration_ms=1000 * (end - start),
                    measurement_kind="service",
                    config_id=config,
                    **fields,
                    **work,
                ),
            )

    def strength(c):
        return trial.strength if c == trial.target else 1.0

    async def request(row):
        scheduled, r, p, d, link, ids, budget = row
        emit(
            "arrivals",
            dict(
                request_id=r,
                arrival_s=scheduled,
                prompt_tokens=len(ids),
                max_new_tokens=budget,
                prefill_id=p,
                decode_id=d,
                path_id=link,
                config_id=config,
            ),
        )
        last = scheduled
        output = []
        fields = dict(prompt_tokens=len(ids), batch_size=1)

        def token(stage, c, tok, index, start, end, applied):
            nonlocal last
            # Client token receipt clock. The first token comes from prefill.
            now = available()
            generated = len(output)
            output.append(tok)
            if now <= duration:
                ledger["applied_stages"] += int(applied)
                emit(
                    "slo_events",
                    dict(
                        request_id=r,
                        event_s=now,
                        metric="ttft_ms" if not generated else "itl_ms",
                        value_ms=1000 * (now - last),
                    ),
                    when=now,
                )
                if generated + 1 == budget:
                    ledger["completed"] += 1
                    if cluster.text is not None:
                        generations[r] = (ids, list(output), p, d)
            last = now
            op(
                stage,
                c,
                r,
                start,
                end,
                fields,
                uncached_tokens=len(ids) if stage == "prefill" else None,
                context_tokens=len(ids) + generated,
                generated_so_far=generated,
            )

        async def engine(c, stage, count, inputs):
            async with locks[c]:
                if available() >= duration:
                    return None
                job = dict(
                    rid=r + "-" + c,
                    mode=stage,
                    ids=inputs,
                    original=ids,
                    count=count,
                    origin=origin,
                    interval=interval,
                    factor=strength(c),
                )

                def receive(*args):
                    loop.call_soon_threadsafe(token, stage, c, *args)

                return await loop.run_in_executor(
                    cluster.pool, cluster.rpc, c, job, receive
                )

        if await engine(p, "prefill", 1, ids) is None:
            return
        key = cache_key(ids)
        src, dst = cluster.root / p / key, cluster.root / d / key
        try:
            async with locks[link]:
                if available() >= duration:
                    return
                start, end, size, applied = await loop.run_in_executor(
                    cluster.pool, move_kv, src, dst, origin, interval, strength(link)
                )
                ledger["applied_stages"] += int(applied and end - origin <= duration)
                op("transfer", link, r, start, end, fields, kv_bytes=size)
            if await engine(d, "decode", budget - 1, ids + output[:1]) is not None:
                ledger["verified_loads"] += 1
        finally:
            for tmp in (src, dst):
                if tmp.exists():
                    shutil.rmtree(tmp)

    async def clock():
        for t in ticks(duration, cfg):
            await asyncio.sleep(max(0, t - available()))
            if fp:
                fp.write(json.dumps(["tick", t]) + "\n")

    sampler = None
    if observed and cluster.backend in ("vllm", "vllm-http"):
        # This side channel is ancillary evidence and never crosses the model's input boundary.
        sampler = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,uuid,utilization.gpu,memory.used,power.draw,temperature.gpu",
                "--format=csv",
                "-l",
                "1",
                "-f",
                str(path / "gpu.csv"),
            ]
        )
    usage0 = resource.getrusage(resource.RUSAGE_SELF)
    origin = time.monotonic()
    tasks = []
    try:
        tasks.append(asyncio.create_task(clock()))
        for row in workload:
            await asyncio.sleep(max(0, row[0] - available()))
            if available() >= duration:
                break
            tasks.append(asyncio.create_task(request(row)))
        await asyncio.gather(*tasks)
    finally:
        if fp:
            fp.write('["end",null]\n')
            fp.close()
        if sampler:
            sampler.terminate()
            sampler.wait(timeout=10)
        if follower:
            follower.join(120)
            if follower.is_alive():
                follower.terminate()
                follower.join()
            if follower.exitcode != 0:
                raise RuntimeError("Live observer failed")
    usage1 = resource.getrusage(resource.RUSAGE_SELF)
    view = frames(trace)
    n = len(view["arrivals"])
    completed = ledger["completed"]
    meta = dict(
        run_id=rid,
        config_id=config,
        duration_s=duration,
        n_completed_requests=completed,
        n_right_censored_requests=n - completed,
        offered=len(workload),
        accepted=n,
        verified_loads=ledger["verified_loads"],
        applied_stage_operations=ledger["applied_stages"],
        trace_bytes=journal.stat().st_size if observed else 0,
        collector_cpu_s=usage1.ru_utime
        + usage1.ru_stime
        - usage0.ru_utime
        - usage0.ru_stime,
        coordinator_peak_rss_kib=usage1.ru_maxrss,
        synthetic_only=False,
        backend=cluster.backend,
    )
    if follower:
        stats = path / "observer_stats.json"
        if stats.exists():
            meta.update(json.loads(stats.read_text()))
    if cluster.text is not None:
        import gzip

        with gzip.open(path / "generations.jsonl.gz", "wt") as output_file:
            for r, (ids, tokens, p, d) in sorted(generations.items()):
                answer = dict(
                    request_id=r,
                    prefill=p,
                    decode=d,
                    prompt_tokens=len(ids),
                    prompt_sha256=hashlib.sha256(
                        np.asarray(ids, dtype="<i4").tobytes()
                    ).hexdigest(),
                    output_token_ids=tokens,
                    output_text=cluster.text.tok.decode(
                        tokens, skip_special_tokens=True
                    ),
                )
                if int(r[1:]) < 3:
                    answer["prompt_text"] = cluster.text.tok.decode(ids)
                output_file.write(json.dumps(answer) + "\n")
    return view, meta


def collect(cluster, trial, duration, rate, path, cfg, baseline=None, observed=True):
    return asyncio.run(
        measure(cluster, trial, duration, rate, path, cfg, baseline, observed)
    )


def experiment(cluster, a, cfg):
    import joblib
    from .pipeline import suite

    a.out.mkdir(parents=True, exist_ok=True)
    pilot = cluster.pilot()
    write_json(a.out / "pilot.json", pilot)
    if a.command == "pilot":
        print(json.dumps(pilot, indent=2))
        return
    if a.command == "suite":
        suite(a, cfg, cluster)
        if not a.with_overhead:
            return
        from copy import copy

        a = copy(a)
        a.baseline = a.out / "baseline.joblib"
        a.out = a.out / "overhead"
        a.out.mkdir(exist_ok=True)
    if not a.baseline:
        raise ValueError("overhead needs --baseline from a measured suite")
    baseline = joblib.load(a.baseline)
    if baseline.model.configs != {cluster.config_id}:
        raise ValueError("Baseline does not match this measured testbed")
    cfg = baseline.cfg
    rows = []
    for i in range(a.reps):
        for enabled in [False, True] if i % 2 == 0 else [True, False]:
            rid = f"overhead-{i}-{'on' if enabled else 'off'}"
            path = a.out / rid
            from .data import load_trace

            path.mkdir(exist_ok=True)
            spec = dict(
                config_id=cluster.config_id,
                seed=a.seed + i,
                rate=a.rate,
                duration=a.duration,
                observer=enabled,
                baseline_sha256=hashlib.sha256(a.baseline.read_bytes()).hexdigest(),
            )
            if (path / "complete.json").exists():
                meta = json.loads((path / "complete.json").read_text())
                if meta.get("overhead_spec") != spec:
                    raise ValueError(
                        "Existing overhead run differs; choose a fresh --out"
                    )
                view = load_trace(path, rid)
            else:
                view, meta = collect(
                    cluster,
                    Trial(rid, "overhead", a.seed + i, "healthy"),
                    a.duration,
                    a.rate,
                    path,
                    cfg,
                    baseline=baseline,
                    observed=enabled,
                )
                validate(view, rid)
                meta["overhead_spec"] = spec
                save_trace(path, view)
                write_json(path / "complete.json", meta)
            rows.append(
                {
                    "rep": i,
                    "observer": enabled,
                    **meta,
                    **service_stats(view, meta, cfg, a.duration),
                }
            )
    f = pd.DataFrame(rows)
    f.to_csv(a.out / "overhead.csv", index=False)
    paired = []
    for rep, g in f.groupby("rep"):
        off, on = g[~g.observer].iloc[0], g[g.observer].iloc[0]
        row = {"rep": rep}
        for key in ["completed_rps", "ttft_p99_ms", "itl_p99_ms"]:
            row[key + "_change_pct"] = (
                100 * (on[key] / off[key] - 1) if off[key] > 0 else np.nan
            )
        paired.append(row)
    pd.DataFrame(paired).to_csv(a.out / "overhead_paired.csv", index=False)
