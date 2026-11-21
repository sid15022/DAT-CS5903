import pandas as pd
import numpy as np
from collections import deque
import heapq


def simulate_run(
    run_id,
    seed,
    duration_s=180,
    regime="normal",
    fault_component=None,
    fault_start_s=75.0,
    fault_end_s=145.0,
    slowdown=3.0,
    workload_shift_s=None,
    sparse=False,
    missing_transfer=False,
):
    """Produce reproducible *synthetic* PD telemetry, never measurements of vLLM.

    Each worker/path is a FIFO server. Decode schedules one token at a time,
    re-enqueuing the request after each token, a simplified round-robin model.
    No continuous batching, GPU kernels, network topology, or real model is
    simulated. batch_size=1; 8–24 tokens bound the generated response lengths.
    Stage observations measure service only; TTFT/ITL also include queue waits.
    TTFT ends at the first decode token (a simplifying serving convention).

    All random work and service noise are pre-drawn before scheduling, making
    same-seed fault/control runs share underlying randomness. A fault applies
    to service starting within [fault_start_s, fault_end_s); an in-flight task
    retains its assigned service duration. event_s is local completion/receipt
    time. Only observations received by duration_s are returned; pending tasks
    are right-censored. No hidden fault labels enter observations or arrivals.
    """
    regimes = {"normal", "long", "burst", "cache"}
    components = ("p1", "p2", "d1", "d2", "l11", "l12", "l21", "l22")
    if regime not in regimes:
        raise ValueError(f"regime must be one of {sorted(regimes)}")
    if fault_component is not None and fault_component not in components:
        raise ValueError(f"Unknown fault component: {fault_component}")
    if duration_s <= 0 or slowdown < 1:
        raise ValueError("duration_s must be positive and slowdown must be >= 1")
    if fault_component is not None and fault_end_s <= fault_start_s:
        raise ValueError("fault_end_s must exceed fault_start_s")

    rng = np.random.default_rng(seed)
    # A common candidate process is thinned by the workload regime. The fault
    # never changes candidates, routes, request features, or random draws.
    candidate_rate = 7.5
    candidate_times = []
    t = 0.0
    while True:
        t += rng.exponential(1.0 / candidate_rate)
        if t >= duration_s:
            break
        candidate_times.append(t)
    candidate_times = np.asarray(candidate_times, dtype=float)
    n_candidates = len(candidate_times)
    regime_active = (
        np.ones(n_candidates, dtype=bool)
        if workload_shift_s is None
        else candidate_times >= workload_shift_s
    )
    arrival_rate = np.where(regime_active & (regime == "burst"), 5.0, 2.6)
    keep = rng.random(n_candidates) < arrival_rate / candidate_rate
    arrival_times = candidate_times[keep]
    active = regime_active[keep]
    n = len(arrival_times)

    base_prompt = rng.lognormal(np.log(1024.0), 0.85, n)
    prompt_multiplier = np.where(active & (regime == "long"), 2.1, 1.0)
    prompts = (
        np.round(np.clip(base_prompt * prompt_multiplier, 128.0, 8192.0) / 32.0) * 32.0
    ).astype(int)
    cache_draw = rng.random(n)
    normal_hit = np.select(
        [cache_draw < 0.55, cache_draw < 0.75, cache_draw < 0.95],
        [0.0, 0.25, 0.75],
        default=0.95,
    )
    shifted_hit = np.select(
        [cache_draw < 0.10, cache_draw < 0.25, cache_draw < 0.65],
        [0.0, 0.50, 0.85],
        default=0.95,
    )
    hit_fraction = np.where(active & (regime == "cache"), shifted_hit, normal_hit)
    uncached = np.maximum(1, np.round(prompts * (1.0 - hit_fraction))).astype(int)
    # Abstract byte volume for conditioning transfer time; not a model KV size.
    kv_bytes = prompts.astype(np.int64) * 8192
    response_lengths = rng.integers(8, 25, n)
    route_draw_p, route_draw_d = rng.random(n), rng.random(n)
    first_probability = 0.985 if sparse else 0.5
    p_idx = np.where(route_draw_p < first_probability, 1, 2)
    d_idx = np.where(route_draw_d < first_probability, 1, 2)
    p_ids = np.array([f"p{x}" for x in p_idx])
    d_ids = np.array([f"d{x}" for x in d_idx])
    path_ids = np.array([f"l{p}{d}" for p, d in zip(p_idx, d_idx)])
    prefill_noise = rng.lognormal(-0.5 * 0.12**2, 0.12, n)
    transfer_noise = rng.lognormal(-0.5 * 0.10**2, 0.10, n)
    decode_noise = rng.lognormal(-0.5 * 0.09**2, 0.09, (n, 24))
    prefill_base_ms = (
        15.0 + 0.024 * uncached + 0.0015 * uncached**2 / 8192.0
    ) * prefill_noise
    transfer_base_ms = (3.0 + kv_bytes / 1_500_000.0) * transfer_noise
    token_positions = np.arange(24)[None, :]
    decode_base_ms = (
        18.0 + 0.0028 * (prompts[:, None] + token_positions)
    ) * decode_noise
    request_ids = np.array([f"{run_id}-r{i:05d}" for i in range(n)])
    arrivals = pd.DataFrame(
        {
            "run_id": run_id,
            "request_id": request_ids,
            "arrival_s": arrival_times,
            "prompt_tokens": prompts,
            "max_new_tokens": response_lengths,
            "prefill_id": p_ids,
            "decode_id": d_ids,
            "path_id": path_ids,
            "config_id": "synthetic-v1",
        }
    )

    queue = {component: deque() for component in components}
    busy = {component: False for component in components}
    events = []
    sequence = 0
    observations, slo_rows = [], []
    previous_token_end = {}
    completed_requests = set()

    def push_event(when, kind, payload):
        nonlocal sequence
        sequence += 1
        heapq.heappush(events, (float(when), sequence, kind, payload))

    def start_next(component, now):
        if busy[component] or not queue[component]:
            return
        i, stage, token = queue[component].popleft()
        busy[component] = True
        if stage == "prefill":
            base_ms = prefill_base_ms[i]
        elif stage == "transfer":
            base_ms = transfer_base_ms[i]
        else:
            base_ms = decode_base_ms[i, token]
        degraded = component == fault_component and fault_start_s <= now < fault_end_s
        duration_ms = float(base_ms * (slowdown if degraded else 1.0))
        push_event(
            now + duration_ms / 1000.0,
            "completion",
            (i, stage, token, component, now, duration_ms),
        )

    def enqueue(component, task, now):
        queue[component].append(task)
        start_next(component, now)

    for i, when in enumerate(arrival_times):
        push_event(when, "arrival", i)

    while events:
        now, _, kind, payload = heapq.heappop(events)
        if now > duration_s:
            break
        if kind == "arrival":
            i = payload
            enqueue(p_ids[i], (i, "prefill", 0), now)
            continue
        i, stage, token, component, start, duration_ms = payload
        busy[component] = False
        if not (missing_transfer and stage == "transfer"):
            obs_id = f"{request_ids[i]}-{stage}-{token}"
            observations.append(
                {
                    "run_id": run_id,
                    "obs_id": obs_id,
                    "request_id": request_ids[i],
                    "stage": stage,
                    "component": component,
                    "start_s": float(start),
                    "event_s": float(now),
                    "duration_ms": duration_ms,
                    "prompt_tokens": int(prompts[i]),
                    "uncached_tokens": int(uncached[i]),
                    "kv_bytes": int(kv_bytes[i]),
                    "context_tokens": int(
                        prompts[i] + (token if stage == "decode" else 0)
                    ),
                    "batch_size": 1,
                    "generated_so_far": int(token),
                    "config_id": "synthetic-v1",
                    "measurement_kind": "service",
                }
            )
        if stage == "prefill":
            enqueue(path_ids[i], (i, "transfer", 0), now)
        elif stage == "transfer":
            enqueue(d_ids[i], (i, "decode", 0), now)
        else:
            metric = "ttft_ms" if token == 0 else "itl_ms"
            reference_time = arrival_times[i] if token == 0 else previous_token_end[i]
            slo_rows.append(
                {
                    "run_id": run_id,
                    "request_id": request_ids[i],
                    "event_s": float(now),
                    "metric": metric,
                    "value_ms": float((now - reference_time) * 1000.0),
                }
            )
            previous_token_end[i] = now
            if token + 1 < response_lengths[i]:
                enqueue(d_ids[i], (i, "decode", token + 1), now)
            else:
                completed_requests.add(i)
        start_next(component, now)

    observation_columns = [
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
    obs = pd.DataFrame(observations, columns=observation_columns)
    if not obs.empty:
        obs = obs.sort_values(["event_s", "obs_id"]).reset_index(drop=True)
    slo_events = pd.DataFrame(
        slo_rows, columns=["run_id", "request_id", "event_s", "metric", "value_ms"]
    )
    if not slo_events.empty:
        slo_events = slo_events.sort_values("event_s", kind="stable").reset_index(
            drop=True
        )
    manifest_rows = []
    if fault_component is not None:
        manifest_rows.append(
            {
                "run_id": run_id,
                "episode_id": f"{run_id}-episode-1",
                "onset_s": float(fault_start_s),
                "end_s": float(fault_end_s),
                "true_component": fault_component,
                "kind": "mixed" if regime != "normal" else "degradation",
            }
        )
    manifest = pd.DataFrame(
        manifest_rows,
        columns=["run_id", "episode_id", "onset_s", "end_s", "true_component", "kind"],
    )
    return {
        "observations": obs,
        "arrivals": arrivals,
        "slo_events": slo_events,
        "manifest": manifest,
        "metadata": {
            "run_id": run_id,
            "seed": int(seed),
            "duration_s": float(duration_s),
            "regime": regime,
            "workload_shift_s": workload_shift_s,
            "fault_component": fault_component,
            "fault_start_s": fault_start_s,
            "fault_end_s": fault_end_s,
            "slowdown": float(slowdown),
            "sparse": bool(sparse),
            "missing_transfer": bool(missing_transfer),
            "n_arrivals": n,
            "n_completed_requests": len(completed_requests),
            "n_right_censored_requests": n - len(completed_requests),
            "synthetic_only": True,
        },
    }
