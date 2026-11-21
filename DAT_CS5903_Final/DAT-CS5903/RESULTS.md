# Experiment results

## Measured four-GPU SageMaker study

On four NVIDIA A10Gs, the conditioned detector ranked the correct component first in 8 of 12 held-out GPU incidents. It abstained on the two sparse-worker and two hidden-transfer incidents. The median time to a correct top ranking was 12.5 s among detected incidents. Three false episodes in 16 monitored healthy minutes exceeded the provisional one-per-hour target. The one-hour schedule includes calibration and injected trials. The false-alarm denominator counts only the 16 no-fault evaluation trials after their 30-second warm-ups.

| Method | Correct first / 12 | Abstentions | Wrong first | Median first correct¹ | False episodes / 16 min |
|---|---:|---:|---:|---:|---:|
| Conditioned | 8 | 4 | 0 | 12.5 s | 3 |
| Historical | 8 | 4 | 0 | 12.5 s | 4 |
| Work rate | 7 | 3 | 2 | 17.5 s | 2 |
| Matched peer | 8 | 4 | 0 | 32.5 s | 0 |

¹ Median among the eight incidents per method that eventually received a correct top ranking. The conditioned method identified both severities of prefill, decode and mixed faults after 12.5 s, and both visible transfer faults after 22.5 s. It matched historical latency on first-choice accuracy and median delay. With one repeat per case and severity the experiment establishes no general performance ordering.

All 40 trial IDs have matching completion records. There were 10,209 offered, 10,208 accepted and 10,198 completed requests, with 10 right-censored after acceptance. Paired runs matched on all 2,821 common completed prompts and output token sequences. All 784 held-out live rankings and explanations match the recorded offline results. Only three stable SLO-alarming windows were eligible; one explanation was correct. GPU monitoring overhead requires a separate paired on/off study.

[Run manifest](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/dat-vllm-experiment/study-2x2-r1-t90-q2.6/study.json) · [Per-incident results](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/dat-vllm-experiment/study-2x2-r1-t90-q2.6/incidents.csv) · [Healthy exposure](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/dat-vllm-experiment/study-2x2-r1-t90-q2.6/healthy_runs.csv) · [Method summary](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/dat-vllm-experiment/study-2x2-r1-t90-q2.6/summary.csv) · [Paired service impact](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/dat-vllm-experiment/study-2x2-r1-t90-q2.6/paired_impact.csv)

## Actual open-weight Qwen model on vLLM

A single vLLM CPU engine served **240 study requests** using the published Qwen2.5-0.5B-Instruct checkpoint. The collector recorded **934 model forwards**. Weights matched the published SHA-256. Two additional prompts outside the study produced “The capital of France is Paris.” and “7 times 8 is 56.”

| Held-out fault | First post-onset ranking | First correct top-1 after onset | Matched no-fault run |
|---|---|---:|---|
| Prefill forward slowed 3× | Prefill | 2.64 s | 1 false episode |
| Decode forward slowed 3× | Prefill | 2.14 s | 1 false episode |

The first ranking identified **one of two faults**. On the decode run, the detector switched from prefill to decode on the next one-second evaluation. On the prefill run, an earlier prefill alarm was already active before injection, cleared, then returned. The short controls show that a correct component label alone provides weak evidence of reliable fault attribution. All 64 matched fault/control request pairs had identical prompt hashes and output token IDs.

There were 48 healthy fitting requests, 32 for scale calibration, 32 for threshold calibration, and 128 held-out requests across two fault/control pairs. Each intervention occupied its chosen component during 16 successive requests. The detector used four-second windows, one-second cadence and two consecutive threshold crossings. The single CPU engine shares hardware between stages and exports no transfer telemetry. It supports a real-model timing and diagnosis check. The multi-GPU server study supplies separate workers and real KV handoffs. Recorded data and the frozen baseline are under `results/qwen-vllm-cpu/`.

## Simulator comparison

**The conditioned detector localized 24 of 36 simulated incidents on its first ranking.** The matched-peer method localized the same 24. The median time to a new diagnosis was 15 seconds for the conditioned detector and 25 seconds for matched-peer. Over two simulated healthy cluster-hours, conditioned raised six false episodes and matched-peer raised one. The evidence favors different methods depending on the cost of a false accusation versus a delayed diagnosis.

## Held-out simulation

The experiment used the DAT simulator, new base seed 41201, 12 independent healthy fitting/calibration runs and 84 held-out runs. The held-out set comprised 36 injected incidents, 36 paired controls and 12 healthy or workload-only runs. Each ran for 180 simulated seconds. Monitored healthy exposure excludes the initial 30-second window in each of the 48 no-fault runs.

| Method | First top-1 | First top-3 | No ranking | Wrong first | Median new episode delay | False episodes / 2 h |
|---|---:|---:|---:|---:|---:|---:|
| Conditioned quantiles | 24/36 | 24/36 | 12 | 0 | 15 s | 6 |
| Matched peer | 24/36 | 24/36 | 12 | 0 | 25 s | 1 |
| Historical latency | 12/36 | 12/36 | 24 | 0 | 25 s | 3 |
| Work rate | 16/36 | 16/36 | 17 | 3 | 25 s | 11 |

Each cell below retains all three independently seeded incidents for that case and severity. A miss counts against localization even when abstaining is the appropriate engineering response.

| Injected case | Target | 1.5× first top-1 | 3× first top-1 |
|---|---|---:|---:|
| Prefill worker | p1 | 3/3 | 3/3 |
| Decode worker | d1 | 3/3 | 3/3 |
| KV transfer path | l11 | 3/3 | 3/3 |
| Longer prompts plus decode slowdown | d1 | 3/3 | 3/3 |
| Sparse traffic to the affected worker | p2 | 0/3 | 0/3 |
| Transfer telemetry withheld | l11 | 0/3 | 0/3 |

The conditioned method abstained throughout the diagnosis windows of all 12 sparse or missing-telemetry incidents. Its eligible run-wide explanations were correct in 142 of 217 stable SLO-alarming windows (65.4%), and it assigned a substantive explanation in 148 (68.2%). Overlapping windows are correlated. The 15-second delay describes observed episodes and excludes unresolved incidents.

The conditioned rate of **3.0 false episodes per simulated healthy cluster-hour** exceeds the project's provisional target of one. An illustrative exact Poisson interval is 1.10–6.53 per hour under the assumption of independent events at a constant rate. Matched-peer's point estimate was 0.5 per hour with an interval of 0.013–2.79. The operating points differ, and the healthy exposure is too short to validate a production rate. Matched-peer also uses the fitted healthy median and needs a same-stage comparison component.

## Measured CPU execution

A separate run exercised the scheduler, process boundaries, KV serialization, copy and import, token timing, intervention harness, live observer and evaluation path. It used deterministic, untrained causal attention with NumPy on CPU. On each of the four worker pairs, split and unsplit execution produced the identical eight-token sequence. Each handoff copied 33,514 bytes and the decode worker loaded 129 cached tokens.

The 40-run short integration study completed 7,122 requests within their horizons and right-censored six. The conditioned method ranked the true component first in 7/12 incidents, emitted one wrong first ranking, and left four unresolved. During the mild prefill intervention, the first ranking incorrectly selected l22. It emitted ten false episodes in only 48 monitored healthy seconds. There were zero eligible stable SLO-alarming windows for explanation accuracy. The 1-second windows and single repetition per case are intended to test integration.

Across the 28 held-out CPU runs, 1,456 live decision records agreed with replay. Three paired, alternating 12-second on/off trials gave the following percentage changes with monitoring enabled. Positive values mean increases.

| Pair | Completed throughput | p99 TTFT | p99 ITL |
|---|---:|---:|---:|
| 1 | +0.22% | −5.41% | −11.45% |
| 2 | 0.00% | −48.99% | +18.21% |
| 3 | 0.00% | −7.54% | −11.22% |

Timing variation between short CPU runs is large. The second pair exceeds the 5% ITL target while the other two show lower measured ITL with the observer enabled. No GPU overhead conclusion follows from these trials.

## GPU evidence and reproducibility

The GPU implementation launches native vLLM 0.10.2 OpenAI-compatible servers, with Qwen2.5-0.5B-Instruct and the V1 shared-storage connector on one host. The executed [SageMaker_vLLM.ipynb](SageMaker_vLLM.ipynb) shows the pilot in section 3 and the full study and evaluation in sections 4–5. The study archive contains the underlying manifests, service measurements and paired generations. GPU monitoring overhead remains unmeasured.

Two complete executions using the same simulation seeds reproduced the same values in all eight evaluation tables, including every component score and decision.

The included results preserve all split assignments, thresholds, frozen models, per-incident outcomes, method decisions, support scores, SLOs and matched service-impact tables. All 40 measured CPU runs and the six overhead runs include canonical traces. The simulation includes completion records for all 96 runs and the raw trace pair for one decode fault and its control. The simulator regenerates the full raw dataset with the commands in README.md. One CPU journal preserves the original interleaving for live-tailing inspection.

The original repository notebook used a smaller 12-incident synthetic experiment. Its historical 8/12 result and the new 24/36 result come from different study designs. Arithmetic agreement in the percentages supplies no evidence of improvement between versions.
