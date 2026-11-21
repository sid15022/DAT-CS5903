# Evidence, acceptance and handover

The [archived GPU run](https://github.com/sid15022/DAT-CS5903/commit/e20f53712de648b8c03784d2b1c8496146fc2d16) contains the held-out results, per-request traces and all 40 completion records. The previously saved standalone pilot is a separate invocation from the pilot inside the suite.

| Expected or planned test | Observed outcome | Artifact |
|---|---|---|
| Exact greedy IDs and verified external prefix on every GPU route | 4/4 routes, 128 loaded tokens each, eight identical IDs each | [Extracted measurements](../evaluation/gpu_pilot_routes.csv) and [stdlib extractor](../evaluation/check_gpu_pilot.py) |
| Keep label and future data out of the detector | Label rejection, timing and future-invariance tests pass | [Tests](../DAT_CS5903_Final/DAT-CS5903/tests/test_pipeline.py), [transcript](../evaluation/verification.log) |
| Refuse missing or ambiguous native forward evidence | Loopback tests reject invalid token/timing joins and broken streams | [Native tests](../DAT_CS5903_Final/DAT-CS5903/tests/test_native_server.py) |
| Unseen valid input and clean end-to-end path | New CPU fixture with seed 59317, 83 prompt tokens, 9 output tokens passed; fresh four-route CPU pilot passed | [Unseen case](../evaluation/unseen_handoff.py), [transcript](../evaluation/verification.log) |
| Retain all 36 simulated incidents including sparse and missing-path cases | 24 first-choice hits; 12 abstentions for conditioned detector | [Incident table](../DAT_CS5903_Final/DAT-CS5903/results/synthetic/incidents.csv) |
| At most one false episode per healthy cluster-hour (provisional) | Conditioned simulation 3/h; GPU 11.25/h (3 episodes in 16 min); provisional target exceeded | [Simulation summary](../DAT_CS5903_Final/DAT-CS5903/results/synthetic/summary.csv), [issue 2](https://github.com/sid15022/DAT-CS5903/issues/2) |
| GPU overhead at most 5% throughput loss or p99 TTFT/ITL increase | Hardware comparison disabled in this run; CPU trials included | [CPU overhead](../DAT_CS5903_Final/DAT-CS5903/results/cpu-overhead/overhead_paired.csv), [issue 3](https://github.com/sid15022/DAT-CS5903/issues/3) |
| Full GPU fault/control localization | 12 injected incidents, 12 paired controls, four other held-out runs and 12 calibration runs verified. Conditioned 8/12, four abstentions | [Issue 1](https://github.com/sid15022/DAT-CS5903/issues/1) |

## Audit the reported numbers

From repository root, run `python evaluation/check_gpu_pilot.py`. Its CSV should match evaluation/gpu_pilot_routes.csv. From the project directory use `python evaluation/acceptance.py` to check the frozen simulator and Qwen CPU summaries. Recreate all simulator tables with `python -m failslow suite --backend sim --workers 2 --reps 3 --seed 41201 --out local-runs/reproduced`. Compare summary.csv and incidents.csv with results/synthetic. For an incident replay use `python -m failslow replay --baseline results/synthetic/baseline.joblib --trace results/synthetic/runs/test-0-decode-1 --out local-runs/replay`. Use only the trusted baseline and its pinned scikit-learn environment.

Run `python evaluation/check_gpu_study.py` from the repository root. It checks trial identity, calibration membership and baseline digest, applied interventions, raw paired generations, held-out live rankings and explanations, and the method summaries. [Verification output](../evaluation/gpu_study_verification.txt).

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


## Progress, feedback and viva

* 9 October: progress report by Siddharth V H records the conditioned-duration method, baseline comparisons, sparse/missing evidence cases, 8/12 initial first-choice hits and a future SageMaker testbed. The [original notebook](../Fail_Slow_LLM_Inference_Prototype.ipynb) provides reproducible prototype evidence.
* 16 October: original notebook repository milestone. 21 November: reproducible simulator and CPU evidence, native GPU pilot and [operator-uploaded notebook](https://github.com/sid15022/DAT-CS5903/commit/60b014c451cdcbe01be30ca16df4eb5e2312e083). The [subsequent archive commit](https://github.com/sid15022/DAT-CS5903/commit/e20f53712de648b8c03784d2b1c8496146fc2d16) supplies all 40 GPU trials. Exact week-by-week logs, formal approval and scored mentor feedback were unavailable. No scored feedback item can be marked resolved on this evidence.
* Viva demonstration: trace one p1→d2 request from client TTFT to worker service spans. Explain why its first inter-token gap also includes transfer. Replay a held-out incident, identify the first accusation and inspect its preceding window. Explain why matched-peer needs another same-stage worker and why sparse traffic can imply unknown. Explain why 8/12 first-choice hits and three false alarms can coexist; compare matched-peer at the same exposure.
