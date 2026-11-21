# Evidence, acceptance and handover

The run under review is [60b014c](https://github.com/sid15022/DAT-CS5903/commit/60b014c451cdcbe01be30ca16df4eb5e2312e083). In the linked [executed notebook](https://github.com/sid15022/DAT-CS5903/blob/60b014c451cdcbe01be30ca16df4eb5e2312e083/DAT_CS5903_Final/DAT-CS5903/SageMaker_vLLM.ipynb), section 3 contains the pilot JSON and route table. Section 4 contains the first collection marker. Section 5 has no saved output. The operator reports the study completed. Counts below describe recoverable evidence.

| Expected or planned test | Observed outcome | Artifact |
|---|---|---|
| Exact greedy IDs and verified external prefix on every GPU route | 4/4 routes, 128 loaded tokens each, eight identical IDs each | [Extracted measurements](../evaluation/gpu_pilot_routes.csv) and [stdlib extractor](../evaluation/check_gpu_pilot.py) |
| Keep label and future data out of the detector | Label rejection, timing and future-invariance tests pass | [Tests](../DAT_CS5903_Final/DAT-CS5903/tests/test_pipeline.py), [transcript](../evaluation/verification.log) |
| Refuse missing or ambiguous native forward evidence | Loopback tests reject invalid token/timing joins and broken streams | [Native tests](../DAT_CS5903_Final/DAT-CS5903/tests/test_native_server.py) |
| Unseen valid input and clean end-to-end path | New CPU fixture with seed 59317, 83 prompt tokens, 9 output tokens passed; fresh four-route CPU pilot passed | [Unseen case](../evaluation/unseen_handoff.py), [transcript](../evaluation/verification.log) |
| Retain all 36 simulated incidents including sparse and missing-path cases | 24 first-choice hits; 12 abstentions for conditioned detector | [Incident table](../DAT_CS5903_Final/DAT-CS5903/results/synthetic/incidents.csv) |
| At most one false episode per healthy cluster-hour (provisional) | Conditioned simulation 3/h; GPU rate unverified | [Simulation summary](../DAT_CS5903_Final/DAT-CS5903/results/synthetic/summary.csv), [issue 2](https://github.com/sid15022/DAT-CS5903/issues/2) |
| GPU overhead at most 5% throughput loss or p99 TTFT/ITL increase | Hardware comparison disabled in this run; CPU trials included | [CPU overhead](../DAT_CS5903_Final/DAT-CS5903/results/cpu-overhead/overhead_paired.csv), [issue 3](https://github.com/sid15022/DAT-CS5903/issues/3) |
| Full GPU fault/control localization | 12 intended injected incidents, 12 paired controls, four other held-out runs and 12 calibration runs; only first progress marker retained | [Issue 1](https://github.com/sid15022/DAT-CS5903/issues/1) |

## Audit the reported numbers

From repository root, run `python evaluation/check_gpu_pilot.py`. Its CSV should match evaluation/gpu_pilot_routes.csv. From the project directory use `python evaluation/acceptance.py` to check the frozen simulator and Qwen CPU summaries. Recreate all simulator tables with `python -m failslow suite --backend sim --workers 2 --reps 3 --seed 41201 --out local-runs/reproduced`. Compare summary.csv and incidents.csv with results/synthetic. For an incident replay use `python -m failslow replay --baseline results/synthetic/baseline.joblib --trace results/synthetic/runs/test-0-decode-1 --out local-runs/replay`. Use only the trusted baseline and its pinned scikit-learn environment.

For the missing GPU evidence, copy the existing study-2x2-r1-t90-q2.6 folder from dat-vllm-experiment on Studio. Retain study.json, calibration.json, pilot.json, run completion records, generation_pairs.csv, incidents.csv, healthy_runs.csv and summary.csv. Check the manifest backend, 40 trial IDs, paired request equality, real intervention counts and correct observation horizon before using a headline metric. Do not combine GPU traces with CPU or simulation baselines.

## Progress, feedback and viva

* 9 October: progress report by Siddharth V H records the conditioned-duration method, baseline comparisons, sparse/missing evidence cases, 8/12 initial first-choice hits and a future SageMaker testbed. The [original notebook](../Fail_Slow_LLM_Inference_Prototype.ipynb) provides reproducible prototype evidence.
* 16 October: original notebook repository milestone. 21 November: reproducible simulator and CPU evidence, native GPU pilot and [operator-uploaded notebook](https://github.com/sid15022/DAT-CS5903/commit/60b014c451cdcbe01be30ca16df4eb5e2312e083). Exact week-by-week logs, formal approval and scored mentor feedback were unavailable. No scored feedback item can be marked resolved on this evidence.
* Viva demonstration: trace one p1→d2 request from client TTFT to worker service spans. Explain why its first inter-token gap also includes transfer. Replay a held-out incident, identify the first accusation and inspect its preceding window. Explain why matched-peer needs another same-stage worker and why sparse traffic can imply unknown. Contrast the four verified GPU handoffs with the currently unverified GPU fault outcomes.
