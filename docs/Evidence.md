# Evidence, acceptance and handover

The [archived GPU run](https://github.com/sid15022/DAT-CS5903/commit/e20f53712de648b8c03784d2b1c8496146fc2d16) contains the held-out results, per-request traces and all 40 completion records. The standalone pilot and the study pilot are separate runs.

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

The 40-trial study on four NVIDIA A10G GPUs included 12 injected incidents. The 16 held-out no-fault trials each contributed 60 monitored seconds after warm-up. Fault and control runs matched on 2,821 completed generations. All 784 live rankings and explanations agreed with offline replay. [Method comparison](../DAT_CS5903_Final/DAT-CS5903/RESULTS.md#measured-four-gpu-sagemaker-study)

[Run manifest](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/dat-vllm-experiment/study-2x2-r1-t90-q2.6/study.json) · [Per-incident results](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/dat-vllm-experiment/study-2x2-r1-t90-q2.6/incidents.csv) · [Healthy exposure](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/dat-vllm-experiment/study-2x2-r1-t90-q2.6/healthy_runs.csv) · [Method summary](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/dat-vllm-experiment/study-2x2-r1-t90-q2.6/summary.csv) · [Paired service impact](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/dat-vllm-experiment/study-2x2-r1-t90-q2.6/paired_impact.csv)


## Progress, feedback and viva

| Date | Milestone | Record |
|---|---|---|
| 9 October 2026 | The simulator localized 8 of 12 incidents. The progress report proposed a SageMaker test of the same method. | [Progress report](Progress_Report_2026-10-09.docx) |
| 16 October 2026 | The prototype implementation was archived. | [Prototype notebook](../Fail_Slow_LLM_Inference_Prototype.ipynb) |
| 21 November 2026 | The native vLLM pilot verified KV reuse on four routes. The subsequent study recorded 40 GPU trials. | [SageMaker_vLLM.ipynb](../DAT_CS5903_Final/DAT-CS5903/SageMaker_vLLM.ipynb) · [Study archive](https://github.com/sid15022/DAT-CS5903/commit/e20f53712de648b8c03784d2b1c8496146fc2d16) |

Weekly logs and scored mentor feedback are unavailable. ADR-001 records the methodology decision retrospectively. [Architecture decisions](Architecture_Decision_Records.md)

For the demonstration, compare trial `test-0-decode-1` with its matched control. The first two threshold crossings occurred at 40 s and 50 s. The second triggered the first diagnosis, 12.5 s after the intervention began. The corresponding trace appears on page 7 of the [report](Project_Report.pdf).

## Release tag

[cs5903-final-2026](https://github.com/sid15022/DAT-CS5903/releases/tag/cs5903-final-2026) · [e0d9ee69ab107454cc464f5a012282cf5b7890ae](https://github.com/sid15022/DAT-CS5903/commit/e0d9ee69ab107454cc464f5a012282cf5b7890ae)

Verify the published revision from a repository clone:

```sh
git fetch origin refs/tags/cs5903-final-2026
git rev-parse 'FETCH_HEAD^{commit}'
```

The [historical GPU implementation bundle](Submission_Tag.bundle) records the earlier frozen implementation.
