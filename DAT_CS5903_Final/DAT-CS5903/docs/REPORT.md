# Final implementation report

Historical implementation tag: cs5903-final-2026. Tagged commit: 98c63b2c87757865c2300959c129461e3b186b95 (expanded in the submission ZIP). In a Git checkout, run git rev-parse 'cs5903-final-2026^{commit}' to obtain the same SHA.

## Research question

Can passive observations separate changes in offered work from a slow prefill worker, decode worker or transfer path? The observer measures excess stage service time after conditioning on the work presented to that stage. The [walkthrough](PIPELINE.md) follows an individual request through collection, calibration and diagnosis.

## Outcomes

The conditioned method identified the correct first component in 24 of 36 simulated incidents. It recorded six false episodes over two healthy simulated hours. The matched-peer baseline achieved the same first-choice count with one false episode, but its median new-episode delay was 25 seconds compared with 15 seconds for the conditioned method. Sparse traffic and withheld transfer telemetry led to abstention. No single method dominated these measures.

The real Qwen CPU experiment recorded 240 requests and 934 model forwards. In two held-out interventions it identified prefill first in the prefill trial and initially accused prefill in the decode trial. It ranked decode correctly after 2.14 seconds. Both paired controls produced a false episode. This small single-engine study cannot establish accuracy on a disaggregated GPU deployment. The separate NumPy pilot verifies cache equivalence with controlled worker processes. The subsequent four-GPU study is documented below.

## Milestone evidence and traceability

| Recorded milestone | Evidence | Requirement addressed |
|---|---|---|
| October 16 prototype | Git commit 5295ac7 and Fail_Slow_LLM_Inference_Prototype.ipynb | Original simulator and diagnostic concept |
| November 21 pipeline | Commit 1ade442 and results/synthetic | Frozen splits, baseline comparisons, paired evaluation and reproducible CPU integration |
| November 21 native model | Commit febc300 and results/qwen-vllm-cpu | Retain simulator and diagnose actual open-weight vLLM inference |
| Submission handover | Tagged checkout, evaluator script and operational guide | Reproducible evaluation and independently restorable source |

The dates reflect available evidence. Intervening weeks have no separately verified weekly logs. See [numerical results](../RESULTS.md), [architecture decisions](DECISIONS.md), [operations](HANDOVER.md) and [unresolved work](OPEN_WORK.md).

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

[Current canonical report](../../../docs/Project_Report.pdf)
