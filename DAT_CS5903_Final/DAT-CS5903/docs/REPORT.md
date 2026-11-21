# Final implementation report

**Final submission** · [cs5903-final-2026](https://github.com/sid15022/DAT-CS5903/releases/tag/cs5903-final-2026) · [e0d9ee69ab107454cc464f5a012282cf5b7890ae](https://github.com/sid15022/DAT-CS5903/commit/e0d9ee69ab107454cc464f5a012282cf5b7890ae)

## Research question

Can passive observations separate changes in offered work from a slow prefill worker, decode worker or transfer path? The observer measures excess stage service time after conditioning on the work presented to that stage. The [walkthrough](PIPELINE.md) follows an individual request through collection, calibration and diagnosis.

## Outcomes

The conditioned method identified the correct first component in 24 of 36 simulated incidents. It recorded six false episodes over two healthy simulated hours. The matched-peer baseline achieved the same first-choice count with one false episode, but its median new-episode delay was 25 seconds compared with 15 seconds for the conditioned method. Sparse traffic and withheld transfer telemetry led to abstention. No single method dominated these measures.

The real Qwen CPU experiment recorded 240 requests and 934 model forwards. In two held-out interventions it identified prefill first in the prefill trial and initially ranked prefill first in the decode trial. It ranked decode correctly after 2.14 seconds. Both paired controls produced a false episode. This small single-engine study cannot establish accuracy on a disaggregated GPU deployment. The separate NumPy pilot verifies cache equivalence with controlled worker processes. The subsequent four-GPU study is documented below.

## Milestone evidence and traceability

| Recorded milestone | Evidence | Requirement addressed |
|---|---|---|
| October 16 prototype | Git commit 5295ac7 and Fail_Slow_LLM_Inference_Prototype.ipynb | Original simulator and diagnostic concept |
| November 21 pipeline | Commit 1ade442 and results/synthetic | Frozen splits, baseline comparisons, paired evaluation and reproducible CPU integration |
| November 21 native model | Commit febc300 and results/qwen-vllm-cpu | Retain simulator and diagnose actual open-weight vLLM inference |
| Submission handover | Tagged checkout, evaluator script and operational guide | Reproducible evaluation and independently restorable source |

Related documentation: [numerical results](../RESULTS.md), [architecture decisions](DECISIONS.md), [operations](HANDOVER.md) and [unresolved work](OPEN_WORK.md).

## Measured four-GPU SageMaker study

The pilot and full study ran in [SageMaker_vLLM.ipynb](../SageMaker_vLLM.ipynb). Sections 3–5 contain the recorded outputs.

Across 12 held-out GPU incidents, the conditioned detector and matched-peer baseline each ranked the affected component first eight times. Median time to the first correct ranking was 12.5 s for the conditioned detector and 32.5 s for matched-peer. Over 16 healthy minutes the methods raised three and zero false alarms, respectively. [Method comparison and raw data](../RESULTS.md#measured-four-gpu-sagemaker-study) · [Final project report](../../../docs/Project_Report.pdf)
