# Final implementation report

Submission tag: cs5903-final-2026. Tagged commit: 98c63b2c87757865c2300959c129461e3b186b95 (expanded in the submission ZIP). In a Git checkout, run git rev-parse 'cs5903-final-2026^{commit}' to obtain the same SHA.

## Research question

Can passive observations separate changes in offered work from a slow prefill worker, decode worker or transfer path? The observer measures excess stage service time after conditioning on the work presented to that stage. The [walkthrough](PIPELINE.md) follows an individual request through collection, calibration and diagnosis.

## Outcomes

The conditioned method identified the correct first component in 24 of 36 simulated incidents. It recorded six false episodes over two healthy simulated hours. The matched-peer baseline achieved the same first-choice count with one false episode, but its median new-episode delay was 25 seconds compared with 15 seconds for the conditioned method. Sparse traffic and withheld transfer telemetry led to abstention. No single method dominated these measures.

The real Qwen CPU experiment recorded 240 requests and 934 model forwards. In two held-out interventions it identified prefill first in the prefill trial and initially accused prefill in the decode trial. It ranked decode correctly after 2.14 seconds. Both paired controls produced a false episode. This small single-engine study cannot establish accuracy on a disaggregated GPU deployment. The separate NumPy pilot verifies cache equivalence with controlled worker processes. GPU localization, transport performance and overhead await hardware execution.

## Milestone evidence and traceability

| Recorded milestone | Evidence | Requirement addressed |
|---|---|---|
| October 16 prototype | Git commit 5295ac7 and Fail_Slow_LLM_Inference_Prototype.ipynb | Original simulator and diagnostic concept |
| November 21 pipeline | Commit 1ade442 and results/synthetic | Frozen splits, baseline comparisons, paired evaluation and reproducible CPU integration |
| November 21 native model | Commit febc300 and results/qwen-vllm-cpu | Retain simulator and diagnose actual open-weight vLLM inference |
| Submission handover | Tagged checkout, evaluator script and operational guide | Reproducible evaluation and independently restorable source |

The dates reflect available evidence. Intervening weeks have no separately verified weekly logs. See [numerical results](../RESULTS.md), [architecture decisions](DECISIONS.md), [operations](HANDOVER.md) and [unresolved work](OPEN_WORK.md).
