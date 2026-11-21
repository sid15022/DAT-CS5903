# Architecture decision records

Passive Fail Slow Diagnosis in Disaggregated LLM Inference · CS5903 · Siddharth V H, CS24MTECH15022. This log was documented on **21 November 2026**. Records are retrospective where the original decision timestamp is unavailable. The October progress report and the corresponding [prototype](../Fail_Slow_LLM_Inference_Prototype.ipynb) are the historical evidence. Implementation links point into this repository revision.

## ADR-001 · Retain healthy-conditioned service duration as the diagnosis method

**Date, status and scope.** Recorded 21 November. Accepted in the implementation. The progress report dated 9 October describes this method and reports results from it. The final implementation reaffirms that direction. A timestamped declaration by 9 October at 11:59 PM IST and formal mentor sign-off were not supplied. No earlier ADR is superseded.

**Context and drivers.** A healthy long prompt and a slowed GPU can both inflate TTFT. Fault labels are scarce, and the observer must operate on signals available at decision time.

**Options.** (1) Alert on historical component latency. Cheap, but a changed work mix shifts latency. (2) Rank excess above a work-conditioned healthy reference, with an evidence threshold. Requires healthy calibration. (3) Train a supervised fault classifier. This would require representative faults and risks learning the injection policy.

**Decision and rationale.** Fit median and p95 log-service quantiles on healthy runs, then require supported, persistent excess before ranking. Measure user-facing latency and offered demand independently. This keeps training feasible and makes a long prompt an explicit covariate.

**Consequences.** Added disjoint fitting, scale and threshold partitions. The implementation moved from synthetic validation to actual Qwen inference, as proposed in October. The shorter single-host GPU study reduces exposure and repeats. It retains the standard window and incident denominator. Unsupported observations can lead to unknown. Frozen evidence gave conditioned and matched-peer identical first-choice accuracy and different false-alarm costs.

**Evidence and validation.** [Model](../DAT_CS5903_Final/DAT-CS5903/failslow/model.py), [detector](../DAT_CS5903_Final/DAT-CS5903/failslow/detector.py), [36-incident summary](../DAT_CS5903_Final/DAT-CS5903/results/synthetic/summary.csv).

**Revisit trigger.** Sustained false episodes above one per healthy GPU hour on an independent test, or systematic abstention on supported workload strata. Recalibrate on development data and reserve fresh test runs for any revised rule.

## ADR-002 · Use passive, request-correlated stage observations

**Date, status and scope.** Recorded 21 November. Accepted for GPU collection and monitoring. No predecessor.

**Context and drivers.** Queueing, transfer and decode all influence the first inter-token gap. Client latency alone permits several competing diagnoses.

**Options.** Client-only TTFT/ITL has the lowest integration cost. Worker-tagged service spans and client receipts require instrumentation but can separate a worker from its route. Active probes add independent signals at the cost of extra traffic.

**Decision and rationale.** Join each streamed token to a worker forward and its request. Record transfer bytes and duration separately. Native endpoints bind to loopback. The workload generator supplies test traffic; the detector reads its observations passively.

**Consequences.** Missing or ambiguous joins invalidate a measured trial. Batch size one makes correlation tractable and limits generalization to production scheduling. Test fault labels remain isolated from training and prediction. A shared host monotonic clock removes the need to model cross-host skew for this experiment.

**Evidence and validation.** [Collector](../DAT_CS5903_Final/DAT-CS5903/failslow/collect.py), [native protocol tests](../DAT_CS5903_Final/DAT-CS5903/tests/test_native_server.py), [label and clock tests](../DAT_CS5903_Final/DAT-CS5903/tests/test_pipeline.py).

**Revisit trigger.** Multiple sequences sharing a forward, server-side token grouping, or moving a worker to a separate clock domain. Require a matching receipt and clock-uncertainty test before extending comparisons.

## ADR-003 · Pin Qwen on native vLLM and verify external KV reuse

**Date, status and scope.** Recorded 21 November. Accepted for the single-host GPU testbed. The simulator remains a separate execution mode.

**Context and drivers.** Reproducible ground truth is easy to generate in simulation. The project also needs proof that actual model computation can move between workers while preserving outputs.

**Options.** Simulation only maximizes control. An embedded engine simplifies process management. Native per-GPU vLLM services exercise the HTTP stream, distinct processes and a connector. A multi-host RDMA testbed would add network realism and placement and instrumentation requirements.

**Decision and rationale.** Use Qwen2.5-0.5B-Instruct with one vLLM 0.10.2 process per GPU. Use file-backed, block-aligned KV transfer on one host. Gate each route on a worker load receipt, positive transfer size and equality of split and unsplit greedy token IDs.

**Consequences.** The small checkpoint supports four replicas. Tested cache-copy latency includes host filesystem behavior. It cannot identify a physical NIC bottleneck. Simulation and CPU fixtures remain available when GPU quotas or model access are blocked.

**Evidence and validation.** [Qwen server](../DAT_CS5903_Final/DAT-CS5903/failslow/native_server.py), [connector](../DAT_CS5903_Final/DAT-CS5903/failslow/connector.py), [four observed A10G handoffs](../evaluation/gpu_pilot_routes.csv).

**Revisit trigger.** A multi-node deployment, connector or engine upgrade, or a model with a different cache layout. Repeat equivalence, receipt and timing tests on every new route and refit the baseline.

## ADR-004 · Separate manifests, observations and offline evidence

**Date, status and scope.** Recorded 21 November. Accepted for data handling, evaluation and handover.

**Context and drivers.** Labels must remain unavailable to the observer. Measurements can arrive late, and overlapping windows create strongly correlated data.

**Options.** A single labeled table eases plotting and invites leakage. A managed database adds persistence and operational dependencies. Isolated manifests and append-only per-run traces support local replay and independent auditing.

**Decision and rationale.** Keep the intervention manifest and paired seeds outside the detector view. Persist arrivals, service spans, SLO events, prediction journals and completion markers in run directories. Compare independent methods against the same held-out incidents. Record misses and false episodes alongside delay.

**Consequences.** No database migration or account is needed. Interrupted studies can resume; copying a notebook alone can omit trace files. Serialized joblib baselines require a trusted source. Generated prompts are synthetic. External access, retention and redaction policy are prerequisites before adapting the collector to private workloads.

**Evidence and validation.** [Schema and validation](../DAT_CS5903_Final/DAT-CS5903/failslow/data.py), [evaluation](../DAT_CS5903_Final/DAT-CS5903/failslow/evaluate.py), [archive recovery and acceptance criteria](https://github.com/sid15022/DAT-CS5903/issues/1).

**Revisit trigger.** Missing run artifacts after handover, incompatible schemas or untrusted serialized models. Introduce atomic export with a manifest and checksums, and evaluate a safer model interchange format before cross-tenant use.

## ADR-005 · Use matched fault/control runs and report operating-point trade-offs

**Date, status and scope.** Recorded 21 November. Accepted for the experimental design and provisional acceptance criteria.

**Context and drivers.** Demand and route choice affect which stages are observable. A correct diagnosis in an already-alarming window can overstate responsiveness.

**Options.** Compare aggregate latency before and after injection, which is simple but confounds changing demand. Pair fault and no-fault runs under identical offered workloads and routes, which costs extra runtime but exposes false accusations. Unpaired multi-seed testing samples more workloads within the same budget.

**Decision and rationale.** Pair controls and interventions, preserve whole-run splits, retain all incidents in the accuracy denominator and report healthy exposure separately. Count episode starts instead of repeatedly counting an active alarm. Report delays conditional on an observed episode and flag alarms already active at onset.

**Consequences.** The one-hour GPU configuration has one repetition per case/severity and excludes overhead pairs by default. A single short study cannot certify a long-run false-alarm rate. A rerun with another configuration needs its own manifest and baseline.

**Evidence and validation.** [Trial design](../DAT_CS5903_Final/DAT-CS5903/failslow/study.py), [matched controls](../DAT_CS5903_Final/DAT-CS5903/results/synthetic/healthy_runs.csv), [real CPU trials](../DAT_CS5903_Final/DAT-CS5903/results/qwen-vllm-cpu/summary.csv).

**Revisit trigger.** Too few eligible windows to calibrate or report a metric, or a wide exposure-based confidence interval spanning the proposed target. Collect additional independent runs before making a production claim.

## Validation update following GPU archive publication

On four NVIDIA A10Gs, the conditioned detector ranked the correct component first in 8 of 12 held-out GPU incidents. It abstained on the two sparse-worker and two hidden-transfer incidents. The median time to a correct top ranking was 12.5 s among detected incidents. Three false episodes in 16 monitored healthy minutes exceeded the provisional one-per-hour target. ADR-001 and ADR-005 retain their operating-point and support caveats. Paired controls yielded 2,821 matching generations. Live and offline logs agree on 784 rankings and explanations. [GPU evidence](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/dat-vllm-experiment/study-2x2-r1-t90-q2.6/summary.csv).
