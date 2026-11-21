# Architecture decision record

| Decision | Reason | Consequence |
|---|---|---|
| Retain the simulator alongside native vLLM | Repeatable interventions and ground truth make algorithm comparisons possible | Simulation accuracy must be reported separately from hardware accuracy |
| Pin Qwen and use one native server per GPU | A small open checkpoint makes independent prefill and decode workers practical | Single-sequence measurements do not characterize continuous batching |
| Use block-aligned file-backed KV transfer with a load receipt | A single host can verify state reuse and greedy equivalence on every route | Transfer measurements describe local file copies. RDMA requires another connector and validation |
| Fit only on healthy runs and freeze calibration | Fault labels should enter evaluation only | A new topology or workload beyond training support needs new healthy evidence |
| Preserve abstentions and matched controls | A silent detector or an always-alarming detector can otherwise look successful | Count misses and false episodes alongside localization and delay |
