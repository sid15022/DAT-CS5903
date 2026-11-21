# Final submission message

**Subject** · CS5903 final project submission — Passive Fail Slow Diagnosis in Disaggregated LLM Inference

| Course field | Submission |
|---|---|
| Project ID | [enter official project ID] |
| Group number | [enter official group number] |
| Student | Siddharth V H (CS24MTECH15022) |
| Repository | https://github.com/sid15022/DAT-CS5903 |
| Stable submission | `cs5903-final-2026-gpu` · use `git rev-parse HEAD` in this frozen checkout for its full commit ID |
| Evaluated GPU run | [e20f53712de648b8c03784d2b1c8496146fc2d16](https://github.com/sid15022/DAT-CS5903/commit/e20f53712de648b8c03784d2b1c8496146fc2d16) |
| Demo | [Executed SageMaker notebook](../DAT_CS5903_Final/DAT-CS5903/SageMaker_vLLM.ipynb) and [offline verification output](../evaluation/gpu_study_verification.txt) |

The study ran Qwen2.5-0.5B-Instruct on vLLM across four NVIDIA A10G GPUs for 40 trials. The conditioned detector ranked the affected component first in 8 of 12 incidents, abstained in four and made zero wrong first accusations. Its median time to a first correct ranking was 12.5 seconds among detected incidents. Matched-peer also scored 8/12, with a 32.5-second median. During 16 monitored healthy minutes the conditioned method raised three false episodes and matched-peer raised none. GPU monitoring overhead remains unmeasured.

[Final report](Project_Report.pdf) · [Architecture decision records](Architecture_Decision_Records.md) · [Evidence, reproduction commands and limitations](Evidence.md)

## Before sending

Fill in the two course allocation fields and confirm the member list. An evaluator can verify the archived four-GPU study from the repository root with `python evaluation/check_gpu_study.py`.
