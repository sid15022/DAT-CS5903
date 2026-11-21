# Final submission message

**Subject** · CS5903 final project submission (Project 25, Group 5) — Passive Fail Slow Diagnosis in Disaggregated LLM Inference

| Course field | Submission |
|---|---|
| Project ID | 25 |
| Group number | 5 |
| Student | Siddharth V H (CS24MTECH15022) |
| Repository | https://github.com/sid15022/DAT-CS5903 |
| Stable submission | [cs5903-final-2026](https://github.com/sid15022/DAT-CS5903/releases/tag/cs5903-final-2026) · [e0d9ee69ab107454cc464f5a012282cf5b7890ae](https://github.com/sid15022/DAT-CS5903/commit/e0d9ee69ab107454cc464f5a012282cf5b7890ae) |
| Evaluated GPU run | [e20f53712de648b8c03784d2b1c8496146fc2d16](https://github.com/sid15022/DAT-CS5903/commit/e20f53712de648b8c03784d2b1c8496146fc2d16) |
| Executed GPU notebook | [SageMaker_vLLM.ipynb](../DAT_CS5903_Final/DAT-CS5903/SageMaker_vLLM.ipynb) · Sections 3–5 contain the saved pilot, 40-trial study and evaluation outputs. [Offline verification](../evaluation/gpu_study_verification.txt) |

The four-GPU vLLM study included 12 held-out incidents. The detector ranked the affected component first in eight and abstained in four. Among the eight detected incidents, median time to the first correct ranking was 12.5 s, compared with 32.5 s for the matched-peer baseline. Over 16 monitored healthy minutes, the detector raised three false alarms and matched-peer raised none.

[Final report](Project_Report.pdf) · [Architecture decisions](Architecture_Decision_Records.md) · [Reproduction and evidence](Evidence.md) · [Historical GPU implementation bundle](Submission_Tag.bundle)
