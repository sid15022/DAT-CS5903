# Final submission message

**Subject** · CS5903 final project submission (Project 25, Group 5) — Passive Fail Slow Diagnosis in Disaggregated LLM Inference

| Course field | Submission |
|---|---|
| Project ID | 25 |
| Group number | 5 |
| Student | Siddharth V H (CS24MTECH15022) |
| Repository | https://github.com/sid15022/DAT-CS5903 |
| Stable submission | `cs5903-final-2026-gpu` · [dd51969ee5c8b6d78a137d29129f61f6a0790b07](https://github.com/sid15022/DAT-CS5903/tree/dd51969ee5c8b6d78a137d29129f61f6a0790b07) |
| Evaluated GPU run | [e20f53712de648b8c03784d2b1c8496146fc2d16](https://github.com/sid15022/DAT-CS5903/commit/e20f53712de648b8c03784d2b1c8496146fc2d16) |
| Demo | [Executed SageMaker notebook](../DAT_CS5903_Final/DAT-CS5903/SageMaker_vLLM.ipynb) and [offline verification output](../evaluation/gpu_study_verification.txt) |

The study ran Qwen2.5-0.5B-Instruct on vLLM across four NVIDIA A10G GPUs for 40 trials. The conditioned detector ranked the affected component first in 8 of 12 incidents, abstained in four and made zero wrong first accusations. Its median time to a first correct ranking was 12.5 seconds among detected incidents. Matched-peer also scored 8/12, with a 32.5-second median. During 16 monitored healthy minutes the conditioned method raised three false episodes and matched-peer raised none. GPU monitoring overhead remains unmeasured.

[Final report](Project_Report.pdf) · [Architecture decision records](Architecture_Decision_Records.md) · [Evidence, reproduction commands and limitations](Evidence.md)

## Before sending

An evaluator can verify the archived four-GPU study from the repository root with `python evaluation/check_gpu_study.py`.

The stable checkout above is available on GitHub. The annotated tag is included in [Submission_Tag.bundle](Submission_Tag.bundle). To publish the tag from an authenticated repository clone, run the following from its root.

```sh
git pull --ff-only origin main
git fetch ./docs/Submission_Tag.bundle refs/tags/cs5903-final-2026-gpu:refs/tags/cs5903-final-2026-gpu
git rev-parse 'cs5903-final-2026-gpu^{commit}'
git push origin refs/tags/cs5903-final-2026-gpu
git ls-remote origin 'refs/tags/cs5903-final-2026-gpu*'
```

The peeled tag must match the stable submission SHA above.
