# Final submission message

**Subject:** CS5903 final project submission — Passive Fail Slow Diagnosis in Disaggregated LLM Inference

Project ID: [enter official project ID]  
Group number: [enter official group number]  
Student: Siddharth V H (CS24MTECH15022)  
Repository: https://github.com/sid15022/DAT-CS5903  
Stable submission commit: a9b8036733f8dccf7720c6badc6f0c288ff09b47  
Tag name: cs5903-final-2026-handover  
Evaluated run commit: 60b014c451cdcbe01be30ca16df4eb5e2312e083  
[Project report](Project_Report.pdf) · [Architecture decision records](Architecture_Decision_Records.md) · [Stable checkout](https://github.com/sid15022/DAT-CS5903/tree/a9b8036733f8dccf7720c6badc6f0c288ff09b47)  
Demo: [executed SageMaker notebook](https://github.com/sid15022/DAT-CS5903/blob/60b014c451cdcbe01be30ca16df4eb5e2312e083/DAT_CS5903_Final/DAT-CS5903/SageMaker_vLLM.ipynb). A public inference endpoint is not required.

The GPU pilot verifies four prefill/decode routes with matching output tokens. GPU diagnostic accuracy from the full 40-trial study awaits the Studio artifact export. The report gives separate simulator and measured CPU comparisons.

## Owner checklist before sending

Fill in the two course allocation fields above and confirm the member list. The annotated tag is preserved in [Submission_Tag.bundle](Submission_Tag.bundle). From an authenticated full clone on the current main branch, publish it with:

```sh
git fetch origin main
git fetch ./docs/Submission_Tag.bundle refs/tags/cs5903-final-2026-handover:refs/tags/cs5903-final-2026-handover
git rev-parse 'cs5903-final-2026-handover^{commit}'
git push origin refs/tags/cs5903-final-2026-handover
git ls-remote --tags origin 'cs5903-final-2026-handover*'
```

The peeled tag must resolve to the stable submission commit above. The previously distributed October-prototype-derived package carried a separate historical tag. Confirm evaluator access to the stable checkout before sending. This text is prepared for the student; it has not been sent to course staff.
