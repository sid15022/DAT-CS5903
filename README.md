# Passive fail-slow diagnosis in disaggregated LLM inference

**CS5903 · Project 25 · Group 5 · Evaluator handover · Siddharth V H (CS24MTECH15022)**

Open the [project report](docs/Project_Report.pdf), [architecture decisions](docs/Architecture_Decision_Records.md) and [evidence index and handover](docs/Evidence.md). The [source and full runbook](DAT_CS5903_Final/DAT-CS5903/README.md) are under DAT_CS5903_Final/DAT-CS5903. On four NVIDIA A10Gs, the conditioned detector ranked the correct component first in 8 of 12 held-out GPU incidents. It abstained on the two sparse-worker and two hidden-transfer incidents. The median time to a correct top ranking was 12.5 s among detected incidents. Three false episodes in 16 monitored healthy minutes exceeded the provisional one-per-hour target. [Full GPU results](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/dat-vllm-experiment/study-2x2-r1-t90-q2.6/summary.csv).

## Quick demonstration

From a clean clone on Linux with Python 3.12:

```sh
python evaluation/check_gpu_pilot.py
python evaluation/check_gpu_study.py
cd DAT_CS5903_Final/DAT-CS5903
python -m venv .venv
. .venv/bin/activate
python -m pip install -r locks/cpu-py312-linux.txt
python -m unittest discover -s tests -v
python evaluation/acceptance.py
python ../../evaluation/unseen_handoff.py
python -m failslow pilot --backend numpy --workers 2 --out local-runs/demo-pilot
```

The first command reads the recorded notebook and prints four rows with 128 loaded tokens and exact matches. The second verifies all 40 GPU trials, 2,821 matched generations and 784 live/offline decisions. [Recorded GPU verification](evaluation/gpu_study_verification.txt). The CPU commands should pass 11 tests, verify the frozen results, exercise a seeded 83-token prompt and produce matching split/unsplit token IDs on every pilot route. [Recorded checks](evaluation/verification.log).

## Run, reproduce and operate

The [pipeline walkthrough](DAT_CS5903_Final/DAT-CS5903/docs/PIPELINE.md) follows a request from prefill to KV copy and decode, through time-ordered scoring to incident evaluation. [Runbook](DAT_CS5903_Final/DAT-CS5903/README.md) covers pinned requirements, CLI commands and simulator regeneration. [Operations](DAT_CS5903_Final/DAT-CS5903/docs/HANDOVER.md) covers logs, termination and restart. [Safe environment example](DAT_CS5903_Final/DAT-CS5903/.env.example) sets worker count, results, selected engine, KV spool, model cache and container image. The experiment writes files. Seeded manifests replace database seeding, so database migration is inapplicable.

Use the [existing SageMaker notebook](DAT_CS5903_Final/DAT-CS5903/SageMaker_vLLM.ipynb) for the budgeted GPU study. On four GPUs, its defaults schedule 40 × 90 s, or one hour of traffic plus startup and evaluation. For a pilot only, set PILOT_ONLY to True. Studio needs GPU quota, storage and access to the checkpoint and packages. Stop the GPU space after saving outputs. Generated notes in the original source archive describe an earlier handover. Use the repository-root report and evidence index for the current status.

[GPU study archive and verification](https://github.com/sid15022/DAT-CS5903/issues/1) · [False-alarm evaluation](https://github.com/sid15022/DAT-CS5903/issues/2) · [Monitoring overhead](https://github.com/sid15022/DAT-CS5903/issues/3)
