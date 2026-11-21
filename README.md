# Passive fail-slow diagnosis in disaggregated LLM inference

**CS5903 · Project 25 · Group 5 · Siddharth V H (CS24MTECH15022)**

The detector estimates healthy service time from the work performed by each prefill worker, decode worker and key-value cache transfer path. It ranks components with persistent excess duration and abstains when observations are insufficient.

In a four-GPU vLLM experiment, the detector ranked the affected component first in 8 of 12 incidents. It abstained on four incidents with sparse traffic or missing transfer measurements. Median time to the first correct ranking was 12.5 s among the eight detected incidents. Three false alarms in 16 monitored healthy minutes exceeded the target of one per hour. [GPU results](DAT_CS5903_Final/DAT-CS5903/dat-vllm-experiment/study-2x2-r1-t90-q2.6/summary.csv)

[Project report](docs/Project_Report.pdf) · [Architecture decisions](docs/Architecture_Decision_Records.md) · [Evidence and reproduction](docs/Evidence.md) · [Installation and runbook](DAT_CS5903_Final/DAT-CS5903/README.md)

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

Use the [SageMaker notebook](DAT_CS5903_Final/DAT-CS5903/SageMaker_vLLM.ipynb) for the budgeted GPU study. On four GPUs, its defaults schedule 40 × 90 s, or one hour of traffic plus startup and evaluation. For a pilot only, set PILOT_ONLY to True. Studio needs GPU quota, storage and access to the checkpoint and packages. Stop the GPU space after saving outputs.

[GPU study archive and verification](https://github.com/sid15022/DAT-CS5903/issues/1) · [False-alarm evaluation](https://github.com/sid15022/DAT-CS5903/issues/2) · [Monitoring overhead](https://github.com/sid15022/DAT-CS5903/issues/3)
