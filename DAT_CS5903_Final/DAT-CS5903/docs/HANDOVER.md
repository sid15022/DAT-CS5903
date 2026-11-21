# Evaluator and operator handover

## First demonstration

From the repository root use Python 3.12 on Linux. Create a virtual environment and install the CPU lock with `python -m pip install -r locks/cpu-py312-linux.txt`. Run `python evaluation/acceptance.py` for the frozen evidence and `python -m unittest discover -s tests -v` for executable checks. Expect PASS and 11 successful tests. Open Study_Results.ipynb for the worked results. Run the README simulator command to regenerate tables in a fresh directory. Seeds, split membership, thresholds and run lengths are in results/synthetic/study.json and calibration.json. The default study seed is 41201.

## What the tests cover

test_pipeline.py checks label isolation, duplicate rejection, observation availability, timestamp semantics, unsupported evidence, paired workloads, lack of same-stage peers and an end-to-end NumPy worker/cache handoff. test_native_server.py exercises a real loopback HTTP connection with fixture responses and rejects ambiguous token chunks, incomplete streams, missing forward records and mismatched identities. The hardware acceptance test is the native pilot. A passing fixture alone gives no GPU timing result.

## Start, stop and reset

For Studio upload SageMaker_vLLM.ipynb to an appropriately sized GPU space. Run the setup and pilot cells first. Use Run All for calibration and incident evaluation. For Docker use the README pilot and suite commands. The job owns its local HTTP workers. Normal exit and exception cleanup terminate worker process groups. If interrupted in Studio, inspect running processes with ps and stop the experiment subprocess and its owned worker groups, or stop the Studio space. For Docker identify the experiment with docker ps and use docker stop on that container. For a submitted managed job use aws sagemaker stop-training-job --training-job-name NAME. Stop the Studio space separately after collecting results to end compute charges.

Use a fresh output directory to reset. Reusing an existing directory with identical configuration resumes completed trials. Preserve results before removing temporary spool files and only clear an abandoned spool after confirming its workers have exited. There is no database, migration, or seed service. The seeded workload generator and run manifests provide the experiment data.

## Failure diagnosis and access

* Startup failure: inspect server-logs/p1.log and d1.log under the pilot or suite directory. Check nvidia-smi, GPU count, architecture, free GPU memory and the pinned vLLM version. A single T4 cannot run this split configuration.
* Weight or image download failure: check registry and Hugging Face connectivity, free disk, and the model cache mount. The model is public. Docker needs NVIDIA runtime access. Studio needs an AWS account with GPU instance quota, appropriate region capacity and space permissions. Managed jobs also need ECR pull, execution-role and S3 write access.
* Handoff mismatch: inspect the failing route, connector version, prefix-cache settings, spool mount and worker log. Retain both token sequences and the loaded-token receipt. A failed pilot invalidates the hardware experiment.
* Empty rankings: check offered traffic, per-component support, thresholds and missing observations in component_scores.csv.gz. Compare paired controls. An abstention is an evaluated outcome.
* Budget: the four-GPU notebook default schedules 80 minutes of traffic plus setup and analysis. The full three-repetition four-worker CLI study schedules 4.8 hours plus overhead. GPU time, storage and managed output costs depend on region and account. A pilot-only run is available.
