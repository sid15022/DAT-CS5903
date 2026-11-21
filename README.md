# Passive fail-slow diagnosis in disaggregated LLM inference

**Final submission** · [cs5903-final-2026](https://github.com/sid15022/DAT-CS5903/releases/tag/cs5903-final-2026) · [e0d9ee69ab107454cc464f5a012282cf5b7890ae](https://github.com/sid15022/DAT-CS5903/commit/e0d9ee69ab107454cc464f5a012282cf5b7890ae)

**CS5903 · Project 25 · Group 5 · Siddharth V H (CS24MTECH15022)**

The detector estimates healthy service time from the work performed by each prefill worker, decode worker and key-value cache transfer path. It ranks components with persistent excess duration and abstains when observations are insufficient.

In a four-GPU vLLM experiment, the detector ranked the affected component first in 8 of 12 incidents. It abstained on four incidents with sparse traffic or missing transfer measurements. Median time to the first correct ranking was 12.5 s among the eight detected incidents. Three false alarms in 16 monitored healthy minutes exceeded the target of one per hour. [GPU results](DAT_CS5903_Final/DAT-CS5903/dat-vllm-experiment/study-2x2-r1-t90-q2.6/summary.csv)

[Project report](docs/Project_Report.pdf) · [Final presentation](docs/Final_Project_Presentation.pptx) · [Experimental evaluation and results](docs/Experimental_Evaluation_and_Results.md) · [Mathematical methods and worked GPU example](docs/Mathematical_Methods.md) · [Architecture decisions](docs/Architecture_Decision_Records.md) · [Evidence and reproduction](docs/Evidence.md) · [Installation and runbook](DAT_CS5903_Final/DAT-CS5903/README.md)

## Repository guide

| Looking for | Start here |
|---|---|
| Documentation | [Submission documents](docs/) include the [project report](docs/Project_Report.pdf) and [architecture decisions](docs/Architecture_Decision_Records.md). The [pipeline walkthrough](DAT_CS5903_Final/DAT-CS5903/docs/PIPELINE.md) traces one request through diagnosis. |
| Evaluation | [GPU evidence checks](evaluation/) verify the archived pilot and study. The [CPU acceptance script](DAT_CS5903_Final/DAT-CS5903/evaluation/acceptance.py) checks frozen CPU and simulator results. |
| Tests | [Test suite](DAT_CS5903_Final/DAT-CS5903/tests/) covers the diagnosis pipeline, KV handoff and HTTP streaming. See [Quick demonstration](#quick-demonstration) for the test command and working directory. |
| Source | [Python package](DAT_CS5903_Final/DAT-CS5903/failslow/) contains the collector, workers and detector. Start at the [CLI entry point](DAT_CS5903_Final/DAT-CS5903/failslow/__main__.py) to follow a pilot, study or replay. |
| Operational handover | [Evaluator and operator guide](DAT_CS5903_Final/DAT-CS5903/docs/HANDOVER.md) covers service lifecycle, recovery, logs, access and costs. |

### Operating a live run

- **Start, stop or reset.** From `DAT_CS5903_Final/DAT-CS5903`, start the Docker pilot with `bash run_gpu.sh pilot`. To interrupt it, identify the experiment container with `docker ps` and run `docker stop <container-id>`. Use a fresh output directory for a clean run. Preserve completed results and confirm the workers have exited before clearing an abandoned KV spool. The handover also covers stopping Studio spaces and managed training jobs. [Start, stop and reset](DAT_CS5903_Final/DAT-CS5903/docs/HANDOVER.md#start-stop-and-reset)
- **Locate logs and troubleshoot.** Open `server-logs/p1.log` and `server-logs/d1.log` inside the pilot or suite output directory. For startup failures, check visible GPUs and free memory. For a KV mismatch, inspect the failing route’s worker logs, spool mount and loaded-token receipt. For an empty ranking, check observation counts in `component_scores.csv.gz`. [Failure diagnosis](DAT_CS5903_Final/DAT-CS5903/docs/HANDOVER.md#failure-diagnosis-and-access)
- **Accounts, quotas and costs.** The archived checks run locally without an AWS account or GPU. A fresh GPU run needs compatible hardware and access to packages and public model weights, or populated caches. Studio requires an AWS account, space permissions, GPU quota and regional capacity. Managed jobs also require an execution role, ECR image access and S3 output permissions. The four-GPU notebook schedules 60 minutes of traffic plus setup and evaluation. Budget for compute and storage, export results and stop the space. If GPU access is unavailable, use the [archived study demo](#review-the-executed-gpu-study). [Access requirements and budget](DAT_CS5903_Final/DAT-CS5903/docs/HANDOVER.md#failure-diagnosis-and-access)

## Review the executed GPU study

Open [SageMaker_vLLM.ipynb](DAT_CS5903_Final/DAT-CS5903/SageMaker_vLLM.ipynb) to inspect the saved SageMaker run without starting a GPU. Section 3 verifies KV-cache reuse on all four routes. Section 4 shows the 40 completed trials and compares the four diagnosis methods. Section 5 shows example generations, first-ranked components, detection delays and matched controls.

### Reproduce the study on SageMaker

1. **Prepare the space.** Start a SageMaker Studio JupyterLab GPU space with four visible NVIDIA A10Gs to reproduce the submitted topology. The notebook also supports two Ampere-or-newer GPUs for a smaller one-route run. Allow access to Python packages and the Qwen checkpoint, and provide writable storage for the environment, model cache and traces.
2. **Open the notebook in a fresh working directory.** Download a copy of [SageMaker_vLLM.ipynb](DAT_CS5903_Final/DAT-CS5903/SageMaker_vLLM.ipynb) there to keep the archived evidence separate. Run sections 1–5 in order in the same kernel. Section 1 extracts the bundled experiment source. Section 2 checks the GPUs and installs the pinned dependencies in an isolated Python 3.10–3.12 environment. It can provision Python 3.12 if the kernel uses a different version.
3. **Choose the run in section 2 before executing it.** For a handoff check alone, set `PILOT_ONLY = True`. To reproduce the submitted study, use the settings below. On four GPUs, `None` selects 90 seconds per trial.

   ```python
   PILOT_ONLY = False
   REPETITIONS = 1
   SECONDS_PER_TRIAL = None
   MEASURE_OVERHEAD = False
   REQUESTS_PER_SECOND = 2.6
   ```

4. **Verify serving in section 3.** Four private vLLM servers run Qwen2.5-0.5B-Instruct in FP16. GPUs 0 and 1 handle prefill, and GPUs 2 and 3 handle decode. For each of the four routes, check that decode loaded 128 cached tokens from the 129-token prompt and that all eight generated token IDs matched unsplit execution. If the pilot fails, inspect its `server-logs/` before proceeding.
5. **Run the study and inspect the results.** Section 4 schedules 12 healthy calibration trials, 12 injected incidents, 12 matched controls and four additional healthy trials. The 40 × 90 seconds cover **60 minutes of scheduled traffic**. Environment installation, model startup, pilot checks and evaluation take additional time. Section 5 reads the resulting tables. Open `study.json` for the configuration, `incidents.csv` for individual diagnoses and `summary.csv` for the baseline comparison. The submitted run ranked the target first in 8 of 12 incidents. [Archived study directory](DAT_CS5903_Final/DAT-CS5903/dat-vllm-experiment/study-2x2-r1-t90-q2.6)
6. **Save and stop.** Save the executed notebook and the `dat-vllm-experiment/` output directory before stopping the Studio GPU space to end compute charges.

## Quick demonstration

Start at the repository root on Linux with Python 3.12. The first two checks read archived outputs using the Python standard library. The remaining commands install the CPU environment and exercise the implementation:

```sh
python evaluation/check_gpu_pilot.py
python evaluation/check_gpu_study.py
cd DAT_CS5903_Final/DAT-CS5903
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r locks/cpu-py312-linux.txt
python -m unittest discover -s tests -v
python evaluation/acceptance.py
python ../../evaluation/unseen_handoff.py
python -m failslow pilot --backend numpy --workers 2 --out local-runs/demo-pilot
```

The first command reads the recorded notebook and prints four rows with 128 loaded tokens and exact matches. The second verifies all 40 GPU trials, 2,821 matched generations and 784 live/offline decisions. [Recorded GPU verification](evaluation/gpu_study_verification.txt). The CPU commands should pass 11 tests, verify the frozen results, exercise a seeded 83-token prompt and produce matching split/unsplit token IDs on every pilot route. [Recorded checks](evaluation/verification.log).

## Architecture and example diagnosis

A coordinator routes each prompt to a prefill worker. The worker produces the first token and exports its KV cache. The coordinator copies the cache files to the decode worker, which loads the cached prefix and streams the remaining tokens. Request IDs join model-forward timings, copy durations and client token receipts.

The detector fits expected stage durations on healthy traffic, scales the excess durations and calibrates alert thresholds on separate healthy runs. Every 10 seconds it examines the preceding 30 seconds and ranks eligible workers and paths. Sparse traffic or missing measurements can produce an abstention. Fault labels enter during offline evaluation. [Request and diagnosis walkthrough](DAT_CS5903_Final/DAT-CS5903/docs/PIPELINE.md)

For a reproducible input/output example, run `python evaluation/check_gpu_study.py` from the repository root. Its input is the committed 40-trial study. The conditioned-detector line should read:

```text
conditioned: 8/12 first correct, 4 abstentions, 3 false episodes in 16 min; median correct delay 12.5 s
```

## Dependencies and configuration

The CPU demo and GPU serving runtime use separate environments. The Linux Python 3.12 [CPU lock](DAT_CS5903_Final/DAT-CS5903/locks/cpu-py312-linux.txt) pins NumPy 2.3.5 and scikit-learn 1.8.0. The [GPU lock](DAT_CS5903_Final/DAT-CS5903/locks/gpu-py312-linux.txt) pins vLLM 0.10.2 and Transformers 4.55.4. The GPU study uses Qwen2.5-0.5B-Instruct at revision `7ae557604adf67be50417f59c2c2f167def9a775`.

For the Docker GPU launcher, install Docker and the NVIDIA Container Toolkit on Linux with a compatible driver. From `DAT_CS5903_Final/DAT-CS5903`, run `bash run_gpu.sh pilot` for one prefill and one decode worker on two GPUs. The launcher builds the image on first use. The following optional shell variables apply to this launcher. [Example shell configuration](DAT_CS5903_Final/DAT-CS5903/.env.example)

| Variable | Default | Purpose |
|---|---|---|
| `WORKERS` | `1` | Workers per stage. Set to 2 for four GPUs. |
| `DAT_BACKEND` | `vllm-http` | Serving backend passed to the experiment. |
| `DAT_RESULTS` | `./gpu-runs` | Host directory for outputs. |
| `DAT_KV_SPOOL` | `$DAT_RESULTS/spool` | Writable host directory for KV transfers. |
| `DAT_MODEL_CACHE` | `$HOME/.cache/huggingface` | Persistent model download cache. |
| `DAT_IMAGE` | `dat-failslow:http-v0.10.2` | Docker image tag to build or reuse. |
| `CUDA_VISIBLE_DEVICES` | Unset | Optional visible GPU selection. |

## Data setup and reproducibility

The experiment stores traces, manifests, fitted baselines and result tables in files. Database provisioning, schema migrations and database seeding are unnecessary. Archived checks read the committed fixtures directly. To generate fresh seeded simulator data after the CPU setup above, run the following from `DAT_CS5903_Final/DAT-CS5903`:

```sh
python -m failslow suite --backend sim --out local-runs/synthetic
```

## Limitations and open work

The four-GPU study used one host, file-backed KV transfers, serialized requests per worker and controlled software delays. Its path measurements include filesystem effects. Physical NIC/RDMA faults, multi-host timing and continuous batching need separate experiments. Healthy monitoring covered 16 minutes, and the submitted run did not measure observer overhead. The 12.5-second median covers the eight detected incidents. [Experimental evaluation](docs/Experimental_Evaluation_and_Results.md) · [Operator handover](DAT_CS5903_Final/DAT-CS5903/docs/HANDOVER.md)

[GPU study archive and verification](https://github.com/sid15022/DAT-CS5903/issues/1) · [False-alarm evaluation](https://github.com/sid15022/DAT-CS5903/issues/2) · [Monitoring overhead](https://github.com/sid15022/DAT-CS5903/issues/3)
