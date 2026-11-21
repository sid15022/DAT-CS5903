# Passive fail-slow diagnosis in disaggregated LLM inference

**CS5903 · Project 25 · Group 5 · Siddharth V H (CS24MTECH15022)**

The detector estimates healthy service time from the work performed by each prefill worker, decode worker and key-value cache transfer path. It ranks components with persistent excess duration and abstains when observations are insufficient.

In a four-GPU vLLM experiment, the detector ranked the affected component first in 8 of 12 incidents. It abstained on four incidents with sparse traffic or missing transfer measurements. Median time to the first correct ranking was 12.5 s among the eight detected incidents. Three false alarms in 16 monitored healthy minutes exceeded the target of one per hour. [GPU results](DAT_CS5903_Final/DAT-CS5903/dat-vllm-experiment/study-2x2-r1-t90-q2.6/summary.csv)

[Project report](docs/Project_Report.pdf) · [Final presentation](docs/Final_Project_Presentation.pptx) · [Experimental evaluation and results](docs/Experimental_Evaluation_and_Results.md) · [Mathematical methods and worked GPU example](docs/Mathematical_Methods.md) · [Architecture decisions](docs/Architecture_Decision_Records.md) · [Evidence and reproduction](docs/Evidence.md) · [Installation and runbook](DAT_CS5903_Final/DAT-CS5903/README.md)

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

[GPU study archive and verification](https://github.com/sid15022/DAT-CS5903/issues/1) · [False-alarm evaluation](https://github.com/sid15022/DAT-CS5903/issues/2) · [Monitoring overhead](https://github.com/sid15022/DAT-CS5903/issues/3)
