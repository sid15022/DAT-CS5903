# Passive Fail-Slow Diagnosis in Disaggregated LLM Inference

**Submission guide.** Start with the [step-by-step pipeline](docs/PIPELINE.md), [final report](docs/REPORT.md), [architecture decisions](docs/DECISIONS.md), [operator and evaluator handover](docs/HANDOVER.md), and [open work](docs/OPEN_WORK.md). The simulation and Qwen CPU results are measured. Native multi-GPU acceptance awaits SageMaker execution.

**Five-minute demo.** From the project root run `python evaluation/acceptance.py`, then open Study_Results.ipynb. The frozen evaluator uses only the Python standard library. For a hardware demonstration open SageMaker_vLLM.ipynb and set PILOT_ONLY to True. The 129-token pilot prompt should yield the same eight generated token IDs on the split and unsplit paths and a receipt for 128 loaded tokens. Generated text and exact timings are recorded by the run.

**Configuration.** The shell launcher reads the variables in .env.example. Source it with automatic export to override defaults. WORKERS specifies workers per stage. DAT_BACKEND selects the engine, DAT_RESULTS stores outputs, DAT_KV_SPOOL holds temporary KV files, DAT_MODEL_CACHE stores downloaded weights and DAT_IMAGE selects the container. CUDA_VISIBLE_DEVICES selects devices. Linux Python 3.12 dependency resolutions for the simulator and Studio GPU runtime are under locks/. To install the fully pinned GPU resolution use `python -m pip install -r locks/gpu-py312-linux.txt` in a dedicated GPU environment. The experiment uses files and needs no database migration.

A request visits a prefill worker, transfers its KV cache, and streams tokens from a decode worker. The observer sees the route, offered work, stage durations and token receipt times. It learns healthy service time from independent runs, ranks persistently slow components, and returns **unknown** when the available evidence cannot support a diagnosis.

**Start on SageMaker** by opening **SageMaker_vLLM.ipynb** in a GPU JupyterLab space. The standalone notebook includes the project source, installs an isolated vLLM runtime, launches native OpenAI-compatible servers, runs the disaggregated pilot and evaluates a held-out experiment. No Docker build or ECR repository is needed for this Studio workflow. The managed-job workflow is documented below.

Open **Study_Results.ipynb** to explore the included experiments, or read **RESULTS.md** for the numerical findings. The original DAT prototype notebook is also included. The GPU suite writes the same result tables and can be opened in the same results notebook.

## Run on NVIDIA GPUs

Use Linux, Docker with NVIDIA Container Toolkit, a compatible NVIDIA driver, local writable disk for KV files, and two distinct Ampere-or-newer NVIDIA GPUs for the pilot. The four-GPU configuration assigns one device to each of two prefill and two decode workers. The pinned testbed uses the FlashAttention KV layout and FP16. A single T4 cannot run this configuration. The first launch needs access to the container registry, Python packages and the public model weights, or their prepopulated caches.

Launch the pilot and full four-GPU study from the extracted project directory.

```bash
bash run_gpu.sh pilot
WORKERS=2 bash run_gpu.sh suite --with-overhead
```

The first command compares eight greedy tokens from split execution with the same eight tokens from unsplit execution. It requires positive transfer bytes, a worker-side receipt confirming the expected number of cached tokens, and exact token agreement on every configured route. The suite repeats that gate, then runs the workload, healthy fitting, calibration, fault/control pairs, passive diagnosis and evaluation. `gpu-runs/suite-2x2/summary.csv` holds the method comparison. `overhead/overhead_paired.csv` holds the on/off changes. **GPU performance and correctness still require this pilot on your hardware.**

To run the whole study on two GPUs, select one worker per stage.

```bash
WORKERS=1 bash run_gpu.sh suite --with-overhead
```

The 1×1 setup covers one worker at each stage and one logical path. A matched-peer comparison is unavailable and the sparse-second-worker case is omitted. With four GPUs the default study has 96 runs of 180 seconds each, or 4.8 hours of collection. Three optional overhead pairs add 18 minutes. Allow additional time for image/model download, engine startup, drain, fitting and reporting. A 2×2 run with one repetition uses 40 runs. Rerunning the same suite in the same output directory reuses completed trials. Use a different `--out /results/name` for changes to the experiment.

`CUDA_VISIBLE_DEVICES=2,3` selects a pair on a larger host. `DAT_RESULTS`, `DAT_KV_SPOOL` and `DAT_MODEL_CACHE` control host paths. If you change the image source after its first build, run `docker build -f Dockerfile.gpu -t dat-failslow:http-v0.10.2 .` again. The helper reuses the local image when present.

If SageMaker Studio already provides the pinned vLLM runtime and exposes the required GPUs, run the CLI directly from the extracted root. Install pandas, scikit-learn and joblib in that runtime first.

```bash
python -m failslow pilot --backend vllm-http --workers 1 --out local-runs/gpu-pilot
python -m failslow suite --backend vllm-http --workers 2 --out local-runs/gpu-study --with-overhead
```

## What the GPU run actually measures

The default SageMaker and Docker backend is `vllm-http`. The coordinator submits streaming `POST /v1/completions` requests to one private vLLM server per GPU. It verifies the served model and server version, joins every streamed token ID to a request-tagged model forward, and fits a new healthy baseline from that deployment. The injector holds the selected model forward on its serving worker. Client streaming delays never supply the worker's service-time measurement. [vLLM server API](https://docs.vllm.ai/en/v0.10.2/serving/openai_compatible_server.html)

The engine is vLLM **0.10.2**, with **Qwen2.5-0.5B-Instruct** pinned to revision `7ae557604adf67be50417f59c2c2f167def9a775`. Each worker is a separate process with one visible GPU. The model size keeps replicated per-worker weights small. The experiment serializes requests per worker and allows distinct workers and paths to run concurrently. The single-host vLLM V1 shared-storage connector exports block-aligned KV tensors. A route-specific copy moves the serialized tensors between worker directories. The decode connector writes its receipt after completing the load. This design follows the published offline connector example. [vLLM example](https://docs.vllm.ai/en/v0.10.2/examples/offline_inference/disaggregated-prefill-v1.html) · [Pinned connector source](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/distributed/kv_transfer/kv_connector/v1/shared_storage_connector.py) · [Model revision](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775)

| Measurement | Boundaries in the GPU harness |
|---|---|
| Prefill service | Synchronized, request-tagged model forward including KV export. Sampling, HTTP and queue wait lie outside this span. |
| Logical transfer | Host-side copy of actual serialized KV files, from the start of copying through completion, including any controlled delay. Bytes include serialization headers. Filesystem/page-cache behavior affects this measurement. |
| Decode service | One synchronized model forward at batch size one. The first continuation step includes KV import plus any uncached tail tokens. |
| TTFT | Scheduled client offer through receipt of the first token from prefill at the local driver. |
| ITL | Consecutive individual token receipt times at that driver. The first gap includes the handoff and decode admission. |
| Ancillary evidence | Configuration fingerprint, model revision, worker placement, GPU utilization/power/memory/temperature samples, right-censored requests, actual applied-delay counts and verified KV loads. |

A slowdown holds the selected worker or path occupied for extra time proportional to the measured operation that began inside the intervention interval. Output tokens and the offered schedule are preserved. These are **controlled software service delays**. The transfer experiment measures a single-host file-backed logical path. NIC/RDMA degradation, physical link attribution, continuous batching, cache reuse across requests and multi-host clock alignment require a different deployment and new calibration. The native study tokenizes natural-language questions with the pinned Qwen chat template and varies context length, arrival rate and output budget. It records generated IDs and decoded text. A chat-template completion may end mid-sentence because the experiment fixes the token budget. The earlier embedded vLLM engine remains available with `--backend vllm`. The reference simulator runs with `--backend sim`.

## Actual Qwen CPU experiment

The additional measured study in `results/qwen-vllm-cpu/` uses Qwen2.5-0.5B-Instruct on the vLLM 0.17.0 CPU engine. It times real synchronous model forwards and injects a 3× service delay inside a forward hook. The same healthy-model, calibration and ranking code evaluates separate prefill and decode incidents. There is one local engine, so the canonical transfer stage has no observations, and physical path localization is unavailable. Study parameters, controls, predictions, generated IDs, examples and raw timings are included. The full multi-GPU study uses the pinned 0.10.2 connector and native HTTP servers.

To repeat the CPU measurement in its own Linux x86-64 Python 3.12 environment, install the official vLLM CPU wheel, then invoke the collector. Use an empty output directory. [vLLM CPU wheel instructions](https://docs.vllm.ai/en/stable/getting_started/installation/cpu/)

```bash
uv venv .venv-cpu-vllm --python 3.12
uv pip install --python .venv-cpu-vllm/bin/python --torch-backend cpu \
  'https://github.com/vllm-project/vllm/releases/download/v0.17.0/vllm-0.17.0%2Bcpu-cp38-abi3-manylinux_2_35_x86_64.whl' \
  pandas==2.2.3 scikit-learn==1.7.2 joblib==1.5.2
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 LD_PRELOAD="$PWD/.venv-cpu-vllm/lib/libiomp5.so" \
  .venv-cpu-vllm/bin/python -m failslow.cpu_study --out local-runs/qwen-cpu
```

The recorded CPU run used `--local-collectives` for single-rank PyTorch process-group initialization in this socket-limited host. On a standard CPU host the default uses vLLM's normal process group. The inference math, KV state, greedy sampling and weights are handled by vLLM in both cases.

## Reproduce the simulator and CPU integration fixture

Use Python 3.12 on Linux. The standard simulator has FIFO prefill/transfer queues, round-robin decode, a cache-mix regime, and independent service noise. The measured CPU backend uses a small deterministic NumPy causal-attention model with real cache export, transfer and reuse. Its short windows exercise the collector and live observer.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m unittest discover -s tests -v
python -m failslow suite --out local-runs/synthetic
python -m failslow suite --backend numpy --workers 2 --quick --reps 1 --out local-runs/cpu
python -m failslow overhead --backend numpy --workers 2 --quick --duration 12 --reps 3 \
  --baseline local-runs/cpu/baseline.joblib --out local-runs/overhead
```

The frozen baseline can replay the included decode incident.

```bash
python -m failslow replay --baseline results/synthetic/baseline.joblib \
  --trace results/synthetic/runs/test-0-decode-1 --out local-runs/replay
```

Use JupyterLab to open **Study_Results.ipynb** in the project root. Set its first cell's `study_dir` to `Path("gpu-runs/suite-2x2")` to analyze your run. The GPU container fits its own baseline with its installed scikit-learn version. Use that same image when replaying its serialized model.

## Diagnosis and evaluation

1. **Split by whole runs.** Four healthy runs fit the stage regressors, four calibrate component score scales, and four calibrate the threshold on the maximum across eligible components. The normal-demand subset sets the demand reference. Fault labels are excluded from the three observer tables.
2. **Condition on work.** Fit median and p95 log-duration gradient-boosted regressors. Prefill features are prompt length, uncached tokens and batch size. Transfer uses bytes. Decode uses context length, token position and batch size. The positive excess is `max(0, (log(duration_ms) - q95) / max(q95 - q50, 0.08))`, with q95 floored at q50 + 0.08. Scores are averaged within each request and then across requests.
3. **Respect availability.** At tick t, consider completed records in (t − 30 s, t] whose collector receipt time is at most t. Require at least five supported requests and 80% supported operations per component. Configuration, timing semantics and padded training-feature ranges gate support. Compare scores against independently calibrated healthy ranges. Two consecutive threshold crossings produce a ranked candidate. The historical, work-rate and matched-peer baselines have their own thresholds.
4. **Explain independently.** Arrival rate, prompt length and output budget provide workload evidence. TTFT and ITL violation fractions provide SLO evidence. Default demonstration targets are 300 ms and 100 ms, respectively, with >1% violations. `--ttft-ms` and `--itl-ms` change targets before a study. The output can be workload, degradation, mixed, unknown, or no diagnosis. A supported candidate may still be emitted with partial coverage.
5. **Score after prediction.** Retain every injected incident. The first ranking in the interval from onset through fault end plus one window determines top-1/top-3. Count a fresh diagnostic episode only on a transition from clear to accusatory. Record any accusation already active before onset separately. Count false episode starts on the held-out healthy, workload-only and paired no-fault runs, excluding each run's initial window from exposure. Delays are conditional on an observed event. Stable SLO-alarming windows alone enter explanation accuracy.
6. **Compare paired runs.** Each intervention and control share a seed, offered work and route choices. Held-out tests cover normal traffic, workload shifts, prefill/decode/path slowdowns, a simultaneous workload shift and decode slowdown, sparse routing, and hidden transfer telemetry. Both 1.5× and 3× delays are tested. The simulator varies cache mix. Measured runs vary output budget. On real hardware the achieved slowdown includes scheduling noise. For an intervention's realized exposure inspect `applied_stage_operations` in its completion record.

The default clock advances every 10 seconds. The quick integration profile uses a 1-second window, 250-ms cadence, two supported requests, and smaller SLO sample minima. Results across these profiles have different meanings. The live observer runs in a separate process, prunes old records each tick and records decisions. The collector retains each bounded trial for reproducible replay. In-flight work at the horizon is right-censored. Diagnosis uses completed spans. An individual stalled request with sufficient unrelated healthy completions can still escape detection.

## Output files

| Path | Contents |
|---|---|
| `study.json`, `calibration.json`, `baseline.joblib` | Study specification, topology, seeds, split membership, frozen thresholds and model digest, and serialized healthy model |
| `runs/<id>/` | Canonical arrivals, stage observations, client SLO events, and completion counts. Measured runs also journal receipt order. |
| `decisions.csv.gz`, `component_scores.csv.gz` | Each tick's rankings, explanation, SLO state, component evidence, support counts and reasons |
| `incidents.csv`, `summary.csv`, `healthy_runs.csv`, `explanations.csv` | Incident-level outcomes and delays, method comparisons, healthy exposure and explanation outcomes |
| `service.csv`, `paired_impact.csv` | Completed throughput, censoring, p99 TTFT/ITL, violation fractions and fault-minus-control changes |
| `runs/<id>/generations.jsonl.gz`, `generation_pairs.csv` | Completed native HTTP generations with prompt hashes, token IDs, decoded text, and matched-pair output checks. Counts distinguish common and one-sided completions. |
| `pilot.json`, `runs/<id>/live.jsonl`, `runs/<id>/gpu.csv` | Token-equivalence and load proof, live predictions, and sampled GPU counters for the measured GPU run |
| `overhead.csv`, `overhead_paired.csv` | Matched, alternating observer-on/off measurements, resource use and paired percentage changes |

The overhead comparison enables stage journaling, board telemetry and live analysis together. Both arms keep the load driver, token timing and correctness instrumentation. On native vLLM, both also keep the worker timing hook and CUDA synchronization. The comparison measures the incremental cost of recording traces and running the observer. Separate 12-second CPU runs in the release exercise this path. The measured GPU overhead target is ≤5% throughput loss or increase in p99 TTFT/ITL.

## Run as a finite SageMaker job

Build **Dockerfile.gpu**, push the image to your ECR repository and supply a SageMaker execution role and S3 output prefix. Run the following from a machine with AWS CLI and your configured region. The Python helper renders the job request. The AWS command submits it and starts billable compute.

```bash
python sagemaker_job.py --image "$ECR_IMAGE" --role "$SAGEMAKER_ROLE" \
  --s3-output "$S3_OUTPUT" --mode suite --workers 2 --with-overhead > job.json
aws sagemaker create-training-job --cli-input-json file://job.json
```

The default job uses one `ml.g5.12xlarge` with four A10G GPUs and a seven-hour stopping limit. The execution role needs image-pull and output-write access, and the container needs access to the pinned weights or a baked model cache. Use `--mode pilot --workers 1` for a handoff-only job. SageMaker packages the output under `/opt/ml/model/failslow` into the S3 model artifact. [Instance specification](https://aws.amazon.com/sagemaker/ai/pricing/) · [Training container arguments](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_AlgorithmSpecification.html) · [S3 output contract](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-output.html)
