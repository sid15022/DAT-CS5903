# Project video recording script

**Passive Fail Slow Diagnosis in Disaggregated LLM Inference**

CS5903 · Project 25 · Group 5 · 15 slides · Reference recording 10:25

Read the paragraphs aloud. Italicized directions and code blocks are screen cues. The timestamps mark the corresponding chapters in the [reference recording](Project_Demonstration.mp4), which uses a synthetic English voice. The [PowerPoint](Final_Project_Presentation.pptx) follows the same order.

## Before recording

Open the presentation and the [executed SageMaker notebook](../DAT_CS5903_Final/DAT-CS5903/SageMaker_vLLM.ipynb). Prepare two terminals with the [CPU dependencies](../DAT_CS5903_Final/DAT-CS5903/locks/cpu-py312-linux.txt) installed. Keep one at the repository root and the other in `DAT_CS5903_Final/DAT-CS5903`. The pilot, replay and verification commands read archived GPU evidence. The simulator executes on CPU.

For slide 14, capture the simulator in an unused output directory so that it runs all trials. The command below used `/tmp/cs5903-recording-sim` on Linux. The recorded run took 68 seconds. During editing, advance from its first trials to the final comparison table.

## 1. Passive Fail Slow Diagnosis in Disaggregated LLM Inference

**0:00–0:32**

Imagine a language model that keeps returning the right words, but sometimes makes the user wait much longer for the next word. The service is still running, so a simple health check can pass.

This project asks whether ordinary inference requests carry enough evidence to identify the slow worker or cache transfer. The study tests that idea in a simulator and on a small open-weight model running on four GPUs. The demonstration replays the recorded SageMaker measurements and runs the simulator on CPU.

## 2. Why a slow response is hard to diagnose

**0:32–1:04**

Consider two pauses a user might notice. The first is the wait after submitting a prompt. Time to first token measures that interval. A long prompt, a long queue, or a slow prefill worker can all increase it.

The second is a pause between two output tokens. In this system, the gap after the first token also includes moving the attention cache to the decode worker. A client can tell us when the pause happened. To find its cause, the monitor needs timings inside the serving path.

## 3. Proposed solution for localizing slow components

**1:04–1:42**

A large cache copy should normally take longer than a small one. The useful question is how long this copy took compared with healthy copies of a similar size. The same idea applies to processing a prompt and generating a token.

The detector learns a healthy reference for each stage. During serving, it scores excess time, groups the evidence by worker or path, and looks for a delay that persists across requests. Diagnosis is passive because it uses traffic already in flight. Only the experimental harness injects a slowdown, and only the evaluator sees its label.

## 4. System architecture on four GPUs

**1:42–2:21**

Read the top row from left to right. One SageMaker host runs four copies of Qwen on vLLM, one per GPU. Two workers handle prefill and two handle decode. Each request visits one worker from each pool.

Prefill processes the prompt and emits the first token. The coordinator copies the cached attention keys and values to the decode worker, which reuses them to generate the remaining tokens. Request identifiers and a common host clock let the observer join the three stages. These transfer measurements are file copies on one host, so they include filesystem and host costs.

## 5. vLLM implementation and request tracing

**2:21–3:08**

Here is one actual request from the handoff pilot. A 129-token prompt requested eight output tokens. Prefill produced token one. The decode worker loaded 128 cached prompt tokens, the portion aligned to complete cache blocks, and produced tokens two through eight. A worker hook timed each forward pass and the collector joined it to a streamed token.

*Screen cue. Switch to the repository-root terminal. Run the pilot check and leave the four route rows visible.*

```sh
python evaluation/check_gpu_pilot.py
```

This command checks the recorded GPU pilot. Each row is one prefill-to-decode route. Look at the loaded-token count and the final column. All four routes loaded 128 tokens, and every route produced the same eight token IDs as the unsplit model. That establishes that the handoff preserved the output before diagnosis was evaluated.

## 6. How the detector scores a single operation

**3:08–3:50**

The two equations answer different questions. The first fits the healthy reference. For each stage, gradient-boosted trees estimate the median and the ninety-fifth percentile of log duration for the amount of work observed. At the higher percentile, predicting too low carries more loss, so the model learns the upper end of normal service time.

The second equation scores a new operation. Subtract the predicted upper bound from its measured log duration, keep only a positive excess, and divide by the healthy spread. The floor of zero point zero eight stabilizes that division. An operation inside the healthy range contributes zero.

## 7. When does a component enter the ranking?

**3:50–4:29**

Every ten seconds the monitor looks back over thirty seconds of available measurements. It first averages operation scores within each request, then averages across requests for a component. That gives a short and a long generation equal weight.

The second equation places this window average on the component’s healthy scale. Separate healthy runs set the alarm threshold. A component enters the ranking only after two consecutive crossings. A window needs at least five supported requests and eighty percent supported observations. Without that evidence, an empty ranking is a meaningful outcome.

## 8. How the monitor explains a latency alarm

**4:29–5:12**

A slow component and a change in demand can occur together. The monitor therefore tracks arrival rate, prompt length, and requested output length separately from worker timings. It also checks two client objectives: three hundred milliseconds to the first token and one hundred milliseconds between tokens.

With enough samples, a window breaches an objective when more than one percent exceed its limit. A persistent component ranking indicates degradation. A simultaneous demand shift changes the label to mixed. With full coverage, a demand shift and a client breach can indicate workload alone. If evidence is incomplete or conflicting, the monitor can report unknown.

## 9. The 60-minute SageMaker experiment

**5:12–5:56**

The GPU study ran forty trials of ninety seconds each, giving one hour of traffic. Twelve healthy trials were reserved for fitting the models, scaling component scores, and setting thresholds. Those decisions were frozen before the held-out incidents.

*Screen cue. Show the executed SageMaker notebook. Scroll to the study cell with the suite launch and its saved 40-trial output.*

The saved SageMaker notebook shows the command that launched the study and its recorded output. There were six kinds of fault, each tested at one and a half times and three times normal duration. Each incident had a control with the same offered work and route. Four more trials supplied healthy exposure. For a threefold slowdown, a five-millisecond forward receives roughly ten milliseconds of added wait. The observer receives the resulting timings.

## 10. A decode slowdown localized in 12.5 seconds

**5:56–6:42**

Follow the first two points after the intervention. Worker D one began slowing at thirty-seven and a half seconds. One forward took about forty-seven milliseconds against a healthy upper bound of about fifteen. Other requests on the same worker then contributed to the window score.

*Screen cue. Switch to the repository-root terminal. Run the replay and point to the empty ranking at 40 seconds and d1 at 50 seconds.*

```sh
python evaluation/replay_formula_example.py
```

The replay command refits the healthy reference from the archived measurements and reconstructs this decision. At forty seconds, the score is forty-nine point three eight, above a threshold of three point two seven. The empty brackets mean that the first crossing produced no ranking. At fifty seconds, the score crosses again and D one appears. Fifty minus thirty-seven and a half gives a diagnosis delay of twelve and a half seconds.

## 11. GPU localization against three baselines

**6:42–7:24**

The first result column counts all twelve injected incidents, including the cases where a method abstained. The proposed method, historical latency, and matched peer each identified the target first in eight cases. Work rate did so in seven.

The difference is clearer in delay and false alarms. The proposed method and historical latency took twelve and a half seconds at the median among eventually correct diagnoses. Matched peer took thirty-two and a half seconds. On healthy traffic, these three methods raised three, four, and zero false episodes, respectively. The faster ranking came with a measurable false-alarm cost.

## 12. The detector localized four of six GPU fault types

**7:24–8:05**

The case breakdown shows where the evidence came from. At both severities, the monitor identified prefill, decode, and mixed demand with slow decode in twelve and a half seconds. A visible cache transfer took one more scoring tick, or twenty-two and a half seconds in total.

The two right columns measure what the user experienced. At threefold slowdown, prefill added about forty-seven milliseconds to the ninety-ninth percentile of first-token latency. Decode added about forty-eight milliseconds to the corresponding token gap. The detector abstained when too few requests visited a worker or the affected transfer had no timing records.

## 13. False alarms and workload explanations

**8:05–8:45**

The false-alarm denominator is sixteen monitored healthy minutes. Three episodes in that time give eleven and a quarter per hour, above the target of one per hour. Matched peer observed zero episodes, but sixteen minutes is too short to establish its long-run rate.

The explanation test uses a different denominator: three eligible windows with a client latency alarm. The monitor correctly identified the decode case. It missed the demand shift in the mixed case and returned unknown for the workload-only case. That is one correct explanation out of three, separate from the eight correct incident rankings.

## 14. Simulator comparison and execution checks

**8:45–9:43**

The simulator repeats each case and severity three times. It exercises the same diagnosis and evaluation logic with generated timings. It also lets the entire experiment run on a CPU.

*Screen cue. Show the simulator run from the inference package directory. Show its start, then advance to the completed four-method table.*

```sh
python -m failslow suite --backend sim --workers 2 --reps 3 --out /tmp/cs5903-recording-sim
```

This is the simulator suite running from the command line. The recording advances through the run to its result table. Across thirty-six injected incidents, the conditioned method and matched peer each ranked the right component first twenty-four times. The conditioned method raised six false episodes in two healthy hours. Matched peer raised one.

*Screen cue. Return to the repository-root terminal. Run the GPU archive check and leave the final verification line visible.*

```sh
python evaluation/check_gpu_study.py
```

The final command verifies the separate GPU archive. It checks the trial inventory and paired outputs, then compares recorded live decisions with offline replay. All forty GPU trials are present. The archive contains 2,821 matching generations, and all 784 rankings and explanations agree.

## 15. What the study established

**9:43–10:25**

The experiment shows that ordinary serving traces can identify a specific slow worker or cache path when the affected component is observed often enough. On real Qwen inference, the detector found eight of twelve incidents. It reached correct rankings sooner than matched peer in this study, while raising more false alarms.

The next tests follow directly from those limits: longer healthy runs, repeated GPU trials, paired measurements of monitoring overhead, and transfers between hosts under continuous batching. The repository contains the implementation, raw traces, and replay commands behind the results. Thank you.
