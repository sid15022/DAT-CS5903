# Mathematical formulation and implementation

## A.1 Measurement and healthy service time

### What the clock measures

For request r, let a denote the scheduled arrival time and bⱼ the client receipt time of output token j. For stage operation o, let sₒ and fₒ denote the worker or copy start and finish times. All timestamps use the same host clock and are measured in seconds. The factor 1,000 converts differences to milliseconds.

$$
T_o=1000(f_o-s_o),\qquad \mathrm{TTFT}_r=1000(b_1-a),\qquad \mathrm{ITL}_{r,j}=1000(b_j-b_{j-1}),\quad j\geq 2. \tag{1}
$$

TTFT includes queueing. The first inter-token gap also includes the KV handoff. Stage time measures a recorded forward or file copy. The conditioned model fits stage time; client latencies are evaluated separately. [Timestamp collection](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/failslow/collect.py#L321-L338) · [Token receipts](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/failslow/collect.py#L358-L388)

### Conditional quantile regression

For each stage, xₒ contains the measured work. Prefill uses prompt tokens, uncached tokens and batch size. Transfer uses KV bytes. Decode uses context tokens, batch size and tokens generated so far. The response yₒ is the natural logarithm of the numerical duration in milliseconds. The elementwise log transform of the inputs also admits zero values.

$$
z_o=\log(1+x_o),\qquad y_o=\log(T_o),\qquad \rho_\tau(u)=u\bigl(\tau-\mathbf{1}[u<0]\bigr). \tag{2}
$$

$$
\widehat f_{s,\tau}\;\approx\;\underset{f\in\mathcal{F}}{\arg\min}\sum_{o\in H_s}\rho_\tau\!\left(y_o-f(z_o)\right),\qquad \tau\in\{0.50,0.95\}. \tag{3}
$$

Hₛ is the sample of healthy fit operations at stage s. The indicator equals one when its condition holds. At the 95th percentile, underpredicting a log duration by one costs 0.95; overpredicting by one costs 0.05. The two fitted functions estimate typical and upper-tail service for comparable work. Gradient boosting approximates this objective using 60 depth-three trees per fit, a learning rate of 0.06 and up to 6,000 rows per stage. [HealthyModels.fit](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/failslow/model.py#L62-L103) · [Quantile regression, Koenker and Hallock (2001)](https://doi.org/10.1257/jep.15.4.143)

### Operation-level excess

$$
m_o=\widehat f_{s,.50}(z_o),\quad u_o=\max\{\widehat f_{s,.95}(z_o),m_o+0.08\},\quad e_o=\frac{[y_o-u_o]_+}{u_o-m_o}. \tag{4}
$$

Here [v]₊ means max(v, 0). The 0.08 floor lifts the upper bound before the subtraction and bounds the denominator. A duration at or below exp(uₒ) contributes zero. A score of one means that the log duration exceeds the adjusted upper bound by one healthy spread. Scores are dimensionless. Inputs outside the fitted workload range or with different configuration or timing semantics receive no score. [HealthyModels.score](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/failslow/model.py#L118-L153)

## A.2 Window evidence and comparison methods

### Availability and request weighting

At tick t, observations enter a 30-second window by operation finish time fₒ. The separate collector availability time vₒ prevents late records from contributing before arrival. For component c, let O be all such observations and S the supported subset. R contains the distinct request IDs in S. Sᵣ contains the supported operations for request r.

$$
O_c(t)=\{o:c_o=c,\ t-W<f_o\leq t,\ v_o\leq t\},\qquad W=30\ \mathrm{s}. \tag{5}
$$

$$
\bar e_{r,c,t}=\frac{1}{|S_r|}\sum_{o\in S_r} e_o,\qquad A_{c,t}=\frac{1}{|R|}\sum_{r\in R}\bar e_{r,c,t}. \tag{6}
$$

A component needs at least five supported requests and |S|/|O| ≥ 0.80. An empty window is ineligible. Averaging within requests first gives a two-token request and a twenty-token request the same weight in a component window. A missing score cannot count as evidence of healthy service. [Window selection](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/failslow/detector.py#L22-L29) · [raw_window](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/failslow/detector.py#L32-L69)

### Historical and work-rate baselines

For historical scoring, gₒ is log duration. For work-rate scoring, gₒ is log duration per unit of work. The work unit wₒ is the uncached token count for prefill, bytes for transfer, and one generated token for decode. Prefill and transfer work counts are floored at one. Let Q denote empirical quantiles over healthy fit observations of the same stage.

$$
g_o^{\mathrm{hist}}=\log T_o,\qquad g_o^{\mathrm{rate}}=\log(T_o/w_o),\qquad e_o^g=\frac{[g_o-Q_{.95}(g)]_+}{\max\{Q_{.95}(g)-Q_{.50}(g),0.08\}}. \tag{7}
$$

These baseline scores use the same supported observations and the request weighting in (6). Their cutoffs are unconditional on the full work vector. Only their denominators receive the 0.08 floor. [Work units and healthy references](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/failslow/model.py#L54-L60) · [Baseline score calculation](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/failslow/model.py#L154-L168)

### Matched-peer baseline

Let M be the median across requests of their mean log residual relative to the healthy median prediction mₒ. P contains the other eligible components of the same stage in this window.

$$
M_{c,t}=\mathrm{median}_{r\in R}\;\frac{1}{|S_r|}\sum_{o\in S_r}(y_o-m_o),\qquad A^{\mathrm{peer}}_{c,t}=\left[M_{c,t}-\mathrm{median}_{p\in P}M_{p,t}\right]_+. \tag{8}
$$

This method detects a component that is unusually slow relative to its contemporaneous peers. A shared slowdown can leave little peer contrast. An empty peer set yields no score. Each method uses its own calibration in A.3. [Peer residual calculation](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/failslow/detector.py#L36-L42) · [Peer selection](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/failslow/detector.py#L69-L81)

## A.3 Healthy calibration and persistent ranking

### From a window average to a component score

The window score A is indexed by method m, component c and tick t. The healthy data are split by whole runs. The fit split trains the operation model. A separate scale split estimates the center B and scale D of each component and method from at least ten finite window scores.

$$
B_{mc}=Q_{.50}^{\mathrm{scale}}(A_{mc}),\quad D_{mc}=\max\{Q_{.95}^{\mathrm{scale}}(A_{mc})-B_{mc},0.025\},\quad Z_{mct}=\frac{A_{mct}-B_{mc}}{D_{mc}}. \tag{9}
$$

A negative normalized score Z is possible even when every operation-level excess is nonnegative. It means that the window average lies below the healthy window median. The 0.025 floor controls amplification when healthy window scores vary little. It acts after aggregation and differs from the log-duration floor in (4). [Baseline.train](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/failslow/detector.py#L132-L179)

### A threshold across the monitored components

The third healthy split calibrates one threshold per method. Each calibration tick first contributes its largest normalized score across eligible components Eₘₜ. Empty ticks do not contribute. At least ten finite maxima are required.

$$
V_{mt}=\max_{c\in E_{mt}}Z_{mct},\qquad \theta_m=\max\{0,Q^{\mathrm{threshold}}_{.99}(V_m)\}. \tag{10}
$$

Taking the maximum accounts empirically for selecting any worker or path at a tick. The calibrated percentile is a window statistic. Overlapping windows and repeated alarms prevent interpreting it as an hourly false-alarm guarantee. The recorded conditioned threshold was 3.26693. [Threshold calibration](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/failslow/detector.py#L164-L183) · [Frozen thresholds](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/dat-vllm-experiment/study-2x2-r1-t90-q2.6/calibration.json)

### Two successive crossings

Let h be one for an eligible score strictly above its method threshold, and zero otherwise. The streak k starts at zero for every method and component. The subscript t−Δ denotes the previous tick, ten seconds earlier.

$$
h_{mct}=\mathbf{1}[c\in E_{mt},\ Z_{mct}>\theta_m],\qquad k_{mct}=h_{mct}(1+k_{mc,t-\Delta}),\qquad \Delta=10\ \mathrm{s}. \tag{11}
$$

The monitor ranks components with k ≥ 2 by decreasing current Z, breaking ties by component ID and retaining at most three. A missing score resets that component’s streak. At the first crossing the component is still absent from the ranking. In the worked incident, the second crossing at 50 s produces the first diagnosis. [Monitor.step](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/failslow/detector.py#L221-L273) · [Live and replay monitor](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/failslow/online.py#L16-L55)

## A.4 Demand, client impact and explanation

### Workload shift

For each window, form a three-element vector d of arrivals per second, median prompt length and median requested output budget. Normal-demand windows in the fit split supply the coordinatewise median μ and scale s. The floors have units of requests/s, prompt tokens and output tokens, respectively.

$$
s_j=\max\{Q_{.95}^{\mathrm{normal\ fit}}(|d_j-\mu_j|),\ \epsilon_j\},\quad \epsilon=(0.3,100,2),\quad G_t=\max_j\frac{|d_{tj}-\mu_j|}{s_j}. \tag{12}
$$

The demand threshold is the larger of 2 and the empirical 99th percentile of G in normal-demand threshold windows. The archived threshold was 2.0. A demand shift requires two consecutive finite scores strictly above it. Absolute deviations allow both rising and falling demand. Workload evidence and component scores are evaluated independently. [Demand vector](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/failslow/detector.py#L83-L90) · [Center and threshold](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/failslow/detector.py#L184-L211) · [Persistence](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/failslow/detector.py#L226-L233)

### Client latency objective

Let Nₖ be the number of observed client values for metric k in the window. Compute the violation fraction only when observations exist. A metric can raise an alarm only if it meets its minimum count.

$$
v_k(t)=\frac{1}{N_k}\sum_{i=1}^{N_k}\mathbf{1}[L_{ki}>b_k],\qquad B_t=\mathbf{1}[\exists k:\ N_k\geq n_k,\ v_k(t)>0.01]. \tag{13}
$$

The pair (bₖ, nₖ) is (300 ms, 10 receipts) for TTFT and (100 ms, 30 gaps) for ITL. One valid metric can establish a breach. In a window without a breach both metrics require sufficient samples to establish valid SLO status. The p99 curve is descriptive; the decision uses (13). [Client SLO calculation](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/failslow/detector.py#L91-L105)

### Explanation rule in evaluation order

A persistent component ranking implies **degradation**, or **mixed** when a demand shift also persists. With an empty ranking, incomplete component coverage or invalid SLO evidence yields **unknown**. With full coverage and valid evidence, a window with neither an SLO alarm nor an instantaneous component crossing yields **no diagnosis**. A breached objective with persistent demand shift and no instantaneous crossing yields **workload**. All other unresolved cases yield **unknown**. A component can thus be ranked while the client objective is still satisfied. [Explanation logic](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/failslow/detector.py#L268-L301)

### Controlled slowdown

For a targeted operation that starts inside the scheduled interval, the harness adds (γ − 1) times its elapsed service time as a wait. For γ = 3, a 5 ms operation receives approximately 10 ms of additional delay. Scheduling and timer overhead can change the realized multiplier. Fault identity and timing enter only the experimental harness and offline evaluation. [Delay implementation](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/failslow/workers.py#L13-L17)

## A.5 Incident and service evaluation

### First ranking and time to the correct diagnosis

For incident i with onset uᵢ, end vᵢ and trial horizon Hᵢ, the evaluation interval is [uᵢ, min(Hᵢ, vᵢ + W)]. Let Fᵢ be its first nonempty ranked list, or the empty list if none occurs. Let cᵢ be the injected target and Rᵢ(t) the ranked list at tick t. An empty first list contributes zero to top-k accuracy.

$$
\mathrm{Top}@k=\frac{1}{N}\sum_{i=1}^N\mathbf{1}[c_i\in F_i[1{:}k]],\qquad d_i=\min_{t:\ R_i(t)[1]=c_i}(t-u_i). \tag{14}
$$

The minimum is over eligible ticks with a nonempty list. Report the median delay only for incidents with a correct first-position diagnosis at some tick. A later correction can contribute a delay while first-choice accuracy remains zero. A new-episode delay uses a transition from empty to nonempty, regardless of correctness. In (14), Fᵢ[1:k] denotes the first k entries, including position k. GPU first-choice accuracy was 8/12 (66.7%), with four abstentions. [Incident scoring](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/failslow/evaluate.py#L37-L75)

### False episodes and exposure

Let Aⱼₜ indicate a nonempty ranking in no-fault trial j, with initial state zero. Count starts of episodes; consecutive active ticks contribute one start. The first window is excluded from each trial’s monitored duration, in seconds.

$$
K=\sum_{j,t}A_{jt}(1-A_{j,t-\Delta}),\quad E=\sum_j(H_j-W)/3600,\quad \widehat\lambda=K/E. \tag{15}
$$

$$
\mathrm{CI}_{.95}=\left[\chi^2_{2K,.025}/(2E),\ \chi^2_{2(K+1),.975}/(2E)\right]. \tag{16}
$$

The lower limit is zero for K = 0. The chi-square subscripts denote degrees of freedom and lower-tail quantile probability. For the conditioned detector, E = 16 × (90 − 30)/3600 = 0.2667 hours and K = 3, giving 11.25 episodes/h (2.32–32.88). This interval assumes a constant Poisson rate. Short exposure and correlated traffic limit interpretation. [Exposure](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/failslow/evaluate.py#L20-L37) · [Rate and interval](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/failslow/evaluate.py#L123-L150) · [Garwood (1936)](https://doi.org/10.1093/biomet/28.3-4.437)

### Paired service impact and explanation accuracy

$$
\Delta M_i=M_i^{\mathrm{fault}}-M_i^{\mathrm{control}},\qquad \mathrm{RPS}_i=N_i^{\mathrm{completed}}/H_i. \tag{17}
$$

For the 3× decode pair, whole-trial p99 ITL rose from 42.564 to 90.653 ms (paired difference +48.090 ms). Throughput was 206/90 = 2.289 requests/s in both trials. Explanation accuracy counts correct labels among stable SLO-alarming windows. These run from onset plus one full window to intervention end, or to the horizon for a workload-only change. Only 1 of 3 eligible GPU windows had a correct explanation. [Paired metrics](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/failslow/evaluate.py#L105-L122) · [Service statistics](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/failslow/evaluate.py#L163-L176) · [Explanation denominator](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/failslow/evaluate.py#L76-L95)

## A.6 Worked GPU example

### One forward operation

In trial `test-0-decode-1` the target was decode worker `d1`. Observation `r00081-decode-1` had 66 context tokens, batch size one and one prior generated token. The forward ended at 37.81057 s and became available at 37.81237 s. It therefore entered the 40 s window. Its measured duration was 46.95191 ms. Refitting the archived healthy runs gives a predicted median of 14.24523 ms and an adjusted upper bound of 15.43168 ms.

$$
e_o=\frac{\log(46.95191)-2.736422}{2.736422-2.656422}\approx13.90877. \tag{18}
$$

The denominator was the 0.08 log-scale floor. This operation exceeded the upper bound by about 1.113 log units. Its score is dimensionless. Other operations in the same request bring the request mean in (6) to 12.95142. [Raw observations](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/dat-vllm-experiment/study-2x2-r1-t90-q2.6/runs/test-0-decode-1/observations.csv.gz) · [Fitted score](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/failslow/model.py#L145-L153)

### From two windows to one diagnosis

At 40 s the component had 181 supported operations from 31 requests. At 50 s it had 205 from 35 requests. The healthy window median was 0.00016645 and the calibration scale hit its 0.025 floor.

$$
Z_{40}=\frac{1.234655-0.00016645}{0.025}\approx49.38,\qquad Z_{50}=\frac{6.175653-0.00016645}{0.025}\approx247.02. \tag{19}
$$

Both scores exceeded θ = 3.26693. At 40 s this was the first crossing, so the ranked list was empty. At 50 s the streak reached two and the first ranked component was d1. With onset at 37.5 s, the delay was 50 − 37.5 = 12.5 s. The window score reflects all supported requests; it cannot be reconstructed from the single forward in (18) alone. [Recorded scores](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/dat-vllm-experiment/study-2x2-r1-t90-q2.6/component_scores.csv.gz) · [Live rankings](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/dat-vllm-experiment/study-2x2-r1-t90-q2.6/runs/test-0-decode-1/live.jsonl) · [Incident summary](https://github.com/sid15022/DAT-CS5903/blob/e20f53712de648b8c03784d2b1c8496146fc2d16/DAT_CS5903_Final/DAT-CS5903/dat-vllm-experiment/study-2x2-r1-t90-q2.6/incidents.csv)

### Reproduction on CPU

From the repository root, with the project’s CPU dependencies installed, run `python evaluation/replay_formula_example.py`. The script refits the model using the 12 archived healthy runs, reconstructs the two request-weighted scores from raw measurements and checks them against the archived scores and live rankings. It also checks the fitted method threshold against `calibration.json`. The GPU study used scikit-learn 1.7.2; the report’s numerical reconstruction also agreed under the pinned CPU environment using 1.8.0. [Reconstruction script](../evaluation/replay_formula_example.py)
