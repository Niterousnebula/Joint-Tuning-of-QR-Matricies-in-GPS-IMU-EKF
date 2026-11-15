# Joint Tuning of Q and R in a GPS–IMU EKF

BTP I (MEMEC702), Netaji Subhas University of Technology, Mechanical Engineering.
Team: Anish Verma, Raghav Anthwal, Adrishikhar Deka, Sanvi Shokeen. Supervisor: Dr Vijayant Agarwal.

We tune the process noise **Q** and the measurement noise **R** together, offline, for one fixed 16-state GPS–IMU Extended Kalman Filter. Q and R are set once at the start of a run and stay constant. Four optimizers are benchmarked under identical conditions against the hand-set starting values (`THETA0`):

- Genetic Algorithm (GA)
- Particle Swarm Optimization (PSO)
- Differential Evolution (DE)
- Grey Wolf Optimizer (GWO)

**Methodology.** Q/R optimization is performed offline using the complete recorded flight. The primary optimization objective is whole-flight horizontal position RMSE. Robustness is evaluated separately by applying predefined sensor perturbations and measuring whole-flight RMSE degradation.

The experiment evaluates the ability of the optimization algorithms to identify Q/R parameters for the given ETH Zurich flight under a fixed computational budget. It does not test generalization to unseen flights.

## Branches

| Branch | Holds |
|---|---|
| `main` | Data, the EKF, the canonical fitness function and this README. Everyone uses it unchanged |
| `ga`, `pso`, `de`, `gwo` | One optimizer each: its code and its config file, built on `main` |
| `hisa` | The Heuristic Informed Search Algorithm (`HISA.py`, `HISA_config.py`, `plot_hisa.py`), built on `main` |
| `results` | All four optimizers merged in, `run_benchmark.py`, runs of every optimizer against the baseline, and the evaluation figures |

Changes to the data, the filter or the fitness function go into `main` first and are then merged into every branch, so all four optimizers always score candidates in the same way.

## PSO settings (from the `pso` branch)

`PSO.py` runs global-best Particle Swarm Optimization on the 7-D log10 decision vector within `[LOWER, UPPER]`. Its settings are in `PSO_config.py`.

| Setting | Value |
|---|---|
| Swarm size × iterations | 200 × 50 = 10,000 evaluations per seed |
| Seeds | 1, 2, 3, 4, 5 (50,000 evaluations in total) |
| Objective backend | GPU: each iteration's 200 particles are one batched EKF pass (`ekf/gpu.py`) |
| Inertia w | linear from 0.9 to 0.4 |
| c1, c2 | 1.49445, 1.49445 |
| Velocity clamp | 20 % of each parameter's bound range |
| Bounds handling | absorbing walls: position clipped, outward velocity set to 0 |
| Initial swarm | uniform within the bounds (THETA0 is not seeded in) |

Every candidate is scored through `ekf.make_cost_fn`, so the budget, the convergence curve and the timing match the other optimizers.

While it runs, every evaluation is appended to `results/PSO_gpu/pso_log.csv` as soon as its iteration is scored. Each row holds `seed, iteration, particle, k, generation, fitness, best_so_far, elapsed` and the 7 θ values, so you can open the file mid-run to watch progress. Rerunning a seed replaces that seed's rows. A progress line is also printed after each iteration. `results/PSO/` holds the earlier 20 × 50 CPU runs (seeds 0–4) and is not overwritten.

```bash
python PSO.py                                        # all seeds -> results/PSO_gpu/
python PSO.py --seeds 1 --max-evals 40 --swarm 8     # quick check
python PSO.py --backend cpu --workers 10             # filterpy reference on the CPU
```

## HISA: Heuristic Informed Search Algorithm (`hisa` branch)

HISA is a deterministic coarse-to-fine grid search. Each candidate is one joint combination of the six **initial bias states** and **R**, scored by the same EKF and the same whole-flight RMSE_h. The Q noise densities stay at `THETA0`. The code is `HISA.py`, the settings are in `HISA_config.py`, and `plot_hisa.py` draws the figures from the CSV outputs only.

```text
signed log grid ±10^e, e ∈ {−3,…,2}, zero excluded          4,826,809 → 2,985,984 bias patterns
accelerometer axes within 0.5 decade (one decade); gyro free; all 64 signs   → 82,944 bias candidates
× R: log10 r_xy ∈ {0,1,2}, log10 r_z ∈ {0.5,1.5,2.5}, r_vel = r_xy/10 (9)  → 746,496 joint candidates
Int25: all 746,496        → best 10 % (74,650)
Int10: 74,650             → best 10 % (7,465)
Int1:  7,465 (canonical filter and fitness) → winner
```

| Item | Definition |
|---|---|
| Bias values | Set the filter's initial `b_a`, `b_g` (`x0[10:16]`); the EKF keeps estimating them |
| R | Searched jointly with the biases; never searched after the bias search or fixed at `THETA0` |
| IntN | Every N IMU intervals are integrated into one prediction (dt-weighted mean of accel and gyro, dt = block length), followed by the last GPS fix of the block. The filter is unchanged; only its input stream is coarser. Int1 is the canonical run |
| Ranking | Whole-flight RMSE_h at that level; ties broken by `candidate_id`; scores are compared only within one level |
| Best so far | The best of each level is evaluated first on the next level, then compared with that level's candidates |
| Final metrics | The Int1 winner is re-scored by the filterpy reference with the benchmark's RMSE_h, robustness and NIS |
| Validation | At every level, the best, the worst and a spread of candidates are re-scored by filterpy on the same stream; the run fails if any difference exceeds `TOL_RMSE` |

**Workload is reported in three separate ways, and they are never mixed:**

| Measure | Int25 | Int10 | Int1 | Total |
|---|---:|---:|---:|---:|
| Literal candidate evaluations | 746,496 | 74,650 | 7,465 | 828,611 |
| Int1-normalized cost ($n/N$) | 29,859.84 | 7,465 | 7,465 | 44,789.84 |
| EKF steps per candidate (predictions + GPS updates) | 1,082 + 1,082 | 2,705 + 2,705 | 27,047 + 13,529 | |

The spec's $1/N$ normalization counts predictions only. Coarse streams keep one GPS update per block, so the measured step count of an Int25 evaluation is about 1/19 of an Int1 evaluation, not 1/25. `stage_summary.csv` records both, together with wall-clock time.

```bash
python HISA.py                                   # -> results/HISA/  (refuses to overwrite)
python HISA.py --limit 3000 --out results/HISA_smoke
python plot_hisa.py --hisa results/HISA --benchmark results/benchmark_<stamp>
```

Outputs: `candidates_all.csv.gz`, `int25_results.csv.gz`, `int10_results.csv`, `int1_results.csv`, `best_so_far.csv`, `final_best.csv`, `stage_summary.csv`, `config_snapshot.json`, `runs.csv` (benchmark schema), `ekf_error_timeseries.csv`, `validation/`, and `figures/`. The two largest tables are gzip-compressed (pandas reads them directly) to stay under GitHub's file-size limit.

**Comparison with GA/PSO/DE/GWO.** The methods differ in decision vector (6 initial biases + R vs 7 log-variances) and in literal evaluation counts, and HISA runs once because it is deterministic. The fair comparison is therefore the final metrics, all computed by the filterpy reference, together with the measured workload and wall-clock time on the same GPU.

## Layout

| Path | Purpose |
|---|---|
| `data/` | The cleaned ETH Zurich flight. Do not clean it again |
| `ekf/data.py` | `load_data`, plus `perturb` and `PERTURBATIONS` for the robustness tests |
| `ekf/filter.py` | The EKF, built on `filterpy.kalman.ExtendedKalmanFilter` |
| `ekf/tuning.py` | Decision vector, bounds, `theta_to_QR`, the canonical `fitness`, `FitnessTracker`, `robustness` and the result helpers |
| `ekf/gpu.py` | The same EKF batched over candidates on the GPU (PyTorch, float64), used by `FitnessTracker(backend="gpu")`, and `compare_with_cpu` |
| `run_baseline.py` | Evaluates `THETA0` through the same fitness and writes `results/baseline/` |

## Quick start

```bash
pip install -r requirements.txt
python run_baseline.py
```

```python
import numpy as np
import ekf

data = ekf.load_data()
J = ekf.fitness(data, ekf.THETA0)                 # whole-flight horizontal RMSE, metres

# What every optimizer calls: same fitness, plus budget, best-so-far history and timing
seed = 0
cost_fn = ekf.make_cost_fn(data, max_evals=1000, n_workers=8)
population = np.random.default_rng(seed).uniform(ekf.LOWER, ekf.UPPER, (20, 7))
costs = cost_fn.evaluate_many(population)         # or cost_fn(theta) for one candidate
# ... the optimizer's own loop, until cost_fn.exhausted ...
row, _ = ekf.run_summary(data, cost_fn, "GA", seed, rmse_baseline=J)
ekf.save_run("results/GA", row, cost_fn)          # runs.csv + convergence_seed<seed>.csv
cost_fn.close()
```

With `n_workers > 1` the script must run its optimizer under `if __name__ == "__main__":`, which Windows multiprocessing requires.

To score each population as one batch on the GPU instead, pass `backend="gpu"` (needs a CUDA build of PyTorch, see `requirements.txt`):

```python
cost_fn = ekf.make_cost_fn(data, max_evals=10000, backend="gpu")
costs = cost_fn.evaluate_many(population)         # 200 candidates -> one batched EKF pass
```

## Results summary

Two runs on one RTX 4060 Laptop GPU, both on the complete ETH Zurich flight. All numbers below are final values re-scored by the filterpy reference filter. The full files are on the [`results` branch](https://github.com/Niterousnebula/Joint-Tuning-of-QR-Matricies-in-GPS-IMU-EKF/tree/results/results).

- **GPU benchmark** (`results/benchmark_20261114_233504/`): GA, PSO, DE and GWO, population 200, 10,000 evaluations per seed, seeds 1–5, 7-D log-variance decision vector.
- **HISA** (`results/HISA/`): one deterministic coarse-to-fine run over 746,496 joint initial-bias + R candidates (Int25 → Int10 → Int1), with Q at `THETA0`.

### Headline numbers

Mean ± std over 5 seeds (HISA: one run). Raw GPS is 3.604 m.

| Method | Whole-flight RMSE_h (m) | vs baseline EKF (m) | vs raw GPS (m) | Mean NIS (expect 6) | Robustness mean ΔRMSE_h (m) | Evals to 99 % of improvement | EKF steps | Wall clock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (`THETA0`) | 3.775 | – | +0.171 | 0.97 | +0.022 | – | 4.1 × 10⁴ | 5 s |
| GA | 3.546 ± 0.016 | −0.229 | −0.058 | 1.98 ± 1.59 | +0.284 ± 0.080 | 8,421 ± 1,136 | 4.1 × 10⁸ per seed | 11.8 min per seed |
| **PSO** | **3.499 ± 0.001** | **−0.276** | **−0.105** | 3.68 ± 0.32 | +0.373 ± 0.010 | 6,690 ± 1,660 | 4.1 × 10⁸ per seed | 11.8 min per seed |
| DE | 3.538 ± 0.012 | −0.237 | −0.066 | 2.75 ± 0.84 | +0.282 ± 0.014 | 7,672 ± 1,458 | 4.1 × 10⁸ per seed | 11.8 min per seed |
| **GWO** | **3.501 ± 0.001** | **−0.274** | **−0.103** | 3.79 ± 0.13 | +0.381 ± 0.027 | **5,726 ± 1,564** | 4.1 × 10⁸ per seed | 11.8 min per seed |
| HISA | 3.740 | −0.036 | +0.136 | 3.25 | **+0.032** | – | 2.3 × 10⁹ | 45.2 min |

### Best values found

The best seed of each optimizer, and the HISA winner. Values are log10 of the noise densities and variances.

| Method (seed) | RMSE_h (m) | `log_q_acc` | `log_q_gyro` | `log_q_ba` | `log_q_bg` | `log_r_xy` | `log_r_z` | `log_r_vel` | Mean NIS | Robustness mean / max ΔRMSE_h (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `THETA0` | 3.775 | 0.20 | −3.00 | −5.00 | −8.00 | 1.00 | 1.50 | 0.00 | 0.97 | +0.022 / +0.049 |
| GA (seed 1) | 3.527 | 0.07 | −6.00 | −8.00 | −9.61 | −0.28 | 0.39 | 1.85 | 4.19 | +0.416 / +1.089 |
| **PSO (seed 5)** | **3.499** | 0.18 | −6.00 | −7.59 | −10.00 | −0.41 | 2.74 | 2.00 | 3.85 | +0.377 / +0.746 |
| DE (seed 4) | 3.520 | 0.35 | −5.88 | −5.02 | −10.35 | −0.50 | 3.50 | 1.80 | 2.98 | +0.303 / +0.621 |
| GWO (seed 4) | 3.499 | 0.20 | −6.00 | −6.69 | −10.01 | −0.38 | 1.41 | 1.78 | 3.79 | +0.364 / +0.736 |
| HISA | 3.740 | 0.20 | −3.00 | −5.00 | −8.00 | 0.00 | 0.50 | −1.00 | 3.25 | +0.032 / +0.061 |

HISA winner's initial biases: $b_a$ = (−0.1, +0.1, −0.1) m/s², $b_g$ = (−0.001, +0.1, −0.1) rad/s. Its Q noise terms are `THETA0` by design.

### What the results show

1. **PSO and GWO are the most accurate.** Both reach 3.499–3.501 m with almost no spread between seeds, 0.276 m better than the baseline EKF and 0.105 m better than raw GPS. GWO converges fastest (median 5,401 evaluations to 99 % of its improvement). GA and DE end 0.04–0.05 m higher and vary more between seeds.
2. **Accuracy is bought with robustness.** All four optimizers lower `r_xy` below 1 m² and so track GPS position more tightly. Under 2 m of added GPS noise their RMSE rises by 0.62–0.73 m, against 0.05 m for the baseline. HISA keeps the baseline's robustness (+0.032 m) but gains only 0.036 m of accuracy.
3. **The optimizers push several parameters to the search bounds.** `log_q_gyro` = −6 is its lower bound for GA, PSO and GWO (DE: −5.88), and `log_r_vel` approaches its upper bound of 2 (PSO reaches 2.00), so GPS velocity is almost ignored. Wider bounds could move the optimum further.
4. **HISA's search space misses the accurate region.** The filter re-estimates the initial biases within seconds, so all 7,465 Int1 finalists lie within 0.025 m of each other. Every finalist has `r_xy` = 1 m², the lowest value on HISA's grid, while `r_vel = r_xy/10` and the fixed Q exclude what the other optimizers found. Among the finalists, Int25 rank does not predict Int1 rank (Spearman ≈ 0.00).
5. **Compute.** Each population optimizer used exactly 10,000 evaluations per seed (4.1 × 10⁸ EKF steps, 11.8 min on the GPU). HISA used 828,611 literal evaluations, 44,790 Int1-normalized, and 2.3 × 10⁹ EKF steps in 45 min: about 1.15 × the work of all five seeds of one optimizer.
6. **Batched GPU evaluation matches the CPU reference:** within 7.4 × 10⁻⁸ m on the benchmark's 200-candidate check, and within 5.6 × 10⁻¹⁰ m on HISA's checks.

### Graphs

**Convergence: best-so-far whole-flight RMSE_h against evaluations (median and every seed)**

![Convergence](https://raw.githubusercontent.com/Niterousnebula/Joint-Tuning-of-QR-Matricies-in-GPS-IMU-EKF/results/results/benchmark_20261114_233504/convergence.png)

**Robustness: clean → perturbed RMSE_h, and ΔRMSE_h per perturbation**

![Robustness](https://raw.githubusercontent.com/Niterousnebula/Joint-Tuning-of-QR-Matricies-in-GPS-IMU-EKF/results/results/benchmark_20261114_233504/robustness.png)

**4D response: accuracy × robustness × convergence, coloured by compute** (interactive version: [response_4d.html](https://github.com/Niterousnebula/Joint-Tuning-of-QR-Matricies-in-GPS-IMU-EKF/blob/results/results/benchmark_20261114_233504/response_4d.html), download it and open it in a browser)

![4D response](https://raw.githubusercontent.com/Niterousnebula/Joint-Tuning-of-QR-Matricies-in-GPS-IMU-EKF/results/results/benchmark_20261114_233504/response_4d.png)

**HISA against the benchmark optimizers (marker size ∝ measured wall clock)**

![HISA comparison](https://raw.githubusercontent.com/Niterousnebula/Joint-Tuning-of-QR-Matricies-in-GPS-IMU-EKF/results/results/HISA/figures/comparison.png)

**HISA: Int25 → Int10 → Int1 progression and screening fidelity**

![HISA stage progression](https://raw.githubusercontent.com/Niterousnebula/Joint-Tuning-of-QR-Matricies-in-GPS-IMU-EKF/results/results/HISA/figures/stage_progression.png)

**HISA: best score for each R combination at each level**

![HISA R search](https://raw.githubusercontent.com/Niterousnebula/Joint-Tuning-of-QR-Matricies-in-GPS-IMU-EKF/results/results/HISA/figures/qr_search.png)

**Whole-flight horizontal error: raw GPS, baseline EKF, HISA winner**

![EKF error comparison](https://raw.githubusercontent.com/Niterousnebula/Joint-Tuning-of-QR-Matricies-in-GPS-IMU-EKF/results/results/HISA/figures/ekf_error_comparison.png)

More HISA figures (score against iteration, bias distributions, best-so-far) are in [`results/HISA/figures/`](https://github.com/Niterousnebula/Joint-Tuning-of-QR-Matricies-in-GPS-IMU-EKF/tree/results/results/HISA/figures).

## Reference results

Whole-flight horizontal RMSE against ground truth (`python run_baseline.py`):

| Source | RMSE_h | vs raw GPS | RMSE_v | RMSE_3D | Mean NIS (expected 6) |
|---|---|---|---|---|---|
| Raw GPS, no filter | 3.604 m | – | – | – | – |
| EKF, `THETA0` | 3.775 m | +0.171 m | 5.802 m | 6.922 m | 0.97 |

The baseline EKF is **worse than raw GPS** on whole-flight horizontal RMSE. Most of the GPS error against ground truth is a slowly varying bias that a GPS–IMU filter cannot see, so an optimizer can improve on the baseline EKF and still not beat raw GPS. Results always report both comparisons. A mean NIS of 0.97 against the expected 6 means the baseline is underconfident: Q or R is larger than the innovations support.

Robustness of `THETA0`, $\Delta\text{RMSE}_h$ against its clean 3.775 m:

| Perturbation | RMSE_h | ΔRMSE_h | ΔRMSE_h % |
|---|---|---|---|
| `gps_noise_1m` | 3.781 m | +0.006 m | +0.16 % |
| `gps_noise_2m` | 3.824 m | +0.049 m | +1.30 % |
| `gps_vel_noise` | 3.782 m | +0.007 m | +0.19 % |
| `imu_noise` | 3.776 m | +0.000 m | +0.01 % |
| `gps_drop_50` | 3.822 m | +0.047 m | +1.24 % |
| Mean / max | | +0.022 m / +0.049 m | |

One fitness evaluation, a full-flight run, takes about 6–8 s on one core of the machine used here.

## The EKF

The filter is `GpsImuEKF` in `ekf/filter.py`, a subclass of filterpy's `ExtendedKalmanFilter`. filterpy runs the covariance algebra: `predict()` forms $P^- = F P F^\top + Q$, and `update()` forms $S$, $K$, the state correction and the Joseph-form covariance. The subclass supplies the nonlinear motion model, its Jacobian and the per-step $Q$ by overriding `predict_x`.

### Changes from the earlier filter

| Earlier filter | Now | Why |
|---|---|---|
| Hand-written predict and update | filterpy `ExtendedKalmanFilter` | One tested implementation of the Kalman equations |
| Q added per IMU step, with no `dt` | Continuous noise densities, discretised every step as $G Q_c G^\top \Delta t$ | A 1.1 s IMU gap now gets 11 times the noise of a 0.1 s step. Tuned values no longer depend on the IMU rate |
| Additive noise on all four quaternion components | Gyro noise mapped into $q$ through $\tfrac{1}{2}\Xi(q)$ | Noise stays in the tangent space of the unit quaternion, instead of also inflating its norm |
| Euler quaternion step | Exact step for constant body rate | Keeps $q$ close to unit norm at 10 Hz, where rates reach 1 rad/s |
| Position step $p + v\,\Delta t$ | $p + v\,\Delta t + \tfrac{1}{2} a_n \Delta t^2$ | Consistent with the velocity step |
| GPS position only | GPS position **and velocity** | Velocity makes yaw and accelerometer bias observable. The earlier bias drifted to 5.8 m/s² |
| `THETA0` from order-of-magnitude guesses | `THETA0` from the flight's own sensor statistics | See [Decision vector](#decision-vector) |

### State (16)

| Symbol | Indices | Meaning | Units |
|---|---|---|---|
| $p$ | `x[0:3]` | Position, ENU | m |
| $v$ | `x[3:6]` | Velocity, ENU | m/s |
| $q$ | `x[6:10]` | Attitude quaternion, body to navigation, Hamilton, scalar first | – |
| $b_a$ | `x[10:13]` | Accelerometer bias | m/s² |
| $b_g$ | `x[13:16]` | Gyroscope bias | rad/s |

Conventions: ENU navigation frame centred on the first GPS fix; body frame x forward, y right, z down (`az` reads about −1 g); gravity $g = [0, 0, -9.80665]$; $R(q)$ maps body vectors into the navigation frame.

### 1. Initialise (`init_state`)

| State | `x0` | `P0` diagonal |
|---|---|---|
| $p$ | First GPS fix | 25 (5 m) |
| $v$ | GPS velocity at that fix | 2.25 (1.5 m/s) |
| $q$ | Roll and pitch from the mean specific force over the first 1 s, yaw 0 | 0.07 (about 30°) |
| $b_a$ | 0 | 0.25 (0.5 m/s²) |
| $b_g$ | 0 | 0.0027 (about 3 °/s) |

Yaw 0 means the body x axis points east. There is no magnetometer, so yaw is corrected from GPS velocity once the vehicle moves.

### 2. Predict (every IMU sample, `predict_x` then filterpy's `predict`)

With the IMU sample at $k-1$ held over $\Delta t$:

$$a = a_m - b_a, \qquad \omega = \omega_m - b_g, \qquad a_n = R(q)\,a + g$$

$$
\begin{aligned}
p_k &= p + v\,\Delta t + \tfrac{1}{2}\,a_n\,\Delta t^2 \\
v_k &= v + a_n\,\Delta t \\
q_k &= \Phi_q(\omega)\,q, \qquad \Phi_q = \cos\tfrac{\lVert\omega\rVert\Delta t}{2}\,I_4 + \frac{\sin(\lVert\omega\rVert\Delta t/2)}{\lVert\omega\rVert}\,\Omega(\omega) \\
b_{a,k} &= b_a, \qquad b_{g,k} = b_g
\end{aligned}
$$

**Jacobian $F$**, taken at the previous state (rows and columns in the order $p, v, q, b_a, b_g$):

$$
F =
\begin{bmatrix}
I & I\Delta t & \tfrac{1}{2}\Delta t^2\,\dfrac{\partial (R a)}{\partial q} & -\tfrac{1}{2}\Delta t^2 R & 0 \\
0 & I & \Delta t\,\dfrac{\partial (R a)}{\partial q} & -\Delta t\,R & 0 \\
0 & 0 & \Phi_q & 0 & -\dfrac{\partial (\Phi_q q)}{\partial \omega} \\
0 & 0 & 0 & I & 0 \\
0 & 0 & 0 & 0 & I
\end{bmatrix}
$$

$\partial(\Phi_q q)/\partial\omega$ is exact (`d_quat_step_dw`). Every block of $F$ matches central finite differences on random states to better than $10^{-7}$.

**Process noise.** Four white-noise inputs drive the model: accelerometer noise $n_a$, gyroscope noise $n_g$, and the random walks of the two biases, $n_{ba}$ and $n_{bg}$. With

$$
G =
\begin{bmatrix}
0 & 0 & 0 & 0 \\
-R(q) & 0 & 0 & 0 \\
0 & -\tfrac{1}{2}\Xi(q) & 0 & 0 \\
0 & 0 & I & 0 \\
0 & 0 & 0 & I
\end{bmatrix},
\qquad
Q_c = \mathrm{diag}(q_{acc} I_3,\; q_{gyro} I_3,\; q_{ba} I_3,\; q_{bg} I_3)
$$

the per-step noise is $Q = G\,Q_c\,G^\top\,\Delta t$, and filterpy forms $P^- = F P F^\top + Q$.

**Normalise.** $q \leftarrow q/\lVert q\rVert$, and $P$ is projected with $J = (I_4 - \bar q \bar q^\top)/\lVert q\rVert$ on its quaternion rows and columns, so it loses the same part of $q$ that normalising removes.

### 3. Update (each GPS fix, filterpy's `update`)

$$
z = \begin{bmatrix} p_{gps} \\ v_{gps} \end{bmatrix}, \qquad
h(x) = \begin{bmatrix} p \\ v \end{bmatrix}, \qquad
H = \begin{bmatrix} I_6 & 0_{6\times 10} \end{bmatrix}, \qquad
R = \mathrm{diag}(r_{xy}, r_{xy}, r_z, r_{vel}, r_{vel}, r_{vel})
$$

filterpy computes $y = z - h(\hat x^-)$, $S = H P^- H^\top + R$, $K = P^- H^\top S^{-1}$, $\hat x = \hat x^- + K y$ and $P = (I-KH)P^-(I-KH)^\top + K R K^\top$. Then $q$ is normalised as above. NIS is $y^\top S^{-1} y$; a consistent filter averages 6, the size of $z$.

### 4. Score

Estimated positions are interpolated onto the ground-truth times over the **complete flight**, and

$$\text{RMSE}_h = \sqrt{\frac{1}{N}\sum_{i=1}^{N} \big[(e_i - \hat e_i)^2 + (n_i - \hat n_i)^2\big]}$$

where $N$ is every ground-truth sample in the flight. Vertical RMSE, 3D RMSE and mean NIS are reported as secondary diagnostics.

## Q/R tuning

### Decision vector

Seven values, each $\log_{10}$ of a variance or noise density:

| Index | Name | Fills | `THETA0` | Bounds | Units of the variance |
|---|---|---|---|---|---|
| 0 | `log_q_acc` | $q_{acc}$, accelerometer white noise | 0.2 | ±3 | (m/s²)²·s |
| 1 | `log_q_gyro` | $q_{gyro}$, gyroscope white noise | −3 | ±3 | (rad/s)²·s |
| 2 | `log_q_ba` | $q_{ba}$, accelerometer bias random walk | −5 | ±3 | (m/s²)²/s |
| 3 | `log_q_bg` | $q_{bg}$, gyroscope bias random walk | −8 | ±3 | (rad/s)²/s |
| 4 | `log_r_xy` | GPS horizontal position | 1.0 | ±2 | m² |
| 5 | `log_r_z` | GPS vertical position | 1.5 | ±2 | m² |
| 6 | `log_r_vel` | GPS velocity | 0.0 | ±2 | (m/s)² |

Where `THETA0` comes from: the accelerometer norm has a standard deviation of 4.1 m/s² at 10 Hz (vibration and aliasing), giving $4.1^2 \times 0.1 \approx 1.6$; gyroscope noise of about 0.1 rad/s gives $10^{-3}$; GPS position noise of 3 m and 6 m; GPS velocity variance of about 1 (m/s)², as logged in `s_variance`. The two bias values are order-of-magnitude choices. `LOWER_TIGHT` and `UPPER_TIGHT` halve the bounds. In 48 random points inside the full bounds, no run diverged.

### Canonical fitness

There is one fitness function, `ekf.fitness(data, theta)` in `ekf/tuning.py`:

```text
candidate theta
      ↓
theta_to_QR(theta)  →  Q_c, R
      ↓
complete 16-state EKF over the whole flight
      ↓
complete estimated trajectory vs complete ground truth
      ↓
whole-flight horizontal RMSE
      ↓
fitness J(theta)
```

$$J(\theta) = \text{RMSE}_{h,\text{whole}}$$

A run that diverges scores `PENALTY = 1000` m. The baseline `THETA0` and every optimizer (GA, PSO, DE, GWO and any later one) are scored by this function. Optimizers do not reimplement it; they call it through `FitnessTracker` (`ekf.make_cost_fn`), which:

- counts evaluations and stops at `max_evals` (`BudgetExhausted` for single calls; `evaluate_many` scores only what still fits),
- clips candidates to the bounds,
- records $J_k$, the best-so-far $J^*(k)$ and elapsed time after every evaluation,
- can score a population in parallel (`n_workers`), recording results in population order, so the count and the curve are the same as in a serial run,
- or, with `backend="gpu"`, score the whole population as one batched EKF on the GPU (next section),
- numbers each objective call as a generation and records `generation` with every evaluation.

The perturbed copies of the flight are never used inside the fitness.

### GPU batched evaluation (`ekf/gpu.py`)

```text
CPU reference:   candidate 1 -> EKF, candidate 2 -> EKF, ..., candidate 200 -> EKF
GPU backend:     [200 candidates] -> batched EKF -> [200 fitness values]
```

Only the candidate dimension is batched. The time recursion is the reference loop unchanged: the same IMU predictions, the same GPS updates in the same order, quaternion normalisation after each, and the same divergence checks. Each candidate keeps its own state, covariance, $Q_c$ and $R$. All arithmetic is float64. The small quaternion matrices come from constant matrices built from the reference helpers in `filter.py`, so both backends use the same functions. Each IMU step (prediction plus 0, 1 or 2 GPS updates) is recorded once as a CUDA graph and replayed, which removes Python overhead from the 27 048-step loop.

**Equivalence.** `ekf.gpu.compare_with_cpu(data, thetas)` scores the same candidates with both backends; the accepted tolerance is `TOL_RMSE = 1e-6` m and `TOL_NIS_REL = 1e-6`. On the RTX 4060, for GA seed 1's first generation (200 candidates, `results/validation_gpu_vs_cpu/` on the `results` branch), 195 candidates agreed to better than $10^{-12}$ m. The largest fitness difference was $7.4 \times 10^{-8}$ m, for a strongly overconfident candidate (mean NIS 110) whose ill-conditioned updates amplify rounding. The largest relative difference in mean NIS was $7.6 \times 10^{-9}$. A candidate that diverges fails on both backends and scores `PENALTY`.

**Reported values always come from the reference.** `run_summary` re-scores the final best theta with the filterpy filter on the CPU, and the robustness runs are CPU runs too. The GPU only drives the search. `best_fitness_search` is the value the search saw and `search_vs_final_diff` shows the (rounding-level) difference.

**Speed.** One batch of 200 candidates takes about 15 s on an RTX 4060 Laptop GPU, about 0.07 s per evaluation against about 7 s for one CPU evaluation.

### Experimental structure

```text
Original clean flight
        │
        ├── Q/R optimization ── whole-flight horizontal RMSE (fitness)
        │
        └── Robustness evaluation of the tuned theta (no re-tuning)
                ├── GPS position noise (1 m, 2 m)
                ├── GPS velocity noise
                ├── IMU noise
                └── GPS dropout
```

### Rules for a fair comparison

- Same parameterization (`THETA_NAMES`), same bounds (`LOWER`, `UPPER`), same fitness (`make_cost_fn`) and same budget (`max_evals`) for every optimizer.
- Starting population: `np.random.default_rng(seed).uniform(lower, upper, (n_pop, n_dim))`; `THETA0` is not included.
- Budget: every fitness call counts, including the first population. Stop as soon as `n_evals` reaches `max_evals`.
- Seeds are fixed and recorded, so every run can be reproduced.
- Each run is written with `run_summary` and `save_run`, so every optimizer produces the same files: `results/<NAME>/runs.csv` (one row per seed) and `results/<NAME>/convergence_seed<seed>.csv`.
- Each optimizer branch puts its settings (population size, budget, seeds and algorithm-specific values) in its own config file.

## Evaluation framework

Optimizers are compared on four dimensions, always in this order:

**Accuracy → Robustness → Convergence efficiency → Computational cost**

| # | Dimension | Question | Primary metric | Column in `runs.csv` |
|---|---|---|---|---|
| 1 | Accuracy | Is the final solution good? | $\text{RMSE}_{h,\text{whole}}$ of the best theta | `rmse_h` |
| 2 | Robustness | Does it stay good under perturbations? | $\Delta\text{RMSE}_h$ | `delta_rmse_mean`, `delta_<perturbation>` |
| 3 | Convergence efficiency | How quickly was a good solution found? | $J^*(k)$ against $k$ | `convergence_seed<seed>.csv`; summary `evals_to_converge` |
| 4 | Computational cost | What effort was needed? | $N_{eval}$ | `n_evals`; secondary `t_total`, `t_eval` |

### 1. Accuracy

$\text{RMSE}_{h,\text{whole}}$ is the primary metric. Two comparisons are always reported separately:

$$\Delta\text{RMSE}_{EKF} = \text{RMSE}_{optimized} - \text{RMSE}_{baseline} \qquad (\texttt{delta\_vs\_baseline}, \texttt{improvement\_vs\_baseline\_pct})$$

$$\Delta\text{RMSE}_{GPS} = \text{RMSE}_{optimized} - \text{RMSE}_{GPS} \qquad (\texttt{delta\_vs\_gps})$$

An optimizer that improves on the baseline EKF but has $\Delta\text{RMSE}_{GPS} > 0$ is still worse than raw GPS and is not described as improving overall navigation. Secondary diagnostics: `rmse_v`, `rmse_3d`, `mean_nis`.

**NIS.** The GPS measurement is $z = [p_x, p_y, p_z, v_x, v_y, v_z]^\top$, so $m = 6$ (`NIS_DOF`). A correctly specified filter has $E[\text{NIS}] \approx 6$. Values well below 6 mean the filter is underconfident and values well above 6 mean it is overconfident. A filter that copies GPS can have low RMSE with NIS far from 6, so NIS is read alongside RMSE.

### 2. Robustness

The tuned theta is held fixed. For each perturbation in `PERTURBATIONS` the complete flight is run again, and

$$\Delta\text{RMSE}_h = \text{RMSE}_{h,\text{perturbed}} - \text{RMSE}_{h,\text{clean}}, \qquad \Delta\text{RMSE}_{h,\%} = 100\,\frac{\text{RMSE}_{h,\text{perturbed}} - \text{RMSE}_{h,\text{clean}}}{\text{RMSE}_{h,\text{clean}}}$$

Smaller is better; $\approx 0$ means little degradation. Both RMSE values cover the entire flight. `delta_rmse_mean` (mean over the perturbations) is the summary score and `delta_rmse_max` is the worst case.

| Name | Change |
|---|---|
| `gps_noise_1m`, `gps_noise_2m` | White noise of 1 m and 2 m added to GPS position |
| `gps_vel_noise` | White noise of 0.3 m/s added to GPS velocity |
| `imu_noise` | 0.5 m/s² added to the accelerometer and 1 °/s to the gyroscope |
| `gps_drop_50` | Half the GPS fixes removed at random |

Each perturbation uses its own fixed seed, so every optimizer sees the same noise. These are stress tests of one flight, not validation on an independent flight.

### 3. Convergence efficiency

For each run, the best-so-far fitness after every evaluation is recorded:

$$J^*(k) = \min_{i \le k} J_i$$

The convergence curve, best-so-far whole-flight RMSE against evaluation count $k$, is the primary representation and is plotted for every optimizer and seed. As a single number, `evals_to_converge` is the first $k$ at which the run has made 99 % of its total improvement from $J^*(1)$ to $J^*(N)$; `evals_to_best` is the $k$ of the final best. Two optimizers with the same final RMSE can differ widely here.

### 4. Computational cost

The primary, hardware-independent measure is $N_{eval}$:

> One evaluation = one complete execution of the 16-state EKF over the full flight for one candidate Q/R parameter set, followed by calculation of the fitness.

Every optimizer gets the same `max_evals`, so $N_{eval}$ is the same for all of them by design. `n_evals` (evaluations recorded) and `n_objective_evals` (candidates the backend actually scored) are counted independently, and both must equal `max_evals`. `n_generations` is the number of objective calls and `population_size` the configured population.

Secondary, hardware-dependent measures: $T_{total}$ (`t_total`, wall-clock seconds for the run), $T_{eval} = T_{total}/N_{eval}$ (`t_eval`), and $T_{obj}$ (`t_objective`, time inside the objective only; `t_objective_per_eval`). Wall-clock values are only comparable when measured on one machine with the same `backend` and `n_workers`.

### Result table

Every run produces a row with these values, so the comparison can be laid out as:

| Optimizer | Whole-flight RMSE_h | vs baseline | vs GPS | Robustness ΔRMSE_h | Evals to converge | N_eval | T_total | T_eval |
|---|---|---|---|---|---|---|---|---|
| Baseline | 3.775 m | – | +0.171 m | +0.022 m | – | 1 | – | – |
| GA, PSO, DE, GWO | from `results/<NAME>/runs.csv` | | | | | | | |

## Results (this branch)

| Path | Holds |
|---|---|
| `results/baseline/` | `THETA0` through the canonical fitness: `runs.csv`, `convergence_seed0.csv`, trajectory, innovations, GPS vs ground truth, error time series and error summary |
| `results/<GA, PSO, DE, GWO>/` | Each optimizer's runs, merged in from its branch: `runs.csv` (one row per seed, written by `ekf.save_run`) and `convergence_seed<seed>.csv` |
| `results/results_table.md`, `.csv` | Mean ± std per optimizer, in the order accuracy, robustness, convergence, compute, with comparisons against both the baseline EKF and raw GPS |
| `results/convergence.png` | Best-so-far whole-flight RMSE $J^*(k)$ against evaluations $k$, every seed and the median per optimizer, with the baseline and raw GPS as reference lines |
| `results/robustness.png` | Clean → perturbed whole-flight RMSE, and $\Delta\text{RMSE}_h$ per perturbation |
| `results/response_4d.png`, `.html` | 4D response graph: accuracy (whole-flight RMSE_h), robustness (mean $\Delta\text{RMSE}_h$) and convergence (evals to 99 % of improvement) on the axes, computational cost as colour |

The colour of the 4D graph is $N_{eval}$ when it differs between optimizers. Under the fixed budget it is the same for all of them, so the graph shows $T_{total}$ instead and states $N_{eval}$ in the colour-bar label.

To refresh after adding optimizer results:

```bash
python run_baseline.py
python plot_response.py
```

`plot_response.py` needs `matplotlib`, and `plotly` for the interactive graph.

### GPU benchmark (`run_benchmark.py`)

This branch holds all four optimizers, merged from `ga`, `pso`, `de` and `gwo`. `run_benchmark.py` runs them one after another on the GPU objective (`ekf/gpu.py`) under one fixed budget:

| Setting | Value |
|---|---|
| Population | 200 for every optimizer |
| Evaluations per seed | 10,000 (50 generations of 200 candidates; DE: initial population plus 49 trial generations) |
| Seeds | 1, 2, 3, 4, 5 |
| Order | GA (all seeds, saved) → PSO → DE → GWO; one optimizer at a time, in one process |

```bash
python run_benchmark.py                   # full benchmark -> results/benchmark_<stamp>/
python run_benchmark.py --validate-only   # only the CPU vs GPU check
python run_benchmark.py --smoke           # tiny budget, tests the pipeline end to end
python plot_response.py --root results/benchmark_<stamp>   # rebuild table and figures
```

Each run gets its own folder, `results/benchmark_<YYYYmmdd_HHMMSS>/`, and an existing folder is never reused, so earlier runs are never overwritten:

| Path | Holds |
|---|---|
| `manifest.json` | Budget, every optimizer's config, git commit, Python, PyTorch, CUDA and GPU, start and finish times, validation summary, per-seed accounting checks |
| `benchmark.log` | Everything printed during the run |
| `validation/` | `validation_table.csv` (one row per candidate: CPU and GPU RMSE_h, RMSE_v, RMSE_3D, mean NIS, ok, and differences) and `validation_summary.json` |
| `baseline/` | `THETA0` through the same fitness |
| `GA/`, `PSO/`, `DE/`, `GWO/` | `runs.csv` (one row per seed), `convergence_seed<s>.csv` (one row per evaluation, with `generation`), and the PSO, DE and GWO live logs |
| `results_table.*`, `convergence.png`, `robustness.png`, `response_4d.*` | Written by `plot_response.py --root <folder>` at the end |

Before any optimizer runs, the benchmark checks the GPU objective. GA's first generation for seed 1 (200 candidates) goes through the real path (optimizer → `FitnessTracker` → batched GPU EKF), and the same candidates are then scored by the filterpy reference on the CPU. The benchmark stops if any fitness, RMSE_v or RMSE_3D difference exceeds `TOL_RMSE = 1e-6` m, if any mean NIS differs by more than `TOL_NIS_REL = 1e-6` (relative), or if a candidate fails on one backend but not the other.

After each optimizer, every seed is checked to have `n_evals` (evaluations recorded), `n_objective_evals` (candidates the GPU actually scored) and convergence-curve rows all equal to 10,000, so the evaluation count is measured, not inferred from the number of generations. The final RMSE, robustness and NIS in `runs.csv` come from the filterpy reference re-scoring the best θ (see "GPU batched evaluation"). `t_total`, `t_objective` and `n_generations` give the compute cost.

The benchmark refuses to start unless the `results` branch is checked out (`--allow-any-branch` overrides this) and all four configs set population 200, 10,000 evaluations, seeds 1–5 and `BACKEND = "gpu"`. A lock file, `results/.benchmark.lock`, stops a second benchmark from sharing the GPU.

### Results: GPU benchmark and HISA

| Folder | Run |
|---|---|
| `results/benchmark_20261114_233504/` | GA, PSO, DE, GWO: population 200, 10,000 evaluations × seeds 1–5, RTX 4060 Laptop GPU; 4 h 10 min in total. Validation passed (max GPU–CPU difference 7.4e-8 m); every seed used exactly 10,000 evaluations |
| `results/HISA/` | HISA: 746,496 → 74,650 → 7,465 candidates (Int25 → Int10 → Int1), same GPU; 45.2 min. Validation passed at all levels (max difference 5.6e-10 m). Includes `figures/` and `comparison_with_benchmark.md` |

All final metrics are re-scored by the filterpy reference. Mean ± std over 5 seeds; HISA is one deterministic run.

| Method | Whole-flight RMSE_h (m) | vs baseline EKF (m) | vs raw GPS (m) | Mean NIS (expect 6) | Robustness mean ΔRMSE_h (m) | Robustness max ΔRMSE_h (m) | Literal evaluations | EKF steps (predictions + updates) | T_total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (THETA0) | 3.775 | – | +0.171 | 0.97 | +0.022 | +0.049 | 1 | 40,576 | 5 s |
| GA | 3.546 ± 0.016 | −0.229 | −0.058 | 1.98 ± 1.59 | +0.284 ± 0.080 | +0.733 | 10,000 | 4.06 × 10⁸ | 11.8 min per seed |
| PSO | 3.499 ± 0.001 | −0.276 | −0.105 | 3.68 ± 0.32 | +0.373 ± 0.010 | +0.721 | 10,000 | 4.06 × 10⁸ | 11.8 min per seed |
| DE | 3.538 ± 0.012 | −0.237 | −0.066 | 2.75 ± 0.84 | +0.282 ± 0.014 | +0.623 | 10,000 | 4.06 × 10⁸ | 11.8 min per seed |
| GWO | 3.501 ± 0.001 | −0.274 | −0.103 | 3.79 ± 0.13 | +0.381 ± 0.027 | +0.730 | 10,000 | 4.06 × 10⁸ | 11.8 min per seed |
| HISA | 3.740 | −0.036 | +0.136 | 3.25 | +0.032 | +0.061 | 828,611 (44,790 Int1-normalized) | 2.32 × 10⁹ | 45.2 min |

What the HISA run shows:

- **Accuracy.** HISA improves on the baseline EKF by 0.036 m but stays 0.136 m worse than raw GPS. The four population optimizers reach 3.50–3.55 m, below raw GPS. HISA searches the initial bias states and R with Q fixed at `THETA0`. The filter re-estimates the initial biases within seconds, so they barely change the whole-flight error: all 7,465 Int1 finalists lie within about 0.025 m of each other.
- **R decides the result, and the winner sits on the edge of the grid.** Every Int1 finalist has `log10 r_xy = 0`, the lowest value searched; the winner has `log10 r_z = 0.5`, also the lowest. The four optimizers settled where HISA's grid cannot reach: `log10 r_xy` ≈ −0.3 to −0.65 (more trust in GPS position), `log10 r_vel` ≈ 1.7 to 2.0 (GPS velocity almost ignored, whereas HISA ties `r_vel = r_xy/10`), and `log10 q_gyro` ≈ −5.5 to −6 (THETA0: −3). A wider R grid, an untied `r_vel`, or searching Q as well would be the next steps.
- **Robustness.** The HISA winner degrades by only +0.032 m under the perturbations, close to the baseline, against +0.28 to +0.38 m for the four optimizers. Their lower `r_xy` makes the estimate follow GPS position more closely, so it also follows added GPS noise. Accuracy and robustness trade off along this axis.
- **Screening fidelity.** Among the Int1 finalists, the Spearman rank correlation between Int25 and Int1 scores is about 0.00, and between Int10 and Int1 it is −0.21 (`figures/stage_progression.png`). In this flat region the coarse levels do not preserve the order. They do separate the R values: the Int25 → Int10 cut kept almost only `r_xy = 1`.
- **Compute.** With 9 R combinations, HISA does 828,611 literal evaluations, 44,790 Int1-normalized, and 2.32 × 10⁹ EKF steps. That is about 5.7 times the steps of one 10,000-evaluation seed, and 1.15 times the steps of all five seeds of one optimizer. HISA's single run took 45 min of wall-clock time, against about 61 min for the five seeds of each other optimizer.

## Data

Files hold one ETH Zurich flight of about 2,714 s. They keep the original dataset's names but **hold cleaned values**: SI units, a local ENU frame, repeated GPS rows removed, sensors starting together and ground-truth timestamps corrected. Unusual samples are kept and flagged, because handling them is the filter's job.

| File | Rows | Rate | Columns |
|---|---|---|---|
| `RawAccel ETH` | 27,048 | 10 Hz | `t, ax, ay, az, is_gap` |
| `RawGyro ETH` | 27,048 | 10 Hz, same timestamps | `t, wx, wy, wz, gyro_gap` |
| `OnboardGPS ETH` | 13,530 | About 5 Hz | `t, imgid, lat, lon, alt, e, n, u, vel_e, vel_n, vel_u, s_variance, c_variance, is_jump` |
| `GroundTruth ETH` | 2,689 | About 1 Hz | `t, stretch, imgid, e, n, u, omega, phi, kappa, motion` |

| Column | Meaning |
|---|---|
| `t` | Time in seconds on the sensor clock, shared by all four files |
| `ax, ay, az` | Specific force, m/s², body frame |
| `wx, wy, wz` | Angular rate, rad/s, body frame, interpolated onto the accelerometer timestamps |
| `is_gap` | The next IMU sample is more than 0.5 s away |
| `gyro_gap` | Width in seconds of the gyro interval this sample was interpolated across |
| `imgid` | Camera frame number |
| `lat, lon, alt` | Original GPS position, degrees and metres |
| `e, n, u` | Local ENU position in metres; the first GPS fix is the origin |
| `vel_e, vel_n, vel_u` | GPS velocity, m/s, ENU |
| `s_variance, c_variance` | GPS speed and course variance, as logged |
| `is_jump` | Position jumped by more than 10 m/s from the previous fix |
| `stretch` | Part of the flight this ground-truth sample belongs to, 0 to 3 |
| `omega, phi, kappa` | Ground-truth attitude in degrees. These wrap at ±180° and `phi` nears ±90° |
| `motion` | `turn`, `hover` or `cruise` (159, 256 and 2,274 samples) |

## Known limits

1. **Single flight.** Calibration uses the complete ETH Zurich flight and there is no independent flight for validation. Results show how well each optimizer calibrates Q/R for this flight under a fixed budget, not how the tuned values generalize to other flights. The perturbation tests are not a substitute for an independent flight.
2. **GPS bias.** Ground truth differs from GPS by a slowly varying offset that no GPS–IMU filter can observe, which caps how far any tuning can get below raw GPS.
3. **RMSE alone can favour following GPS.** Very large process noise makes the filter copy GPS. Read `mean_nis` (consistent ≈ 6) alongside RMSE.
4. **Sensor-log breaks** at t = 774.49, 1772.16 and 2616.03 s, where GPS jumps by 4–13 m (`is_jump`). The filter has no outlier gate and takes a few seconds to recover.
5. **Ground-truth timing** is a fit, good to about ±0.3 s over most of the flight; the last 102 rows (`stretch` 3) are less certain.
6. **10 Hz IMU.** The IMU log is sparse and heavily aliased, so the accelerometer noise density is large and the prediction is weak between fixes.
7. **Wall-clock time** depends on the machine, the `backend` and `n_workers`; compare optimizers on $N_{eval}$ and on convergence curves first.
