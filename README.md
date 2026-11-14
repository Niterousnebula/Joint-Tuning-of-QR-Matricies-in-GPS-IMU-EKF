# Joint Tuning of Q and R in a GPS–IMU EKF

BTP I (MEMEC702), Netaji Subhas University of Technology, Mechanical Engineering.
Team: Anish Verma, Raghav Anthwal, Adrishikhar Deka, Sanvi Shokeen. Supervisor: Dr Vijayant Agarwal.

We tune the process noise **Q** and the measurement noise **R** together, offline, for one fixed 16-state GPS–IMU Extended Kalman Filter. Q and R are set once at the start of a run and stay constant. Four optimizers are benchmarked under identical conditions against the hand-set starting values (`THETA0`):

- Genetic Algorithm (GA)
- Particle Swarm Optimization (PSO)
- Differential Evolution (DE)
- Grey Wolf Optimizer (GWO)

## Branches

| Branch | Holds |
|---|---|
| `main` | Data, the EKF, the shared cost function and this README. Everyone uses it unchanged |
| `ga`, `pso`, `de`, `gwo` | One optimizer each: its code and its config file, built on `main` |
| `results` | Runs of every optimizer against the baseline, and the 4D response graph |

Changes to the data, the filter or the cost function go into `main` first and are then merged into every branch, so all four optimizers always score candidates in the same way.

## Layout

| Path | Purpose |
|---|---|
| `data/` | The cleaned ETH Zurich flight. Do not clean it again |
| `ekf/data.py` | `load_data`, plus `perturb` and `ROBUSTNESS_DATASETS` for the robustness copies of the flight |
| `ekf/filter.py` | The EKF, built on `filterpy.kalman.ExtendedKalmanFilter` |
| `ekf/tuning.py` | Decision vector, bounds, `theta_to_QR`, `make_cost_fn` and the scoring functions |
| `run_baseline.py` | Runs `THETA0` and writes `results/baseline/` |

## Quick start

```bash
pip install -r requirements.txt
python run_baseline.py
```

```python
import ekf

data = ekf.load_data()
Q_c, R = ekf.theta_to_QR(ekf.THETA0)
x_log, innovation_log, mean_nis, ok = ekf.run_ekf(data, Q_c, R)
rmse_h = ekf.position_error(data["t_imu"], x_log, data["t_true"], data["p_true"])

cost_fn = ekf.make_cost_fn(data)      # what every optimizer calls
cost = cost_fn(ekf.THETA0)
```

## Reference results

Horizontal RMS error against ground truth (`python run_baseline.py`):

| Source | Whole flight | Tuning part (`t < T_SPLIT`) | Held-out part | Mean NIS |
|---|---|---|---|---|
| Raw GPS, no filter | 3.604 m | 3.812 m | 3.382 m | – |
| EKF, `THETA0` | 3.775 m | 3.939 m | 3.603 m | 0.97 |

Robustness of `THETA0`, held-out RMSE on each copy of the flight: clean 3.603 m, GPS noise 1 m 3.608 m, GPS noise 2 m 3.671 m, GPS velocity noise 3.607 m, IMU noise 3.602 m, half the GPS fixes dropped 3.661 m. Their standard deviation, the robustness score, is 0.029 m.

One full-flight run takes about 6 s and a tuning run (half the flight) about 3 s on one core.

Most of the GPS error against ground truth is slowly varying bias, which a GPS–IMU filter cannot see. The filter can therefore only beat raw GPS by a small margin on this flight, and the benchmark compares optimizers against `THETA0` rather than against GPS.

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

Estimated positions are interpolated onto the ground-truth times, and

$$\text{RMSE}_h = \sqrt{\frac{1}{N}\sum_i \big[(p_{e,i} - p^{true}_{e,i})^2 + (p_{n,i} - p^{true}_{n,i})^2\big]}$$

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

### Cost function

`make_cost_fn(data)` returns `cost_fn(theta)`, which runs the filter up to `T_SPLIT = 1364.04 s` and returns $\text{RMSE}_h$ over that part, or `PENALTY = 1000` m if the run fails. The second half of the flight is held out and never seen by the optimizers.

### Benchmark metrics

The `results` branch plots every optimizer run against the baseline on four axes:

| Axis | Measured as |
|---|---|
| RMSE | Held-out $\text{RMSE}_h$ of the tuned filter |
| Robustness | `ekf.robustness(data, theta)["std"]`: standard deviation of held-out $\text{RMSE}_h$ over the datasets in `ROBUSTNESS_DATASETS`. Lower is more stable |
| Compute | `evals_to_best`: filter runs until the best result was found |
| Time | `wall_time` of one optimizer run, in seconds, on one machine |

`ROBUSTNESS_DATASETS` (in `ekf/data.py`) are copies of the flight with ground truth unchanged. Each copy uses its own fixed seed, so every optimizer sees the same noise:

| Name | Change |
|---|---|
| `clean` | None |
| `gps_noise_1m`, `gps_noise_2m` | White noise of 1 m and 2 m added to GPS position |
| `gps_vel_noise` | White noise of 0.3 m/s added to GPS velocity |
| `imu_noise` | 0.5 m/s² added to the accelerometer and 1 °/s to the gyroscope |
| `gps_drop_50` | Half the GPS fixes removed at random |

### Rules for a fair comparison

- Starting population: `np.random.default_rng(seed).uniform(lower, upper, (n_pop, n_dim))`; `THETA0` is not included.
- Budget: every `cost_fn` call counts, including the first population; stop as soon as `n_evals` reaches `max_evals`.
- Bounds: values outside them are clipped to the nearest bound.
- Scoring: always `ekf.make_cost_fn(data)`.
- Each optimizer branch puts its settings (population size, budget, seeds and algorithm-specific values) in its own config file.

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

1. **Sensor-log breaks** at t = 774.49, 1772.16 and 2616.03 s, where GPS jumps by 4–13 m (`is_jump`). The filter has no outlier gate and takes a few seconds to recover.
2. **Ground-truth timing** is a fit, good to about ±0.3 s over most of the flight; the last 102 rows (`stretch` 3) are less certain.
3. **GPS bias.** Ground truth differs from GPS by a slowly varying offset that no GPS–IMU filter can observe, which caps how far any tuning can get below raw GPS.
4. **RMSE alone can favour following GPS.** Very large process noise makes the filter copy GPS. Check `mean_nis` (consistent ≈ 6) when reading results.
5. **10 Hz IMU.** The IMU log is sparse and heavily aliased, so the accelerometer noise density is large and the prediction is weak between fixes.
6. **Single flight.** Tuning, held-out and robustness datasets all come from the same flight.
