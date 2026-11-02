# Joint Tuning of Q and R in a GPS–IMU EKF: Team Baseline

BTP I (MEMEC702), Netaji Subhas University of Technology, Mechanical Engineering.
Team: Anish Verma, Raghav Anthwal, Adrishikhar Deka, Sanvi Shokeen. Supervisor: Dr Vijayant Agarwal.

We tune the process noise covariance **Q** and the measurement noise covariance **R** together, offline, for one fixed 16-state GPS–IMU Extended Kalman Filter. `Q` and `R` are set once at the start of a run and stay constant; adaptive Q and R are future work. Four optimizers are benchmarked under identical conditions, against the filter's starting values:

- Genetic Algorithm (GA)
- Particle Swarm Optimization (PSO)
- Differential Evolution (DE)
- Grey Wolf Optimizer (GWO)

**This package is our common starting point, so that everyone uses the same data, variables, filter and results format.** Use the data and `config.py` unchanged. If you think either needs changing, raise it with the team so that everyone changes together.

## What's in this package

| Item | Purpose |
|---|---|
| `README.md` | Names, EKF steps, rules, and the outputs every optimizer must produce |
| `config.py` | The EKF steps and parameters, shared by everyone |
| `data/` | The cleaned ETH Zurich flight, already processed. Do not clean it again |

You write your optimizer and save its [required outputs](#required-outputs-for-each-optimizer). You need Python with `numpy` and `pandas`.

## Quick start

Run this from the package folder:

```python
import pandas as pd
import config as cfg


def load_data(data_dir="data"):
    accel = pd.read_csv(f"{data_dir}/RawAccel ETH")
    gyro = pd.read_csv(f"{data_dir}/RawGyro ETH")
    gps = pd.read_csv(f"{data_dir}/OnboardGPS ETH")
    gt = pd.read_csv(f"{data_dir}/GroundTruth ETH")
    return {
        "t_imu": accel["t"].to_numpy(),
        "accel": accel[["ax", "ay", "az"]].to_numpy(),
        "gyro": gyro[["wx", "wy", "wz"]].to_numpy(),
        "t_gps": gps["t"].to_numpy(),
        "p_gps": gps[["e", "n", "u"]].to_numpy(),
        "v_gps": gps[["vel_e", "vel_n", "vel_u"]].to_numpy(),
        "t_true": gt["t"].to_numpy(),
        "p_true": gt[["e", "n", "u"]].to_numpy(),
        "motion": gt["motion"].to_numpy(),
    }


data = load_data()
x0, P0 = cfg.init_state(data)
Q, R = cfg.theta_to_QR(cfg.THETA0)
x_log, innovation_log, mean_nis, ok = cfg.run_ekf(data, Q, R, x0, P0)
rmse_h = cfg.position_error(data["t_imu"], x_log, data["t_true"], data["p_true"])
```

**First run:** the EKF code in `config.py` has not yet been run in this form. It comes from a tested version of the filter, but the first person to run it should share the `rmse_h` above with the team. Everyone's result must then match it. In the earlier test version, this `theta0` gave a mean NIS of 0.22 and accelerometer bias no larger than about 1.5 m/s².

## Reference results

Horizontal RMS error against ground truth on this data:

| Source | Whole flight | Tuning part (`t < T_SPLIT`) | Held-out part | Notes |
|---|---|---|---|---|
| Raw GPS, no filter | 3.604 m | 3.812 m | 3.382 m | Check this with `gps_vs_ground_truth.csv` |
| This filter, `theta0` | – | – | – | First run fills this in. This is the value for the optimizers to beat |

**Do not compare with earlier numbers.** The deck's results and the earlier pipeline's used ground-truth timestamps that have since been corrected, so the 4.66 m for GPS and 4.558 m on slide 20 no longer apply.

## Data

### Files

Files have the same names as the original dataset but **hold cleaned values**, so do not mix them with the original files. They are comma-separated text, with no file extension. They hold one ETH Zurich flight of about 2,714 s.

| File | Rows | Rate | Columns |
|---|---|---|---|
| `RawAccel ETH` | 27,048 | 10 Hz | `t, ax, ay, az, is_gap` |
| `RawGyro ETH` | 27,048 | 10 Hz, same timestamps | `t, wx, wy, wz, gyro_gap` |
| `OnboardGPS ETH` | 13,530 | About 5 Hz | `t, imgid, lat, lon, alt, e, n, u, vel_e, vel_n, vel_u, s_variance, c_variance, is_jump` |
| `GroundTruth ETH` | 2,689 | About 1 Hz | `t, stretch, imgid, e, n, u, omega, phi, kappa, motion` |

| Column | Meaning |
|---|---|
| `t` | Time in seconds on the sensor clock. All four files share this clock |
| `ax, ay, az` | Specific force in m/s², body frame |
| `wx, wy, wz` | Angular rate in rad/s, body frame, interpolated onto the accelerometer timestamps |
| `is_gap` | The next IMU sample is more than 0.5 s away |
| `gyro_gap` | Width in seconds of the gyro interval this sample was interpolated across |
| `imgid` | Camera frame number |
| `lat, lon, alt` | Original GPS position, in degrees and metres |
| `e, n, u` | Local ENU position in metres. The first GPS fix is the origin |
| `vel_e, vel_n, vel_u` | GPS velocity in m/s, ENU |
| `s_variance, c_variance` | GPS speed and course variance, as logged |
| `is_jump` | The position jumped by more than 10 m/s from the previous fix |
| `stretch` | Which part of the flight this ground-truth sample belongs to, 0 to 3 |
| `omega, phi, kappa` | Ground-truth attitude in degrees. These wrap at ±180° and `phi` nears ±90°, so do not difference them directly |
| `motion` | `turn`, `hover` or `cruise` |

### Cleaning

The data is already cleaned, so do not clean it again. Cleaning converted values to SI units and a local ENU frame, removed repeated GPS rows, started all sensors at the same time, and corrected the ground-truth timestamps. Unusual samples are kept and flagged with `is_gap` and `is_jump`, because handling them is the filter's job.

## Conventions

| Item | Convention |
|---|---|
| Navigation frame | ENU: x east, y north, z up, centred on the first GPS fix |
| Body frame | x forward, y right, z down. `az` reads about -1 g, so z points down. x and y are a baseline choice; turning them only shifts the initial yaw |
| Gravity | `g = [0, 0, -GRAVITY]` with `GRAVITY = 9.80665` m/s² |
| Body-to-navigation rotation | $R(q)$ maps a body-frame vector into the navigation frame |
| Quaternion | Hamilton, scalar first: `q = [qw, qx, qy, qz]`, unit norm |
| Units | m, m/s, m/s², rad, rad/s, s |
| Time index | `k` for IMU samples, `j` for GPS fixes |
| Estimate suffix | `_pred` for before the GPS update ($\hat{x}^-$), no suffix for after it ($\hat{x}$) |

## Variable names

### State vector (16 states)

| Symbol | Code name | Indices | Size | Meaning | Units |
|---|---|---|---|---|---|
| $p$ | `p` | `x[0:3]` | 3 | Position | m |
| $v$ | `v` | `x[3:6]` | 3 | Velocity | m/s |
| $q$ | `q` | `x[6:10]` | 4 | Attitude quaternion | – |
| $b_a$ | `b_a` | `x[10:13]` | 3 | Accelerometer bias | m/s² |
| $b_g$ | `b_g` | `x[13:16]` | 3 | Gyroscope bias | rad/s |

### Sensor inputs and data

| Symbol | Code name | Size | Meaning |
|---|---|---|---|
| $a_m$ | `a_m` | 3 | One accelerometer reading (specific force, body frame) |
| $\omega_m$ | `w_m` | 3 | One gyroscope reading (body frame) |
| $\Delta t$ | `dt` | 1 | Time between IMU samples, from timestamps |
| $z$ | `z` | 3 | GPS position measurement in the local frame |
| – | `data` | – | The flight, as a dictionary with the keys below |
| – | `t_imu`, `t_gps`, `t_true` | n | Timestamps in seconds |
| – | `accel`, `gyro` | n × 3 | IMU readings in m/s² and rad/s |
| – | `p_gps`, `v_gps` | n × 3 | GPS fixes and velocities in the local frame |
| – | `p_true` | n × 3 | Ground-truth positions in the local frame |
| – | `motion` | n | Motion label of each ground-truth sample |

### EKF quantities

| Symbol | Code name | Size | Meaning |
|---|---|---|---|
| $\hat{x}$, $\hat{x}^-$ | `x`, `x_pred` | 16 | State estimate after and before the update |
| $P$, $P^-$ | `P`, `P_pred` | 16 × 16 | State covariance after and before the update |
| $Q$ | `Q` | 16 × 16 | Process noise covariance, per step |
| $R$ | `R` | 3 × 3 | GPS measurement noise covariance |
| $F$ | `F` | 16 × 16 | Discrete state transition Jacobian |
| $H$ | `H` | 3 × 16 | Measurement Jacobian |
| $y$ | `y` | 3 | Innovation, $z - h(\hat{x}^-)$ |
| $S$ | `S` | 3 × 3 | Innovation covariance |
| $K$ | `K` | 16 × 3 | Kalman gain |
| – | `nis` | 1 | Normalised innovation squared, $y^\top S^{-1} y$. A consistent filter averages 3 |
| – | `mean_nis` | 1 | `nis` averaged over a run |
| $R(q)$ | `R_nb` | 3 × 3 | Body-to-navigation rotation matrix |
| $\Omega(\omega)$ | `omega_matrix(w)` | 4 × 4 | Quaternion rate matrix |
| $\Xi(q)$ | `xi_matrix(q)` | 4 × 3 | Quaternion rate matrix, written in terms of $q$ |
| $x_0$, $P_0$ | `x0`, `P0` | – | Initial state and covariance |
| – | `x_log` | n × 16 | State after each IMU sample, in `STATE_COLUMNS` order |
| – | `innovation_log` | n × 8 | One row per GPS update, in `INNOVATION_COLUMNS` order |

## EKF steps

IMU samples drive the prediction and GPS fixes drive the correction. Prediction runs on every IMU sample. The update runs only when a GPS fix arrives. All of this is already in `config.py`, so this section explains it rather than asking you to write it.

### 1. Initialise (`init_state`)

| State | `x0` | `P0` diagonal | Equivalent standard deviation |
|---|---|---|---|
| `p` | First GPS fix, which is the origin | 25 | 5 m |
| `v` | `[vel_e, vel_n, vel_u]` at that fix | 2.25 | 1.5 m/s |
| `q` | Roll and pitch from the accelerometer mean over the first 1 s, yaw 0 | 0.07 | About 30° |
| `b_a` | Zero | 0.25 | 0.5 m/s² |
| `b_g` | Zero | 0.0027 | About 3 °/s |

Yaw 0 means the body x axis points east. With z down in the body and z up in ENU, level is `q = [0, 1, 0, 0]`, not the identity. Yaw has no magnetometer to set it, so the filter corrects it from GPS. The first fix sets `x0`, so the first update uses the second fix.

### 2. Predict (`ekf_predict`, every IMU sample)

These run in this order, because $F$ must use the previous state, not the newly integrated one.

**1. Corrected IMU readings**, using the sample at $k-1$, which is held over $\Delta t$:

$$a = a_{m,k-1} - b_{a,k-1}, \qquad \omega = \omega_{m,k-1} - b_{g,k-1}$$

**2. Jacobian at the previous state (`jacobian_F`)**

$$F = I + A\,\Delta t$$

$$
A =
\begin{bmatrix}
0 & I_3 & 0 & 0 & 0 \\
0 & 0 & \dfrac{\partial\,(R(q)\,a)}{\partial q} & -R(q) & 0 \\
0 & 0 & \tfrac{1}{2}\,\Omega(\omega) & 0 & -\tfrac{1}{2}\,\Xi(q) \\
0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0
\end{bmatrix}
$$

Rows and columns are in state order: $p, v, q, b_a, b_g$, and $q$ here is $q_{k-1}$.

**3. Euler integration, $\hat{x}^- = f(\hat{x}, u)$ (`f_state`)**, from this continuous model:

$$\dot{p} = v, \qquad \dot{v} = R(q)\,a + g, \qquad \dot{q} = \tfrac{1}{2}\,\Omega(\omega)\,q, \qquad \dot{b}_a = 0, \qquad \dot{b}_g = 0$$

$$
\begin{aligned}
p_k &= p_{k-1} + v_{k-1}\,\Delta t \\
v_k &= v_{k-1} + \big(R(q_{k-1})\,a + g\big)\,\Delta t \\
q_k &= q_{k-1} + \tfrac{1}{2}\,\Omega(\omega)\,q_{k-1}\,\Delta t \\
b_{a,k} &= b_{a,k-1} \\
b_{g,k} &= b_{g,k-1}
\end{aligned}
$$

**4. Covariance**

$$P^- = F\,P\,F^\top + Q$$

**5. Normalise $q$ and project $P$ (`normalize_state`)**

Normalising removes part of $q$, so $P$ must lose the same part:

$$\bar{q} = \frac{q}{\lVert q \rVert}, \qquad J = \frac{I_4 - \bar{q}\,\bar{q}^\top}{\lVert q \rVert}, \qquad P \leftarrow G\,P\,G^\top$$

$G$ is $I_{16}$ with its quaternion block, rows and columns 6 to 9, replaced by $J$. Without this, the filter keeps believing in corrections that normalising threw away. In a test run, accelerometer bias z climbed to 7.14 m/s² without it, and stayed within -0.03 to 0.57 m/s² with it.

### 3. Update (`ekf_update`, when a GPS fix arrives)

**Measurement model**

$$h(x) = p, \qquad H = \begin{bmatrix} I_3 & 0_{3\times 13} \end{bmatrix}$$

**Correction**

$$
\begin{aligned}
y &= z - h(\hat{x}^-) \\
S &= H\,P^-\,H^\top + R \\
K &= P^-\,H^\top\,S^{-1} \\
\hat{x} &= \hat{x}^- + K\,y
\end{aligned}
$$

**Covariance, in Joseph form.** It gives the same result as $(I - KH)P^-$ for this gain, but holds up better when an optimizer tries extreme Q and R values:

$$P = (I - KH)\,P^-\,(I - KH)^\top + K\,R\,K^\top$$

Then normalise $q$ and project $P$ again, as in the predict step.

### 4. Score (`position_error`)

Interpolate the estimated positions to `t_true`, then compute the horizontal RMS error:

$$\text{RMSE}_h = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\Big[(p_{x,i} - p^{\text{true}}_{x,i})^2 + (p_{y,i} - p^{\text{true}}_{y,i})^2\Big]}$$

### Matrix definitions

With $q = [q_w, q_x, q_y, q_z]$, $e = [q_x, q_y, q_z]^\top$ and $\omega = [\omega_x, \omega_y, \omega_z]^\top$:

$$
R(q) =
\begin{bmatrix}
q_w^2 + q_x^2 - q_y^2 - q_z^2 & 2(q_x q_y - q_w q_z) & 2(q_x q_z + q_w q_y) \\
2(q_x q_y + q_w q_z) & q_w^2 - q_x^2 + q_y^2 - q_z^2 & 2(q_y q_z - q_w q_x) \\
2(q_x q_z - q_w q_y) & 2(q_y q_z + q_w q_x) & q_w^2 - q_x^2 - q_y^2 + q_z^2
\end{bmatrix}
$$

$$
\Omega(\omega) =
\begin{bmatrix}
0 & -\omega_x & -\omega_y & -\omega_z \\
\omega_x & 0 & \omega_z & -\omega_y \\
\omega_y & -\omega_z & 0 & \omega_x \\
\omega_z & \omega_y & -\omega_x & 0
\end{bmatrix}
\qquad
\Xi(q) =
\begin{bmatrix}
-q_x & -q_y & -q_z \\
q_w & -q_z & q_y \\
q_z & q_w & -q_x \\
-q_y & q_x & q_w
\end{bmatrix}
$$

$$
\frac{\partial\,(R(q)\,a)}{\partial q} =
2\begin{bmatrix}
q_w\,a + e \times a & (e^\top a)\,I_3 + e\,a^\top - a\,e^\top - q_w\,[a]_\times
\end{bmatrix}
$$

The first block is the 3 × 1 column for $q_w$ and the second is the 3 × 3 block for $e$. $[a]_\times$ is the skew-symmetric cross-product matrix of $a$.

These formulas for $F$ were checked against central finite differences on 200 random states. The largest difference was about $2\times 10^{-10}$.

## What's in `config.py`

| Name | Returns or holds | Purpose |
|---|---|---|
| `STATE_COLUMNS`, `INNOVATION_COLUMNS` | Lists of names | Column order of the logs and output files |
| `GRAVITY`, `P0_DIAG`, `MAX_POSITION` | Values | Fixed filter parameters |
| `THETA_NAMES`, `THETA0` | Values | The decision vector and its starting point |
| `LOWER`, `UPPER`, `LOWER_TIGHT`, `UPPER_TIGHT` | Arrays | Search bounds |
| `T_SPLIT`, `PENALTY` | Values | Tuning and held-out split, and cost of a failed run |
| `theta_to_QR(theta)` | `Q, R` | Build the matrices from a candidate |
| `init_state(data)` | `x0, P0` | Fixed initial conditions |
| `f_state(x, a_m, w_m, dt)` | `x_pred` | Euler state propagation |
| `jacobian_F(x, a_m, w_m, dt)` | `F` | State transition Jacobian |
| `h_gps(x)`, `jacobian_H()` | `z_pred`, `H` | Measurement model |
| `normalize_state(x, P)` | `x, P` | Normalise $q$ and project $P$ |
| `ekf_predict(x, P, a_m, w_m, dt, Q)` | `x_pred, P_pred` | Full predict step |
| `ekf_update(x_pred, P_pred, z, R)` | `x, P, y, S, nis` | Full update step |
| `run_ekf(data, Q, R, x0, P0, t_end=None)` | `x_log, innovation_log, mean_nis, ok` | Run the filter. `ok` is false if the run failed |
| `position_error(t_est, p_est, t_true, p_true)` | `rmse_h` | Horizontal RMS error at `t_true` |
| `make_cost_fn(data, candidate_log=None)` | `cost_fn` | The one cost function every optimizer calls |
| `quat_to_rot(q)`, `omega_matrix(w)`, `xi_matrix(q)`, `skew(v)` | Matrices | Quaternion helpers |

## Q/R tuning

### Decision vector

Diagonal terms are grouped so that the search has 7 dimensions, with each value stored as $\log_{10}$ of a variance:

| Index | Name | Fills | `THETA0` | `LOWER` | `UPPER` |
|---|---|---|---|---|---|
| 0 | `log_q_pos` | Q diagonal, position | -4 | -7 | -1 |
| 1 | `log_q_vel` | Q diagonal, velocity | -1 | -4 | 2 |
| 2 | `log_q_att` | Q diagonal, quaternion | -4 | -7 | -1 |
| 3 | `log_q_ba` | Q diagonal, accelerometer bias | -6 | -9 | -3 |
| 4 | `log_q_bg` | Q diagonal, gyroscope bias | -8 | -11 | -5 |
| 5 | `log_r_xy` | R diagonal, GPS x and y | 1 | -1 | 3 |
| 6 | `log_r_z` | R diagonal, GPS z | 1.5 | -0.5 | 3.5 |

The log scale lets the search cover several orders of magnitude and keeps every variance positive.

`THETA0` is the starting filter, replacing hand tuning. `log_q_vel` and `log_q_att` come from the sensor noise in the flight, `log_r_xy` and `log_r_z` from GPS noise of 3 m and 6 m, and the other three are order-of-magnitude choices. Bounds are ±3 decades for Q and ±2 decades for R. In test runs of `THETA0`, 28 one-at-a-time changes and 100 random points inside these bounds, no run diverged.

`Q` is tuned per step, exactly as it appears in $P^- = F\,P\,F^\top + Q$, with no scaling by `dt`. Tuned `Q` values therefore belong to one IMU rate: values from ETH data at 10 Hz are not reused on rig data at 100 Hz, which is tuned separately.

### Search settings

| Name | Baseline value | Reason |
|---|---|---|
| `n_pop` | 20 | About three times `n_dim` |
| `max_evals` | 1,000 | Same budget of filter runs for all four, 50 iterations at `n_pop = 20` |
| `seeds` | 0, 1, 2, 3, 4 | Each optimizer is random, so one run says little |
| `T_SPLIT` | 1,364.04 s | Tune on the first half of the flight, and report on the held-out second half |
| `PENALTY` | 1,000 m | Cost of a failed run |

One full-flight filter run took 2.6 s on its own in the test version, and 5.3 s with six running together. Tuning runs cover half the flight. The main benchmark of 4 optimizers × 5 seeds × 1,000 runs is about 14 CPU-hours, or about 2.5 hours on 6 cores, and the tight-bounds case takes the same again.

### Rules for a fair comparison

All four optimizers follow these rules:

- **Starting population:** `np.random.default_rng(seed).uniform(lower, upper, (n_pop, n_dim))`, so the same seed gives every optimizer the same start. `THETA0` is not included.
- **Budget:** every `cost_fn` call counts, including the first population. An optimizer stops as soon as `n_evals` reaches `max_evals`, even mid-iteration.
- **Bounds:** any value outside the bounds is clipped to the nearest bound.
- **Scoring:** always `cfg.make_cost_fn(data)`, on this data.

### Function to write

Name your function by optimizer, with these arguments:

| Function | Returns |
|---|---|
| `run_ga(cost_fn, lower, upper, n_pop, max_evals, seed, settings)` | `result` |
| `run_pso(...)`, `run_de(...)`, `run_gwo(...)` | `result`, same arguments |

`settings` is a dictionary holding the algorithm-specific values below. `result` holds `best_theta`, `best_cost`, `evals_to_best`, `n_evals`, `wall_time` and `seed`.

### Shared variables

| Code name | Meaning |
|---|---|
| `theta` | One candidate, holding log variances |
| `n_dim` | Length of `theta` |
| `lower`, `upper` | Search bounds for `theta` |
| `pop` | All candidates, `n_pop × n_dim` |
| `n_pop` | Population size |
| `costs` | Cost of each candidate |
| `max_evals` | Budget of filter runs |
| `n_evals` | Filter runs used so far |
| `seed`, `rng` | Random seed and generator |
| `best_theta`, `best_cost` | Best solution found so far |
| `evals_to_best` | Value of `n_evals` when `best_cost` was last improved |
| `wall_time` | Seconds taken by one optimizer and seed |
| `settings` | Dictionary of algorithm-specific values |

### Algorithm-specific names

Baseline values are common textbook defaults, not tuned for this problem.

| Algorithm | Symbol | Code name | Baseline | Meaning |
|---|---|---|---|---|
| GA | – | `tournament_size` | 3 | Tournament selection: candidates compared in each pick |
| GA | – | `crossover_rate` | 0.8 | Probability a pair is crossed |
| GA | – | `blx_alpha` | 0.5 | BLX-α crossover on real values |
| GA | – | `mutation_rate` | 0.1 | Probability, per value, of Gaussian mutation |
| GA | – | `mutation_sigma` | 0.1 | Mutation standard deviation, as a fraction of each value's range |
| GA | – | `elite_count` | 2 | Best candidates kept unchanged. They are not re-scored |
| PSO | $\omega$ | `inertia` | 0.7 | Inertia weight, global-best swarm |
| PSO | $c_1$, $c_2$ | `c1`, `c2` | 1.5, 1.5 | Personal and global attraction |
| PSO | – | `v_max_frac` | 0.2 | Velocity limit, as a fraction of each value's range |
| PSO | – | `particle_vel` | Zero at start | Particle velocities, `n_pop × n_dim` |
| PSO | – | `pbest_pos`, `pbest_cost` | – | Best result of each particle |
| PSO | – | `gbest_pos`, `gbest_cost` | – | Best result of the swarm |
| DE | – | Strategy | rand/1/bin | Mutation and crossover scheme |
| DE | $F$ | `de_scale` | 0.5 | Mutation scale factor |
| DE | $CR$ | `de_crossover` | 0.9 | Crossover probability |
| DE | – | `mutant`, `trial` | – | Mutant and trial vectors |
| GWO | $a$ | `gwo_a` | 2 falling to 0 | Control value, falling linearly with `n_evals / max_evals` |
| GWO | $A$, $C$ | `gwo_A`, `gwo_C` | Standard GWO formulas | Coefficient vectors |
| GWO | – | `theta_alpha`, `theta_beta`, `theta_delta` | – | Three best wolves |

### Names that clash

The same letter means different things across the filter and the optimizers. Use the code name in the right-hand column.

| Letter | Filter meaning | Other meaning | Use in code |
|---|---|---|---|
| `R` | Measurement noise covariance | Rotation matrix $R(q)$ | `R`, `R_nb` |
| `F` | State Jacobian | DE scale factor | `F`, `de_scale` |
| `A` | Continuous Jacobian | GWO coefficient | `A`, `gwo_A` |
| `a` | Corrected acceleration | GWO control value | `a`, `gwo_a` |
| `q`, `Q` | Quaternion, process noise | Log variances in `theta` | `q`, `Q`, `log_q_*` |
| `v` | Velocity | PSO particle velocity | `v`, `particle_vel` |
| `w` | Angular rate, or quaternion scalar | PSO inertia weight | `w_m`, `qw`, `inertia` |
| `x` | EKF state | Optimizer candidate | `x`, `theta` |
| `g` | Gravity | PSO global best | `g`, `gbest_pos` |

## Required outputs for each optimizer

Every optimizer produces the same seven CSV files, with these exact names and columns, so that results can be compared directly:

```text
results/
├── GA/
│   ├── ekf_error_comparison.csv
│   ├── final_trajectory.csv
│   ├── gps_vs_ground_truth.csv
│   ├── gt_error_timeseries.csv
│   ├── innovations.csv
│   ├── optimized_parameters.csv
│   └── pre_trajectory.csv
├── PSO/   the same seven files
├── DE/    the same seven files
└── GWO/   the same seven files
```

Write each file with a header row, comma separators and no index column: `table.to_csv(path, index=False)`. Units are as in [Conventions](#conventions).

**Pre** is the filter with `THETA0`. **Final** is the filter with `best_theta` from the selected run. The **selected run** is the seed with the lowest `best_cost`, which uses only the tuning part of the flight, so the held-out part stays unseen. Pre and final trajectories always cover the whole flight.

### Summary of the files

| File | Rows | What it holds | Same for all optimizers? |
|---|---|---|---|
| `pre_trajectory.csv` | 27,048 | Full state of the pre filter | Yes |
| `final_trajectory.csv` | 27,048 | Full state of the final filter | No |
| `gps_vs_ground_truth.csv` | 2,689 | Raw GPS against ground truth, with no filter | Yes |
| `gt_error_timeseries.csv` | 2,689 | Error against time for GPS, pre and final | GPS and pre columns only |
| `innovations.csv` | 13,529 | Innovations of the final filter | No |
| `ekf_error_comparison.csv` | 22 | Error summary, including robustness | GPS and pre rows only |
| `optimized_parameters.csv` | 10 | Tuned values and metrics, per run | No |

Files or columns marked "yes" must match across teammates. If they don't, something differs in your setup.

### `pre_trajectory.csv` and `final_trajectory.csv`

One row per IMU sample, from `x_log`:

`t, p_e, p_n, p_u, v_e, v_n, v_u, qw, qx, qy, qz, b_ax, b_ay, b_az, b_gx, b_gy, b_gz`

### `gps_vs_ground_truth.csv`

One row per ground-truth sample, with GPS interpolated to `t_true`. Errors are GPS minus ground truth. The RMS of `err_h` must come to 3.604 m.

`t, gps_e, gps_n, gps_u, gt_e, gt_n, gt_u, err_e, err_n, err_u, err_h`

### `gt_error_timeseries.csv`

One row per ground-truth sample, holding horizontal error in metres. `part` is `tune` when `t < T_SPLIT`, otherwise `held_out`.

`t, part, motion, gps_err_h, pre_err_h, final_err_h`

### `innovations.csv`

One row per GPS update of the final filter, from `innovation_log`. `s_e`, `s_n` and `s_u` are the diagonal of `S`.

`t, y_e, y_n, y_u, s_e, s_n, s_u, nis`

### `ekf_error_comparison.csv`

One row per `case` and `source`, computed from horizontal error at the ground-truth samples in that case:

`case, source, rmse_h, mean_err_h, median_err_h, p95_err_h, max_err_h, n_samples`

| `case` | `source` | Samples scored |
|---|---|---|
| `whole` | `gps`, `pre`, `final` | All |
| `tune` | `gps`, `pre`, `final` | `t < T_SPLIT` |
| `held_out` | `gps`, `pre`, `final` | `t >= T_SPLIT` |
| `motion_hover`, `motion_cruise`, `motion_turn` | `gps`, `pre`, `final` | That `motion` label |
| `bounds_tight` | `final` | `t >= T_SPLIT` |
| `gps_noise_1m`, `gps_noise_2m`, `imu_noise` | `final` | All |

### `optimized_parameters.csv`

One row per run: 5 seeds with `bounds = baseline` and 5 with `bounds = tight`. `is_selected` is true for the selected run in each group. `best_cost` and `mean_nis` cover the tuning part. Both `rmse_h` columns use that run's `best_theta`.

`optimizer, bounds, seed, is_selected, log_q_pos, log_q_vel, log_q_att, log_q_ba, log_q_bg, log_r_xy, log_r_z, best_cost, mean_nis, rmse_h_held_out, rmse_h_whole, evals_to_best, n_evals, wall_time`

## Benchmark metrics

| Metric | Measured as | Where it comes from |
|---|---|---|
| Computation | `evals_to_best`: filter runs until the best result was found. `n_evals` itself is the same for all four, because the budget is fixed | `optimized_parameters.csv`, mean and standard deviation across seeds |
| Time | `wall_time`, in seconds. This is only comparable on one machine, so final timings are run on one machine with the same number of parallel runs | `optimized_parameters.csv`, mean and standard deviation across seeds |
| RMSE | `rmse_h`, in metres, mainly on the held-out part | `optimized_parameters.csv` across seeds, and `ekf_error_comparison.csv` for the selected run |
| Robustness | `rmse_h` in each case below | `ekf_error_comparison.csv` |

### Robustness cases

| `case` | What changes | Re-tune? |
|---|---|---|
| `bounds_tight` | Search bounds become `LOWER_TIGHT` and `UPPER_TIGHT`: Q ±1.5 decades and R ±1 decade around `THETA0`. Uses the same seeds and the same selection rule | Yes |
| `gps_noise_1m`, `gps_noise_2m` | Add `rng.normal(0, sigma, p_gps.shape)` to `p_gps`, with sigma 1 m and 2 m | No |
| `imu_noise` | Add `rng.normal(0, 0.5, accel.shape)` to `accel`, then `rng.normal(0, radians(1), gyro.shape)` to `gyro` | No |
| `motion_hover`, `motion_cruise`, `motion_turn` | Same run as final, scored only on the samples with that `motion` label | No |

Each noise case starts a fresh `rng = np.random.default_rng(0)`, so every optimizer sees the same noise.

`motion` is already in the data. A turn has a mean $\lvert\omega_z\rvert$ above 5 °/s over the surrounding 2 s, a hover has a ground-truth speed below 0.3 m/s and is not a turn, and everything else is cruise. This gives 159 turn, 256 hover and 2,274 cruise samples.

## Known limits

1. **Sensor-log breaks:** the sensor log has breaks at t = 774.49, 1772.16 and 2616.03 s, and GPS jumps by 4–13 m at each (`is_jump`). The filter sees these jumps with no warning and may take a few seconds to recover.
2. **Ground-truth timing:** the corrected timestamps are a fit, not a measurement, good to about ±0.3 s over most of the flight. The last 102 rows (`stretch` 3) are less certain.
3. **RMSE alone can favour following GPS:** in earlier tests, very high process noise made the filter follow GPS, with `mean_nis` near 0, far below the consistent value of 3. The cost stays `rmse_h` as the baseline, so check `mean_nis` when reading results.
4. **Accelerometer bias:** `b_a` x reached about -1.5 m/s² at `THETA0` in testing. This is large for a real bias, so `b_a` may be absorbing vibration or yaw error rather than measuring a true bias.
5. **IMU gaps:** `Q` is per step, so a 1.1 s gap adds the same `Q` as a 0.1 s step, and Euler integration holds one reading across the gap.
6. **Integration step:** Euler at 10 Hz is coarse, with a 0.1 s step. Rig data at 100 Hz will not behave the same.
7. **Sensor scales:** the accelerometer and gyroscope scales are datasheet ranges checked against the flight, not a calibration.
8. **Single flight:** the tuning and held-out parts come from the same flight, so the held-out score shows less than a separate flight would.
9. **Logging rig:** these scales, axes and tuned values apply to the ETH data. The rig (ESP32, MPU-6500 at 100 Hz, u-blox M6) needs its own.
