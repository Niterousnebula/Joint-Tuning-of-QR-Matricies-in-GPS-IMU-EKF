# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Offline joint tuning of the process noise Q and measurement noise R for one fixed 16-state GPS–IMU EKF, on one recorded flight or drive at a time: the ETH Zurich flight by default, or any other processed dataset the user selects. Four optimizers (GA, PSO, DE, GWO) are compared against the hand-set starting point `THETA0`. The README holds the full methodology and results; read it before changing anything that affects the numbers.

## Commands

No build step, no linter, no test suite. Python with `filterpy`, `numpy`, `pandas`, `matplotlib`, `plotly`; PyTorch only for the GPU backend.

```bash
pip install -r requirements.txt

# 1. Turn raw_data/ into processed_data/<id>/ (the EKF reads only processed_data/)
python -m data_pipeline                    # list datasets with their EKF status, pick one by number or id
python -m data_pipeline --list --filter inavwhu
python -m data_pipeline --select inavwhu/street00   # non-interactive

# 2. Run on a processed dataset (--dataset omitted: asks on a terminal, else uses eth)
python run_baseline.py --dataset eth       # THETA0 through the canonical fitness -> results/baseline/
python PSO.py --dataset inavwhu/street00   # all seeds in PSO_config.py -> results/inavwhu/street00/PSO_gpu/ (needs CUDA torch)
python PSO.py --seeds 1 --max-evals 40 --swarm 8   # quick smoke test
python PSO.py --backend cpu --workers 10   # filterpy reference, no GPU
python plot_response.py --dataset eth      # regenerate results/ tables and figures from runs.csv files
python plot_trajectories.py --dataset inavwhu/street00 --runs results/inavwhu/street00/PSO_gpu/runs.csv   # map: truth, GNSS, EKF
```

Scripts write to relative `results/` paths, so run them from the repo root. `results/` is not in the working tree; it lives on the `results` branch. The default dataset (`eth`) keeps the original `results/<NAME>/` layout; every other dataset writes to `results/<dataset id>/<NAME>/` (`ekf.results_dir`).

Verifying a change to the filter or fitness: `python run_baseline.py --dataset eth` should still report THETA0 RMSE_h ≈ 3.775 m (mean NIS 0.97, robustness mean +0.022 m). A GPU change should also pass `ekf.gpu.compare_with_cpu(data, thetas)`, whose tolerance is 1e-6 m, on a dataset with all six GPS channels and on one without velocity.

## Architecture

The pieces that only make sense together:

- **Fitness is the single source of truth.** `ekf.fitness(data, theta)` (`ekf/tuning.py`) maps the 7-D log10 vector → `theta_to_QR` → full-flight `GpsImuEKF` run → whole-flight horizontal RMSE. Diverged runs return `PENALTY` (1000 m). Every optimizer must score through `ekf.make_cost_fn`, which returns a `FitnessTracker`. That object enforces `max_evals`, clips to bounds, records best-so-far and timing per evaluation, and handles parallel CPU workers or the GPU backend. Changing the fitness, bounds, or budget logic changes every optimizer's results, so it belongs on `main` and is merged everywhere.
- **Two backends, one objective.** `backend="cpu"` uses filterpy (reference). `backend="gpu"` (`ekf/gpu.py`) batches the candidate dimension in float64 and replays a CUDA graph per IMU step. Search may use either, but reported numbers are always re-scored on the CPU reference: `run_summary` calls `evaluate` on `tracker.best_theta`, and robustness runs are CPU as well. `search_vs_final_diff` shows any gap.
- **Filter.** `GpsImuEKF` (`ekf/filter.py`) subclasses filterpy's `ExtendedKalmanFilter`. It overrides only `predict_x` (nonlinear motion model, Jacobian `F`, normalisation) and supplies `H` and `h_gps`; filterpy does the covariance algebra. Q is continuous-time, discretised per step as `G Q_c Gᵀ Δt`, so `THETA0` is independent of IMU rate. A GPS fix updates whichever of the six channels `[e, n, u, ve, vn, vu]` the dataset provides (`ekf.measurement_channels`: finite at every fix; NaN in the processed data means the source has none), using the matching rows of `H` and block of `R`, so NIS has `ekf.nis_dof(data)` degrees of freedom (6 for ETH, 3 for a position-only F9P dataset). The CPU filter and `ekf/gpu.py` implement the same masking and must stay in step. `run_ekf` is the entry point; `init_state` sets x0/P0 from the first second of IMU and the first GPS fix (a missing channel starts at 0, so no velocity means a start at rest). Two behaviours are switched by `ekf.filter.OPTIONS` and recorded in every `runs.csv` row: `init_heading` (on) turns the initial yaw to the GNSS course of the first leg when the dataset has no GPS velocity, assuming the vehicle moves along body x (playground00: start error 91° to 5°, `THETA0` 1.133 to 1.113 m; no effect on ETH); `gate` (off) is a chi-square gate on GPS fixes that was measured to hurt ETH (`THETA0` 3.775 to 5.108 m, the tuned theta 3.499 to 22.3 m), so leave it off. The GPU evaluator reads `OPTIONS` when it is built, so set them before creating it.
- **Decision vector** (`THETA_NAMES`, `THETA0`, `LOWER`/`UPPER`, in `ekf/tuning.py`): `[log_q_acc, log_q_gyro, log_q_ba, log_q_bg, log_r_xy, log_r_z, log_r_vel]`. Bounds are ±3 for the Q terms and ±2 for the R terms around `THETA0`. `LOWER_TIGHT`/`UPPER_TIGHT` exist but are not used by the optimizers. Optimizers must draw starting populations from `np.random.default_rng(seed).uniform(LOWER, UPPER, ...)` without seeding in `THETA0`.
- **Output contract.** Every optimizer writes the same two artifacts through `ekf.run_summary` and `ekf.save_run`: `results/<NAME>/runs.csv` (one row per seed) and `convergence_seed<seed>.csv`. `plot_response.py` reads those for the comparison table and figures. Keep this schema when adding an optimizer.
- **Robustness.** `ekf/data.py` builds perturbed copies of the flight (`PERTURBATIONS`, fixed seeds). These are used only to score the final theta. They are never part of fitness.
- **Dataset pipeline (`data_pipeline/`).** Raw files in `raw_data/` (not in git; large) are read by adapters into one canonical `Recording` (IMU in SI units, GPS and GT in ENU, time rebased to the first IMU sample, a missing channel NaN). `core.py` owns the format, registry, cleaning (gap splitting, GT clipped to the IMU window and decimated to about 10 Hz) and writing, and should not change for new data. Each raw layout is an adapter: `eth.py`, `quadrotor.py`, `inavwhu.py` (increments to rates, NED to ENU, GNSS origin from the stationary start), `shikmona.py`, and `table.py` (declarative `TableSpec` per xlsx/csv file or glob, the place for units, scales, GPS boxes, gap splitting and `notes`; used by `land_vehicle.py` and `remote_control_car.py`). A new file in a known layout is a new `TableSpec` entry; a new layout is a new adapter module that calls `@register`. Every processed dataset has a `meta.json` with its quirks and an `ekf_ready` verdict (GPS, ground truth, length, coverage, static start) with reasons.
- **Dataset loading (`ekf/data.py`).** `ekf.load_data(dataset)` reads `processed_data/<id>/` into the dict the filter uses (`t_imu, accel, gyro, t_gps, p_gps, v_gps, t_true, p_true, motion, dataset`); it processes the dataset from `raw_data/` first if needed and refuses one that is not `ekf_ready` unless `allow_unready=True` (`--force` on `run_baseline.py`). `ekf.select_dataset` is the interactive menu. Dataset differences are kept out of `ekf/` by the processed format, so a new dataset needs an adapter, not a change to the filter.

## Branch layout

The README describes the intended branch split: `main` holds the shared code (filter, fitness, data pipeline) and the README; `ga`, `pso`, `de`, `gwo`, `hisa` each add one optimizer and its config; `results` holds outputs and figures. The `data` branch holds the usable datasets (`raw_data/` and `processed_data/` for ETH and iNav-WHU only), so that about 190 MB of data is not merged into every optimizer branch. The `.gitignore` is the same on every branch and lets only those datasets be added; commit them on `data`, never on `main`. Changes to shared code belong on `main` and are then merged into each optimizer branch.

## Gotchas

- `THETA0`, `LOWER` and `UPPER` were derived from the ETH flight's sensor statistics and are the same for every dataset. On a different sensor they are only a starting point (for the iNav-WHU ADIS16465 data `THETA0` gives mean NIS ≈ 0), and the bounds may be too tight or too loose.
- Long datasets are expensive: iNav-WHU is 265k to 280k IMU steps at 200 Hz, about 10 times ETH, so one evaluation takes about 45 s on the CPU. On an RTX 3050 Laptop GPU a candidate cost about 2 s in batches of 32 (a 10,000-evaluation seed is about 5.5 hours), against about 0.07 s per candidate on ETH on an RTX 4060 per the README.
- `PSO_config.py` holds all PSO settings. Edit it, not the constants in `PSO.py`.
- With `n_workers > 1`, the driving script must guard its entry point with `if __name__ == "__main__":` (Windows multiprocessing).
- `PSO.py` rewrites a seed's rows in its live `pso_log.csv` on rerun, and `save_run` replaces a seed's row in `runs.csv`. Rerunning a seed is safe; rerunning with a different budget mixes results.
- `ekf_env/` is a local virtualenv (ignored). It needs `torch` with CUDA for the GPU backend, plus `scipy` and `openpyxl` (in `requirements.txt`).
- `processed_data/` is generated by `python -m data_pipeline` from `raw_data/`; only the ETH and iNav-WHU outputs are kept in git. Regenerate the rest locally if you need them (the Excel land-vehicle files take about a minute each).
