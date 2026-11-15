# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Offline joint tuning of the process noise Q and measurement noise R for one fixed 16-state GPS–IMU EKF, on one ETH Zurich flight. Four optimizers (GA, PSO, DE, GWO) are compared against the hand-set starting point `THETA0`. The README holds the full methodology and results; read it before changing anything that affects the numbers.

## Commands

No build step, no linter, no test suite. Python with `filterpy`, `numpy`, `pandas`, `matplotlib`, `plotly`; PyTorch only for the GPU backend.

```bash
pip install -r requirements.txt

python run_baseline.py                     # THETA0 through the canonical fitness -> results/baseline/
python PSO.py                              # all seeds in PSO_config.py -> results/PSO_gpu/ (needs CUDA torch)
python PSO.py --seeds 1 --max-evals 40 --swarm 8   # quick smoke test
python PSO.py --backend cpu --workers 10   # filterpy reference, no GPU
python plot_response.py                    # regenerate results/ tables and figures from runs.csv files
```

Scripts write to relative `results/` paths, so run them from the repo root. `results/` is not in the working tree; it lives on the `results` branch.

Verifying a change to the filter or fitness: `python run_baseline.py` should still report THETA0 RMSE_h ≈ 3.775 m (mean NIS 0.97). A GPU change should also pass `ekf.gpu.compare_with_cpu(data, thetas)`, whose tolerance is 1e-6 m.

## Architecture

The pieces that only make sense together:

- **Fitness is the single source of truth.** `ekf.fitness(data, theta)` (`ekf/tuning.py`) maps the 7-D log10 vector → `theta_to_QR` → full-flight `GpsImuEKF` run → whole-flight horizontal RMSE. Diverged runs return `PENALTY` (1000 m). Every optimizer must score through `ekf.make_cost_fn`, which returns a `FitnessTracker`. That object enforces `max_evals`, clips to bounds, records best-so-far and timing per evaluation, and handles parallel CPU workers or the GPU backend. Changing the fitness, bounds, or budget logic changes every optimizer's results, so it belongs on `main` and is merged everywhere.
- **Two backends, one objective.** `backend="cpu"` uses filterpy (reference). `backend="gpu"` (`ekf/gpu.py`) batches the candidate dimension in float64 and replays a CUDA graph per IMU step. Search may use either, but reported numbers are always re-scored on the CPU reference: `run_summary` calls `evaluate` on `tracker.best_theta`, and robustness runs are CPU as well. `search_vs_final_diff` shows any gap.
- **Filter.** `GpsImuEKF` (`ekf/filter.py`) subclasses filterpy's `ExtendedKalmanFilter`. It overrides only `predict_x` (nonlinear motion model, Jacobian `F`, normalisation) and supplies `H` and `h_gps`; filterpy does the covariance algebra. Q is continuous-time, discretised per step as `G Q_c Gᵀ Δt`, so `THETA0` is independent of IMU rate. Each GPS fix updates position and velocity (6-D measurement, so NIS has 6 degrees of freedom). `run_ekf` is the entry point; `init_state` sets x0/P0 from the flight's first second and first GPS fix.
- **Decision vector** (`THETA_NAMES`, `THETA0`, `LOWER`/`UPPER`, in `ekf/tuning.py`): `[log_q_acc, log_q_gyro, log_q_ba, log_q_bg, log_r_xy, log_r_z, log_r_vel]`. Bounds are ±3 for the Q terms and ±2 for the R terms around `THETA0`. `LOWER_TIGHT`/`UPPER_TIGHT` exist but are not used by the optimizers. Optimizers must draw starting populations from `np.random.default_rng(seed).uniform(LOWER, UPPER, ...)` without seeding in `THETA0`.
- **Output contract.** Every optimizer writes the same two artifacts through `ekf.run_summary` and `ekf.save_run`: `results/<NAME>/runs.csv` (one row per seed) and `convergence_seed<seed>.csv`. `plot_response.py` reads those for the comparison table and figures. Keep this schema when adding an optimizer.
- **Robustness.** `ekf/data.py` builds perturbed copies of the flight (`PERTURBATIONS`, fixed seeds). These are used only to score the final theta. They are never part of fitness.
- **Data.** `load_data()` reads four cleaned files in `data/` (IMU at 10 Hz, GPS at about 5 Hz, ground truth at about 1 Hz). They already hold cleaned SI/ENU values, so do not re-clean them.

## Branch layout

The README describes the intended branch split: `main` holds the shared data/filter/fitness/README; `ga`, `pso`, `de`, `gwo`, `hisa` each add one optimizer and its config; `results` holds outputs and figures. The local branch `data` is not part of that table. Changes to shared code belong on `main` and are then merged into each optimizer branch.

## Gotchas

- `PSO_config.py` holds all PSO settings. Edit it, not the constants in `PSO.py`.
- With `n_workers > 1`, the driving script must guard its entry point with `if __name__ == "__main__":` (Windows multiprocessing).
- `PSO.py` rewrites a seed's rows in its live `pso_log.csv` on rerun, and `save_run` replaces a seed's row in `runs.csv`. Rerunning a seed is safe; rerunning with a different budget mixes results.
- `ekf_env/` is an untracked local virtualenv that is not in `.gitignore`. Do not commit it.
