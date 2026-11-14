"""Particle Swarm Optimization (PSO) for joint Q/R tuning of the GPS-IMU EKF.

    python PSO.py                         # every seed in PSO_config.SEEDS
    python PSO.py --seeds 0 --max-evals 40 --swarm 8   # quick check
    python PSO.py --backend cpu           # filterpy reference instead of the GPU batch

Global-best PSO over the 7-D log10 decision vector inside [ekf.LOWER, ekf.UPPER].
Every candidate is scored by the canonical fitness through ekf.make_cost_fn, which
enforces the evaluation budget and records the convergence curve. Results go to
<OUTPUT_DIR>/runs.csv (one row per seed) and <OUTPUT_DIR>/convergence_seed<s>.csv.
While a run is going, every evaluation is appended to <OUTPUT_DIR>/<LOG_FILE>.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import ekf
import PSO_config as cfg


class RunLog:
    """Live CSV log: one row per evaluation, appended as soon as a batch is scored."""

    def __init__(self, path, seed):
        self.path = Path(path)
        self.seed = seed
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():                       # rerunning a seed replaces its old rows
            old = pd.read_csv(self.path)
            old[old["seed"] != seed].to_csv(self.path, index=False)

    def write(self, cost_fn, iteration, n_new):
        rows = pd.DataFrame(cost_fn.history[cost_fn.n_evals - n_new:])
        rows.insert(0, "particle", np.arange(n_new))
        rows.insert(0, "iteration", iteration)
        rows.insert(0, "seed", self.seed)
        rows.to_csv(self.path, mode="a", index=False, header=not self.path.exists())
        print(f"seed {self.seed}  iter {iteration:3d}  k {cost_fn.n_evals:5d}  "
              f"best RMSE_h {cost_fn.best_cost:.4f} m  elapsed {cost_fn.t_total:7.0f} s", flush=True)


def pso(cost_fn, rng, swarm_size=cfg.SWARM_SIZE, log=None):
    """Minimise cost_fn until its budget runs out. Returns (best_theta, best_cost)."""
    lower, upper = cost_fn.lower, cost_fn.upper
    span = upper - lower
    v_max = cfg.VELOCITY_CLAMP_FRACTION * span
    dim = lower.size
    n_iter = max(1, int(np.ceil(cost_fn.max_evals / swarm_size)))

    # Initial swarm: uniform positions in the bounds, small random velocities
    x = lower + rng.random((swarm_size, dim)) * span
    v = rng.uniform(-v_max, v_max, (swarm_size, dim))
    cost = cost_fn.evaluate_many(x)
    if log:
        log.write(cost_fn, 0, len(cost))
    x = x[:len(cost)]
    v = v[:len(cost)]
    p_best, p_cost = x.copy(), cost.copy()
    g = int(np.argmin(p_cost))
    g_best, g_cost = p_best[g].copy(), p_cost[g]

    it = 1
    while not cost_fn.exhausted:
        w = cfg.INERTIA_START + (cfg.INERTIA_END - cfg.INERTIA_START) * min(it / max(n_iter - 1, 1), 1.0)
        r1, r2 = rng.random(x.shape), rng.random(x.shape)
        v = w * v + cfg.COGNITIVE * r1 * (p_best - x) + cfg.SOCIAL * r2 * (g_best - x)
        v = np.clip(v, -v_max, v_max)
        x = x + v

        # Absorbing walls: stop at the bound and zero the outward velocity
        out = (x < lower) | (x > upper)
        x = np.clip(x, lower, upper)
        v[out] = 0.0

        cost = cost_fn.evaluate_many(x)
        n = len(cost)
        if log:
            log.write(cost_fn, it, n)
        improved = cost < p_cost[:n]
        p_best[:n][improved] = x[:n][improved]
        p_cost[:n][improved] = cost[improved]
        g = int(np.argmin(p_cost))
        if p_cost[g] < g_cost:
            g_best, g_cost = p_best[g].copy(), p_cost[g]
        it += 1
    return g_best, g_cost


def run_experiment(max_evals=None, seeds=None, out_dir=None, population_size=None,
                   n_workers=None, backend=None, data=None):
    """Run every seed and save benchmark-compatible outputs (same interface as GA, DE, GWO)."""
    max_evals = int(cfg.MAX_EVALS if max_evals is None else max_evals)
    seeds = tuple(cfg.SEEDS if seeds is None else seeds)
    out_dir = cfg.OUTPUT_DIR if out_dir is None else out_dir
    swarm_size = int(cfg.SWARM_SIZE if population_size is None else population_size)
    n_workers = int(cfg.N_WORKERS if n_workers is None else n_workers)
    backend = cfg.BACKEND if backend is None else backend

    if data is None:
        data = ekf.load_data()
    rmse_gps = ekf.gps_rmse(data)
    rmse_baseline = ekf.fitness(data, ekf.THETA0)   # reference only, not part of the budget
    print(f"Raw GPS RMSE_h {rmse_gps:.3f} m, baseline EKF RMSE_h {rmse_baseline:.3f} m", flush=True)

    for seed in seeds:
        print(f"\n=== PSO | seed={seed} | budget={max_evals} | backend={backend} ===", flush=True)
        rng = np.random.default_rng(cfg.RANDOM_SEED + seed)
        log = RunLog(Path(out_dir) / cfg.LOG_FILE, seed)
        with ekf.make_cost_fn(data, max_evals=max_evals, n_workers=n_workers, backend=backend) as cost_fn:
            best_theta, best_cost = pso(cost_fn, rng, swarm_size, log)
            row, _ = ekf.run_summary(data, cost_fn, "PSO", seed,
                                     rmse_gps=rmse_gps, rmse_baseline=rmse_baseline,
                                     population_size=swarm_size)
            ekf.save_run(out_dir, row, cost_fn)
        print(f"seed {seed}: RMSE_h {row['rmse_h']:.3f} m "
              f"(vs baseline {row['delta_vs_baseline']:+.3f} m, vs GPS {row['delta_vs_gps']:+.3f} m), "
              f"robustness mean {row['delta_rmse_mean']:+.3f} m, "
              f"converged at k = {row['evals_to_converge']}, N_eval {row['n_evals']}, "
              f"T_total {row['t_total']:.0f} s (objective {row['t_objective']:.0f} s)", flush=True)


def main(seeds, max_evals, swarm_size, n_workers, out_dir, backend=None):
    run_experiment(max_evals, seeds, out_dir, swarm_size, n_workers, backend)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(cfg.SEEDS))
    parser.add_argument("--max-evals", type=int, default=cfg.MAX_EVALS)
    parser.add_argument("--swarm", type=int, default=cfg.SWARM_SIZE)
    parser.add_argument("--workers", type=int, default=cfg.N_WORKERS)
    parser.add_argument("--backend", choices=ekf.BACKENDS, default=cfg.BACKEND)
    parser.add_argument("--out", default=cfg.OUTPUT_DIR)
    a = parser.parse_args()
    main(a.seeds, a.max_evals, a.swarm, a.workers, a.out, a.backend)
