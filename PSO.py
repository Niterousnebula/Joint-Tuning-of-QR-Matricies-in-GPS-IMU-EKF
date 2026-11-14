"""Particle Swarm Optimization (PSO) for joint Q/R tuning of the GPS-IMU EKF.

    python PSO.py                         # every seed in PSO_config.SEEDS
    python PSO.py --seeds 0 --max-evals 40 --swarm 8   # quick check

Global-best PSO over the 7-D log10 decision vector inside [ekf.LOWER, ekf.UPPER].
Every candidate is scored by the canonical fitness through ekf.make_cost_fn, which
enforces the evaluation budget and records the convergence curve. Results go to
<OUTPUT_DIR>/runs.csv (one row per seed) and <OUTPUT_DIR>/convergence_seed<s>.csv.
"""
import argparse

import numpy as np

import ekf
import PSO_config as cfg


def pso(cost_fn, rng, swarm_size=cfg.SWARM_SIZE):
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
        improved = cost < p_cost[:n]
        p_best[:n][improved] = x[:n][improved]
        p_cost[:n][improved] = cost[improved]
        g = int(np.argmin(p_cost))
        if p_cost[g] < g_cost:
            g_best, g_cost = p_best[g].copy(), p_cost[g]
        it += 1
    return g_best, g_cost


def main(seeds, max_evals, swarm_size, n_workers, out_dir):
    data = ekf.load_data()
    rmse_gps = ekf.gps_rmse(data)
    rmse_baseline = ekf.fitness(data, ekf.THETA0)   # reference only, not part of the budget
    print(f"Raw GPS RMSE_h {rmse_gps:.3f} m, baseline EKF RMSE_h {rmse_baseline:.3f} m")

    for seed in seeds:
        rng = np.random.default_rng(cfg.RANDOM_SEED + seed)
        with ekf.make_cost_fn(data, max_evals=max_evals, n_workers=n_workers) as cost_fn:
            best_theta, best_cost = pso(cost_fn, rng, swarm_size)
            row, _ = ekf.run_summary(data, cost_fn, "PSO", seed,
                                     rmse_gps=rmse_gps, rmse_baseline=rmse_baseline)
            ekf.save_run(out_dir, row, cost_fn)
        print(f"seed {seed}: RMSE_h {row['rmse_h']:.3f} m "
              f"(vs baseline {row['delta_vs_baseline']:+.3f} m, vs GPS {row['delta_vs_gps']:+.3f} m), "
              f"robustness mean {row['delta_rmse_mean']:+.3f} m, "
              f"converged at k = {row['evals_to_converge']}, N_eval {row['n_evals']}, "
              f"T_total {row['t_total']:.0f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(cfg.SEEDS))
    parser.add_argument("--max-evals", type=int, default=cfg.MAX_EVALS)
    parser.add_argument("--swarm", type=int, default=cfg.SWARM_SIZE)
    parser.add_argument("--workers", type=int, default=cfg.N_WORKERS)
    parser.add_argument("--out", default=cfg.OUTPUT_DIR)
    a = parser.parse_args()
    main(a.seeds, a.max_evals, a.swarm, a.workers, a.out)
