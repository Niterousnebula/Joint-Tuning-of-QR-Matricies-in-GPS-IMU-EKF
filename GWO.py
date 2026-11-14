"""Grey Wolf Optimizer for whole-flight joint GPS-IMU EKF Q/R tuning.

Only GWO.py and GWO_config.py are optimizer-specific. The shared EKF,
data loader, parameter bounds and RMSE objective are imported from `ekf`.
Run from the repository root, e.g.:
    python GWO.py --max-evals 40 --seeds 0 --out results/GWO_smoke
"""
import argparse
from pathlib import Path

import numpy as np

import ekf
import GWO_config as cfg


def _append_evaluation_log(log_path, tracker, start_index, seed, iteration, candidate_offset=0):
    """Append newly completed FitnessTracker evaluations to a live CSV log."""
    if log_path is None:
        return
    import csv
    from pathlib import Path
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["seed", "evaluation", "iteration", "candidate", "fitness", "best_so_far", "elapsed",
              "log_q_acc", "log_q_gyro", "log_q_ba", "log_q_bg", "log_r_xy", "log_r_z", "log_r_vel"]
    write_header = not log_path.exists() or log_path.stat().st_size == 0
    new_rows = tracker.history[start_index:]
    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for offset, item in enumerate(new_rows):
            row = {"seed": seed, "evaluation": item.get("k", start_index + offset + 1),
                   "iteration": iteration, "candidate": candidate_offset + offset,
                   "fitness": item.get("fitness"), "best_so_far": item.get("best_so_far"),
                   "elapsed": item.get("elapsed")}
            for name in fields[7:]:
                row[name] = item.get(name)
            writer.writerow(row)
        f.flush()

def _initial_population(population_size, lower, upper, rng):
    """Uniform initial pack, with the common THETA0 starting point included."""
    wolves = rng.uniform(lower, upper, size=(population_size, lower.size))
    wolves[0] = np.clip(ekf.THETA0, lower, upper)
    return wolves


def _update_pack(wolves, costs, lower, upper, rng, a):
    """Apply the canonical GWO encircling/hunting update from alpha, beta, delta."""
    order = np.argsort(costs)
    leaders = wolves[order[:3]].copy()
    updated = np.empty_like(wolves)
    for i, wolf in enumerate(wolves):
        proposals = []
        for leader in leaders:
            r1 = rng.random(wolf.size)
            r2 = rng.random(wolf.size)
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            D = np.abs(C * leader - wolf)
            proposals.append(leader - A * D)
        updated[i] = np.mean(proposals, axis=0)
    return np.clip(updated, lower, upper)


def run_gwo(tracker, seed=0, population_size=None, log_path=None):
    """Run bounded GWO using only the shared FitnessTracker for evaluations.

    One candidate evaluation is one complete EKF run. evaluate_many enforces
    the exact evaluation budget and can use multiple worker processes.
    """
    population_size = int(cfg.POPULATION_SIZE if population_size is None else population_size)
    if population_size < 3:
        raise ValueError("GWO population_size must be at least 3 (alpha, beta, delta).")
    lower = np.asarray(tracker.lower, dtype=float)
    upper = np.asarray(tracker.upper, dtype=float)
    if lower.shape != upper.shape or lower.ndim != 1 or np.any(lower >= upper):
        raise ValueError("Invalid parameter bounds in FitnessTracker")

    rng = np.random.default_rng(seed)
    wolves = _initial_population(population_size, lower, upper, rng)
    generation = 0
    while not tracker.exhausted:
        start_index = tracker.n_evals
        costs = tracker.evaluate_many(wolves)
        _append_evaluation_log(log_path, tracker, start_index, seed, generation)
        if costs.size == 0:
            break
        if costs.size != population_size:
            break  # final partial pack: budget is consumed exactly, no incomplete update
        if tracker.exhausted:
            break
        generation += 1
        # Linear control parameter decreases from 2 to 0 across configured generations.
        # The evaluation budget determines a conservative generation horizon.
        max_generations = max(1, int(np.ceil(max(tracker.max_evals, population_size) / population_size)))
        a = max(0.0, 2.0 * (1.0 - generation / max_generations))
        wolves = _update_pack(wolves, costs, lower, upper, rng, a)
    return tracker


def run_experiment(max_evals=None, seeds=None, out_dir=None, population_size=None, n_workers=None):
    max_evals = int(cfg.MAX_EVALS if max_evals is None else max_evals)
    seeds = tuple(cfg.SEEDS if seeds is None else seeds)
    out_dir = Path(cfg.OUTPUT_DIR if out_dir is None else out_dir)
    population_size = int(cfg.POPULATION_SIZE if population_size is None else population_size)
    n_workers = int(cfg.N_WORKERS if n_workers is None else n_workers)
    if max_evals < 1 or not seeds or n_workers < 1:
        raise ValueError("Require max_evals >= 1, at least one seed, and n_workers >= 1")

    print("Loading ETH flight data...")
    data = ekf.load_data()
    print("Computing baseline references...")
    rmse_baseline = ekf.fitness(data, ekf.THETA0)
    rmse_gps = ekf.gps_rmse(data)
    print(f"Baseline EKF (THETA0): {rmse_baseline:.4f} m")
    print(f"Raw GPS:               {rmse_gps:.4f} m")

    for seed in seeds:
        print(f"\n=== GWO | seed={seed} | budget={max_evals} ===")
        tracker = ekf.make_cost_fn(data, max_evals=max_evals, n_workers=n_workers)
        try:
            log_path = out_dir / "gwo_log.csv"
            run_gwo(tracker, seed=seed, population_size=population_size,
                    log_path=log_path)
            if tracker.best_theta is None:
                raise RuntimeError("GWO completed without evaluating a candidate")
            row, _ = ekf.run_summary(data, tracker, "GWO", seed,
                                     rmse_gps=rmse_gps, rmse_baseline=rmse_baseline)
            ekf.save_run(out_dir, row, tracker)
            print(f"Best whole-flight RMSE_h: {row['rmse_h']:.4f} m")
            print(f"Change vs baseline EKF:   {row['delta_vs_baseline']:+.4f} m")
            print(f"Change vs raw GPS:        {row['delta_vs_gps']:+.4f} m")
            print(f"Evaluations:              {row['n_evals']}/{max_evals}")
            print(f"Time:                     {row['t_total']:.2f} s")
        finally:
            tracker.close()
    print(f"\nSaved GWO run files to: {out_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="GWO for whole-flight EKF Q/R tuning")
    parser.add_argument("--max-evals", type=int, default=cfg.MAX_EVALS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(cfg.SEEDS))
    parser.add_argument("--population-size", type=int, default=cfg.POPULATION_SIZE)
    parser.add_argument("--workers", type=int, default=cfg.N_WORKERS)
    parser.add_argument("--out", default=cfg.OUTPUT_DIR)
    args = parser.parse_args()
    run_experiment(args.max_evals, args.seeds, args.out, args.population_size, args.workers)


if __name__ == "__main__":
    main()
