"""Differential Evolution for whole-flight joint GPS-IMU EKF Q/R tuning.

Uses canonical DE/rand/1/bin mutation, binomial crossover and greedy selection.
Only DE.py and DE_config.py are optimizer-specific; the shared EKF, bounds and
horizontal-RMSE objective are imported from `ekf`.
Run from the repository root, e.g.:
    python DE.py --max-evals 40 --seeds 0 --out results/DE_smoke
"""
import argparse
from pathlib import Path

import numpy as np

import ekf
import DE_config as cfg


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
    population = rng.uniform(lower, upper, size=(population_size, lower.size))
    population[0] = np.clip(ekf.THETA0, lower, upper)
    return population


def _make_trial(target_index, population, lower, upper, rng, differential_weight, crossover_rate):
    """Create one DE/rand/1/bin trial vector; donor indices exclude the target."""
    n_pop, n_dim = population.shape
    choices = np.delete(np.arange(n_pop), target_index)
    r1, r2, r3 = rng.choice(choices, size=3, replace=False)
    mutant = population[r1] + differential_weight * (population[r2] - population[r3])
    mutant = np.clip(mutant, lower, upper)
    mask = rng.random(n_dim) < crossover_rate
    mask[rng.integers(n_dim)] = True  # guarantee at least one mutant component
    trial = np.where(mask, mutant, population[target_index])
    return np.clip(trial, lower, upper)


def run_de(tracker, seed=0, population_size=None, differential_weight=None, crossover_rate=None, log_path=None):
    """Run bounded DE under the tracker's strict evaluation budget."""
    population_size = int(cfg.POPULATION_SIZE if population_size is None else population_size)
    differential_weight = float(cfg.DIFFERENTIAL_WEIGHT if differential_weight is None else differential_weight)
    crossover_rate = float(cfg.CROSSOVER_RATE if crossover_rate is None else crossover_rate)
    if population_size < 4:
        raise ValueError("DE population_size must be at least 4 for three distinct donor vectors.")
    if not 0.0 < differential_weight <= 2.0:
        raise ValueError("DIFFERENTIAL_WEIGHT must be in (0, 2].")
    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("CROSSOVER_RATE must be in [0, 1].")

    lower = np.asarray(tracker.lower, dtype=float)
    upper = np.asarray(tracker.upper, dtype=float)
    if lower.shape != upper.shape or lower.ndim != 1 or np.any(lower >= upper):
        raise ValueError("Invalid parameter bounds in FitnessTracker")
    rng = np.random.default_rng(seed)
    population = _initial_population(population_size, lower, upper, rng)
    start_index = tracker.n_evals
    costs = tracker.evaluate_many(population)
    _append_evaluation_log(log_path, tracker, start_index, seed, 0)
    if costs.size != population_size:
        return tracker

    iteration = 0
    while not tracker.exhausted:
        iteration += 1
        trials = np.vstack([
            _make_trial(i, population, lower, upper, rng, differential_weight, crossover_rate)
            for i in range(population_size)
        ])
        start_index = tracker.n_evals
        trial_costs = tracker.evaluate_many(trials)
        _append_evaluation_log(log_path, tracker, start_index, seed, iteration)
        if trial_costs.size == 0:
            break
        # Greedy replacement for all evaluated trials, including a possible final partial batch.
        for i, trial_cost in enumerate(trial_costs):
            if trial_cost <= costs[i]:
                population[i] = trials[i]
                costs[i] = trial_cost
        if trial_costs.size != population_size:
            break
    return tracker


def run_experiment(max_evals=None, seeds=None, out_dir=None, population_size=None,
                   n_workers=None, differential_weight=None, crossover_rate=None):
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
        print(f"\n=== DE | seed={seed} | budget={max_evals} ===")
        tracker = ekf.make_cost_fn(data, max_evals=max_evals, n_workers=n_workers)
        try:
            log_path = out_dir / "de_log.csv"
            run_de(tracker, seed=seed, population_size=population_size,
                   differential_weight=differential_weight, crossover_rate=crossover_rate,
                   log_path=log_path)
            if tracker.best_theta is None:
                raise RuntimeError("DE completed without evaluating a candidate")
            row, _ = ekf.run_summary(data, tracker, "DE", seed,
                                     rmse_gps=rmse_gps, rmse_baseline=rmse_baseline)
            ekf.save_run(out_dir, row, tracker)
            print(f"Best whole-flight RMSE_h: {row['rmse_h']:.4f} m")
            print(f"Change vs baseline EKF:   {row['delta_vs_baseline']:+.4f} m")
            print(f"Change vs raw GPS:        {row['delta_vs_gps']:+.4f} m")
            print(f"Evaluations:              {row['n_evals']}/{max_evals}")
            print(f"Time:                     {row['t_total']:.2f} s")
        finally:
            tracker.close()
    print(f"\nSaved DE run files to: {out_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="DE for whole-flight EKF Q/R tuning")
    parser.add_argument("--max-evals", type=int, default=cfg.MAX_EVALS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(cfg.SEEDS))
    parser.add_argument("--population-size", type=int, default=cfg.POPULATION_SIZE)
    parser.add_argument("--workers", type=int, default=cfg.N_WORKERS)
    parser.add_argument("--differential-weight", type=float, default=cfg.DIFFERENTIAL_WEIGHT)
    parser.add_argument("--crossover-rate", type=float, default=cfg.CROSSOVER_RATE)
    parser.add_argument("--out", default=cfg.OUTPUT_DIR)
    args = parser.parse_args()
    run_experiment(args.max_evals, args.seeds, args.out, args.population_size,
                   args.workers, args.differential_weight, args.crossover_rate)


if __name__ == "__main__":
    main()
