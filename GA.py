"""Genetic Algorithm for whole-flight joint Q/R tuning.

Run from the repository root:
    python GA.py
    python GA.py --max-evals 20 --seeds 0 --out results/GA_smoke
    python GA.py --backend cpu        # filterpy reference instead of the GPU batch
    python GA.py --dataset inavwhu/street00

Only this file and GA_config.py are GA-specific. The shared EKF and objective are
imported from the existing `ekf` package and are not modified.
"""
import argparse
from pathlib import Path

import numpy as np

import ekf
import GA_config as cfg


def _tournament_select(population, costs, rng, tournament_size):
    """Return one parent using tournament selection (minimization)."""
    size = min(int(tournament_size), len(population))
    candidates = rng.choice(len(population), size=size, replace=False)
    winner = candidates[np.argmin(costs[candidates])]
    return population[winner].copy()


def _blx_alpha_crossover(parent_a, parent_b, rng, alpha):
    """Blend crossover: sample each gene from an expanded parent interval."""
    lo = np.minimum(parent_a, parent_b)
    hi = np.maximum(parent_a, parent_b)
    span = hi - lo
    child_a = rng.uniform(lo - alpha * span, hi + alpha * span)
    child_b = rng.uniform(lo - alpha * span, hi + alpha * span)
    return child_a, child_b


def _mutate(candidate, rng, lower, upper, probability, sigma_fraction):
    """Apply independent Gaussian mutation to genes, then enforce bounds."""
    child = candidate.copy()
    sigma = sigma_fraction * (upper - lower)
    mask = rng.random(child.size) < probability
    if np.any(mask):
        child[mask] += rng.normal(0.0, sigma[mask])
    return np.clip(child, lower, upper)


def _initial_population(population_size, lower, upper, rng):
    """Uniform random population with THETA0 inserted as the first individual."""
    population = rng.uniform(lower, upper, size=(population_size, lower.size))
    population[0] = np.clip(ekf.THETA0, lower, upper)
    return population


def _make_offspring(population, costs, lower, upper, rng, population_size):
    """Create a complete next generation, preserving the best individuals."""
    elite_count = min(int(cfg.ELITE_COUNT), population_size)
    elite_indices = np.argsort(costs)[:elite_count]
    next_population = [population[i].copy() for i in elite_indices]

    while len(next_population) < population_size:
        parent_a = _tournament_select(
            population, costs, rng, cfg.TOURNAMENT_SIZE
        )
        parent_b = _tournament_select(
            population, costs, rng, cfg.TOURNAMENT_SIZE
        )

        if rng.random() < cfg.CROSSOVER_PROBABILITY:
            child_a, child_b = _blx_alpha_crossover(
                parent_a, parent_b, rng, cfg.BLX_ALPHA
            )
        else:
            child_a, child_b = parent_a.copy(), parent_b.copy()

        child_a = _mutate(
            child_a, rng, lower, upper,
            cfg.MUTATION_PROBABILITY, cfg.MUTATION_SIGMA_FRACTION
        )
        next_population.append(np.clip(child_a, lower, upper))

        if len(next_population) < population_size:
            child_b = _mutate(
                child_b, rng, lower, upper,
                cfg.MUTATION_PROBABILITY, cfg.MUTATION_SIGMA_FRACTION
            )
            next_population.append(np.clip(child_b, lower, upper))

    return np.asarray(next_population, dtype=float)


def run_ga(data, tracker, seed=0, population_size=None):
    """Run GA using the supplied shared FitnessTracker.

    The tracker is the sole evaluator. Its `evaluate_many` method enforces the
    evaluation budget, including when the final generation is only partly evaluated.
    Returns the tracker after the run.
    """
    population_size = int(
        cfg.POPULATION_SIZE if population_size is None else population_size
    )
    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if cfg.ELITE_COUNT < 1 or cfg.ELITE_COUNT > population_size:
        raise ValueError("ELITE_COUNT must be between 1 and population_size")
    if cfg.TOURNAMENT_SIZE < 1:
        raise ValueError("TOURNAMENT_SIZE must be at least 1")
    if cfg.MUTATION_PROBABILITY < 0 or cfg.MUTATION_PROBABILITY > 1:
        raise ValueError("MUTATION_PROBABILITY must be in [0, 1]")

    lower = np.asarray(tracker.lower, dtype=float)
    upper = np.asarray(tracker.upper, dtype=float)
    if lower.shape != upper.shape or np.any(lower >= upper):
        raise ValueError("Invalid parameter bounds in FitnessTracker")

    rng = np.random.default_rng(seed)
    population = _initial_population(population_size, lower, upper, rng)

    while not tracker.exhausted:
        # evaluate_many truncates the final generation to the remaining budget.
        costs = tracker.evaluate_many(population)
        if costs.size == 0 or tracker.exhausted:
            break

        # A partial generation means the evaluation budget ended mid-generation.
        # Do not attempt selection from an incompletely scored population.
        if costs.size != population_size:
            break

        population = _make_offspring(
            population, costs, lower, upper, rng, population_size
        )

    return tracker


def run_experiment(max_evals=None, seeds=None, out_dir=None,
                   population_size=None, n_workers=None, backend=None, data=None, dataset=None):
    """Run configured independent seeds and save benchmark-compatible outputs."""
    max_evals = int(cfg.MAX_EVALS if max_evals is None else max_evals)
    seeds = tuple(cfg.SEEDS if seeds is None else seeds)
    population_size = int(
        cfg.POPULATION_SIZE if population_size is None else population_size
    )
    n_workers = int(cfg.N_WORKERS if n_workers is None else n_workers)
    backend = cfg.BACKEND if backend is None else backend

    if max_evals < 1:
        raise ValueError("max_evals must be at least 1")
    if not seeds:
        raise ValueError("At least one seed is required")
    if n_workers < 1:
        raise ValueError("n_workers must be at least 1")

    data, dataset, out_dir = ekf.prepare_run(data, dataset, out_dir, Path(cfg.OUTPUT_DIR).name)
    print(f"Dataset: {dataset}; results -> {out_dir}")

    # Reference values are computed once per experiment, outside each GA budget.
    print("Computing baseline references...")
    rmse_baseline = ekf.evaluate(data, ekf.THETA0)["rmse_h"]
    rmse_gps = ekf.gps_rmse(data)
    print(f"Baseline EKF (THETA0): {rmse_baseline:.4f} m")
    print(f"Raw GPS:               {rmse_gps:.4f} m")

    for seed in seeds:
        print(f"\n=== GA | seed={seed} | budget={max_evals} | backend={backend} ===")
        tracker = ekf.make_cost_fn(
            data, max_evals=max_evals, n_workers=n_workers, backend=backend
        )
        try:
            run_ga(data, tracker, seed=seed, population_size=population_size)
            if tracker.best_theta is None:
                raise RuntimeError("GA completed without evaluating a candidate")

            row, _ = ekf.run_summary(
                data, tracker, "GA", seed,
                rmse_gps=rmse_gps,
                rmse_baseline=rmse_baseline,
                population_size=population_size
            )
            ekf.save_run(out_dir, row, tracker)

            print(f"Best whole-flight RMSE_h: {row['rmse_h']:.4f} m")
            print(f"Change vs baseline EKF:   {row['delta_vs_baseline']:+.4f} m")
            print(f"Change vs raw GPS:        {row['delta_vs_gps']:+.4f} m")
            print(f"Evaluations:              {row['n_evals']}/{max_evals}")
            print(f"Time:                     {row['t_total']:.2f} s "
                  f"(objective {row['t_objective']:.2f} s, {row['n_generations']} generations)")
            print("Best parameter vector (log10):")
            for name in ekf.THETA_NAMES:
                print(f"  {name:<12} {row[name]: .6f}")
        finally:
            tracker.close()

    print(f"\nSaved GA run files to: {out_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="GA for whole-flight EKF Q/R tuning")
    parser.add_argument("--max-evals", type=int, default=cfg.MAX_EVALS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(cfg.SEEDS))
    parser.add_argument("--population-size", type=int, default=cfg.POPULATION_SIZE)
    parser.add_argument("--workers", type=int, default=cfg.N_WORKERS)
    parser.add_argument("--backend", choices=ekf.BACKENDS, default=cfg.BACKEND)
    parser.add_argument("--dataset", help="processed dataset id; asked for on a terminal if omitted")
    parser.add_argument("--out", help="output folder (default: results/[<dataset>/]<NAME>)")
    args = parser.parse_args()

    run_experiment(
        max_evals=args.max_evals,
        seeds=args.seeds,
        out_dir=args.out,
        population_size=args.population_size,
        n_workers=args.workers,
        backend=args.backend,
        dataset=args.dataset
    )


if __name__ == "__main__":
    main()
