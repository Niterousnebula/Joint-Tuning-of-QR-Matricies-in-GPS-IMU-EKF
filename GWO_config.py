"""Configuration for Grey Wolf Optimization (GWO). Shared EKF files are untouched."""

# 100 wolves x 10 generations = 1,000 evaluations per seed
POPULATION_SIZE = 100
MAX_EVALS = 1000
SEEDS = (1, 2, 3, 4, 5)
N_WORKERS = 1              # CPU backend only
BACKEND = "gpu"            # "gpu": each generation is one batched EKF pass; "cpu": filterpy reference

# GWO uses its standard alpha/beta/delta leadership and linearly decreasing a.
OUTPUT_DIR = "results/GWO"    # standalone runs: results/[<dataset>/]GWO
