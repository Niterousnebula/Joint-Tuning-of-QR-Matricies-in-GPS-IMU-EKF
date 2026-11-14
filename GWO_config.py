"""Configuration for Grey Wolf Optimization (GWO). Shared EKF files are untouched."""

POPULATION_SIZE = 20
MAX_EVALS = 1000
SEEDS = (0, 1, 2, 3, 4)
N_WORKERS = 1

# GWO uses its standard alpha/beta/delta leadership and linearly decreasing a.
OUTPUT_DIR = "results/GWO"
