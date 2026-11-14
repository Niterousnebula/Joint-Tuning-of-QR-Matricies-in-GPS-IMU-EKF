"""Configuration for Differential Evolution (DE). Shared EKF files are untouched."""

POPULATION_SIZE = 20
MAX_EVALS = 1000
SEEDS = (0, 1, 2, 3, 4)
N_WORKERS = 1

# Canonical DE/rand/1/bin controls
DIFFERENTIAL_WEIGHT = 0.7  # mutation scale F
CROSSOVER_RATE = 0.9       # binomial crossover probability CR

OUTPUT_DIR = "results/DE"
