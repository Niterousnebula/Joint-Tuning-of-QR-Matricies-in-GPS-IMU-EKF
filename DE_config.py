"""Configuration for Differential Evolution (DE). Shared EKF files are untouched."""

# 200 candidates: initial population + 9 trial generations = 2,000 evaluations per seed
POPULATION_SIZE = 200
MAX_EVALS = 2000
SEEDS = (1, 2, 3, 4, 5)
N_WORKERS = 1              # CPU backend only
BACKEND = "gpu"            # "gpu": each generation is one batched EKF pass; "cpu": filterpy reference

# Canonical DE/rand/1/bin controls
DIFFERENTIAL_WEIGHT = 0.7  # mutation scale F
CROSSOVER_RATE = 0.9       # binomial crossover probability CR

OUTPUT_DIR = "results/DE"     # standalone runs: results/[<dataset>/]DE
