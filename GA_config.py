"""Configuration for the Genetic Algorithm (GA).

These settings apply only to GA.py. The shared EKF, data loader and fitness function
remain in the existing `ekf` package and are not modified by this file.
"""

# Experiment budget: 200 candidates x 10 generations = 2,000 evaluations per seed
POPULATION_SIZE = 200
MAX_EVALS = 2000
SEEDS = (1, 2, 3, 4, 5)
N_WORKERS = 1             # CPU backend only
BACKEND = "gpu"           # "gpu": each generation is one batched EKF pass; "cpu": filterpy reference

# GA operators
TOURNAMENT_SIZE = 3
ELITE_COUNT = 2
CROSSOVER_PROBABILITY = 0.90
BLX_ALPHA = 0.50
MUTATION_PROBABILITY = 0.20       # probability of mutating each gene
MUTATION_SIGMA_FRACTION = 0.10    # sigma as a fraction of each parameter's bound range

# Reproducibility and output
RANDOM_SEED = 0
OUTPUT_DIR = "results/GA"     # standalone runs: results/[<dataset>/]GA
