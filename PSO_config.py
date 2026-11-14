"""Configuration for Particle Swarm Optimization (PSO).

These settings apply only to PSO.py. The shared EKF, data loader and fitness function
remain in the existing `ekf` package and are not modified by this file.
"""

# Experiment budget (same as GA: 1000 evaluations per seed, 5 seeds)
SWARM_SIZE = 20             # 20 particles x 50 iterations = 1000 evaluations
MAX_EVALS = 1000
SEEDS = (0, 1, 2, 3, 4)
N_WORKERS = 10

# PSO coefficients (global-best topology, linearly decreasing inertia)
INERTIA_START = 0.90        # w at the first iteration
INERTIA_END = 0.40          # w at the last iteration
COGNITIVE = 1.49445         # c1, pull toward each particle's personal best
SOCIAL = 1.49445            # c2, pull toward the swarm's global best
VELOCITY_CLAMP_FRACTION = 0.20   # |v| limit as a fraction of each parameter's bound range

# Reproducibility and output
RANDOM_SEED = 0             # run seed s uses RANDOM_SEED + s
OUTPUT_DIR = "results/PSO"
LOG_FILE = "pso_log.csv"    # in OUTPUT_DIR; one row per evaluation, written live
