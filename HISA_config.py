"""Configuration for the Heuristic Informed Search Algorithm (HISA).

These settings apply only to HISA.py. The shared EKF, data loader and fitness function
remain in the `ekf` package. Every value here is written to config_snapshot.json.
"""

# ---------------------------------------------------------------------------
# Search space: joint candidate [b_ax, b_ay, b_az, b_gx, b_gy, b_gz] + R
# ---------------------------------------------------------------------------
# Bias values are signed powers of ten, +-10^e; zero is excluded. Each candidate's six
# values are the filter's initial bias states, which the EKF then keeps estimating.
BIAS_EXPONENTS = (-3, -2, -1, 0, 1, 2)
BIAS_SIGNS = (-1, 1)

# Heuristic admissibility
ACCEL_CLUSTER_MAX_SPREAD = 0.5   # decades: max(e_a) - min(e_a) <= 0.5 -> one decade for all 3 axes
GYRO_CLUSTER_MAX_SPREAD = None   # no gyro-axis clustering: 6^3 = 216 magnitude patterns
ACCEL_GYRO_MAX_SPACING = None    # no accelerometer <-> gyroscope spacing rule
# All 2^6 = 64 sign combinations are allowed: 6 x 216 x 64 = 82,944 bias candidates.

# R, searched jointly with the biases (log10 of variances). Three independent values each,
# spanning the shared search bounds (ekf.LOWER, THETA0, ekf.UPPER): 3 x 3 x 3 = 27.
R_LOG_XY = (-1.0, 0.0, 1.0)      # GPS horizontal position, m^2
R_LOG_Z = (-0.5, 1.5, 3.5)       # GPS vertical position, m^2
R_LOG_VEL = (-2.0, 0.0, 2.0)     # GPS velocity, (m/s)^2
# 82,944 x 27 = 2,239,488 joint candidates.

# Q noise densities are not searched: they stay at ekf.THETA0[0:4]
# (log_q_acc, log_q_gyro, log_q_ba, log_q_bg).

# ---------------------------------------------------------------------------
# Coarse-to-fine schedule
# ---------------------------------------------------------------------------
# IntN: every N consecutive IMU intervals are integrated into one prediction (dt-weighted
# mean accel and gyro, dt = block length), followed by the last GPS fix in that block.
# Int1 is the unchanged canonical filter and fitness.
LEVELS = (50, 1)

# Candidates kept after each non-final level (the best by that level's score; ties broken
# by candidate_id). The level's winner is always kept.
RETAIN = {50: 5210}

# Int1-equivalent budget: sum over levels of candidates / N.
#   Int50: 2,239,488 / 50 = 44,789.76;  Int1: 5,210  ->  49,999.76 <= 50,000
INT1_EQUIVALENT_BUDGET = 50_000

# GPU batch size per level (memory: steps x batch x 24 bytes for the position log)
BATCH = {50: 8192, 1: 1000}

# ---------------------------------------------------------------------------
# Reproducibility and output
# ---------------------------------------------------------------------------
SEED = 0                         # the search is deterministic; recorded for completeness
VALIDATE_PER_LEVEL = 8           # candidates per level re-scored by the filterpy reference
OUTPUT_DIR = "results/HISA"
MAX_FILE_MB = 90                 # result tables larger than this are split into parts (GitHub limit 100 MB)
