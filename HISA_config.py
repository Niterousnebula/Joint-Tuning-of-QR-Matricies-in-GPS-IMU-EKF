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
# All 2^6 = 64 sign combinations are allowed.

# R, searched jointly with the biases (log10 of variances)
R_LOG_XY = (0.0, 1.0, 2.0)       # horizontal GPS position, m^2: THETA0 decade and one either side
R_LOG_Z = (0.5, 1.5, 2.5)        # vertical GPS position, m^2
R_VEL_OFFSET = -1.0              # log10 r_vel = log10 r_xy - 1 (THETA0's 10:1 position:velocity ratio)

# Q noise densities are not searched: they stay at ekf.THETA0[0:4]
# (log_q_acc, log_q_gyro, log_q_ba, log_q_bg).

# ---------------------------------------------------------------------------
# Coarse-to-fine schedule
# ---------------------------------------------------------------------------
# IntN: every N consecutive IMU intervals are integrated into one prediction (dt-weighted
# mean accel and gyro, dt = block length), followed by the last GPS fix in that block.
# Int1 is the unchanged canonical filter and fitness.
LEVELS = (25, 10, 1)
RETAIN_FRACTION = 0.10           # best 10 % of each level go to the next level
# retained = round(RETAIN_FRACTION * n); ties in score are broken by candidate_id

# GPU batch size per level (memory: steps x batch x 24 bytes for the position log)
BATCH = {25: 8192, 10: 4096, 1: 1000}

# ---------------------------------------------------------------------------
# Reproducibility and output
# ---------------------------------------------------------------------------
SEED = 0                         # the search is deterministic; recorded for completeness
VALIDATE_PER_LEVEL = 8           # candidates per level re-scored by the filterpy reference
OUTPUT_DIR = "results/HISA"
