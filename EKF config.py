"""Common EKF steps and parameters. Everyone uses this file unchanged. See README.md."""
import numpy as np

# ---------------------------------------------------------------------------
# State and logs
# ---------------------------------------------------------------------------
N_STATES = 16
STATE_COLUMNS = ["p_e", "p_n", "p_u", "v_e", "v_n", "v_u", "qw", "qx", "qy", "qz",
                 "b_ax", "b_ay", "b_az", "b_gx", "b_gy", "b_gz"]
INNOVATION_COLUMNS = ["t", "y_e", "y_n", "y_u", "s_e", "s_n", "s_u", "nis"]

# ---------------------------------------------------------------------------
# Fixed filter parameters
# ---------------------------------------------------------------------------
GRAVITY = 9.80665                       # m/s^2
G_NAV = np.array([0.0, 0.0, -GRAVITY])  # ENU, z up

# Initial covariance diagonal: p, v, q, b_a, b_g
P0_DIAG = np.r_[np.full(3, 25.0), np.full(3, 2.25), np.full(4, 0.07), np.full(3, 0.25), np.full(3, 0.0027)]
INIT_ATTITUDE_WINDOW = 1.0              # s of accelerometer data averaged for roll and pitch

MAX_POSITION = 1e6                      # m; any position above this counts as diverged

# ---------------------------------------------------------------------------
# Q and R: log10 of variances, tuned jointly
# ---------------------------------------------------------------------------
THETA_NAMES = ["log_q_pos", "log_q_vel", "log_q_att", "log_q_ba", "log_q_bg", "log_r_xy", "log_r_z"]
THETA0 = np.array([-4.0, -1.0, -4.0, -6.0, -8.0, 1.0, 1.5])
LOWER = THETA0 - np.array([3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0])
UPPER = THETA0 + np.array([3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0])
LOWER_TIGHT = THETA0 - np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.0, 1.0])
UPPER_TIGHT = THETA0 + np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.0, 1.0])

T_SPLIT = 1364.04                       # s; tune before this time, hold out after it
PENALTY = 1000.0                        # m; cost of a run that fails


def theta_to_QR(theta):
    """Build the per-step Q and the GPS R from a candidate."""
    variance = 10.0 ** np.asarray(theta, dtype=float)
    Q = np.diag(np.r_[np.full(3, variance[0]), np.full(3, variance[1]), np.full(4, variance[2]),
                      np.full(3, variance[3]), np.full(3, variance[4])])
    R = np.diag([variance[5], variance[5], variance[6]])
    return Q, R


# ---------------------------------------------------------------------------
# Quaternion helpers: Hamilton, scalar first, body to navigation
# ---------------------------------------------------------------------------
def skew(v):
    """Cross-product matrix of a 3-vector."""
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def quat_to_rot(q):
    """Rotation matrix R(q), body to navigation."""
    qw, qx, qy, qz = q
    return np.array([
        [qw * qw + qx * qx - qy * qy - qz * qz, 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), qw * qw - qx * qx + qy * qy - qz * qz, 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), qw * qw - qx * qx - qy * qy + qz * qz]])


def omega_matrix(w):
    """Quaternion rate matrix: q_dot = 0.5 * omega_matrix(w) @ q."""
    wx, wy, wz = w
    return np.array([[0.0, -wx, -wy, -wz],
                     [wx, 0.0, wz, -wy],
                     [wy, -wz, 0.0, wx],
                     [wz, wy, -wx, 0.0]])


def xi_matrix(q):
    """Quaternion rate matrix in terms of q: q_dot = 0.5 * xi_matrix(q) @ w."""
    qw, qx, qy, qz = q
    return np.array([[-qx, -qy, -qz],
                     [qw, -qz, qy],
                     [qz, qw, -qx],
                     [-qy, qx, qw]])


def _rot_to_quat(R_nb):
    """Unit quaternion from a rotation matrix."""
    trace = np.trace(R_nb)
    s = np.array([1 + trace, 1 + R_nb[0, 0] - R_nb[1, 1] - R_nb[2, 2],
                  1 - R_nb[0, 0] + R_nb[1, 1] - R_nb[2, 2], 1 - R_nb[0, 0] - R_nb[1, 1] + R_nb[2, 2]])
    i = int(np.argmax(s))
    q = np.zeros(4)
    q[i] = np.sqrt(s[i]) / 2
    if i == 0:
        q[1:] = np.array([R_nb[2, 1] - R_nb[1, 2], R_nb[0, 2] - R_nb[2, 0], R_nb[1, 0] - R_nb[0, 1]]) / (4 * q[0])
    elif i == 1:
        q[[0, 2, 3]] = np.array([R_nb[2, 1] - R_nb[1, 2], R_nb[0, 1] + R_nb[1, 0], R_nb[0, 2] + R_nb[2, 0]]) / (4 * q[1])
    elif i == 2:
        q[[0, 1, 3]] = np.array([R_nb[0, 2] - R_nb[2, 0], R_nb[0, 1] + R_nb[1, 0], R_nb[1, 2] + R_nb[2, 1]]) / (4 * q[2])
    else:
        q[[0, 1, 2]] = np.array([R_nb[1, 0] - R_nb[0, 1], R_nb[0, 2] + R_nb[2, 0], R_nb[1, 2] + R_nb[2, 1]]) / (4 * q[3])
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# EKF step 2: initialise
# ---------------------------------------------------------------------------
def init_state(data):
    """Fixed initial state and covariance. Yaw 0 means the body x axis points east."""
    t_imu, accel = data["t_imu"], data["accel"]
    up = accel[t_imu < t_imu[0] + INIT_ATTITUDE_WINDOW].mean(axis=0)
    up = up / np.linalg.norm(up)
    x_axis = np.array([1.0, 0.0, 0.0])
    east = x_axis - (x_axis @ up) * up
    east = east / np.linalg.norm(east)
    north = np.cross(up, east)

    x0 = np.zeros(N_STATES)
    x0[:3] = data["p_gps"][0]
    x0[3:6] = data["v_gps"][0]
    x0[6:10] = _rot_to_quat(np.vstack([east, north, up]))
    return x0, np.diag(P0_DIAG)


# ---------------------------------------------------------------------------
# EKF step 3: predict
# ---------------------------------------------------------------------------
def f_state(x, a_m, w_m, dt):
    """Euler state propagation, with no normalisation."""
    p, v, q, b_a, b_g = x[:3], x[3:6], x[6:10], x[10:13], x[13:16]
    a = a_m - b_a
    w = w_m - b_g
    return np.r_[p + v * dt,
                 v + (quat_to_rot(q) @ a + G_NAV) * dt,
                 q + 0.5 * omega_matrix(w) @ q * dt,
                 b_a,
                 b_g]


def jacobian_F(x, a_m, w_m, dt):
    """State transition Jacobian at the state passed in. Call it before f_state."""
    q, b_a, b_g = x[6:10], x[10:13], x[13:16]
    a = a_m - b_a
    w = w_m - b_g
    qw, e = q[0], q[1:]
    d_Ra_d_q = np.zeros((3, 4))
    d_Ra_d_q[:, 0] = qw * a + np.cross(e, a)
    d_Ra_d_q[:, 1:] = (e @ a) * np.eye(3) + np.outer(e, a) - np.outer(a, e) - qw * skew(a)

    F = np.eye(N_STATES)
    F[0:3, 3:6] = np.eye(3) * dt
    F[3:6, 6:10] = 2.0 * d_Ra_d_q * dt
    F[3:6, 10:13] = -quat_to_rot(q) * dt
    F[6:10, 6:10] += 0.5 * omega_matrix(w) * dt
    F[6:10, 13:16] = -0.5 * xi_matrix(q) * dt
    return F


def normalize_state(x, P):
    """Normalise q and project P, so that P loses the same part of q that normalising removes."""
    x, P = x.copy(), P.copy()
    norm = np.linalg.norm(x[6:10])
    q = x[6:10] / norm
    x[6:10] = q
    J = (np.eye(4) - np.outer(q, q)) / norm
    P[6:10, :] = J @ P[6:10, :]
    P[:, 6:10] = P[:, 6:10] @ J.T
    return x, P


def ekf_predict(x, P, a_m, w_m, dt, Q):
    """Full predict step, with F taken at the previous state."""
    F = jacobian_F(x, a_m, w_m, dt)
    x_pred = f_state(x, a_m, w_m, dt)
    P_pred = F @ P @ F.T + Q
    return normalize_state(x_pred, P_pred)


# ---------------------------------------------------------------------------
# EKF step 4: update
# ---------------------------------------------------------------------------
def h_gps(x):
    """Predicted GPS position."""
    return x[:3]


def jacobian_H():
    """Measurement Jacobian."""
    return np.hstack([np.eye(3), np.zeros((3, N_STATES - 3))])


def ekf_update(x_pred, P_pred, z, R):
    """Full update step, with the Joseph-form covariance."""
    H = jacobian_H()
    y = z - h_gps(x_pred)
    S = H @ P_pred @ H.T + R
    S_inv = np.linalg.inv(S)
    K = P_pred @ H.T @ S_inv
    x = x_pred + K @ y
    I_KH = np.eye(N_STATES) - K @ H
    P = I_KH @ P_pred @ I_KH.T + K @ R @ K.T
    nis = float(y @ S_inv @ y)
    x, P = normalize_state(x, P)
    return x, P, y, S, nis


# ---------------------------------------------------------------------------
# Whole run
# ---------------------------------------------------------------------------
def run_ekf(data, Q, R, x0, P0, t_end=None):
    """Run the filter up to t_end, or over the whole flight.

    Returns x_log (one row per IMU sample processed, in STATE_COLUMNS order),
    innovation_log (one row per GPS update, in INNOVATION_COLUMNS order),
    mean_nis, and ok, which is False if the run failed.
    """
    t_imu, accel, gyro = data["t_imu"], data["accel"], data["gyro"]
    t_gps, p_gps = data["t_gps"], data["p_gps"]
    n_imu, n_gps = len(t_imu), len(t_gps)

    x, P = x0.copy(), P0.copy()
    x_log = np.zeros((n_imu, N_STATES))
    x_log[0] = x
    innovation_log = np.zeros((n_gps, len(INNOVATION_COLUMNS)))
    n_updates = 0
    j = 1                               # the first fix already set x0
    last = n_imu - 1
    ok = True
    for k in range(1, n_imu):
        if t_end is not None and t_imu[k] > t_end:
            last = k - 1
            break
        dt = t_imu[k] - t_imu[k - 1]
        x, P = ekf_predict(x, P, accel[k - 1], gyro[k - 1], dt, Q)
        try:
            while j < n_gps and t_gps[j] <= t_imu[k]:
                x, P, y, S, nis = ekf_update(x, P, p_gps[j], R)
                innovation_log[n_updates] = np.r_[t_gps[j], y, np.diag(S), nis]
                n_updates += 1
                j += 1
        except np.linalg.LinAlgError:
            ok, last = False, k - 1
            break
        x_log[k] = x
        if not np.isfinite(x).all() or np.abs(x[:3]).max() > MAX_POSITION:
            ok, last = False, k
            break
    innovation_log = innovation_log[:n_updates]
    mean_nis = float(innovation_log[:, -1].mean()) if n_updates else float("nan")
    return x_log[:last + 1], innovation_log, mean_nis, ok


# ---------------------------------------------------------------------------
# EKF step 5: score
# ---------------------------------------------------------------------------
def position_error(t_est, p_est, t_true, p_true):
    """Horizontal RMS error, with the estimate interpolated to the ground-truth times."""
    east = np.interp(t_true, t_est, p_est[:, 0]) - p_true[:, 0]
    north = np.interp(t_true, t_est, p_est[:, 1]) - p_true[:, 1]
    return float(np.sqrt(np.mean(east ** 2 + north ** 2)))


def make_cost_fn(data, candidate_log=None):
    """Bind the data to the one cost function that every optimizer calls.

    cost_fn(theta) runs the filter to T_SPLIT and returns rmse_h over that part,
    or PENALTY if the run failed. Each call is added to candidate_log if given.
    """
    x0, P0 = init_state(data)
    tune = data["t_true"] < T_SPLIT
    t_true, p_true = data["t_true"][tune], data["p_true"][tune]

    def cost_fn(theta):
        Q, R = theta_to_QR(theta)
        x_log, _, mean_nis, ok = run_ekf(data, Q, R, x0, P0, t_end=T_SPLIT)
        cost = position_error(data["t_imu"][:len(x_log)], x_log, t_true, p_true) if ok else PENALTY
        if candidate_log is not None:
            candidate_log.append({"theta": np.array(theta, dtype=float), "cost": cost, "mean_nis": mean_nis})
        return cost

    return cost_fn
