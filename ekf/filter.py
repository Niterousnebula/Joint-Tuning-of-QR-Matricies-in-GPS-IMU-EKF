"""16-state GPS-IMU EKF built on filterpy's ExtendedKalmanFilter.

State x = [p (3), v (3), q (4), b_a (3), b_g (3)], ENU navigation frame,
Hamilton scalar-first quaternion q mapping body vectors into the navigation frame.
The IMU drives the prediction; GPS position and velocity drive the update.
"""
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter

N_STATES = 16
N_MEAS = 6
STATE_COLUMNS = ["p_e", "p_n", "p_u", "v_e", "v_n", "v_u", "qw", "qx", "qy", "qz",
                 "b_ax", "b_ay", "b_az", "b_gx", "b_gy", "b_gz"]
INNOVATION_COLUMNS = ["t", "y_pe", "y_pn", "y_pu", "y_ve", "y_vn", "y_vu",
                      "s_pe", "s_pn", "s_pu", "s_ve", "s_vn", "s_vu", "nis"]

GRAVITY = 9.80665
G_NAV = np.array([0.0, 0.0, -GRAVITY])

# Initial covariance diagonal: p, v, q, b_a, b_g
P0_DIAG = np.r_[np.full(3, 25.0), np.full(3, 2.25), np.full(4, 0.07),
                np.full(3, 0.25), np.full(3, 0.0027)]
INIT_ATTITUDE_WINDOW = 1.0      # s of accelerometer data averaged for roll and pitch
MAX_POSITION = 1e6              # m; any position beyond this counts as diverged

SL_P, SL_V, SL_Q, SL_BA, SL_BG = slice(0, 3), slice(3, 6), slice(6, 10), slice(10, 13), slice(13, 16)
H_GPS = np.zeros((N_MEAS, N_STATES))
H_GPS[:, :6] = np.eye(6)


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------
def skew(v):
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
    """Right-multiplication matrix: q ⊗ [0, w] = omega_matrix(w) @ q."""
    wx, wy, wz = w
    return np.array([[0.0, -wx, -wy, -wz],
                     [wx, 0.0, wz, -wy],
                     [wy, -wz, 0.0, wx],
                     [wz, wy, -wx, 0.0]])


def xi_matrix(q):
    """q ⊗ [0, w] = xi_matrix(q) @ w."""
    qw, qx, qy, qz = q
    return np.array([[-qx, -qy, -qz],
                     [qw, -qz, qy],
                     [qz, qw, -qx],
                     [-qy, qx, qw]])


def d_rot_times_a_dq(q, a):
    """Jacobian of R(q) @ a with respect to q (3 x 4)."""
    qw, e = q[0], q[1:]
    J = np.empty((3, 4))
    J[:, 0] = qw * a + np.cross(e, a)
    J[:, 1:] = (e @ a) * np.eye(3) + np.outer(e, a) - np.outer(a, e) - qw * skew(a)
    return 2.0 * J


def quat_step_matrix(w, dt):
    """Exact transition for constant body rate: q_k = quat_step_matrix(w, dt) @ q_{k-1}."""
    angle = np.linalg.norm(w) * dt
    half = 0.5 * angle
    if angle < 1e-9:
        return np.eye(4) + 0.5 * omega_matrix(w) * dt
    return np.cos(half) * np.eye(4) + (np.sin(half) / np.linalg.norm(w)) * omega_matrix(w)


def left_matrix(q):
    """Left-multiplication matrix: q ⊗ p = left_matrix(q) @ p."""
    qw, qx, qy, qz = q
    return np.array([[qw, -qx, -qy, -qz],
                     [qx, qw, -qz, qy],
                     [qy, qz, qw, -qx],
                     [qz, -qy, qx, qw]])


def d_quat_step_dw(q, w, dt):
    """Jacobian of quat_step_matrix(w, dt) @ q with respect to w (4 x 3).

    The step is q ⊗ dq(w) with dq = [cos(n dt / 2), sin(n dt / 2) w / n], n = |w|.
    """
    n = np.linalg.norm(w)
    if n * dt < 1e-6:
        d_dq = np.vstack([-0.25 * dt * dt * w, 0.5 * dt * np.eye(3)])
    else:
        half = 0.5 * n * dt
        s = np.sin(half) / n
        ds_dn = (0.5 * dt * np.cos(half) * n - np.sin(half)) / (n * n)
        u = w / n
        d_dq = np.vstack([-0.5 * dt * np.sin(half) * u, s * np.eye(3) + ds_dn * np.outer(w, u)])
    return left_matrix(q) @ d_dq


def rot_to_quat(R_nb):
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
# Measurement model (GPS position and velocity)
# ---------------------------------------------------------------------------
def h_gps(x):
    return x[:6]


def jacobian_H(x):
    return H_GPS


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------
class GpsImuEKF(ExtendedKalmanFilter):
    """filterpy EKF with the strapdown IMU model as its nonlinear prediction.

    Q_c holds the continuous noise densities, in order: accelerometer white noise,
    gyroscope white noise, accelerometer bias random walk and gyroscope bias random walk,
    each a 3-vector. They are turned into a discrete Q for every IMU step from that
    step's dt, so gaps in the IMU log get the right amount of process noise.
    """

    def __init__(self, Q_c, R):
        super().__init__(dim_x=N_STATES, dim_z=N_MEAS)
        self.Q_c = np.asarray(Q_c, dtype=float)
        self.R = np.asarray(R, dtype=float)

    # filterpy calls predict_x and then P = F P F' + Q, so F and Q are set here,
    # at the previous state, before x is propagated.
    def predict_x(self, u):
        a_m, w_m, dt = u
        x = self.x
        p, v, q, b_a, b_g = x[SL_P], x[SL_V], x[SL_Q], x[SL_BA], x[SL_BG]
        a = a_m - b_a
        w = w_m - b_g
        R_nb = quat_to_rot(q)
        acc_nav = R_nb @ a + G_NAV
        Phi_q = quat_step_matrix(w, dt)
        dRa_dq = d_rot_times_a_dq(q, a)
        Xi = xi_matrix(q)

        F = np.eye(N_STATES)
        F[SL_P, SL_V] = np.eye(3) * dt
        F[SL_P, SL_Q] = 0.5 * dt * dt * dRa_dq
        F[SL_P, SL_BA] = -0.5 * dt * dt * R_nb
        F[SL_V, SL_Q] = dRa_dq * dt
        F[SL_V, SL_BA] = -R_nb * dt
        F[SL_Q, SL_Q] = Phi_q
        F[SL_Q, SL_BG] = -d_quat_step_dw(q, w, dt)
        self.F = F

        # Noise input matrix G: [n_a, n_g, n_ba, n_bg] -> state rates
        G = np.zeros((N_STATES, 12))
        G[SL_V, 0:3] = -R_nb
        G[SL_Q, 3:6] = -0.5 * Xi
        G[SL_BA, 6:9] = np.eye(3)
        G[SL_BG, 9:12] = np.eye(3)
        self.Q = (G * self.Q_c) @ G.T * dt

        x_new = x.copy()
        x_new[SL_P] = p + v * dt + 0.5 * acc_nav * dt * dt
        x_new[SL_V] = v + acc_nav * dt
        x_new[SL_Q] = Phi_q @ q
        self.x = x_new

    def normalize(self):
        """Normalise q and remove the matching part of P."""
        q = self.x[SL_Q]
        norm = np.linalg.norm(q)
        q_bar = q / norm
        self.x[SL_Q] = q_bar
        J = (np.eye(4) - np.outer(q_bar, q_bar)) / norm
        self.P[SL_Q, :] = J @ self.P[SL_Q, :]
        self.P[:, SL_Q] = self.P[:, SL_Q] @ J.T
        self.P = 0.5 * (self.P + self.P.T)

    def step_predict(self, a_m, w_m, dt):
        self.predict(u=(a_m, w_m, dt))
        self.normalize()

    def step_update(self, z):
        self.update(z, jacobian_H, h_gps)
        self.normalize()
        return float(self.y @ self.SI @ self.y)


def init_state(data):
    """Initial state and covariance.

    Position and velocity come from the first GPS fix. Roll and pitch come from the
    mean specific force over the first second; yaw starts at 0 (body x pointing east)
    and is corrected from GPS velocity once the vehicle moves.
    """
    t_imu, accel = data["t_imu"], data["accel"]
    up = accel[t_imu < t_imu[0] + INIT_ATTITUDE_WINDOW].mean(axis=0)
    up = up / np.linalg.norm(up)
    x_axis = np.array([1.0, 0.0, 0.0])
    east = x_axis - (x_axis @ up) * up
    east = east / np.linalg.norm(east)
    north = np.cross(up, east)

    x0 = np.zeros(N_STATES)
    x0[SL_P] = data["p_gps"][0]
    x0[SL_V] = data["v_gps"][0]
    x0[SL_Q] = rot_to_quat(np.vstack([east, north, up]))
    return x0, np.diag(P0_DIAG)


def run_ekf(data, Q_c, R, x0=None, P0=None, t_end=None):
    """Run the filter over the flight, or up to t_end.

    Returns x_log (one row per IMU sample, STATE_COLUMNS order), innovation_log
    (one row per GPS update, INNOVATION_COLUMNS order), mean_nis and ok, which is
    False if the run diverged or a matrix inversion failed.
    """
    if x0 is None or P0 is None:
        x0, P0 = init_state(data)
    t_imu, accel, gyro = data["t_imu"], data["accel"], data["gyro"]
    t_gps = data["t_gps"]
    z_gps = np.hstack([data["p_gps"], data["v_gps"]])
    n_imu, n_gps = len(t_imu), len(t_gps)

    kf = GpsImuEKF(Q_c, R)
    kf.x, kf.P = x0.copy(), P0.copy()

    x_log = np.zeros((n_imu, N_STATES))
    x_log[0] = kf.x
    innovation_log = np.zeros((n_gps, len(INNOVATION_COLUMNS)))
    n_updates = 0
    j = 1                               # the first fix already set x0
    while j < n_gps and t_gps[j] <= t_imu[0]:
        j += 1
    last = n_imu - 1
    ok = True
    for k in range(1, n_imu):
        if t_end is not None and t_imu[k] > t_end:
            last = k - 1
            break
        try:
            kf.step_predict(accel[k - 1], gyro[k - 1], t_imu[k] - t_imu[k - 1])
            while j < n_gps and t_gps[j] <= t_imu[k]:
                nis = kf.step_update(z_gps[j])
                innovation_log[n_updates] = np.r_[t_gps[j], kf.y, np.diag(kf.S), nis]
                n_updates += 1
                j += 1
        except np.linalg.LinAlgError:
            ok, last = False, k - 1
            break
        x_log[k] = kf.x
        if not np.isfinite(kf.x).all() or np.abs(kf.x[SL_P]).max() > MAX_POSITION:
            ok, last = False, k
            break
    innovation_log = innovation_log[:n_updates]
    mean_nis = float(innovation_log[:, -1].mean()) if n_updates else float("nan")
    return x_log[:last + 1], innovation_log, mean_nis, ok
