"""Decision vector, bounds, cost function and scoring shared by every optimizer."""
import time

import numpy as np

from .data import robustness_datasets
from .filter import init_state, run_ekf

# log10 of the continuous noise densities and GPS variances
THETA_NAMES = ["log_q_acc", "log_q_gyro", "log_q_ba", "log_q_bg",
               "log_r_xy", "log_r_z", "log_r_vel"]
THETA0 = np.array([0.2, -3.0, -5.0, -8.0, 1.0, 1.5, 0.0])
LOWER = THETA0 - np.array([3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0])
UPPER = THETA0 + np.array([3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0])
LOWER_TIGHT = THETA0 - np.array([1.5, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0])
UPPER_TIGHT = THETA0 + np.array([1.5, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0])

T_SPLIT = 1364.04       # s; tune before this time, hold out after it
PENALTY = 1000.0        # m; cost of a run that fails


def theta_to_QR(theta):
    """Continuous noise densities Q_c (12) and the GPS R (6 x 6) from a candidate."""
    var = 10.0 ** np.asarray(theta, dtype=float)
    Q_c = np.r_[np.full(3, var[0]), np.full(3, var[1]), np.full(3, var[2]), np.full(3, var[3])]
    R = np.diag([var[4], var[4], var[5], var[6], var[6], var[6]])
    return Q_c, R


def horizontal_errors(t_est, p_est, t_true, p_true):
    """Horizontal error at each ground-truth time, estimate interpolated onto t_true."""
    east = np.interp(t_true, t_est, p_est[:, 0]) - p_true[:, 0]
    north = np.interp(t_true, t_est, p_est[:, 1]) - p_true[:, 1]
    return np.hypot(east, north)


def rmse(err):
    return float(np.sqrt(np.mean(np.square(err))))


def position_error(t_est, p_est, t_true, p_true):
    """Horizontal RMS error against ground truth."""
    return rmse(horizontal_errors(t_est, p_est, t_true, p_true))


def make_cost_fn(data, candidate_log=None):
    """The one cost function every optimizer calls.

    cost_fn(theta) runs the filter up to T_SPLIT and returns the horizontal RMSE over
    that part, or PENALTY if the run failed.
    """
    x0, P0 = init_state(data)
    tune = data["t_true"] < T_SPLIT
    t_true, p_true = data["t_true"][tune], data["p_true"][tune]

    def cost_fn(theta):
        Q_c, R = theta_to_QR(theta)
        x_log, _, mean_nis, ok = run_ekf(data, Q_c, R, x0, P0, t_end=T_SPLIT)
        cost = position_error(data["t_imu"][:len(x_log)], x_log, t_true, p_true) if ok else PENALTY
        if not np.isfinite(cost):
            cost = PENALTY
        if candidate_log is not None:
            candidate_log.append({"theta": np.array(theta, dtype=float), "cost": cost,
                                  "mean_nis": mean_nis})
        return cost

    return cost_fn


def held_out_rmse(data, theta):
    """Run the whole flight and score only t >= T_SPLIT. Returns (rmse_h, mean_nis, ok)."""
    Q_c, R = theta_to_QR(theta)
    x_log, _, mean_nis, ok = run_ekf(data, Q_c, R)
    if not ok:
        return PENALTY, mean_nis, False
    held = data["t_true"] >= T_SPLIT
    err = horizontal_errors(data["t_imu"][:len(x_log)], x_log, data["t_true"][held], data["p_true"][held])
    return rmse(err), mean_nis, True


def robustness(data, theta, datasets=None):
    """Held-out RMSE of one theta on every robustness copy of the flight.

    Returns a dict with per-dataset RMSE, their mean, and their standard deviation,
    which is the robustness score (lower is more stable).
    """
    datasets = robustness_datasets(data) if datasets is None else datasets
    per_dataset = {name: held_out_rmse(d, theta)[0] for name, d in datasets.items()}
    values = np.array(list(per_dataset.values()))
    return {"per_dataset": per_dataset, "mean": float(values.mean()), "std": float(values.std())}


def timed(fn, *args, **kwargs):
    """Call fn and return (result, wall_time_seconds)."""
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - start
