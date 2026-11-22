"""Batched GPU evaluation of the canonical fitness: a whole population in one pass.

The EKF in `filter.py` (filterpy) is the reference. This module runs the same filter
for B candidates at once with PyTorch in float64:

    [B candidates] -> batched EKF -> [B whole-flight RMSE_h values]

Only the candidate dimension is batched. The time recursion is the reference loop
unchanged: the same IMU predictions, the same GPS updates in the same order, the
same quaternion normalisation after each, the same divergence checks. Each
candidate keeps its own state x, covariance P, continuous noise Q_c and GPS R.
Every formula below mirrors `GpsImuEKF.predict_x`, filterpy's `predict` and
`update` (Joseph form) and `GpsImuEKF.normalize`, written in the same order. The
small quaternion matrices are produced by constant matrices taken from the
reference helpers themselves (see `_basis`), so they are the same functions.

Results agree with the CPU reference to floating-point rounding; `compare_with_cpu`
measures the difference for a set of candidates.

A failed candidate (diverged state or a singular innovation covariance) scores
PENALTY, as in `tuning.evaluate`; its arithmetic keeps running inside the batch but
never touches the other candidates.
"""
import time

import numpy as np
import torch
from scipy.stats import chi2

from .filter import (G_NAV, MAX_POSITION, N_MEAS, N_STATES, OPTIONS, d_rot_times_a_dq, init_state,
                     left_matrix, measurement_channels, omega_matrix, quat_to_rot, xi_matrix)

DTYPE = torch.float64
TOL_RMSE = 1e-6          # largest accepted |RMSE_gpu - RMSE_cpu|: in m up to 1 m, relative above
TOL_NIS_REL = 1e-6       # largest accepted relative difference in mean NIS
NIS_UNSTABLE = 1e3       # mean NIS above this: the filter is numerically chaotic, so rounding decides its
                         # result on any backend (CPU alone moves by tens of metres for a 1e-12 change)


# ---------------------------------------------------------------------------
# Constant matrices that reproduce the reference quaternion helpers
# ---------------------------------------------------------------------------
def _basis():
    """Linear, bilinear and quadratic forms of the helpers in filter.py.

    omega_matrix, xi_matrix and left_matrix are linear in their argument,
    d_rot_times_a_dq is bilinear in (q, a) and quat_to_rot is quadratic in q, so
    each is exactly one constant matrix applied to w, q, q (x) a or q (x) q.
    """
    e3, e4 = np.eye(3), np.eye(4)
    c_omega = np.stack([omega_matrix(e3[i]).ravel() for i in range(3)])          # 3 x 16
    c_xi = np.stack([xi_matrix(e4[i]).ravel() for i in range(4)])                # 4 x 12
    c_left = np.stack([left_matrix(e4[i]).ravel() for i in range(4)])            # 4 x 16
    c_dra = np.stack([d_rot_times_a_dq(e4[i], e3[j]).ravel()                     # 12 x 12
                      for i in range(4) for j in range(3)])
    c_rot = np.zeros((16, 9))                                                    # 16 x 9
    for i in range(4):
        c_rot[i * 4 + i] = quat_to_rot(e4[i]).ravel()
        for j in range(i + 1, 4):
            c_rot[i * 4 + j] = (quat_to_rot(e4[i] + e4[j]) - quat_to_rot(e4[i])
                                - quat_to_rot(e4[j])).ravel()
    return c_omega, c_xi, c_left, c_dra, c_rot


def _theta_to_qr(thetas):
    """Batched tuning.theta_to_QR: Q_c (B x 12) and R (B x 6 x 6)."""
    var = 10.0 ** np.asarray(thetas, dtype=float)
    Q_c = np.repeat(var[:, :4], 3, axis=1)
    R = np.zeros((len(var), N_MEAS, N_MEAS))
    R[:, [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]] = var[:, [4, 4, 5, 6, 6, 6]]
    return Q_c, R


class _Batch:
    """Persistent GPU buffers and (optionally) CUDA graphs for one batch size."""

    def __init__(self, ev, B):
        dev = ev.device
        self.B = B
        z = lambda *s: torch.zeros(*s, dtype=DTYPE, device=dev)
        self.x, self.P = z(B, N_STATES), z(B, N_STATES, N_STATES)
        self.Q_c, self.R = z(B, 12), z(B, ev.m, ev.m)
        self.F = ev.I16.expand(B, N_STATES, N_STATES).clone()
        self.G = z(B, N_STATES, 12)
        self.G[:, 10:13, 6:9] = ev.I3
        self.G[:, 13:16, 9:12] = ev.I3
        self.failed = torch.zeros(B, dtype=torch.bool, device=dev)
        self.nis_sum, self.n_upd = z(B), z(B)
        self.t_last, self.n_rej = z(B), z(B)        # time of the last accepted fix, fixes rejected
        self.p_last = z(B, 2)                       # plausibility rule: position at the last accepted fix
        self.pos_log = z(ev.n_imu, B, 3)
        self.s = torch.zeros(1, dtype=torch.long, device=dev)   # IMU step counter (k - 1)
        self.u = torch.zeros(1, dtype=torch.long, device=dev)   # GPS update counter
        self.graphs = None


class GpuEvaluator:
    """Whole-flight fitness for a batch of candidates on one dataset.

    evaluate(thetas) returns numpy arrays fitness (J, tuning.objective_value), rmse_h (PENALTY on failure),
    rmse_v, rmse_3d, mean_nis and ok, one value per candidate, as tuning.evaluate.
    """

    def __init__(self, data, device="cuda", use_graphs=True, x0=None):
        """x0 (optional) replaces init_state's initial state, e.g. when data is a resampled stream."""
        from .filter import nis_dof
        from .tuning import PENALTY
        self.penalty = PENALTY
        self.dof = nis_dof(data)
        self.device = torch.device(device)
        self.use_graphs = use_graphs and self.device.type == "cuda"
        dev, T = self.device, (lambda a: torch.as_tensor(np.asarray(a, dtype=float), dtype=DTYPE, device=dev))

        t_imu, accel, gyro = data["t_imu"], data["accel"], data["gyro"]
        t_gps = data["t_gps"]
        self.channels = measurement_channels(data)       # the GPS channels this dataset provides
        self.m = len(self.channels)
        self.ch = torch.as_tensor(self.channels, device=dev)
        z_gps = np.hstack([data["p_gps"], data["v_gps"]])[:, self.channels]
        self.n_imu = n_imu = len(t_imu)

        # The reference schedule of run_ekf: which GPS fixes follow each IMU prediction
        j, n_gps = 1, len(t_gps)
        while j < n_gps and t_gps[j] <= t_imu[0]:
            j += 1
        j0, counts = j, np.zeros(n_imu, dtype=int)
        for k in range(1, n_imu):
            while j < n_gps and t_gps[j] <= t_imu[k]:
                counts[k] += 1
                j += 1
        self.counts = counts
        self.n_updates = int(counts.sum())

        # Inputs indexed by the step counter s = k - 1 and the update counter
        self.u_a, self.u_w = T(accel[:-1]), T(gyro[:-1])
        self.u_dt = T(np.diff(t_imu))
        self.z = T(z_gps[j0:j0 + self.n_updates])
        self.zt = T(t_gps[j0:j0 + self.n_updates])
        self.gate = OPTIONS["gate"]                    # read when the evaluator is built
        self.gate_limit = float(chi2.isf(OPTIONS["gate_p"], self.m))
        self.gate_timeout = OPTIONS["gate_timeout_s"]
        self.plausibility = OPTIONS["plausibility"]
        self.max_speed, self.plausible_m = OPTIONS["max_speed_ms"], OPTIONS["plausible_m"]
        self.t_start = float(t_imu[0])

        x_init, P0 = init_state(data)
        self.x0, self.P0 = T(x_init if x0 is None else x0), T(P0)
        self.g_nav = T(G_NAV)
        self.I3 = torch.eye(3, dtype=DTYPE, device=dev)
        self.I4 = torch.eye(4, dtype=DTYPE, device=dev)
        self.I16 = torch.eye(N_STATES, dtype=DTYPE, device=dev)
        self.c_omega, self.c_xi, self.c_left, self.c_dra, self.c_rot = (T(c) for c in _basis())

        # np.interp(t_true, t_imu, p) as index + slope terms, evaluated on the GPU
        t_true = data["t_true"]
        jj = np.clip(np.searchsorted(t_imu, t_true, side="right") - 1, 0, n_imu - 2)
        self.i0 = torch.as_tensor(jj, device=dev)
        self.i1 = torch.as_tensor(jj + 1, device=dev)
        self.dx = T(t_imu[jj + 1] - t_imu[jj]).view(-1, 1, 1)
        self.xo = T(t_true - t_imu[jj]).view(-1, 1, 1)
        self.below = torch.as_tensor(t_true < t_imu[0], device=dev).view(-1, 1, 1)
        self.above = torch.as_tensor(t_true > t_imu[-1], device=dev).view(-1, 1, 1)
        self.p_true = T(data["p_true"]).unsqueeze(1)                 # n_true x 1 x 3

        self._batches = {}
        self.n_evaluated = 0           # candidates scored by this evaluator
        self.t_gpu = 0.0               # wall-clock seconds spent in evaluate()

    # -- one EKF step for the whole batch -------------------------------------
    def _normalize(self, b):
        x, P = b.x, b.P
        q = x[:, 6:10]
        norm = torch.linalg.vector_norm(q, dim=1, keepdim=True)
        q_bar = q / norm
        x[:, 6:10] = q_bar
        J = (self.I4 - q_bar.unsqueeze(2) * q_bar.unsqueeze(1)) / norm.unsqueeze(2)
        P[:, 6:10, :] = J @ P[:, 6:10, :]
        P[:, :, 6:10] = P[:, :, 6:10] @ J.transpose(1, 2)
        P.copy_(0.5 * (P + P.transpose(1, 2)))

    def _predict(self, b):
        """GpsImuEKF.predict_x, then P = F P F' + Q, then normalize."""
        B, x, P = b.B, b.x, b.P
        a_m = self.u_a.index_select(0, b.s)                         # 1 x 3
        w_m = self.u_w.index_select(0, b.s)
        dt = self.u_dt.index_select(0, b.s).view(1, 1)
        dt3 = dt.view(1, 1, 1)
        p, v, q, b_a, b_g = x[:, 0:3], x[:, 3:6], x[:, 6:10], x[:, 10:13], x[:, 13:16]
        a = a_m - b_a
        w = w_m - b_g
        R_nb = ((q.unsqueeze(2) * q.unsqueeze(1)).reshape(B, 16) @ self.c_rot).view(B, 3, 3)
        acc_nav = (R_nb @ a.unsqueeze(2)).squeeze(2) + self.g_nav
        # quat_step_matrix(w, dt)
        n = torch.linalg.vector_norm(w, dim=1, keepdim=True)        # B x 1
        n3 = n.view(B, 1, 1)
        Om = (w @ self.c_omega).view(B, 4, 4)
        angle = n3 * dt3
        half = 0.5 * angle
        Phi_q = torch.where(angle < 1e-9, self.I4 + 0.5 * Om * dt3,
                            torch.cos(half) * self.I4 + (torch.sin(half) / n3) * Om)
        dRa_dq = ((q.unsqueeze(2) * a.unsqueeze(1)).reshape(B, 12) @ self.c_dra).view(B, 3, 4)
        Xi = (q @ self.c_xi).view(B, 4, 3)
        # d_quat_step_dw(q, w, dt)
        small = (n * dt) < 1e-6                                      # B x 1
        h = 0.5 * n * dt
        s = torch.sin(h) / n
        ds_dn = (0.5 * dt * torch.cos(h) * n - torch.sin(h)) / (n * n)
        u = w / n
        top = torch.where(small, -0.25 * dt * dt * w, -0.5 * dt * torch.sin(h) * u)
        low = torch.where(small.view(B, 1, 1), 0.5 * dt3 * self.I3,
                          s.view(B, 1, 1) * self.I3 + ds_dn.view(B, 1, 1) * (w.unsqueeze(2) * u.unsqueeze(1)))
        d_dq = torch.cat([top.unsqueeze(1), low], dim=1)            # B x 4 x 3
        dq_dw = (q @ self.c_left).view(B, 4, 4) @ d_dq

        F = b.F
        F[:, 0:3, 3:6] = self.I3 * dt3
        F[:, 0:3, 6:10] = 0.5 * dt3 * dt3 * dRa_dq
        F[:, 0:3, 10:13] = -0.5 * dt3 * dt3 * R_nb
        F[:, 3:6, 6:10] = dRa_dq * dt3
        F[:, 3:6, 10:13] = -R_nb * dt3
        F[:, 6:10, 6:10] = Phi_q
        F[:, 6:10, 13:16] = -dq_dw
        G = b.G
        G[:, 3:6, 0:3] = -R_nb
        G[:, 6:10, 3:6] = -0.5 * Xi
        Q = ((G * b.Q_c.unsqueeze(1)) @ G.transpose(1, 2)) * dt3

        x_new = torch.cat([p + v * dt + 0.5 * acc_nav * dt * dt,
                           v + acc_nav * dt,
                           (Phi_q @ q.unsqueeze(2)).squeeze(2),
                           b_a, b_g], dim=1)
        P_new = (F @ P) @ F.transpose(1, 2) + Q
        x.copy_(x_new)
        P.copy_(P_new)
        self._normalize(b)

    def _update(self, b):
        """filterpy ExtendedKalmanFilter.update (Joseph form) with the plausibility rule and the
        chi-square gate, then normalize.

        A rejected fix leaves x and P untouched, exactly as run_ekf skips it. Everything is masked,
        never branched, so the step stays valid inside a CUDA graph.
        """
        x, P, R, ch = b.x, b.P, b.R, self.ch
        z = self.z.index_select(0, b.u)                              # 1 x m
        PHT = P.index_select(2, ch)                                  # B x 16 x m
        S = PHT.index_select(1, ch) + R
        SI, info = torch.linalg.inv_ex(S)
        K = PHT @ SI
        y = z - x.index_select(1, ch)
        x_new = x + (K @ y.unsqueeze(2)).squeeze(2)
        I_KH = self.I16.expand_as(P).clone()
        I_KH.index_add_(2, ch, -K)
        P_new = (I_KH @ P) @ I_KH.transpose(1, 2) + (K @ R) @ K.transpose(1, 2)
        nis = ((y.unsqueeze(1) @ SI) @ y.unsqueeze(2)).view(-1)
        b.failed |= info != 0
        alive = ~b.failed
        if self.gate or self.plausibility:
            t_now = self.zt.index_select(0, b.u)                     # 1
            accept = torch.ones_like(alive)
            if self.plausibility:
                jump = torch.linalg.vector_norm(z[:, 0:2] - b.p_last, dim=1)
                accept &= jump <= self.plausible_m + self.max_speed * (t_now - b.t_last)
            if self.gate:
                accept &= (nis <= self.gate_limit) | ((t_now - b.t_last) > self.gate_timeout)
            b.t_last.copy_(torch.where(accept, t_now, b.t_last))
            b.n_rej += (alive & ~accept).to(DTYPE)
            x_new = torch.where(accept.view(-1, 1), x_new, x)
            P_new = torch.where(accept.view(-1, 1, 1), P_new, P)
            b.p_last.copy_(torch.where(accept.view(-1, 1), x_new[:, 0:2], b.p_last))
            counted = alive & accept
        else:
            counted = alive
        b.nis_sum += torch.where(counted, nis, torch.zeros_like(nis))
        b.n_upd += counted.to(DTYPE)
        x.copy_(x_new)
        P.copy_(P_new)
        b.u += 1
        self._normalize(b)

    def _end_step(self, b):
        """Log the position for step k and apply run_ekf's divergence check."""
        b.pos_log.index_copy_(0, b.s + 1, b.x[:, 0:3].unsqueeze(0))
        bad = ~torch.isfinite(b.x).all(dim=1) | (b.x[:, 0:3].abs().amax(dim=1) > MAX_POSITION)
        b.failed |= bad
        b.s += 1

    def _step(self, b, n_upd):
        self._predict(b)
        for _ in range(n_upd):
            self._update(b)
        self._end_step(b)

    # -- batch management -----------------------------------------------------
    def _reset(self, b, thetas, bias0=None):
        Q_c, R = _theta_to_qr(thetas)
        R = R[:, self.channels][:, :, self.channels]                 # the block of the used channels
        b.Q_c.copy_(torch.as_tensor(Q_c, dtype=DTYPE))
        b.R.copy_(torch.as_tensor(R, dtype=DTYPE))
        b.x.copy_(self.x0.expand(b.B, -1))
        if bias0 is not None:
            b.x[:, 10:16] = torch.as_tensor(np.asarray(bias0, dtype=float), dtype=DTYPE)
        b.P.copy_(self.P0.expand(b.B, -1, -1))
        b.failed.zero_()
        b.nis_sum.zero_()
        b.n_upd.zero_()
        b.t_last.fill_(self.t_start)
        b.p_last.copy_(self.x0[0:2].expand(b.B, -1))
        b.n_rej.zero_()
        b.pos_log.zero_()
        b.pos_log[0] = self.x0[0:3]
        b.s.zero_()
        b.u.zero_()

    def _capture(self, b, thetas):
        """CUDA graphs of one full step with 0, 1 and 2 GPS updates."""
        graphs = {}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for n_upd in sorted(set(self.counts[1:].tolist())):
                self._reset(b, thetas)
                for _ in range(2):                                   # warm-up
                    self._step(b, n_upd)
                self._reset(b, thetas)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=stream):
                    self._step(b, n_upd)
                graphs[n_upd] = g
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        b.graphs = graphs

    def _batch(self, B, thetas):
        b = self._batches.get(B)
        if b is None:
            b = self._batches[B] = _Batch(self, B)
            if self.use_graphs:
                self._capture(b, thetas)
        return b

    # -- public ---------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, thetas, bias0=None):
        """bias0 (optional, B x 6) sets each candidate's initial bias states [b_a, b_g]."""
        t0 = time.perf_counter()
        thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
        B = len(thetas)
        b = self._batch(B, thetas)
        self._reset(b, thetas, bias0)
        counts = self.counts
        if b.graphs is not None:
            graphs = b.graphs
            for k in range(1, self.n_imu):
                graphs[counts[k]].replay()
        else:
            for k in range(1, self.n_imu):
                self._step(b, counts[k])

        # Whole-flight errors at the ground-truth times (np.interp of the position log)
        pos = b.pos_log
        p0, p1 = pos.index_select(0, self.i0), pos.index_select(0, self.i1)
        est = (p1 - p0) / self.dx * self.xo + p0
        est = torch.where(self.below, pos[0:1], torch.where(self.above, pos[-1:], est))
        err = est - self.p_true                                      # n_true x B x 3
        rmse_h = torch.sqrt(torch.mean(torch.hypot(err[..., 0], err[..., 1]) ** 2, dim=0))
        rmse_v = torch.sqrt(torch.mean(err[..., 2] ** 2, dim=0))
        rmse_3d = torch.sqrt(torch.mean(torch.linalg.vector_norm(err, dim=2) ** 2, dim=0))
        ok = ~b.failed & torch.isfinite(rmse_h)
        out = {"rmse_h": torch.where(ok, rmse_h, torch.full_like(rmse_h, self.penalty)),
               "rmse_v": torch.where(ok, rmse_v, torch.full_like(rmse_v, float("nan"))),
               "rmse_3d": torch.where(ok, rmse_3d, torch.full_like(rmse_3d, float("nan"))),
               "mean_nis": b.nis_sum / b.n_upd, "ok": ok, "n_rejected": b.n_rej}
        out = {k: v.cpu().numpy() for k, v in out.items()}
        from .tuning import objective_value
        out["fitness"] = np.atleast_1d(objective_value(out["rmse_h"], out["mean_nis"], self.dof, out["ok"]))
        self.n_evaluated += B
        self.t_gpu += time.perf_counter() - t0
        return out

    def fitness(self, thetas):
        return self.evaluate(thetas)["fitness"]


_EVALUATORS = {}


def gpu_evaluator(data, device="cuda", use_graphs=True):
    """One cached evaluator per dataset object (the GPU buffers are reused across seeds)."""
    key = (id(data), device, use_graphs)
    if key not in _EVALUATORS:
        _EVALUATORS[key] = (data, GpuEvaluator(data, device=device, use_graphs=use_graphs))
    return _EVALUATORS[key][1]


def clear_cache():
    """Drop every cached evaluator and free its GPU memory (main.py calls it between datasets)."""
    _EVALUATORS.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compare_with_cpu(data, thetas, n_workers=1, device="cuda"):
    """Score the same candidates with the filterpy reference and the batched GPU EKF.

    Returns (table, summary): one row per candidate with both results and their
    absolute differences, and the largest differences with a pass/fail flag.
    """
    import pandas as pd
    from .tuning import THETA_NAMES, cpu_metrics_many

    thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
    t0 = time.perf_counter()
    cpu = cpu_metrics_many(data, thetas, n_workers=n_workers)
    t_cpu = time.perf_counter() - t0
    ev = GpuEvaluator(data, device=device)
    ev.evaluate(thetas)                    # first call includes graph capture
    t0 = time.perf_counter()
    gpu = ev.evaluate(thetas)
    t_gpu = time.perf_counter() - t0

    table = pd.DataFrame(thetas, columns=THETA_NAMES)
    for key in ("fitness", "rmse_h", "rmse_v", "rmse_3d", "mean_nis", "ok"):
        table[f"cpu_{key}"] = cpu[key]
        table[f"gpu_{key}"] = gpu[key]
    both = table["cpu_ok"] & table["gpu_ok"]
    for key in ("fitness", "rmse_h", "rmse_v", "rmse_3d"):
        table[f"diff_{key}"] = np.where(both, np.abs(table[f"gpu_{key}"] - table[f"cpu_{key}"]), 0.0)
    table["reldiff_mean_nis"] = np.where(
        both, np.abs(table["gpu_mean_nis"] - table["cpu_mean_nis"]) / np.abs(table["cpu_mean_nis"]), 0.0)
    # chaotic candidates are reported but not held to the tolerance; failed ones are checked by the ok flags
    table["stable"] = ~(table["cpu_mean_nis"] > NIS_UNSTABLE)
    st = table[table["stable"]]

    summary = {"n_candidates": len(thetas), "n_failed_cpu": int((~table["cpu_ok"]).sum()),
               "n_unstable_excluded": int((~table["stable"]).sum()), "nis_unstable": NIS_UNSTABLE,
               "max_diff_fitness_unstable": float(table.loc[~table["stable"], "diff_fitness"].max())
               if (~table["stable"]).any() else 0.0,
               "ok_flags_agree": bool((table["cpu_ok"] == table["gpu_ok"]).all()),
               "max_diff_fitness": float(st["diff_fitness"].max()),
               "max_diff_rmse_h": float(st["diff_rmse_h"].max()),
               "max_diff_rmse_v": float(st["diff_rmse_v"].max()),
               "max_diff_rmse_3d": float(st["diff_rmse_3d"].max()),
               "max_reldiff_mean_nis": float(st["reldiff_mean_nis"].max()),
               "tol_rmse": TOL_RMSE, "tol_nis_rel": TOL_NIS_REL,
               "t_cpu": t_cpu, "t_gpu": t_gpu, "cpu_workers": n_workers,
               "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               "torch": torch.__version__}
    # TOL_RMSE in metres up to 1 m, relative above: 64 m GPS noise gives errors of tens of metres
    scaled = {k: float((st[f"diff_{k}"] / np.maximum(1.0, st[f"cpu_{k}"].abs())).max()) if len(st) else 0.0
              for k in ("fitness", "rmse_h", "rmse_v", "rmse_3d")}
    summary["max_scaled_diff"] = max(scaled.values())
    summary["passed"] = bool(summary["ok_flags_agree"]
                             and summary["max_scaled_diff"] <= TOL_RMSE
                             and summary["max_reldiff_mean_nis"] <= TOL_NIS_REL)
    return table, summary
