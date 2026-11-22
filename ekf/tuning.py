"""Decision vector, the canonical fitness function, and the benchmark metrics.

Q/R calibration is offline and uses the complete flight. Every candidate is scored by
one full-flight EKF run on two measures, accuracy (whole-flight horizontal RMSE) and
consistency (mean NIS against its expected value), combined into one number by
`objective_value` (`fitness`). The baseline and every optimizer call this same function,
through `FitnessTracker`, which also counts evaluations, records the best-so-far curve and
times the run.
"""
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .data import PERTURBATIONS, perturb
from .filter import N_MEAS, OPTIONS, init_state, nis_dof, run_ekf

# log10 of the continuous noise densities and GPS variances
THETA_NAMES = ["log_q_acc", "log_q_gyro", "log_q_ba", "log_q_bg",
               "log_r_xy", "log_r_z", "log_r_vel"]
THETA0 = np.array([0.2, -3.0, -5.0, -8.0, 1.0, 1.5, 0.0])
THETA0_ETH = THETA0.copy()      # hand-set from the ETH sensor; ekf.load_data re-centres THETA0 per dataset
HALF_WIDTH = np.array([3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0])
LOWER = THETA0 - HALF_WIDTH
UPPER = THETA0 + HALF_WIDTH
LOWER_TIGHT = THETA0 - HALF_WIDTH / 2
UPPER_TIGHT = THETA0 + HALF_WIDTH / 2


def set_search_space(theta0):
    """Centre THETA0 and the bounds on one dataset (called by ekf.load_data).

    In place, so every reference to ekf.THETA0, LOWER and UPPER (including function defaults
    and FitnessTracker) follows the dataset loaded last.
    """
    THETA0[:] = theta0
    LOWER[:] = THETA0 - HALF_WIDTH
    UPPER[:] = THETA0 + HALF_WIDTH
    LOWER_TIGHT[:] = THETA0 - HALF_WIDTH / 2
    UPPER_TIGHT[:] = THETA0 + HALF_WIDTH / 2

PENALTY = 1e6           # fitness of a run that fails; above any run that does not (|position| < MAX_POSITION = 1e6 m)
# The objective J = RMSE_h * (1 + nis_weight * |log10(mean NIS / dof)|): a consistent filter (mean NIS =
# dof) scores its RMSE_h, one whose NIS is 10x too high or too low counts as nis_weight + 1 times worse.
# nis_weight = 0 is RMSE_h alone. main.py sets it per scenario.
OBJECTIVE = {"nis_weight": 1.0}
NIS_DOF = N_MEAS        # full GPS position + velocity: a consistent filter averages NIS = 6.
                        # A dataset missing channels has fewer: use ekf.nis_dof(data)
CONVERGENCE_FRACTION = 0.99


def theta_to_QR(theta):
    """Continuous noise densities Q_c (12) and the GPS R (6 x 6) from a candidate."""
    var = 10.0 ** np.asarray(theta, dtype=float)
    Q_c = np.r_[np.full(3, var[0]), np.full(3, var[1]), np.full(3, var[2]), np.full(3, var[3])]
    R = np.diag([var[4], var[4], var[5], var[6], var[6], var[6]])
    return Q_c, R


def objective_value(rmse_h, mean_nis, dof, ok=True):
    """The optimizer score J from the two measures, element-wise for arrays; PENALTY for a failed run."""
    rmse_h, mean_nis = np.asarray(rmse_h, dtype=float), np.asarray(mean_nis, dtype=float)
    w = OBJECTIVE["nis_weight"]
    with np.errstate(divide="ignore", invalid="ignore"):
        j = rmse_h * (1.0 + w * np.abs(np.log10(mean_nis / dof))) if w else rmse_h.copy()
    j = np.where(np.asarray(ok) & np.isfinite(j), j, PENALTY)
    return float(j) if j.ndim == 0 else j


def nis_consistency(mean_nis, dof):
    """|log10(mean NIS / dof)|: 0 for a consistent filter, 1 for NIS 10x off either way."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(np.log10(np.asarray(mean_nis, dtype=float) / dof))


def clip_theta(theta, lower=LOWER, upper=UPPER):
    return np.clip(np.asarray(theta, dtype=float), lower, upper)


# ---------------------------------------------------------------------------
# Errors against ground truth
# ---------------------------------------------------------------------------
def position_errors(t_est, p_est, t_true, p_true):
    """ENU error (estimate minus truth) at each ground-truth time, estimate interpolated onto t_true."""
    return np.column_stack([np.interp(t_true, t_est, p_est[:, i]) - p_true[:, i] for i in range(3)])


def horizontal_errors(t_est, p_est, t_true, p_true):
    err = position_errors(t_est, p_est, t_true, p_true)
    return np.hypot(err[:, 0], err[:, 1])


def rmse(err):
    return float(np.sqrt(np.mean(np.square(err))))


def position_error(t_est, p_est, t_true, p_true):
    """Horizontal RMSE against ground truth."""
    return rmse(horizontal_errors(t_est, p_est, t_true, p_true))


def gps_rmse(data):
    """Raw GPS reference: whole-flight horizontal RMSE with no filter."""
    return position_error(data["t_gps"], data["p_gps"], data["t_true"], data["p_true"])


# ---------------------------------------------------------------------------
# Canonical fitness
# ---------------------------------------------------------------------------
def evaluate(data, theta, bias0=None, x0=None):
    """One complete EKF run for theta, with the fitness and the secondary diagnostics.

    Returns a dict: fitness (J, objective_value), rmse_h (PENALTY on failure), rmse_v, rmse_3d,
    mean_nis, ok, x_log, innovation_log and n_rejected (GPS fixes skipped by the gate).

    bias0 (optional, 6 values [b_a, b_g]) sets the initial bias states, which the filter
    then keeps estimating; x0 (optional) replaces the initial state from init_state.
    Without them the run is the canonical one.
    """
    Q_c, R = theta_to_QR(theta)
    x_init, P0 = init_state(data)
    x0 = x_init if x0 is None else np.array(x0, dtype=float)
    if bias0 is not None:
        x0 = x0.copy()
        x0[10:16] = bias0
    x_log, innovation_log, mean_nis, ok, n_rejected = run_ekf(data, Q_c, R, x0, P0)
    out = {"ok": ok, "mean_nis": mean_nis, "x_log": x_log, "innovation_log": innovation_log,
           "n_rejected": n_rejected,
           "rmse_h": PENALTY, "rmse_v": np.nan, "rmse_3d": np.nan}
    if ok:
        err = position_errors(data["t_imu"][:len(x_log)], x_log, data["t_true"], data["p_true"])
        rmse_h = rmse(np.hypot(err[:, 0], err[:, 1]))
        if np.isfinite(rmse_h):
            out.update(rmse_h=rmse_h, rmse_v=rmse(err[:, 2]), rmse_3d=rmse(np.linalg.norm(err, axis=1)))
        else:
            out["ok"] = False
    out["fitness"] = objective_value(out["rmse_h"], mean_nis, nis_dof(data), out["ok"])
    return out


def fitness(data, theta):
    """The optimizer score J(theta): whole-flight RMSE_h weighted by NIS consistency."""
    return evaluate(data, theta)["fitness"]


METRIC_KEYS = ("rmse_h", "rmse_v", "rmse_3d", "mean_nis", "ok", "fitness")


def _metrics(data, theta):
    result = evaluate(data, theta)
    return tuple(result[k] for k in METRIC_KEYS)


class BudgetExhausted(Exception):
    """Raised when a FitnessTracker is called after max_evals evaluations."""


_worker_data = None

# Processes for the perturbed runs of ekf.robustness (not part of main.py's results).
RESCORE_WORKERS = 1


def _init_worker(data, options=None, objective=None):
    """Pool initializer: the dataset, and the parent's filter OPTIONS and OBJECTIVE (spawned workers start fresh)."""
    global _worker_data
    _worker_data = data
    if options:
        OPTIONS.update(options)
    if objective:
        OBJECTIVE.update(objective)


def _worker_perturbed_rmse(args):
    kwargs, theta, bias0 = args
    return evaluate(perturb(_worker_data, **kwargs), theta, bias0=bias0)["rmse_h"]


def _worker_fitness(theta):
    return fitness(_worker_data, theta)


def _worker_metrics(theta):
    return _metrics(_worker_data, theta)


def cpu_metrics_many(data, thetas, n_workers=1):
    """Reference (filterpy) metrics for many candidates: dict of arrays, METRIC_KEYS."""
    thetas = list(np.atleast_2d(np.asarray(thetas, dtype=float)))
    if n_workers > 1:
        with Pool(n_workers, initializer=_init_worker, initargs=(data, dict(OPTIONS), dict(OBJECTIVE))) as pool:
            rows = pool.map(_worker_metrics, thetas)
    else:
        rows = [_metrics(data, t) for t in thetas]
    cols = list(zip(*rows))
    return {k: np.array(c, dtype=bool if k == "ok" else float) for k, c in zip(METRIC_KEYS, cols)}


BACKENDS = ("cpu", "gpu")


class FitnessTracker:
    """The one cost function every optimizer calls, with budget, history and timing.

    tracker(theta) scores one candidate; tracker.evaluate_many(pop) scores a population
    and records results in population order. One candidate is one complete full-flight
    EKF run and counts as one evaluation. Once n_evals reaches max_evals,
    tracker(theta) raises BudgetExhausted and evaluate_many scores only the candidates
    that still fit in the budget.

    backend="cpu" runs the filterpy reference, in parallel processes when n_workers > 1.
    backend="gpu" runs the whole population as one batched EKF on the GPU (ekf.gpu);
    the objective is the same, only the execution differs.

    Each call (one evaluate_many or one single evaluation) is one generation and is
    numbered in the history. t_objective is the wall-clock time spent inside the
    objective; t_total also includes the optimizer's own work between calls.
    """

    def __init__(self, data, max_evals=None, n_workers=1, lower=LOWER, upper=UPPER, backend="cpu"):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        self.data = data
        self.max_evals = max_evals
        self.lower, self.upper = np.asarray(lower, float), np.asarray(upper, float)
        self.n_workers = n_workers
        self.backend = backend
        self._pool = None
        self._gpu = None
        if backend == "gpu":
            from .gpu import gpu_evaluator
            self._gpu = gpu_evaluator(data)
        self.history = []                   # one dict per evaluation
        self.best_theta, self.best_cost = None, np.inf
        self.evals_to_best = 0
        self.n_generations = 0              # objective calls (one per population)
        self.n_objective_evals = 0          # candidates actually scored by the backend
        self.t_objective = 0.0              # s spent inside the objective
        self._t_start = None
        self._t_last = None

    # -- budget ---------------------------------------------------------------
    @property
    def n_evals(self):
        return len(self.history)

    @property
    def remaining(self):
        return np.inf if self.max_evals is None else self.max_evals - self.n_evals

    @property
    def exhausted(self):
        return self.remaining <= 0

    # -- evaluation -----------------------------------------------------------
    def _record(self, theta, cost):
        now = time.perf_counter()
        if self._t_start is None:
            self._t_start = now
        self._t_last = now
        if cost < self.best_cost:
            self.best_cost, self.best_theta = float(cost), np.array(theta, dtype=float)
            self.evals_to_best = self.n_evals + 1
        self.history.append({"k": self.n_evals + 1, "generation": self.n_generations,
                             "fitness": float(cost),
                             "best_so_far": self.best_cost, "elapsed": now - self._t_start,
                             **dict(zip(THETA_NAMES, np.asarray(theta, dtype=float)))})

    def _objective(self, thetas):
        """Score a list of clipped candidates with the configured backend."""
        t0 = time.perf_counter()
        if self.backend == "gpu":
            costs = list(self._gpu.fitness(np.asarray(thetas)))
        elif self.n_workers > 1 and len(thetas) > 1:
            if self._pool is None:
                self._pool = Pool(self.n_workers, initializer=_init_worker, initargs=(self.data, dict(OPTIONS), dict(OBJECTIVE)))
            costs = self._pool.map(_worker_fitness, thetas)
        else:
            costs = [fitness(self.data, t) for t in thetas]
        self.t_objective += time.perf_counter() - t0
        self.n_objective_evals += len(thetas)
        self.n_generations += 1
        return costs

    def __call__(self, theta):
        if self.exhausted:
            raise BudgetExhausted(f"max_evals = {self.max_evals} reached")
        if self._t_start is None:
            self._t_start = time.perf_counter()
        theta = clip_theta(theta, self.lower, self.upper)
        cost = self._objective([theta])[0]
        self._record(theta, cost)
        return cost

    def evaluate_many(self, thetas):
        """Score a population. Returns costs for the candidates scored, in order."""
        thetas = [clip_theta(t, self.lower, self.upper) for t in np.atleast_2d(thetas)]
        n = int(min(len(thetas), self.remaining))
        thetas = thetas[:n]
        if n == 0:
            return np.array([])
        if self._t_start is None:
            self._t_start = time.perf_counter()
        costs = self._objective(thetas)
        for theta, cost in zip(thetas, costs):
            self._record(theta, cost)
        return np.array(costs)

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- results --------------------------------------------------------------
    @property
    def t_total(self):
        """Wall-clock seconds from the first evaluation starting to the last one ending."""
        return 0.0 if self._t_start is None else self._t_last - self._t_start

    @property
    def t_eval(self):
        return self.t_total / self.n_evals if self.n_evals else np.nan

    def convergence(self):
        """Best-so-far curve J*(k) as a DataFrame, one row per evaluation."""
        return pd.DataFrame(self.history)


def make_cost_fn(data, max_evals=None, n_workers=1, lower=LOWER, upper=UPPER, backend="cpu"):
    """The canonical cost function for optimizers: a FitnessTracker on the clean flight."""
    return FitnessTracker(data, max_evals=max_evals, n_workers=n_workers, lower=lower, upper=upper,
                          backend=backend)


def evals_to_converge(best_so_far, fraction=CONVERGENCE_FRACTION):
    """First evaluation k at which the run has made `fraction` of its total improvement.

    Improvement is measured from J*(1) to the final J*(N). A run that never improves
    on its first evaluation converges at k = 1.
    """
    best = np.asarray(best_so_far, dtype=float)
    if best.size == 0:
        return 0
    target = best[-1] + (1.0 - fraction) * (best[0] - best[-1])
    return int(np.argmax(best <= target + 1e-12)) + 1


# ---------------------------------------------------------------------------
# Robustness: whole-flight RMSE degradation under predefined perturbations
# ---------------------------------------------------------------------------
def robustness(data, theta, rmse_clean=None, perturbations=PERTURBATIONS, bias0=None, n_workers=None):
    """Degradation of a fixed theta under each perturbation of the flight.

    theta (and bias0, the optional initial bias states) is never re-tuned here.
    For each perturbation:
        delta     = RMSE_h,perturbed - RMSE_h,clean          (m, lower is better)
        delta_pct = 100 * delta / RMSE_h,clean
    Returns {"rmse_clean", "per_perturbation": {name: {rmse, delta, delta_pct}},
             "delta_mean", "delta_max"}.
    The perturbed runs are independent; with n_workers > 1 (default RESCORE_WORKERS) they run
    in parallel CPU processes, with identical results.
    """
    if rmse_clean is None:
        rmse_clean = evaluate(data, theta, bias0=bias0)["rmse_h"]
    n_workers = RESCORE_WORKERS if n_workers is None else n_workers
    tasks = [(kwargs, theta, bias0) for kwargs in perturbations.values()]
    if n_workers > 1:
        with Pool(min(n_workers, len(tasks)), initializer=_init_worker, initargs=(data, dict(OPTIONS), dict(OBJECTIVE))) as pool:
            values = pool.map(_worker_perturbed_rmse, tasks)
    else:
        values = [evaluate(perturb(data, **kw), t, bias0=b)["rmse_h"] for kw, t, b in tasks]
    per = {}
    for name, value in zip(perturbations, values):
        delta = value - rmse_clean
        per[name] = {"rmse": value, "delta": delta, "delta_pct": 100.0 * delta / rmse_clean}
    deltas = np.array([p["delta"] for p in per.values()])
    return {"rmse_clean": rmse_clean, "per_perturbation": per,
            "delta_mean": float(deltas.mean()), "delta_max": float(deltas.max())}


# ---------------------------------------------------------------------------
# One row of results per run, same schema for the baseline and every optimizer
# ---------------------------------------------------------------------------
def run_summary(data, tracker, optimizer, seed, rmse_gps=None, rmse_baseline=None,
                population_size=None):
    """Score the tracker's best theta: the objective (fitness), accuracy (rmse_h), consistency
    (mean_nis, nis_consistency), convergence
    (evals_to_converge, evals_to_best) and compute (n_evals, t_total, t_eval, t_objective).
    Also reports the comparison with the baseline EKF and with raw GPS separately.
    Robustness is not part of the row; call ekf.robustness for it.

    The final theta is always re-scored with the filterpy reference on the CPU, so
    rmse_h comes from the reference filter whichever backend
    drove the search. rmse_baseline is the baseline's RMSE_h (not its J). best_fitness_search is the J the search saw;
    search_vs_final_diff is their difference (rounding only, for the GPU backend).
    """
    theta = tracker.best_theta
    result = evaluate(data, theta)
    rmse_gps = gps_rmse(data) if rmse_gps is None else rmse_gps
    row = {"optimizer": optimizer, "seed": seed,
           **dict(zip(THETA_NAMES, theta)),
           "fitness": result["fitness"], "nis_weight": OBJECTIVE["nis_weight"],
           "rmse_h": result["rmse_h"], "rmse_v": result["rmse_v"], "rmse_3d": result["rmse_3d"],
           "mean_nis": result["mean_nis"], "nis_dof": nis_dof(data),
           "nis_consistency": float(nis_consistency(result["mean_nis"], nis_dof(data))),
           "rmse_gps": rmse_gps, "delta_vs_gps": result["rmse_h"] - rmse_gps,
           # which filter behaviour produced the row (ekf.filter.OPTIONS)
           "gate_enabled": bool(OPTIONS["gate"]), "init_heading": bool(OPTIONS["init_heading"]),
           "plausibility": bool(OPTIONS["plausibility"]),
           "n_gate_rejected": result["n_rejected"]}
    if rmse_baseline is not None:
        row["delta_vs_baseline"] = result["rmse_h"] - rmse_baseline
        row["improvement_vs_baseline_pct"] = 100.0 * (rmse_baseline - result["rmse_h"]) / rmse_baseline
    row.update({f"theta0_{n}": v for n, v in zip(THETA_NAMES, THETA0)})   # the dataset's starting point
    best = tracker.convergence()["best_so_far"].to_numpy()
    row.update({"evals_to_converge": evals_to_converge(best), "evals_to_best": tracker.evals_to_best,
                "n_evals": tracker.n_evals, "max_evals": tracker.max_evals,
                "t_total": tracker.t_total, "t_eval": tracker.t_eval,
                "backend": tracker.backend, "population_size": population_size,
                "n_generations": tracker.n_generations,
                "n_objective_evals": tracker.n_objective_evals,
                "t_objective": tracker.t_objective,
                "t_objective_per_eval": (tracker.t_objective / tracker.n_objective_evals
                                         if tracker.n_objective_evals else np.nan),
                "best_fitness_search": tracker.best_cost,
                "search_vs_final_diff": result["fitness"] - tracker.best_cost})
    return row, result


def save_run(out_dir, row, tracker):
    """Append one run to <out_dir>/runs.csv and write its convergence curve."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracker.convergence().to_csv(out_dir / f"convergence_seed{row['seed']}.csv", index=False)
    runs_path = out_dir / "runs.csv"
    runs = pd.DataFrame([row])
    if runs_path.exists():
        old = pd.read_csv(runs_path)
        old = old[old["seed"] != row["seed"]]
        runs = pd.concat([old, runs], ignore_index=True)
    runs.to_csv(runs_path, index=False)
