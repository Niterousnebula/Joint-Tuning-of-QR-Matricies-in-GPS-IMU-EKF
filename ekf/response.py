"""Data for the 4D response surfaces: robustness of every best-so-far candidate of a run.

Robustness (mean ΔRMSE_h over PERTURBATIONS) needs five extra full-flight runs per theta, so it
is not computed for every evaluation. A run's best-so-far theta only changes when the search
finds a better candidate (its incumbents), so scoring each incumbent under every perturbation
gives the robustness of the best-so-far theta at every evaluation k:

    accuracy     A(k) = best-so-far whole-flight RMSE_h after k evaluations (from the run)
    robustness   R(k) = mean ΔRMSE_h of the incumbent at k              (this module)
    convergence  k    = evaluations used
    compute      T(k) = wall-clock seconds at k                         (from the run)

The incumbents are scored on the GPU with the same batched EKF as the search (ekf.gpu), one
evaluator per perturbation, all seeds of one optimizer in the same batches.
"""
import numpy as np
import pandas as pd

from .data import PERTURBATIONS, perturb
from .tuning import THETA_NAMES


def incumbents(conv):
    """The rows of a convergence curve where the best-so-far improved (first row included)."""
    best = conv["best_so_far"].to_numpy()
    improved = np.r_[True, best[1:] < best[:-1]]
    return conv[improved].reset_index(drop=True)


def incumbent_robustness(data, curves, batch=200, device="cuda"):
    """Robustness of every incumbent of every seed, on the GPU.

    curves: {seed: convergence DataFrame (ekf.save_run's convergence_seed<s>.csv)}.
    Returns {seed: DataFrame} with k, elapsed, best_so_far (the clean RMSE_h), the theta,
    rmse_<perturbation>, delta_<perturbation>, delta_mean and delta_max for each incumbent.
    """
    import torch

    from .gpu import GpuEvaluator

    tables = {s: incumbents(c) for s, c in curves.items()}
    if not tables:
        return {}
    seeds = list(tables)
    stacked = pd.concat([tables[s].assign(seed=s) for s in seeds], ignore_index=True)
    thetas = stacked[THETA_NAMES].to_numpy(dtype=float)
    n = len(thetas)
    for name, kwargs in PERTURBATIONS.items():
        ev = GpuEvaluator(perturb(data, **kwargs), device=device)
        values = np.empty(n)
        for i in range(0, n, batch):
            idx = np.arange(i, min(i + batch, n))
            size = batch if n > batch else n                     # fixed batch size: one graph capture
            pad = np.r_[idx, np.full(size - len(idx), idx[-1])]
            values[idx] = ev.fitness(thetas[pad])[:len(idx)]
        del ev
        torch.cuda.empty_cache()
        stacked[f"rmse_{name}"] = values
        stacked[f"delta_{name}"] = values - stacked["best_so_far"].to_numpy()
    deltas = stacked[[f"delta_{name}" for name in PERTURBATIONS]]
    stacked["delta_mean"] = deltas.mean(axis=1)
    stacked["delta_max"] = deltas.max(axis=1)
    keep = ["k", "elapsed", "best_so_far", *THETA_NAMES, *[c for c in stacked.columns if c.startswith(("rmse_", "delta_"))]]
    return {s: stacked.loc[stacked["seed"] == s, keep].reset_index(drop=True) for s in seeds}


def response_path(conv, inc):
    """Accuracy, robustness and compute at every evaluation k of one seed.

    conv: convergence curve (k, best_so_far, elapsed); inc: its incumbent_robustness table.
    Returns a DataFrame with k, accuracy, robustness, elapsed.
    """
    k = conv["k"].to_numpy()
    pos = np.searchsorted(inc["k"].to_numpy(), k, side="right") - 1      # incumbent in force at k
    return pd.DataFrame({"k": k, "accuracy": conv["best_so_far"].to_numpy(),
                         "robustness": inc["delta_mean"].to_numpy()[pos],
                         "elapsed": conv["elapsed"].to_numpy()})
