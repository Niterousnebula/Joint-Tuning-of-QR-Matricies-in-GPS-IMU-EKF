"""Filter consistency (mean NIS) of every best-so-far candidate.

A run's best-so-far theta only changes when the search finds a better candidate (its incumbents),
so scoring each incumbent's mean NIS gives the NIS of the best-so-far theta at every evaluation k:

    accuracy     A(k) = best-so-far whole-flight RMSE_h after k evaluations (from the run)
    consistency  N(k) = mean NIS of the incumbent at k                     (this module)
    convergence  k    = evaluations used
    compute      T(k) = wall-clock seconds at k                            (from the run)

The incumbents of all seeds are scored on the GPU in one batched pass (ekf.gpu).
"""
import numpy as np
import pandas as pd

from .tuning import THETA_NAMES


def incumbents(conv):
    """The rows of a convergence curve where the best-so-far improved (first row included)."""
    best = conv["best_so_far"].to_numpy()
    improved = np.r_[True, best[1:] < best[:-1]]
    return conv[improved].reset_index(drop=True)


def incumbent_nis(data, curves, batch=200, device="cuda"):
    """Mean NIS of every incumbent of every seed. curves: {seed: convergence DataFrame}.

    Returns {seed: DataFrame} with k, elapsed, best_so_far, the theta and mean_nis per incumbent.
    """
    from .gpu import gpu_evaluator

    tables = {s: incumbents(c) for s, c in curves.items()}
    if not tables:
        return {}
    stacked = pd.concat([t.assign(seed=s) for s, t in tables.items()], ignore_index=True)
    thetas = stacked[THETA_NAMES].to_numpy(dtype=float)
    ev = gpu_evaluator(data, device=device)                 # the search's own evaluator, already built
    nis = np.concatenate([ev.evaluate(thetas[i:i + batch])["mean_nis"] for i in range(0, len(thetas), batch)])
    stacked["mean_nis"] = nis
    keep = ["k", "elapsed", "best_so_far", *THETA_NAMES, "mean_nis"]
    return {s: stacked.loc[stacked["seed"] == s, keep].reset_index(drop=True) for s in tables}


def response_path(conv, inc):
    """Accuracy, consistency and compute at every evaluation k of one seed."""
    k = conv["k"].to_numpy()
    pos = np.searchsorted(inc["k"].to_numpy(), k, side="right") - 1      # incumbent in force at k
    return pd.DataFrame({"k": k, "accuracy": conv["best_so_far"].to_numpy(),
                         "nis": inc["mean_nis"].to_numpy()[pos], "elapsed": conv["elapsed"].to_numpy()})
