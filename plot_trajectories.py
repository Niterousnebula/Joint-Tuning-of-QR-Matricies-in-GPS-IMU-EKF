"""Trajectory map: ground truth, raw GNSS and the EKF estimate, in local ENU metres.

    python plot_trajectories.py --dataset inavwhu/street00
    python plot_trajectories.py --dataset eth --runs results/PSO_gpu/runs.csv     # adds the best tuned theta
    python plot_trajectories.py --dataset eth --theta 0.18 -6 -7.59 -10 -0.41 2.74 2 --label "PSO seed 5"
    python plot_trajectories.py --dataset inavwhu/street00 --window 0 120         # zoom on a time window

Left: the map (east against north). Right: horizontal error against ground truth over time.
Ground truth is drawn only where the dataset has it; without it the right panel shows the
distance between the EKF estimate and GNSS instead. Writes results/[<dataset>/]trajectories.png.
Datasets the pipeline marked not ekf_ready are drawn as well (the EKF still runs on them).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ekf

COLORS = {"gt": "#222222", "gnss": "#0072B2", "ekf": ["#D55E00", "#009E73", "#CC79A7", "#E69F00"]}


def best_theta(runs_csv):
    runs = pd.read_csv(runs_csv)
    row = runs.loc[runs["rmse_h"].idxmin()]
    return row[ekf.THETA_NAMES].to_numpy(dtype=float), f"{row['optimizer']} seed {int(row['seed'])}"


def estimate(data, theta):
    """EKF run: IMU times, ENU positions, and the run's own result dict."""
    result = ekf.evaluate(data, theta)
    x = result["x_log"]
    return data["t_imu"][:len(x)], x[:, :3], result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", help="processed dataset id; asked for on a terminal if omitted")
    parser.add_argument("--theta", type=float, nargs=7, metavar="V", help="an extra theta to draw (7 log10 values)")
    parser.add_argument("--label", default="tuned", help="legend label for --theta")
    parser.add_argument("--runs", help="a runs.csv from an optimizer: draws its best theta")
    parser.add_argument("--window", type=float, nargs=2, metavar=("T0", "T1"), help="only draw this time window (s)")
    parser.add_argument("--out", help="output png (default: results/[<dataset>/]trajectories.png)")
    a = parser.parse_args()

    dataset = ekf.select_dataset(a.dataset)
    try:
        data = ekf.load_data(dataset, allow_unready=True, require_gt=False)
    except (ekf.DatasetNotReady, FileNotFoundError) as err:
        raise SystemExit(f"error: {err}")
    has_gt = len(data["t_true"]) > 1

    thetas = [("EKF, THETA0 (baseline)", ekf.THETA0)]
    if a.runs:
        theta, label = best_theta(a.runs)
        thetas.append((f"EKF, {label}", theta))
    if a.theta:
        thetas.append((f"EKF, {a.label}", np.array(a.theta)))

    runs = []
    for label, theta in thetas:
        print(f"running the EKF for: {label}", flush=True)
        runs.append((label, *estimate(data, theta)))

    lo, hi = a.window if a.window else (-np.inf, np.inf)
    inside = lambda t: (t >= lo) & (t <= hi)
    t_g, p_g, t_t, p_t = data["t_gps"], data["p_gps"], data["t_true"], data["p_true"]

    fig, (ax_map, ax_err) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.15, 1]})
    if has_gt:
        m = inside(t_t)
        ax_map.plot(p_t[m, 0], p_t[m, 1], color=COLORS["gt"], lw=1.6, label="ground truth", zorder=3)
    m = inside(t_g)
    ax_map.plot(p_g[m, 0], p_g[m, 1], ".", color=COLORS["gnss"], ms=3.5, alpha=0.7, label="GNSS fixes", zorder=2)
    for i, (label, t_e, p_e, _) in enumerate(runs):
        m = inside(t_e)
        ax_map.plot(p_e[m, 0], p_e[m, 1], color=COLORS["ekf"][i], lw=1.2, label=label, zorder=4 + i)
    first = inside(t_g)
    if first.any():
        ax_map.plot(*p_g[first][0, :2], marker="o", mfc="white", mec="black", ms=9, ls="", label="start", zorder=9)
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.set_xlabel("East (m)")
    ax_map.set_ylabel("North (m)")
    ax_map.grid(alpha=0.3)
    ax_map.legend(loc="best", fontsize=8)
    ax_map.set_title("Trajectories in local ENU" + (f" (t = {lo:.0f}-{hi:.0f} s)" if a.window else ""))

    if has_gt:
        e_g = ekf.horizontal_errors(t_g, p_g, t_t, p_t)
        mm = inside(t_t)
        ax_err.plot(t_t[mm], e_g[mm], color=COLORS["gnss"], lw=1.2,
                    label=f"GNSS (RMSE {ekf.rmse(e_g):.3f} m)")
        for i, (label, t_e, p_e, res) in enumerate(runs):
            e = ekf.horizontal_errors(t_e, p_e, t_t, p_t)
            ax_err.plot(t_t[mm], e[mm], color=COLORS["ekf"][i], lw=1.2, label=f"{label} (RMSE {res['rmse_h']:.3f} m)")
        ax_err.set_ylabel("Horizontal error vs ground truth (m)")
    else:
        for i, (label, t_e, p_e, _) in enumerate(runs):
            d = np.column_stack([np.interp(t_g, t_e, p_e[:, j]) - p_g[:, j] for j in range(2)])
            mm = inside(t_g)
            ax_err.plot(t_g[mm], np.hypot(d[mm, 0], d[mm, 1]), color=COLORS["ekf"][i], lw=1.2, label=label)
        ax_err.set_ylabel("Distance between EKF estimate and GNSS (m)")
        ax_err.text(0.02, 0.97, "no ground truth for this dataset", transform=ax_err.transAxes, va="top", fontsize=9)
    ax_err.set_xlabel("Time since the first IMU sample (s)")
    ax_err.grid(alpha=0.3)
    ax_err.legend(loc="best", fontsize=8)
    ax_err.set_title("Error over time")

    fig.suptitle(f"{dataset}: {len(data['t_imu'])} IMU samples, {len(t_g)} GNSS fixes, "
                 f"GPS channels used: {ekf.measurement_channels(data).tolist()}", fontsize=11)
    fig.tight_layout()
    out = Path(a.out) if a.out else ekf.results_dir(dataset, "trajectories.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
