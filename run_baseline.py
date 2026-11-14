"""Run the EKF with THETA0 and write the baseline files that every optimizer is compared against.

    python run_baseline.py            # writes results/baseline/
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import ekf


def error_summary(err, t, motion, source):
    """One row per scoring case for one error series."""
    cases = {
        "whole": np.ones_like(t, dtype=bool),
        "tune": t < ekf.T_SPLIT,
        "held_out": t >= ekf.T_SPLIT,
        "motion_hover": motion == "hover",
        "motion_cruise": motion == "cruise",
        "motion_turn": motion == "turn",
    }
    rows = []
    for case, mask in cases.items():
        e = err[mask]
        rows.append({"case": case, "source": source, "rmse_h": ekf.rmse(e),
                     "mean_err_h": e.mean(), "median_err_h": np.median(e),
                     "p95_err_h": np.percentile(e, 95), "max_err_h": e.max(), "n_samples": int(mask.sum())})
    return rows


def main(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = ekf.load_data()
    t_true, p_true, motion = data["t_true"], data["p_true"], data["motion"]

    # Raw GPS against ground truth
    gps_at_gt = np.column_stack([np.interp(t_true, data["t_gps"], data["p_gps"][:, i]) for i in range(3)])
    gps_err = gps_at_gt - p_true
    gps_err_h = np.hypot(gps_err[:, 0], gps_err[:, 1])
    pd.DataFrame({"t": t_true,
                  "gps_e": gps_at_gt[:, 0], "gps_n": gps_at_gt[:, 1], "gps_u": gps_at_gt[:, 2],
                  "gt_e": p_true[:, 0], "gt_n": p_true[:, 1], "gt_u": p_true[:, 2],
                  "err_e": gps_err[:, 0], "err_n": gps_err[:, 1], "err_u": gps_err[:, 2],
                  "err_h": gps_err_h}).to_csv(out_dir / "gps_vs_ground_truth.csv", index=False)

    # Baseline filter
    Q_c, R = ekf.theta_to_QR(ekf.THETA0)
    start = time.perf_counter()
    x_log, innovation_log, mean_nis, ok = ekf.run_ekf(data, Q_c, R)
    run_time = time.perf_counter() - start
    if not ok:
        raise RuntimeError("baseline filter diverged")
    t_imu = data["t_imu"][:len(x_log)]
    traj = pd.DataFrame(x_log, columns=ekf.STATE_COLUMNS)
    traj.insert(0, "t", t_imu)
    traj.to_csv(out_dir / "baseline_trajectory.csv", index=False)
    pd.DataFrame(innovation_log, columns=ekf.INNOVATION_COLUMNS).to_csv(out_dir / "baseline_innovations.csv", index=False)

    base_err_h = ekf.horizontal_errors(t_imu, x_log, t_true, p_true)
    pd.DataFrame({"t": t_true, "part": np.where(t_true < ekf.T_SPLIT, "tune", "held_out"), "motion": motion,
                  "gps_err_h": gps_err_h, "baseline_err_h": base_err_h}).to_csv(out_dir / "gt_error_timeseries.csv", index=False)

    rows = error_summary(gps_err_h, t_true, motion, "gps") + error_summary(base_err_h, t_true, motion, "baseline")
    rob = ekf.robustness(data, ekf.THETA0)
    for name, value in rob["per_dataset"].items():
        rows.append({"case": f"robust_{name}", "source": "baseline", "rmse_h": value})
    pd.DataFrame(rows).to_csv(out_dir / "ekf_error_comparison.csv", index=False)

    tune_cost = ekf.make_cost_fn(data)(ekf.THETA0)
    summary = dict(zip(ekf.THETA_NAMES, ekf.THETA0))
    summary.update({"optimizer": "baseline", "seed": -1, "best_cost": tune_cost, "mean_nis": mean_nis,
                    "rmse_h_held_out": rob["per_dataset"]["clean"],
                    "rmse_h_whole": ekf.rmse(base_err_h),
                    "robustness_mean": rob["mean"], "robustness_std": rob["std"],
                    "evals_to_best": 0, "n_evals": 1, "wall_time": run_time})
    pd.DataFrame([summary]).to_csv(out_dir / "baseline_parameters.csv", index=False)

    print(f"GPS      rmse_h whole {ekf.rmse(gps_err_h):.3f} m")
    print(f"Baseline rmse_h whole {ekf.rmse(base_err_h):.3f} m, tune {tune_cost:.3f} m, "
          f"held-out {rob['per_dataset']['clean']:.3f} m, mean NIS {mean_nis:.2f}")
    print(f"Robustness (std of held-out rmse_h over {len(rob['per_dataset'])} datasets): {rob['std']:.3f} m")
    for name, value in rob["per_dataset"].items():
        print(f"  {name:<14} {value:.3f} m")
    print(f"One full-flight run: {run_time:.1f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="results/baseline")
    main(parser.parse_args().out)
