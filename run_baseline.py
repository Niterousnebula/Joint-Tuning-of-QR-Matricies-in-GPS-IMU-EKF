"""Evaluate THETA0 with the canonical fitness and write the baseline files.

    python run_baseline.py                       # asks which processed dataset to use
    python run_baseline.py --dataset inavwhu/street00
    python run_baseline.py --dataset eth         # writes results/baseline/ (other datasets: results/<id>/baseline/)

The baseline goes through the same FitnessTracker as every optimizer (one evaluation),
so its row in runs.csv has exactly the same columns as an optimizer run. Datasets come from
processed_data/ (see `python -m data_pipeline`).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import ekf


def error_summary(err_h, motion, source):
    """Secondary diagnostics: horizontal error over the whole flight and per motion label."""
    cases = {"whole": np.ones(len(err_h), dtype=bool),
             "motion_hover": motion == "hover",
             "motion_cruise": motion == "cruise",
             "motion_turn": motion == "turn"}
    rows = []
    for case, mask in cases.items():
        if not mask.any():              # a dataset without motion labels has no hover/cruise/turn cases
            continue
        e = err_h[mask]
        rows.append({"case": case, "source": source, "rmse_h": ekf.rmse(e),
                     "mean_err_h": e.mean(), "median_err_h": np.median(e),
                     "p95_err_h": np.percentile(e, 95), "max_err_h": e.max(), "n_samples": int(mask.sum())})
    return rows


def main(dataset, out_dir=None, allow_unready=False, init_heading=None, gate=None):
    if init_heading is not None:
        ekf.filter.OPTIONS["init_heading"] = init_heading
    if gate is not None:
        ekf.filter.OPTIONS["gate"] = gate
    dataset = ekf.select_dataset(dataset)
    out_dir = Path(out_dir) if out_dir else ekf.results_dir(dataset, "baseline")
    try:
        data = ekf.load_data(dataset, allow_unready=allow_unready)
    except (ekf.DatasetNotReady, FileNotFoundError) as err:
        raise SystemExit(f"error: {err}")
    out_dir.mkdir(parents=True, exist_ok=True)       # only once the dataset is known to load
    print(f"Dataset {dataset}: {len(data['t_imu'])} IMU samples, {len(data['t_gps'])} GPS fixes, "
          f"GPS channels {ekf.measurement_channels(data).tolist()} (e n u ve vn vu = 0..5)")
    t_true, p_true, motion = data["t_true"], data["p_true"], data["motion"]

    # Raw GPS reference, reported separately from the EKF
    gps_err = ekf.position_errors(data["t_gps"], data["p_gps"], t_true, p_true)
    gps_err_h = np.hypot(gps_err[:, 0], gps_err[:, 1])
    rmse_gps = ekf.rmse(gps_err_h)
    pd.DataFrame({"t": t_true,
                  "gps_e": p_true[:, 0] + gps_err[:, 0], "gps_n": p_true[:, 1] + gps_err[:, 1],
                  "gps_u": p_true[:, 2] + gps_err[:, 2],
                  "gt_e": p_true[:, 0], "gt_n": p_true[:, 1], "gt_u": p_true[:, 2],
                  "err_e": gps_err[:, 0], "err_n": gps_err[:, 1], "err_u": gps_err[:, 2],
                  "err_h": gps_err_h}).to_csv(out_dir / "gps_vs_ground_truth.csv", index=False)

    # Baseline through the canonical fitness: one full-flight evaluation of THETA0
    tracker = ekf.make_cost_fn(data, max_evals=1)
    rmse_baseline = tracker(ekf.THETA0)
    row, result = ekf.run_summary(data, tracker, "baseline", 0, rmse_gps=rmse_gps, rmse_baseline=rmse_baseline)
    ekf.save_run(out_dir, row, tracker)

    x_log = result["x_log"]
    t_imu = data["t_imu"][:len(x_log)]
    traj = pd.DataFrame(x_log, columns=ekf.STATE_COLUMNS)
    traj.insert(0, "t", t_imu)
    traj.to_csv(out_dir / "baseline_trajectory.csv", index=False)
    pd.DataFrame(result["innovation_log"], columns=ekf.INNOVATION_COLUMNS).to_csv(
        out_dir / "baseline_innovations.csv", index=False)

    base_err_h = ekf.horizontal_errors(t_imu, x_log, t_true, p_true)
    pd.DataFrame({"t": t_true, "motion": motion, "gps_err_h": gps_err_h,
                  "baseline_err_h": base_err_h}).to_csv(out_dir / "gt_error_timeseries.csv", index=False)
    pd.DataFrame(error_summary(gps_err_h, motion, "gps") + error_summary(base_err_h, motion, "baseline")
                 ).to_csv(out_dir / "ekf_error_comparison.csv", index=False)

    print(f"Raw GPS             RMSE_h whole flight {rmse_gps:.3f} m (reference, no filter)")
    print(f"Baseline EKF THETA0 RMSE_h whole flight {row['rmse_h']:.3f} m   "
          f"(vs GPS {row['delta_vs_gps']:+.3f} m)")
    print(f"  secondary: RMSE_v {row['rmse_v']:.3f} m, RMSE_3D {row['rmse_3d']:.3f} m, "
          f"mean NIS {row['mean_nis']:.2f} (expected {ekf.nis_dof(data)} for m = {ekf.nis_dof(data)})")
    print(f"Robustness, Delta RMSE_h = perturbed - clean (whole flight): "
          f"mean {row['delta_rmse_mean']:+.3f} m, max {row['delta_rmse_max']:+.3f} m")
    for name in ekf.PERTURBATIONS:
        print(f"  {name:<14} {row[f'rmse_{name}']:.3f} m   {row[f'delta_{name}']:+.3f} m "
              f"({row[f'delta_pct_{name}']:+.2f} %)")
    print(f"Compute: N_eval {row['n_evals']}, T_total {row['t_total']:.1f} s, "
          f"T_eval {row['t_eval']:.1f} s (wall clock, hardware-dependent)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", help="processed dataset id; asked for on a terminal if omitted")
    parser.add_argument("--out", help="output folder (default: results/[<dataset>/]baseline)")
    parser.add_argument("--force", action="store_true",
                        help="run even if the pipeline marked the dataset not ekf_ready")
    parser.add_argument("--no-init-heading", action="store_true",
                        help="switch off the GNSS-course heading initialisation (the filter before that change)")
    parser.add_argument("--gate", action="store_true", help="switch on the chi-square GPS gate (off by default)")
    a = parser.parse_args()
    main(a.dataset, a.out, a.force, False if a.no_init_heading else None, True if a.gate else None)
