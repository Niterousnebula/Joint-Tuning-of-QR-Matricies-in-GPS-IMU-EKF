"""Loading a processed dataset and building perturbed copies of it.

The EKF never reads raw files. It reads processed_data/<id>/ (written by `python -m
data_pipeline`), where every dataset has the same format: IMU in SI units, GPS and ground
truth in ENU, time rebased to the first IMU sample. A channel a source does not have is NaN
(for example GPS velocity), and the filter works with whichever channels are present
(`ekf.measurement_channels`). So a new dataset needs an adapter in data_pipeline/ and no
change in this package.

load_data returns one dictionary of numpy arrays: t_imu, accel, gyro, t_gps, p_gps, v_gps,
t_true, p_true, motion, plus the dataset id.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "processed_data"
DEFAULT_DATASET = "eth"


class DatasetNotReady(ValueError):
    """The dataset lacks something a GPS-aided EKF tuned against ground truth needs."""


def _natural(text):
    import re
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", text)]


def available_datasets(processed_dir=PROCESSED_DIR):
    """The processed datasets, from their meta.json: id, label, ready, reasons, channels, duration."""
    rows = []
    for path in Path(processed_dir).rglob("meta.json"):
        meta = json.loads(path.read_text())
        ready = meta.get("ekf_ready", {})
        rows.append({"id": meta["id"], "label": meta.get("label", ""), "ready": bool(ready.get("ready")),
                     "reasons": ready.get("reasons", []), "channels": ready.get("gps_channels", []),
                     "duration_s": meta.get("imu_duration_s")})
    return sorted(rows, key=lambda r: _natural(r["id"]))


def _prepare(dataset, processed_dir):
    """Process one dataset from raw_data/ with the data pipeline, when it has not been processed."""
    import data_pipeline
    try:
        record = data_pipeline.load(dataset)
    except KeyError as err:
        raise FileNotFoundError(
            f"dataset {dataset!r} is not in {processed_dir} and not in raw_data/ either. "
            f"List what exists with `python -m data_pipeline --list`.") from err
    data_pipeline.write(record, processed_dir)


def load_data(dataset=DEFAULT_DATASET, processed_dir=PROCESSED_DIR, allow_unready=False, require_gt=True):
    """Read processed_data/<dataset>/ into the dictionary the filter uses.

    A dataset that is not processed yet is processed first. One the pipeline marked not
    ekf_ready (no GPS, no ground truth, too short, ...) raises DatasetNotReady, unless
    allow_unready is set, which still needs a GPS file to exist. A ground-truth file is also
    required unless require_gt is False (for plotting a run without scoring it): the ground
    truth arrays are then empty and nothing that scores against them can be used.
    """
    folder = Path(processed_dir) / dataset
    if not (folder / "meta.json").is_file():
        _prepare(dataset, processed_dir)
    meta = json.loads((folder / "meta.json").read_text())
    # A derived dataset (make_noisy_gps.py) holds only its own gps.csv; IMU and truth come from its source.
    source = Path(processed_dir) / meta["derived_from"] if "derived_from" in meta else folder
    ready = meta.get("ekf_ready", {})
    if not ready.get("ready") and not allow_unready:
        raise DatasetNotReady(f"dataset {dataset!r} is not ready for the EKF: " + "; ".join(ready.get("reasons", [])))
    for name in ("gps", "gt") if require_gt else ("gps",):
        if not ((folder if name == "gps" else source) / f"{name}.csv").is_file():
            raise DatasetNotReady(f"dataset {dataset!r} has no {name}.csv, so the EKF cannot run on it")

    imu = pd.read_csv(source / "imu.csv")
    gps = pd.read_csv(folder / "gps.csv")
    gt_path = source / "gt.csv"
    gt = pd.read_csv(gt_path) if gt_path.is_file() else pd.DataFrame(columns=["t", "e", "n", "u"], dtype=float)
    data = {
        "t_imu": imu["t"].to_numpy(),
        "accel": imu[["ax", "ay", "az"]].to_numpy(),
        "gyro": imu[["wx", "wy", "wz"]].to_numpy(),
        "t_gps": gps["t"].to_numpy(),
        "p_gps": gps[["e", "n", "u"]].to_numpy(),
        "v_gps": gps[["ve", "vn", "vu"]].to_numpy(),
        "t_true": gt["t"].to_numpy(),
        "p_true": gt[["e", "n", "u"]].to_numpy(),
        "motion": gt["motion"].to_numpy() if "motion" in gt.columns else np.full(len(gt), "unknown"),
        "dataset": dataset,
    }
    from .tuning import set_search_space          # here, not at the top: tuning imports this module
    data["theta0"] = dataset_theta0(data, gps, meta)
    set_search_space(data["theta0"])
    return data


STATIC_WINDOW_S = 10.0     # every ekf_ready dataset starts at rest (the pipeline checks it)


def dataset_theta0(data, gps, meta):
    """THETA0 for one dataset, from its own sensors, so its search is centred on its own noise.

    ETH keeps the hand-set THETA0 (built from the same statistics). Any other source: white-noise
    densities from the IMU's first STATIC_WINDOW_S seconds (variance x sample interval) and position
    variances from the receiver's reported sigmas (std_e/std_n/std_u) when the GPS file has them.
    The bias random walks and r_vel keep the ETH values. Noise added by make_noisy_gps.py
    (meta["gps_noise"]) is added to R, white and Gauss-Markov alike.
    """
    from .tuning import THETA0_ETH
    theta = THETA0_ETH.copy()
    if meta.get("derived_from", meta.get("id", data["dataset"])) != DEFAULT_DATASET:
        t = data["t_imu"]
        still = t < t[0] + STATIC_WINDOW_S
        dt = float(np.median(np.diff(t)))
        theta[0] = np.log10(data["accel"][still].var(axis=0).mean() * dt)
        theta[1] = np.log10(data["gyro"][still].var(axis=0).mean() * dt)
        if {"std_e", "std_n", "std_u"} <= set(gps.columns):
            theta[4] = np.log10(((gps["std_e"] ** 2 + gps["std_n"] ** 2) / 2).mean())
            theta[5] = np.log10((gps["std_u"] ** 2).mean())
    noise = meta.get("gps_noise", {})
    gm2 = noise.get("gm_sigma_m", 0.0) ** 2
    theta[4] = np.log10(10 ** theta[4] + noise.get("sigma_h_m", 0.0) ** 2 + gm2)
    theta[5] = np.log10(10 ** theta[5] + noise.get("sigma_u_m", 0.0) ** 2 + gm2)
    theta[6] = np.log10(10 ** theta[6] + noise.get("sigma_vel_ms", 0.0) ** 2)
    return theta


def select_dataset(dataset=None, processed_dir=PROCESSED_DIR):
    """The dataset id to run on: the one given, else asked for on a terminal, else the default.

    The menu lists the processed datasets the pipeline marked ekf_ready.
    """
    if dataset:
        return dataset
    if not sys.stdin.isatty():
        print(f"No --dataset given and no terminal to ask on: using '{DEFAULT_DATASET}'.", flush=True)
        return DEFAULT_DATASET
    rows = available_datasets(processed_dir)
    ready = [r for r in rows if r["ready"]]
    if not ready:
        print(f"No ready datasets in {processed_dir}; using '{DEFAULT_DATASET}' "
              "(processed from raw_data/ if needed).", flush=True)
        return DEFAULT_DATASET
    print("Processed datasets ready for the EKF:")
    for i, r in enumerate(ready, 1):
        print(f"  {i:3d}. {r['id']:<40} {r['duration_s']:7.0f} s  GPS channels: {','.join(r['channels'])}")
    skipped = len(rows) - len(ready)
    if skipped:
        print(f"  ({skipped} more processed datasets are not ready; see `python -m data_pipeline --list`)")
    while True:
        try:
            choice = input("Select a dataset (number or id): ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit(1)
        if choice.isdigit() and 1 <= int(choice) <= len(ready):
            return ready[int(choice) - 1]["id"]
        if choice in {r["id"] for r in ready}:
            return choice
        print("Not a listed dataset, try again.")


def results_dir(dataset, *parts):
    """Where a dataset's results go: results/<parts> for the default dataset (the layout the
    README and plot_response.py describe), results/<dataset>/<parts> for every other one,
    so runs on different datasets never mix."""
    base = Path("results")
    if dataset != DEFAULT_DATASET:
        base = base / dataset
    return base.joinpath(*parts)


def prepare_run(data=None, dataset=None, out_dir=None, name="RUN"):
    """The dataset and output folder for one optimizer run (shared by GA, PSO, DE and GWO).

    With `data` given (e.g. by main.py) it is used as is; otherwise `dataset` is loaded,
    asked for on a terminal if omitted. `out_dir` defaults to results_dir(dataset, name).
    Returns (data, dataset, out_dir).
    """
    if data is None:
        dataset = select_dataset(dataset)
        try:
            data = load_data(dataset)
        except (DatasetNotReady, FileNotFoundError) as err:
            raise SystemExit(f"error: {err}")
    else:
        dataset = data.get("dataset", DEFAULT_DATASET)
    if out_dir is None:
        out_dir = results_dir(dataset, name)
    return data, dataset, Path(out_dir)


def perturb(data, gps_sigma=0.0, vel_sigma=0.0, accel_sigma=0.0, gyro_sigma=0.0,
            gps_drop=0.0, seed=0):
    """Copy of `data` with extra white noise on the sensors and GPS fixes dropped at random.

    Ground truth is never changed. Each call starts from its own generator, so the same
    arguments always give the same copy. Noise on a channel the dataset lacks (NaN) stays NaN.
    """
    rng = np.random.default_rng(seed)
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    if gps_sigma:
        out["p_gps"] = out["p_gps"] + rng.normal(0.0, gps_sigma, out["p_gps"].shape)
    if vel_sigma:
        out["v_gps"] = out["v_gps"] + rng.normal(0.0, vel_sigma, out["v_gps"].shape)
    if accel_sigma:
        out["accel"] = out["accel"] + rng.normal(0.0, accel_sigma, out["accel"].shape)
    if gyro_sigma:
        out["gyro"] = out["gyro"] + rng.normal(0.0, gyro_sigma, out["gyro"].shape)
    if gps_drop:
        keep = rng.random(len(out["t_gps"])) >= gps_drop
        keep[0] = True                      # the first fix sets the initial state
        for key in ("t_gps", "p_gps", "v_gps"):
            out[key] = out[key][keep]
    return out


def degrade_gps(data, sigma_h, sigma_u=None, sigma_vel=0.0, gm_sigma=0.0, gm_tau=60.0, seed=0, drop_vel=False):
    """Copy of `data` with the GPS degraded to a lower-grade receiver; IMU and ground truth unchanged.

    Each position axis gets white noise (sigma_h east/north, sigma_u up, default 1.5 * sigma_h) plus,
    if gm_sigma > 0, a first-order Gauss-Markov error with steady-state std gm_sigma and correlation
    time gm_tau, discretised exactly over the real interval between fixes:
        b_k = exp(-dt/tau) b_{k-1} + gm_sigma sqrt(1 - exp(-2 dt/tau)) n_k.
    Velocity channels (when present) get white noise sigma_vel. A missing channel (NaN) stays NaN.
    drop_vel removes GPS velocity altogether (NaN), so the filter updates on position only.
    The same seed always gives the same copy.
    """
    rng = np.random.default_rng(seed)
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    n = len(out["t_gps"])
    sigma = np.array([sigma_h, sigma_h, 1.5 * sigma_h if sigma_u is None else sigma_u])
    noise = rng.normal(0.0, 1.0, (n, 3)) * sigma
    if gm_sigma:
        phi = np.exp(-np.diff(out["t_gps"], prepend=out["t_gps"][0]) / gm_tau)
        b = np.zeros((n, 3))
        b[0] = rng.normal(0.0, gm_sigma, 3)
        for k in range(1, n):
            b[k] = phi[k] * b[k - 1] + gm_sigma * np.sqrt(1.0 - phi[k] ** 2) * rng.normal(0.0, 1.0, 3)
        noise += b
    out["p_gps"] = out["p_gps"] + noise
    if sigma_vel:
        out["v_gps"] = out["v_gps"] + rng.normal(0.0, sigma_vel, out["v_gps"].shape)
    if drop_vel:
        out["v_gps"] = np.full_like(out["v_gps"], np.nan)
    return out


# Predefined perturbations used to measure robustness. Each entry is passed to `perturb`
# as keyword arguments. They are evaluated on a fixed, already tuned theta and are never
# part of the optimization objective.
PERTURBATIONS = {
    "gps_noise_1m": {"gps_sigma": 1.0},
    "gps_noise_2m": {"gps_sigma": 2.0},
    "gps_vel_noise": {"vel_sigma": 0.3},
    "imu_noise": {"accel_sigma": 0.5, "gyro_sigma": float(np.radians(1.0))},
    "gps_drop_50": {"gps_drop": 0.5},
}


def perturbed_datasets(data):
    """All perturbed copies of `data`, keyed by name."""
    return {name: perturb(data, **kwargs) for name, kwargs in PERTURBATIONS.items()}
