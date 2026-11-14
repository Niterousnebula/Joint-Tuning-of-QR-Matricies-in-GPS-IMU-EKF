"""Loading the cleaned ETH Zurich flight and building perturbed copies of it."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_data(data_dir=DATA_DIR):
    """Read the four sensor files into one dictionary of numpy arrays."""
    data_dir = Path(data_dir)
    accel = pd.read_csv(data_dir / "RawAccel ETH")
    gyro = pd.read_csv(data_dir / "RawGyro ETH")
    gps = pd.read_csv(data_dir / "OnboardGPS ETH")
    gt = pd.read_csv(data_dir / "GroundTruth ETH")
    return {
        "t_imu": accel["t"].to_numpy(),
        "accel": accel[["ax", "ay", "az"]].to_numpy(),
        "gyro": gyro[["wx", "wy", "wz"]].to_numpy(),
        "t_gps": gps["t"].to_numpy(),
        "p_gps": gps[["e", "n", "u"]].to_numpy(),
        "v_gps": gps[["vel_e", "vel_n", "vel_u"]].to_numpy(),
        "t_true": gt["t"].to_numpy(),
        "p_true": gt[["e", "n", "u"]].to_numpy(),
        "motion": gt["motion"].to_numpy(),
    }


def perturb(data, gps_sigma=0.0, vel_sigma=0.0, accel_sigma=0.0, gyro_sigma=0.0,
            gps_drop=0.0, seed=0):
    """Copy of `data` with extra white noise on the sensors and GPS fixes dropped at random.

    Ground truth is never changed. Each call starts from its own generator, so the same
    arguments always give the same copy.
    """
    rng = np.random.default_rng(seed)
    out = {k: v.copy() for k, v in data.items()}
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


# Copies of the flight used to measure how stable a tuned filter is.
# Each entry is passed to `perturb` as keyword arguments.
ROBUSTNESS_DATASETS = {
    "clean": {},
    "gps_noise_1m": {"gps_sigma": 1.0},
    "gps_noise_2m": {"gps_sigma": 2.0},
    "gps_vel_noise": {"vel_sigma": 0.3},
    "imu_noise": {"accel_sigma": 0.5, "gyro_sigma": float(np.radians(1.0))},
    "gps_drop_50": {"gps_drop": 0.5},
}


def robustness_datasets(data):
    """All robustness copies of `data`, keyed by name."""
    return {name: perturb(data, **kwargs) for name, kwargs in ROBUSTNESS_DATASETS.items()}
