"""Write a derived dataset whose GPS is degraded to a lower-grade receiver (IMU and ground truth unchanged).

The clean datasets have GPS that is far better than their IMU, so a GPS-IMU filter has almost nothing
to add and every Q/R tuning ends at raw GPS. This adds receiver-like noise to the GPS fixes only:
white noise on each position axis plus an optional Gauss-Markov (time-correlated) error, and white
noise on GPS velocity when the dataset has it (see ekf.degrade_gps).

The result is processed_data/<dataset>_gps<sigma>m[_gm<g>m<tau>s]_s<seed>/ holding gps.csv and
meta.json only; ekf.load_data takes the IMU and ground truth from the source dataset. Use the new id
with --dataset in every script (run_baseline.py, GA.py, PSO.py, DE.py, GWO.py, main.py --datasets).
The noise is frozen in the file, so every optimizer is tuned on exactly the same fixes.

Tune on one seed and check the tuned theta on other seeds, otherwise the tuning fits that one
noise realisation:

    python make_noisy_gps.py --dataset inavwhu/street00 --sigma 1 2 4             # tuning sets, seed 0
    python make_noisy_gps.py --dataset inavwhu/street00 --sigma 2 --seed 1 2 3 4   # held-out sets
"""
import argparse
import json

import numpy as np
import pandas as pd

import ekf
from ekf.data import PROCESSED_DIR


def derived_id(dataset, sigma, gm_sigma, gm_tau, seed):
    tag = f"gps{sigma:g}m" + (f"_gm{gm_sigma:g}m{gm_tau:g}s" if gm_sigma else "")
    return f"{dataset}_{tag}_s{seed}"


def make(dataset, sigma, sigma_u=None, sigma_vel=0.0, gm_sigma=0.0, gm_tau=60.0, seed=0):
    data = ekf.load_data(dataset)
    noisy = ekf.degrade_gps(data, sigma, sigma_u, sigma_vel, gm_sigma, gm_tau, seed)
    new_id = derived_id(dataset, sigma, gm_sigma, gm_tau, seed)
    out = PROCESSED_DIR / new_id
    out.mkdir(parents=True, exist_ok=True)

    gps = pd.read_csv(PROCESSED_DIR / dataset / "gps.csv")
    gps[["e", "n", "u"]] = noisy["p_gps"]
    gps[["ve", "vn", "vu"]] = noisy["v_gps"]
    gps.to_csv(out / "gps.csv", index=False)

    meta = json.loads((PROCESSED_DIR / dataset / "meta.json").read_text())
    rmse_clean, rmse_noisy = ekf.gps_rmse(data), ekf.gps_rmse(noisy)
    meta.update({
        "id": new_id,
        "label": f"{meta.get('label', dataset)} + degraded GPS ({sigma:g} m white"
                 + (f", {gm_sigma:g} m / {gm_tau:g} s Gauss-Markov" if gm_sigma else "") + f", seed {seed})",
        "derived_from": dataset,
        "gps_noise": {"sigma_h_m": sigma, "sigma_u_m": 1.5 * sigma if sigma_u is None else sigma_u,
                      "sigma_vel_ms": sigma_vel, "gm_sigma_m": gm_sigma, "gm_tau_s": gm_tau, "seed": seed,
                      "raw_gps_rmse_h_clean_m": rmse_clean, "raw_gps_rmse_h_noisy_m": rmse_noisy},
    })
    if "std_e" in gps.columns:
        meta["gps_noise"]["note"] = "std_e/std_n/std_u are still the receiver's original sigmas"
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{new_id}: raw GPS RMSE_h {rmse_clean:.3f} m -> {rmse_noisy:.3f} m")
    return new_id


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, help="clean processed dataset id, e.g. eth or inavwhu/street00")
    p.add_argument("--sigma", type=float, nargs="+", required=True,
                   help="white position noise per horizontal axis, m (one dataset per value)")
    p.add_argument("--sigma-u", type=float, help="vertical white noise, m (default 1.5 * sigma)")
    p.add_argument("--vel-sigma", type=float, default=0.0, help="white GPS velocity noise, m/s (default 0)")
    p.add_argument("--gm-sigma", type=float, default=0.0, help="Gauss-Markov error std per axis, m (default 0)")
    p.add_argument("--gm-tau", type=float, default=60.0, help="Gauss-Markov correlation time, s (default 60)")
    p.add_argument("--seed", type=int, nargs="+", default=[0])
    a = p.parse_args()
    for sigma in a.sigma:
        for seed in a.seed:
            make(a.dataset, sigma, a.sigma_u, a.vel_sigma, a.gm_sigma, a.gm_tau, seed)


if __name__ == "__main__":
    main()
