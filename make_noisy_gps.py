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

--sections instead keeps the raw GPS and degrades three sections of 250 ft (76.2 m) of travel, at 25 %,
50 % and 75 % of the path: about 5 m added error, about 15 m, and a complete outage (SECTIONS below).
It writes processed_data/<dataset>_outages_s<seed>/, the datasets of main.py's "outages" scenario:

    python make_noisy_gps.py --dataset eth --sections
"""
import argparse
import json

import numpy as np
import pandas as pd

import ekf
from ekf.data import PROCESSED_DIR


def derived_id(dataset, sigma, gm_sigma, gm_tau, seed, drop_vel=False):
    tag = f"gps{sigma:g}m" + (f"_gm{gm_sigma:g}m{gm_tau:g}s" if gm_sigma else "") + ("_nov" if drop_vel else "")
    return f"{dataset}_{tag}_s{seed}"


def make(dataset, sigma, sigma_u=None, sigma_vel=0.0, gm_sigma=0.0, gm_tau=60.0, seed=0, drop_vel=False):
    data = ekf.load_data(dataset)
    noisy = ekf.degrade_gps(data, sigma, sigma_u, sigma_vel, gm_sigma, gm_tau, seed, drop_vel)
    new_id = derived_id(dataset, sigma, gm_sigma, gm_tau, seed, drop_vel)
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
                      "sigma_vel_ms": sigma_vel, "velocity_dropped": bool(drop_vel), "gm_sigma_m": gm_sigma, "gm_tau_s": gm_tau, "seed": seed,
                      "raw_gps_rmse_h_clean_m": rmse_clean, "raw_gps_rmse_h_noisy_m": rmse_noisy},
    })
    if "std_e" in gps.columns:
        meta["gps_noise"]["note"] = "std_e/std_n/std_u are still the receiver's original sigmas"
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{new_id}: raw GPS RMSE_h {rmse_clean:.3f} m -> {rmse_noisy:.3f} m")
    return new_id


# ---------------------------------------------------------------------------
# Raw GPS with degraded sections (main.py scenario "outages")
# ---------------------------------------------------------------------------
SECTION_M = 76.2           # 250 ft of travel per section
# Where each section is centred (share of the ground-truth path length) and what happens there.
# sigma_h_m is the added horizontal error (RMS); vertical gets 1.5x per axis, and GPS velocity
# (when the dataset has it) 0.1 m/s per metre of sigma_h_m. An outage removes the fixes.
SECTIONS = ({"kind": "noise", "sigma_h_m": 5.0, "at": 0.25},
            {"kind": "noise", "sigma_h_m": 15.0, "at": 0.50},
            {"kind": "outage", "at": 0.75})


def sections_id(dataset, seed=0):
    return f"{dataset}_outages_s{seed}"


def fix_errors(t_fix, p_fix, data):
    """Horizontal error of each fix against the truth interpolated to the fix's time."""
    return np.hypot(*(p_fix[:, i] - np.interp(t_fix, data["t_true"], data["p_true"][:, i]) for i in range(2)))


def _section_gps_rmse(t_fix, p_fix, data, t0, t1):
    """Horizontal RMS error of the fixes in a section (NaN without fixes)."""
    return ekf.rmse(fix_errors(t_fix, p_fix, data)) if len(t_fix) else float("nan")


def make_sections(dataset, sections=SECTIONS, length_m=SECTION_M, seed=0):
    """Raw GPS of `dataset` with each section of `length_m` metres of travel degraded or removed."""
    data = ekf.load_data(dataset)
    t, p = data["t_true"], data["p_true"][:, :2]
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    gps = pd.read_csv(PROCESSED_DIR / dataset / "gps.csv")
    rng = np.random.default_rng(seed)
    keep = np.ones(len(gps), bool)
    has_vel = bool(np.isfinite(gps[["ve", "vn", "vu"]].to_numpy()).all())
    out_sections = []
    for sec in sections:
        s0 = sec["at"] * s[-1] - length_m / 2
        t0, t1 = (float(v) for v in np.interp([s0, s0 + length_m], s, t))
        inside = ((gps["t"] >= t0) & (gps["t"] <= t1)).to_numpy()
        tf = gps["t"].to_numpy()[inside]
        raw = _section_gps_rmse(tf, gps.loc[inside, ["e", "n", "u"]].to_numpy(), data, t0, t1)
        if sec["kind"] == "noise":
            axis = sec["sigma_h_m"] / np.sqrt(2)
            gps.loc[inside, ["e", "n", "u"]] += rng.normal(0.0, 1.0, (inside.sum(), 3)) * [axis, axis, 1.5 * axis]
            if has_vel:
                gps.loc[inside, ["ve", "vn", "vu"]] += rng.normal(0.0, 0.1 * sec["sigma_h_m"], (inside.sum(), 3))
            degraded = _section_gps_rmse(tf, gps.loc[inside, ["e", "n", "u"]].to_numpy(), data, t0, t1)
        else:
            keep &= ~inside
            degraded = float("nan")
        out_sections.append({**sec, "t_start_s": t0, "t_end_s": t1, "duration_s": t1 - t0,
                             "start_m": float(s0), "length_m": length_m, "n_fixes": int(inside.sum()),
                             "raw_gps_rmse_h_m": raw, "degraded_gps_rmse_h_m": degraded})
    gps = gps[keep]

    new_id = sections_id(dataset, seed)
    out = PROCESSED_DIR / new_id
    out.mkdir(parents=True, exist_ok=True)
    gps.to_csv(out / "gps.csv", index=False)
    meta = json.loads((PROCESSED_DIR / dataset / "meta.json").read_text())
    noisy = {**data, "t_gps": gps["t"].to_numpy(), "p_gps": gps[["e", "n", "u"]].to_numpy()}
    meta.update({
        "id": new_id,
        "label": f"{meta.get('label', dataset)}, raw GPS with {len(sections)} degraded sections of {length_m:g} m",
        "derived_from": dataset,
        "gps_samples": int(len(gps)),
        "gps_sections": {"seed": seed, "length_m": length_m, "sections": out_sections,
                         "raw_gps_rmse_h_clean_m": ekf.gps_rmse(data), "raw_gps_rmse_h_m": ekf.gps_rmse(noisy)},
    })
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{new_id}: " + "; ".join(
        f"{x['kind']}{' %g m' % x['sigma_h_m'] if x['kind'] == 'noise' else ''} "
        f"t {x['t_start_s']:.0f}-{x['t_end_s']:.0f} s ({x['n_fixes']} fixes, GPS {x['raw_gps_rmse_h_m']:.2f} -> "
        f"{x['degraded_gps_rmse_h_m']:.2f} m)" for x in out_sections))
    return new_id


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, help="clean processed dataset id, e.g. eth or inavwhu/street00")
    p.add_argument("--sections", action="store_true",
                   help="raw GPS with the SECTIONS degraded (5 m, 15 m, outage) instead of noise everywhere")
    p.add_argument("--sigma", type=float, nargs="+",
                   help="white position noise per horizontal axis, m (one dataset per value)")
    p.add_argument("--sigma-u", type=float, help="vertical white noise, m (default 1.5 * sigma)")
    p.add_argument("--vel-sigma", type=float, default=0.0, help="white GPS velocity noise, m/s (default 0)")
    p.add_argument("--gm-sigma", type=float, default=0.0, help="Gauss-Markov error std per axis, m (default 0)")
    p.add_argument("--gm-tau", type=float, default=60.0, help="Gauss-Markov correlation time, s (default 60)")
    p.add_argument("--no-velocity", action="store_true", help="drop GPS velocity (position-only fixes)")
    p.add_argument("--seed", type=int, nargs="+", default=[0])
    a = p.parse_args()
    if a.sections:
        for seed in a.seed:
            make_sections(a.dataset, seed=seed)
        return
    if not a.sigma:
        p.error("give --sigma or --sections")
    for sigma in a.sigma:
        for seed in a.seed:
            make(a.dataset, sigma, a.sigma_u, a.vel_sigma, a.gm_sigma, a.gm_tau, seed, a.no_velocity)


if __name__ == "__main__":
    main()
