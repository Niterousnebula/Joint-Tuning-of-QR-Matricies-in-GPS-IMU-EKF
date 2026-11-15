"""Heuristic Informed Search Algorithm (HISA): joint initial-bias + R search, coarse to fine.

    python HISA.py                       # full search on the GPU -> results/HISA/
    python HISA.py --limit 20000 --out results/HISA_smoke   # evenly spaced subset, quick check

Pipeline (see HISA_config.py and the README):

    signed log grid +-10^e, e in {-3..2}, zero excluded        12^6 = 2,985,984 bias patterns
    accelerometer axes in one decade, gyro unrestricted, all signs   -> 82,944 bias candidates
    x 27 R combinations (r_xy, r_z, r_vel, 3 values each, independent), searched jointly
                                                                     -> 2,239,488 candidates
    Int50: all candidates  -> keep the best 5,210 (the Int50 winner always among them)
    Int1:  survivors (canonical filter and fitness) -> final winner
    Int1-equivalent cost: 2,239,488 / 50 + 5,210 = 49,999.76 <= 50,000

Every level runs on the GPU (ekf.gpu.GpuEvaluator), the whole batch of candidates at once.
IntN integrates every N IMU intervals into one prediction and keeps the last GPS fix of each
block; Int1 is the canonical run. Scores are only compared within one level. The incumbent
(best of the previous level) is re-evaluated first on the next level. The Int1 winner is then
re-scored by the filterpy reference on the CPU, with the same RMSE, robustness and NIS
metrics as the other optimizers.

Workload is reported two ways and never mixed: literal candidate evaluations per level, and
Int1-normalized prediction-update cost (sum of n / N). Measured EKF steps and wall-clock time
are reported next to them.
"""
import argparse
import itertools
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import ekf
import HISA_config as cfg

BIAS_NAMES = ["bax", "bay", "baz", "bgx", "bgy", "bgz"]
EXP_NAMES = ["e_ax", "e_ay", "e_az", "e_gx", "e_gy", "e_gz"]
SIGN_NAMES = ["s_ax", "s_ay", "s_az", "s_gx", "s_gy", "s_gz"]
R_NAMES = ["log_r_xy", "log_r_z", "log_r_vel"]


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
def admissible_magnitudes():
    """Exponent patterns that pass the admissibility rules, and the counts at each step."""
    exps = cfg.BIAS_EXPONENTS
    accel = [e for e in itertools.product(exps, repeat=3)
             if max(e) - min(e) <= cfg.ACCEL_CLUSTER_MAX_SPREAD]
    gyro = [e for e in itertools.product(exps, repeat=3)
            if cfg.GYRO_CLUSTER_MAX_SPREAD is None or max(e) - min(e) <= cfg.GYRO_CLUSTER_MAX_SPREAD]
    if cfg.ACCEL_GYRO_MAX_SPACING is not None:
        raise NotImplementedError("the current HISA specification has no accel/gyro spacing rule")
    return accel, gyro


def build_candidates():
    """All admissible joint candidates, in a fixed order (candidate_id = row index)."""
    accel, gyro = admissible_magnitudes()
    signs = list(itertools.product(cfg.BIAS_SIGNS, repeat=6))
    r_grid = list(itertools.product(cfg.R_LOG_XY, cfg.R_LOG_Z, cfg.R_LOG_VEL))
    ia, ig, isg, ir = (a.ravel() for a in np.meshgrid(np.arange(len(accel)), np.arange(len(gyro)),
                                                       np.arange(len(signs)), np.arange(len(r_grid)),
                                                       indexing="ij"))
    e = np.hstack([np.asarray(accel)[ia], np.asarray(gyro)[ig]]).astype(np.int8)
    s = np.asarray(signs)[isg].astype(np.int8)
    r = np.asarray(r_grid)[ir]
    cand = pd.DataFrame(e, columns=EXP_NAMES)
    for name, col in zip(SIGN_NAMES, s.T):
        cand[name] = col
    bias = s * 10.0 ** e
    for name, col in zip(BIAS_NAMES, bias.T):
        cand[name] = col
    cand["log_r_xy"], cand["log_r_z"], cand["log_r_vel"] = r[:, 0], r[:, 1], r[:, 2]
    for name in ("r_xy", "r_z", "r_vel"):
        cand[name] = 10.0 ** cand[f"log_{name}"]
    cand.insert(0, "candidate_id", np.arange(len(cand)))
    n_exp = len(cfg.BIAS_EXPONENTS)
    counts = {"theoretical_with_zero": (2 * n_exp + 1) ** 6,
              "signed_log_grid": (2 * n_exp) ** 6,
              "accel_magnitude_patterns": len(accel), "gyro_magnitude_patterns": len(gyro),
              "sign_patterns": len(signs), "bias_candidates": len(accel) * len(gyro) * len(signs),
              "r_combinations": len(r_grid), "joint_candidates": len(cand)}
    return cand, counts


def thetas_of(cand):
    """The 7-D theta for each candidate: Q terms from THETA0, R from the candidate."""
    theta = np.tile(ekf.THETA0, (len(cand), 1))
    theta[:, 4:7] = cand[R_NAMES].to_numpy()
    return theta


# ---------------------------------------------------------------------------
# IntN streams: the canonical filter on integrated IMU and decimated GPS
# ---------------------------------------------------------------------------
def integrated_stream(data, n):
    """Int-n data: n IMU intervals per prediction (dt-weighted means), last GPS fix per block."""
    if n == 1:
        return data
    t, accel, gyro = data["t_imu"], data["accel"], data["gyro"]
    n_int = len(t) - 1
    starts = np.arange(0, n_int, n)
    bounds = np.r_[starts, n_int]
    dt = np.diff(t)
    w = np.add.reduceat(dt, starts)
    a = np.add.reduceat(accel[:-1] * dt[:, None], starts) / w[:, None]
    g = np.add.reduceat(gyro[:-1] * dt[:, None], starts) / w[:, None]
    t_new = t[bounds]
    out = dict(data)
    out["t_imu"] = t_new
    out["accel"] = np.vstack([a, a[-1:]])        # the last row is never used by run_ekf
    out["gyro"] = np.vstack([g, g[-1:]])

    t_gps = data["t_gps"]
    j = np.arange(1, len(t_gps))
    j = j[(t_gps[j] > t_new[0]) & (t_gps[j] <= t_new[-1])]
    block = np.searchsorted(t_new, t_gps[j], side="left")      # t_new[b-1] < t <= t_new[b]
    last = np.r_[block[1:] != block[:-1], True]
    keep = np.r_[0, j[last]]
    for key in ("t_gps", "p_gps", "v_gps"):
        out[key] = data[key][keep]
    return out


def stream_workload(stream):
    """Prediction steps and GPS updates per candidate for one stream (run_ekf's schedule)."""
    t_imu, t_gps = stream["t_imu"], stream["t_gps"]
    j = 1
    while j < len(t_gps) and t_gps[j] <= t_imu[0]:
        j += 1
    n_upd = int(np.searchsorted(t_gps, t_imu[-1], side="right") - j)
    return len(t_imu) - 1, n_upd


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_level(ev, cand, batch):
    """Score candidates on one level, in the given order, in fixed-size GPU batches."""
    thetas, bias = thetas_of(cand), cand[BIAS_NAMES].to_numpy()
    n = len(cand)
    out = {k: np.empty(n) for k in ("score", "rmse_v", "rmse_3d", "mean_nis")}
    out["ok"] = np.empty(n, dtype=bool)
    for i in range(0, n, batch):
        idx = np.arange(i, min(i + batch, n))
        pad = np.r_[idx, np.full(batch - len(idx), idx[-1])] if n > batch else idx
        r = ev.evaluate(thetas[pad], bias0=bias[pad])
        m = len(idx)
        out["score"][idx] = r["rmse_h"][:m]
        for k in ("rmse_v", "rmse_3d", "mean_nis", "ok"):
            out[k][idx] = r[k][:m]
        print(f"    {min(i + batch, n):>7,}/{n:,} candidates", flush=True)
    return out


def cpu_check(stream, x0, cand):
    """filterpy reference metrics for a few candidates on one stream."""
    rows = []
    for theta, bias in zip(thetas_of(cand), cand[BIAS_NAMES].to_numpy()):
        r = ekf.evaluate(stream, theta, bias0=bias, x0=x0)
        rows.append([r["rmse_h"], r["rmse_v"], r["rmse_3d"], r["mean_nis"], r["ok"]])
    return np.array(rows, dtype=float)


def write_table(df, out_dir, base):
    """Write <base>.csv (gzip when large); split into parts if a file would exceed MAX_FILE_MB."""
    gz = len(df) > 100_000
    path = out_dir / (base + (".csv.gz" if gz else ".csv"))
    df.to_csv(path, index=False, float_format="%.10g")
    size_mb = path.stat().st_size / 2**20
    if size_mb <= cfg.MAX_FILE_MB:
        return [path.name]
    path.unlink()
    n_parts = int(np.ceil(size_mb / (0.8 * cfg.MAX_FILE_MB)))
    names = []
    for i, part in enumerate(np.array_split(np.arange(len(df)), n_parts), start=1):
        p = out_dir / f"{base}_part{i:02d}.csv.gz"
        df.iloc[part].to_csv(p, index=False, float_format="%.10g")
        names.append(p.name)
    return names


def n_retained(level, n, n_full):
    """How many candidates go on to the next level (scaled down for a --limit subset)."""
    if level == cfg.LEVELS[-1]:
        return 0
    keep = cfg.RETAIN[level]
    if n < n_full:                                   # smoke test on a subset
        keep = max(2, int(round(keep * n / n_full)))
    return min(keep, n)


def level_summary(stage, level, table, incumbent, t_level, n_pred, n_upd, literal_total, norm_total):
    """stage_summary row and best-so-far change points for one evaluated level."""
    order = np.lexsort((table["candidate_id"].to_numpy(), table["score"].to_numpy()))
    best = table.iloc[order[0]]
    inc_score = (float(table.loc[table["candidate_id"] == incumbent, "score"].iloc[0])
                 if incumbent is not None else np.nan)
    row = {"stage": stage, "integration_level": level, "candidates_in": len(table),
           "retained_out": int(table["retained"].sum()), "literal_evals": len(table),
           "int1_normalized_cost": len(table) / level,
           "predictions_per_candidate": n_pred, "gps_updates_per_candidate": n_upd,
           "ekf_steps_total": len(table) * (n_pred + n_upd),
           "failed_candidates": int((~table["ok"].astype(bool)).sum()),
           "best_candidate_id": int(best["candidate_id"]), "best_score": float(best["score"]),
           "incumbent_id": incumbent, "incumbent_score": inc_score, "wall_clock_s": t_level}
    curve = [{"stage": stage, "integration_level": level, "eval_order": int(r["eval_order"]),
              "literal_evals": literal_total + int(r["eval_order"]),
              "int1_normalized": norm_total + r["eval_order"] / level,
              "candidate_id": int(r["candidate_id"]), "score": r["score"],
              "best_so_far_in_level": r["best_so_far"]}
             for _, r in table[table["is_best_so_far"].astype(bool)].iterrows()]
    return row, curve, int(best["candidate_id"]), inc_score


def read_table(out_dir, base):
    """Read a table written by write_table (plain, gzip, or split into parts)."""
    for name in (f"{base}.csv", f"{base}.csv.gz"):
        if (out_dir / name).exists():
            return pd.read_csv(out_dir / name), [out_dir / name]
    parts = sorted(out_dir.glob(f"{base}_part*.csv.gz"))
    if not parts:
        raise FileNotFoundError(f"{base} not found in {out_dir}")
    return pd.concat([pd.read_csv(q) for q in parts], ignore_index=True), parts


def rank(scores, ids):
    """Rank 1 = best; ties broken by candidate_id so the order is reproducible."""
    order = np.lexsort((ids, scores))
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return order, ranks


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def run(out_dir, limit=None):
    import torch
    from ekf.gpu import TOL_NIS_REL, TOL_RMSE, GpuEvaluator

    out_dir = Path(out_dir)
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; HISA never overwrites an earlier run.")
    out_dir.mkdir(parents=True)
    (out_dir / "validation").mkdir()
    t_start = time.perf_counter()
    started = datetime.now().isoformat(timespec="seconds")

    data = ekf.load_data()
    x0, _ = ekf.init_state(data)
    cand, counts = build_candidates()
    if limit:
        cand = cand.iloc[np.linspace(0, len(cand) - 1, limit).round().astype(int)].reset_index(drop=True)
        counts["smoke_limit"] = limit
    print(f"HISA: {len(cand):,} admissible joint candidates "
          f"({counts['bias_candidates']:,} bias x {counts['r_combinations']} R)", flush=True)
    n_full = counts["joint_candidates"]
    planned, n = 0.0, n_full
    for lv in cfg.LEVELS:
        planned += n / lv
        n = cfg.RETAIN.get(lv, 0)
    counts["planned_int1_equivalent"] = planned
    print(f"Planned Int1-equivalent cost: {planned:,.2f} (budget {cfg.INT1_EQUIVALENT_BUDGET:,})", flush=True)
    if planned > cfg.INT1_EQUIVALENT_BUDGET:
        raise SystemExit("HISA_config: the schedule exceeds INT1_EQUIVALENT_BUDGET")
    files = {"candidates_all": write_table(cand, out_dir, "candidates_all")}

    pool = cand
    stages, curve, validation = [], [], []
    incumbent = None              # candidate_id of the best of the previous level
    literal_total, norm_total = 0, 0.0
    for stage, level in enumerate(cfg.LEVELS, start=1):
        stream = integrated_stream(data, level)
        n_pred, n_upd = stream_workload(stream)
        print(f"\n=== Stage {stage}: Int{level} | {len(pool):,} candidates | "
              f"{n_pred:,} predictions + {n_upd:,} GPS updates per candidate ===", flush=True)

        # Evaluation order: first level by candidate_id; later levels by previous rank, so
        # the incumbent is re-evaluated first on the new level. All on the GPU.
        ev = GpuEvaluator(stream, x0=x0)
        t0 = time.perf_counter()
        res = evaluate_level(ev, pool, cfg.BATCH[level])
        t_level = time.perf_counter() - t0
        del ev
        torch.cuda.empty_cache()

        table = pool.copy()
        table.insert(1, "integration_level", level)
        table["eval_order"] = np.arange(1, len(table) + 1)
        table["score"] = res["score"]
        for k in ("rmse_v", "rmse_3d", "mean_nis", "ok"):
            table[k] = res[k]
        order, ranks = rank(table["score"].to_numpy(), table["candidate_id"].to_numpy())
        table["rank"] = ranks
        running = np.minimum.accumulate(table["score"].to_numpy())
        new_best = np.r_[True, running[1:] < running[:-1]]
        table["is_best_so_far"] = new_best
        table["best_so_far"] = running
        n_keep = n_retained(level, len(table), n_full if stage == 1 else len(table))
        table["retained"] = table["rank"] <= n_keep
        best = table.iloc[order[0]]
        if n_keep:
            assert bool(best["retained"]), "the level winner must go on to the next level"

        st_row, st_curve, _, inc_score = level_summary(stage, level, table, incumbent, t_level, n_pred, n_upd,
                                                       literal_total, norm_total)
        if incumbent is not None:
            print(f"  incumbent {incumbent} re-evaluated on Int{level}: {inc_score:.4f} m", flush=True)

        files[f"int{level}_results"] = write_table(table, out_dir, f"int{level}_results")
        curve.extend(st_curve)                 # best-so-far change points, both workload measures
        literal_total += len(table)
        norm_total += len(table) / level

        # GPU vs filterpy on this level: top candidates, the worst ones and a spread in between
        pick = np.unique(np.r_[order[:3], order[-2:],
                               order[np.linspace(0, len(order) - 1, max(cfg.VALIDATE_PER_LEVEL - 5, 1)).astype(int)]])
        sample = table.iloc[pick]
        cpu = cpu_check(stream, x0, sample)
        for (_, r), c in zip(sample.iterrows(), cpu):
            both = bool(r["ok"]) and bool(c[4])
            validation.append({"integration_level": level, "candidate_id": int(r["candidate_id"]),
                               "gpu_score": r["score"], "cpu_score": c[0],
                               "diff_score": abs(r["score"] - c[0]),
                               "diff_rmse_v": abs(r["rmse_v"] - c[1]) if both else 0.0,
                               "diff_rmse_3d": abs(r["rmse_3d"] - c[2]) if both else 0.0,
                               "reldiff_mean_nis": abs(r["mean_nis"] - c[3]) / abs(c[3]) if both else 0.0,
                               "gpu_ok": bool(r["ok"]), "cpu_ok": bool(c[4])})

        stages.append(st_row)
        print(f"  best Int{level}: candidate {int(best['candidate_id'])}, score {best['score']:.4f} m; "
              f"{int((~table['ok']).sum()):,} diverged; {t_level:.0f} s", flush=True)
        incumbent = int(best["candidate_id"])
        if n_keep:
            pool = table.iloc[order[:n_keep]][cand.columns].reset_index(drop=True)
            assert pool["candidate_id"].iloc[0] == incumbent, "the incumbent must lead the next level"

    t_search = time.perf_counter() - t_start
    val = pd.DataFrame(validation)
    val.to_csv(out_dir / "validation" / "validation_table.csv", index=False)
    return finalize(out_dir, data, cand, counts, pd.DataFrame(stages), pd.DataFrame(curve), val, incumbent,
                    literal_total, norm_total, files, started, {"search": t_search}, t_start=t_start)


def finalize(out_dir, data, cand, counts, st, curve, val, incumbent, literal_total, norm_total, files,
             started, timing, t_start=None, note=None):
    """Validation verdict, filterpy re-score of the Int1 winner, and every summary file."""
    import torch
    from ekf.gpu import TOL_NIS_REL, TOL_RMSE

    val_ok = bool((val["gpu_ok"] == val["cpu_ok"]).all() and val["diff_score"].max() <= TOL_RMSE
                  and val["diff_rmse_v"].max() <= TOL_RMSE and val["diff_rmse_3d"].max() <= TOL_RMSE
                  and val["reldiff_mean_nis"].max() <= TOL_NIS_REL)
    print(f"\nGPU vs filterpy on {len(val)} candidates over all levels: max score difference "
          f"{val['diff_score'].max():.3e} m -> {'PASSED' if val_ok else 'FAILED'}", flush=True)

    # Final winner: re-scored by the filterpy reference with the benchmark's metrics
    win = cand[cand["candidate_id"] == incumbent].iloc[0]
    theta, bias = thetas_of(win.to_frame().T)[0], win[BIAS_NAMES].to_numpy(dtype=float)
    print(f"\nFinal winner {incumbent}: re-scoring with filterpy, robustness and NIS...", flush=True)
    t0 = time.perf_counter()
    result = ekf.evaluate(data, theta, bias0=bias)
    rob = ekf.robustness(data, theta, rmse_clean=result["rmse_h"], bias0=bias)
    rmse_gps = ekf.gps_rmse(data)
    rmse_baseline = ekf.fitness(data, ekf.THETA0)
    t_final = time.perf_counter() - t0
    timing["final_rescore"] = t_final
    timing["total"] = (time.perf_counter() - t_start) if t_start is not None else timing["search"] + t_final
    t_total = timing["total"]
    st.to_csv(out_dir / "stage_summary.csv", index=False)
    curve.to_csv(out_dir / "best_so_far.csv", index=False)

    row = {"optimizer": "HISA", "seed": cfg.SEED, **dict(zip(ekf.THETA_NAMES, theta)),
           **dict(zip(BIAS_NAMES, bias)),
           "rmse_h": result["rmse_h"], "rmse_v": result["rmse_v"], "rmse_3d": result["rmse_3d"],
           "mean_nis": result["mean_nis"], "nis_dof": ekf.NIS_DOF,
           "rmse_gps": rmse_gps, "delta_vs_gps": result["rmse_h"] - rmse_gps,
           "delta_vs_baseline": result["rmse_h"] - rmse_baseline,
           "improvement_vs_baseline_pct": 100.0 * (rmse_baseline - result["rmse_h"]) / rmse_baseline,
           "delta_rmse_mean": rob["delta_mean"], "delta_rmse_max": rob["delta_max"]}
    for pname, p in rob["per_perturbation"].items():
        row[f"rmse_{pname}"], row[f"delta_{pname}"], row[f"delta_pct_{pname}"] = p["rmse"], p["delta"], p["delta_pct"]
    row.update({"evals_to_converge": np.nan, "evals_to_best": np.nan,
                "n_evals": literal_total, "max_evals": np.nan,
                "t_total": t_total, "t_eval": t_total / literal_total,
                "backend": "gpu", "population_size": len(cand), "n_generations": len(cfg.LEVELS),
                "n_objective_evals": literal_total, "t_objective": float(st["wall_clock_s"].sum()),
                "t_objective_per_eval": float(st["wall_clock_s"].sum()) / literal_total,
                "best_fitness_search": float(st["best_score"].iloc[-1]),
                "search_vs_final_diff": result["rmse_h"] - float(st["best_score"].iloc[-1]),
                "int1_normalized_evals": norm_total,
                "ekf_steps_total": int(st["ekf_steps_total"].sum()),
                **{f"literal_int{int(s['integration_level'])}": int(s["literal_evals"]) for _, s in st.iterrows()}})
    pd.DataFrame([row]).to_csv(out_dir / "runs.csv", index=False)
    final = {"candidate_id": incumbent, **win.to_dict(), "int1_score_gpu": float(st["best_score"].iloc[-1]),
             "int1_score_cpu": result["rmse_h"], "rmse_v": result["rmse_v"], "rmse_3d": result["rmse_3d"],
             "mean_nis": result["mean_nis"], "delta_rmse_mean": rob["delta_mean"],
             "delta_rmse_max": rob["delta_max"], "rmse_gps": rmse_gps, "rmse_baseline": rmse_baseline}
    pd.DataFrame([final]).to_csv(out_dir / "final_best.csv", index=False)

    # EKF error comparison data: raw GPS, baseline EKF, HISA winner (all filterpy, whole flight)
    t_true, p_true = data["t_true"], data["p_true"]
    gps_err = ekf.position_errors(data["t_gps"], data["p_gps"], t_true, p_true)
    base = ekf.evaluate(data, ekf.THETA0)
    pd.DataFrame({"t": t_true, "motion": data["motion"],
                  "gps_err_h": np.hypot(gps_err[:, 0], gps_err[:, 1]),
                  "baseline_err_h": ekf.horizontal_errors(data["t_imu"], base["x_log"], t_true, p_true),
                  "hisa_err_h": ekf.horizontal_errors(data["t_imu"][:len(result["x_log"])], result["x_log"],
                                                      t_true, p_true)}
                 ).to_csv(out_dir / "ekf_error_timeseries.csv", index=False)

    import subprocess
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    snapshot = {"started": started, "finished": datetime.now().isoformat(timespec="seconds"),
                "git": {"branch": git("rev-parse", "--abbrev-ref", "HEAD"), "commit": git("rev-parse", "HEAD"),
                        "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))},
                "config": {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()},
                "q_terms_from_theta0": dict(zip(ekf.THETA_NAMES[:4], ekf.THETA0[:4].tolist())),
                "candidate_counts": counts,
                "objective": "whole-flight horizontal RMSE (ekf.evaluate / GpuEvaluator), PENALTY "
                             f"{ekf.PENALTY} m on divergence; scores compared within one level only",
                "int_levels": {f"Int{lv}": "canonical filter" if lv == 1 else
                               f"{lv} IMU intervals per prediction (dt-weighted mean), last GPS fix per block"
                               for lv in cfg.LEVELS},
                "workload": {"literal_evals": literal_total, "int1_normalized_evals": norm_total,
                             "int1_equivalent_budget": cfg.INT1_EQUIVALENT_BUDGET,
                             "within_budget": bool(norm_total <= cfg.INT1_EQUIVALENT_BUDGET),
                             "ekf_steps_total": int(st["ekf_steps_total"].sum())},
                "files": files,
                "timing_s": timing,
                "note": note,
                "validation": {"passed": val_ok, "max_diff_score": float(val["diff_score"].max()),
                               "n_candidates": len(val), "tol_rmse": TOL_RMSE, "tol_nis_rel": TOL_NIS_REL},
                "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}}
    (out_dir / "config_snapshot.json").write_text(json.dumps(snapshot, indent=2, default=str))

    print(f"\nHISA winner {incumbent}: RMSE_h {result['rmse_h']:.4f} m (vs baseline "
          f"{row['delta_vs_baseline']:+.4f} m, vs GPS {row['delta_vs_gps']:+.4f} m), "
          f"robustness mean {rob['delta_mean']:+.4f} m, NIS {result['mean_nis']:.2f}", flush=True)
    print(f"Workload: {literal_total:,} literal evaluations, {norm_total:,.2f} Int1-normalized, "
          f"{row['ekf_steps_total']:,} EKF steps; total {t_total / 60:.1f} min", flush=True)
    if not val_ok:
        raise SystemExit("GPU vs filterpy validation exceeded tolerance; see validation/validation_table.csv")
    return out_dir


def finish(out_dir):
    """Complete a run whose search finished but whose final step did not.

    Needs candidates_all, every int<N>_results table and validation/validation_table.csv.
    Re-scores the Int1 winner with filterpy and writes stage_summary, best_so_far, runs,
    final_best, ekf_error_timeseries and config_snapshot. Nothing is re-searched. The per-level
    wall-clock times are taken from the files' modification times and are marked as such.
    """
    out_dir = Path(out_dir)
    done = [f for f in ("stage_summary.csv", "runs.csv", "final_best.csv", "config_snapshot.json")
            if (out_dir / f).exists()]
    if done:
        raise SystemExit(f"{out_dir} already has {done}; nothing to finish.")
    data = ekf.load_data()
    cand, cand_files = read_table(out_dir, "candidates_all")
    _, counts = build_candidates()
    if len(cand) != counts["joint_candidates"]:
        counts["smoke_limit"] = len(cand)
    planned, n = 0.0, counts["joint_candidates"]
    for lv in cfg.LEVELS:
        planned += n / lv
        n = cfg.RETAIN.get(lv, 0)
    counts["planned_int1_equivalent"] = planned
    files = {"candidates_all": [f.name for f in cand_files]}
    val_path = out_dir / "validation" / "validation_table.csv"
    val = pd.read_csv(val_path)

    t_prev = max(f.stat().st_mtime for f in cand_files)
    t_first = t_prev
    stages, curve = [], []
    incumbent, literal_total, norm_total = None, 0, 0.0
    for stage, level in enumerate(cfg.LEVELS, start=1):
        table, paths = read_table(out_dir, f"int{level}_results")
        files[f"int{level}_results"] = [q.name for q in paths]
        n_pred, n_upd = stream_workload(integrated_stream(data, level))
        t_file = max(q.stat().st_mtime for q in paths)
        row, cur, incumbent_next, _ = level_summary(stage, level, table, incumbent, t_file - t_prev,
                                                    n_pred, n_upd, literal_total, norm_total)
        stages.append(row)
        curve.extend(cur)
        t_prev = t_file
        incumbent = incumbent_next
        literal_total += len(table)
        norm_total += len(table) / level
        print(f"Int{level}: {len(table):,} candidates, best {row['best_candidate_id']} "
              f"({row['best_score']:.4f} m), ~{row['wall_clock_s'] / 60:.1f} min", flush=True)

    t_search = val_path.stat().st_mtime - t_first
    started = datetime.fromtimestamp(t_first).isoformat(timespec="seconds")
    note = ("Completed with HISA.py --finish after the original run stopped during the final step. The "
            "search tables and validation are from the original run. Per-level wall_clock_s and "
            "timing_s.search come from file modification times (they include writing each level's table "
            "and the validation checks; the time to build and write candidates_all is not included).")
    return finalize(out_dir, data, cand, counts, pd.DataFrame(stages), pd.DataFrame(curve), val, incumbent,
                    literal_total, norm_total, files, started, {"search": t_search, "source": "file mtimes"},
                    note=note)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=cfg.OUTPUT_DIR)
    parser.add_argument("--limit", type=int, help="evaluate only this many evenly spaced candidates (smoke test)")
    parser.add_argument("--finish", metavar="DIR",
                        help="complete a run in DIR whose search finished but whose final step did not")
    a = parser.parse_args()
    if a.finish:
        finish(a.finish)
    else:
        run(a.out, a.limit)
