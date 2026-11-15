"""HISA figures and the side-by-side comparison with the GA/PSO/DE/GWO benchmark.

Reads only the CSV/JSON files written by HISA.py (and, optionally, a run_benchmark.py
folder); the search itself is never imported or re-run.

    python plot_hisa.py --hisa results/HISA --benchmark results/benchmark_<stamp>

Writes to <hisa>/figures/:
    score_vs_iteration.png      candidate score against evaluation order, per level
    stage_progression.png       Int25 -> Int10 -> Int1 scores of the finalists, rank agreement
    bias_distributions.png      bias exponents and signs: all candidates vs survivors of each level
    qr_search.png               best score per R combination at each level
    best_so_far.png             best-so-far per level against Int1-normalized workload
    ekf_error_comparison.png    whole-flight horizontal error: raw GPS, baseline EKF, HISA winner
and, with --benchmark, to <hisa>/:
    comparison_with_benchmark.csv/.md, and figures/comparison.png
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LEVEL_COLORS = {25: "#9467bd", 10: "#17becf", 1: "#d62728"}
OPT_COLORS = {"baseline": "#555555", "GA": "#1f77b4", "PSO": "#ff7f0e", "DE": "#2ca02c", "GWO": "#8c564b",
              "HISA": "#d62728"}
BIAS_NAMES = ["bax", "bay", "baz", "bgx", "bgy", "bgz"]
EXP_NAMES = ["e_ax", "e_ay", "e_az", "e_gx", "e_gy", "e_gz"]
SIGN_NAMES = ["s_ax", "s_ay", "s_az", "s_gx", "s_gy", "s_gz"]
EKF_STEPS_PER_CANONICAL_EVAL = 27047 + 13529      # Int1: IMU predictions + GPS updates


def read_level(hisa, level):
    for name in (f"int{level}_results.csv", f"int{level}_results.csv.gz"):
        if (hisa / name).exists():
            return pd.read_csv(hisa / name)
    raise FileNotFoundError(f"int{level}_results not found in {hisa}")


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def score_vs_iteration(levels, fig_dir):
    fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 4.2), squeeze=False)
    for ax, (lv, t) in zip(axes[0], levels.items()):
        ok = t[t["ok"]]
        step = max(1, len(ok) // 60000)
        ax.scatter(ok["eval_order"][::step], ok["score"][::step], s=1, alpha=0.3, color=LEVEL_COLORS[lv],
                   rasterized=True, label="candidate score")
        ax.plot(t["eval_order"], t["best_so_far"], color="k", lw=1.5, label="best so far (this level)")
        hi = np.nanpercentile(ok["score"], 99) if len(ok) else 1
        ax.set_ylim(min(ok["score"].min(), t["best_so_far"].min()) - 0.05, hi)
        ax.set_title(f"Int{lv}: {len(t):,} candidates ({(~t['ok']).sum():,} diverged)")
        ax.set_xlabel("evaluation order within level")
        ax.set_ylabel(f"Int{lv} score: whole-flight RMSE_h (m)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("HISA candidate score vs iteration (scores comparable only within one level)")
    save(fig, fig_dir / "score_vs_iteration.png")


def stage_progression(levels, fig_dir):
    lvs = list(levels)
    fin = levels[lvs[-1]][["candidate_id", "score"]].rename(columns={"score": f"Int{lvs[-1]}"})
    for lv in lvs[:-1]:
        fin = fin.merge(levels[lv][["candidate_id", "score"]].rename(columns={"score": f"Int{lv}"}),
                        on="candidate_id")
    cols = [f"Int{lv}" for lv in lvs]
    fin = fin.sort_values(cols[-1])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    top = fin.head(40)
    for _, r in top.iterrows():
        ax1.plot(range(len(cols)), r[cols].to_numpy(dtype=float), color="#d62728", alpha=0.35, lw=1)
    ax1.plot(range(len(cols)), top[cols].median().to_numpy(), color="k", lw=2.2, label="median of top 40")
    ax1.set_xticks(range(len(cols)), cols)
    ax1.set_ylabel("score at that level (m)")
    ax1.set_title("Int25 → Int10 → Int1 scores of the 40 best Int1 candidates")
    ax1.grid(alpha=0.3)
    ax1.legend()
    rk = fin[cols].rank()
    rho = {f"{a} vs {b}": rk[a].corr(rk[b]) for a, b in zip(cols[:-1], cols[1:])}
    rho[f"{cols[0]} vs {cols[-1]}"] = rk[cols[0]].corr(rk[cols[-1]])
    ax2.scatter(fin[cols[0]], fin[cols[-1]], s=3, alpha=0.4, color="#d62728", rasterized=True)
    ax2.set_xlabel(f"{cols[0]} score (m)")
    ax2.set_ylabel(f"{cols[-1]} score (m)")
    ax2.set_title("Screening fidelity, Int1 finalists\n" +
                  ", ".join(f"Spearman {k}: {v:.2f}" for k, v in rho.items()), fontsize=9)
    ax2.grid(alpha=0.3)
    save(fig, fig_dir / "stage_progression.png")
    return rho


def bias_distributions(cand, levels, fig_dir):
    groups = {"all candidates": cand}
    for lv, t in levels.items():
        kept = t[t["retained"]] if t["retained"].any() else t.nsmallest(max(1, len(t) // 10), "score")
        label = f"kept after Int{lv}" if t["retained"].any() else f"best 10 % at Int{lv}"
        groups[label] = kept
    exps = sorted(cand["e_ax"].unique())
    fig, axes = plt.subplots(2, 6, figsize=(20, 7))
    width = 0.8 / len(groups)
    for j, (e, s) in enumerate(zip(EXP_NAMES, SIGN_NAMES)):
        for i, (label, g) in enumerate(groups.items()):
            frac = g[e].value_counts(normalize=True).reindex(exps, fill_value=0)
            axes[0, j].bar(np.arange(len(exps)) + i * width, frac, width, label=label)
            sf = g[s].value_counts(normalize=True).reindex([-1, 1], fill_value=0)
            axes[1, j].bar(np.arange(2) + i * width, sf, width, label=label)
        axes[0, j].set_xticks(np.arange(len(exps)) + 0.4 - width / 2, [f"1e{x}" for x in exps], fontsize=8)
        axes[0, j].set_title(f"|{BIAS_NAMES[j]}| decade")
        axes[1, j].set_xticks(np.arange(2) + 0.4 - width / 2, ["−", "+"])
        axes[1, j].set_title(f"sign of {BIAS_NAMES[j]}")
    axes[0, 0].set_ylabel("fraction of group")
    axes[1, 0].set_ylabel("fraction of group")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Initial-bias candidates: all admissible vs survivors of each level")
    save(fig, fig_dir / "bias_distributions.png")


def qr_search(levels, fig_dir):
    fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 4.2), squeeze=False)
    for ax, (lv, t) in zip(axes[0], levels.items()):
        piv = t.pivot_table(index="log_r_z", columns="log_r_xy", values="score", aggfunc="min")
        cnt = t.pivot_table(index="log_r_z", columns="log_r_xy", values="score", aggfunc="size")
        im = ax.imshow(piv.to_numpy(), origin="lower", cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(piv.shape[1]), [f"{v:g}" for v in piv.columns])
        ax.set_yticks(range(piv.shape[0]), [f"{v:g}" for v in piv.index])
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v, n = piv.iat[i, j], cnt.iat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.3f}\nn={int(n):,}", ha="center", va="center", fontsize=8, color="w")
        ax.set_xlabel("log10 r_xy (r_vel = r_xy / 10)")
        ax.set_ylabel("log10 r_z")
        ax.set_title(f"Int{lv}: best score per R (m)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("R search, joint with the biases (Q noise terms at THETA0)")
    save(fig, fig_dir / "qr_search.png")


def best_so_far(curve, stages, fig_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    start = 0.0
    for _, s in stages.iterrows():
        lv = int(s["integration_level"])
        c = curve[curve["integration_level"] == lv]
        end = start + s["int1_normalized_cost"]
        x = np.r_[c["int1_normalized"].to_numpy(), end]
        y = np.r_[c["best_so_far_in_level"].to_numpy(), c["best_so_far_in_level"].iloc[-1]]
        ax.step(x, y, where="post", color=LEVEL_COLORS[lv], lw=2, label=f"Int{lv} best so far")
        ax.axvline(end, color="grey", ls=":", lw=0.8)
        start = end
    ax.set_xlabel("Int1-normalized workload (Σ candidates / N)")
    ax.set_ylabel("best-so-far score at that level (m)")
    ax.set_title("HISA best-so-far; each level has its own scale, segments are not comparable")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, fig_dir / "best_so_far.png")


def ekf_error(err, fig_dir):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for col, label, color in (("gps_err_h", "raw GPS", "k"), ("baseline_err_h", "baseline EKF (THETA0)", "#555555"),
                              ("hisa_err_h", "HISA winner", "#d62728")):
        rms = np.sqrt(np.mean(err[col] ** 2))
        ax.plot(err["t"], err[col], lw=0.8, color=color, alpha=0.8, label=f"{label}: RMSE_h {rms:.3f} m")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("horizontal error (m)")
    ax.set_title("EKF error comparison, whole flight (filterpy reference)")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, fig_dir / "ekf_error_comparison.png")


def comparison(hisa, bench, fig_dir):
    runs = []
    for name in ("baseline", "GA", "PSO", "DE", "GWO"):
        p = bench / name / "runs.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["optimizer"] = name
            runs.append(df)
    h = pd.read_csv(hisa / "runs.csv")
    runs = pd.concat(runs + [h], ignore_index=True)
    runs["int1_normalized_evals"] = runs["int1_normalized_evals"].fillna(runs["n_evals"])
    runs["ekf_steps_total"] = runs["ekf_steps_total"].fillna(runs["n_evals"] * EKF_STEPS_PER_CANONICAL_EVAL)
    cols = ["rmse_h", "delta_vs_baseline", "delta_vs_gps", "mean_nis", "delta_rmse_mean", "delta_rmse_max",
            "n_evals", "int1_normalized_evals", "ekf_steps_total", "t_total"]
    g = runs.groupby("optimizer", sort=False)
    table = g[cols].mean()
    std = g[cols].std()
    table.insert(0, "runs", g.size())
    for c in cols:
        table[f"{c}_std"] = std[c]
    table.to_csv(hisa / "comparison_with_benchmark.csv")

    labels = ["Whole-flight RMSE_h (m)", "vs baseline EKF (m)", "vs raw GPS (m)", "Mean NIS (expect 6)",
              "Robustness mean ΔRMSE_h (m)", "Robustness max ΔRMSE_h (m)", "Literal evaluations",
              "Int1-normalized evaluations", "EKF steps (predictions + updates)", "T_total (s)"]
    ints = {"n_evals", "int1_normalized_evals", "ekf_steps_total"}
    lines = ["| Method | Runs | " + " | ".join(labels) + " |", "|---|---:|" + "---:|" * len(cols)]
    for name, r in table.iterrows():
        cells = []
        for c in cols:
            v, s = r[c], r[f"{c}_std"]
            f = "{:,.0f}" if c in ints else "{:.3f}"
            cells.append("–" if pd.isna(v) else f.format(v) + ("" if pd.isna(s) or s == 0 else " ± " + f.format(s)))
        lines.append(f"| {name} | {int(r['runs'])} | " + " | ".join(cells) + " |")
    note = ("\nGA, PSO, DE and GWO: 7-D log-variance decision vector, 10,000 canonical (Int1) evaluations "
            "per seed, 5 seeds. HISA: 6 initial biases + R (Q noise at THETA0), one deterministic run, "
            "Int25 → Int10 → Int1. Decision-vector dimensionality and literal evaluation counts differ; "
            "compare the final metrics (all from the filterpy reference) and the measured workload and time.\n")
    (hisa / "comparison_with_benchmark.md").write_text("\n".join(lines) + "\n" + note, encoding="utf-8")
    print(f"wrote {hisa / 'comparison_with_benchmark.md'}")

    fig, ax = plt.subplots(figsize=(9, 6))
    tmax = runs["t_total"].max()
    for name, d in runs.groupby("optimizer", sort=False):
        ax.scatter(d["rmse_h"], d["delta_rmse_mean"], s=40 + 400 * d["t_total"] / tmax,
                   color=OPT_COLORS.get(name), edgecolors="k", linewidths=0.6, alpha=0.8,
                   marker="*" if name in ("baseline", "HISA") else "o",
                   label=f"{name} (T_total {d['t_total'].mean() / 60:.1f} min)")
    ax.axvline(runs["rmse_gps"].iloc[0], color="k", ls=":", lw=1, label="raw GPS RMSE_h")
    ax.set_xlabel("Accuracy: whole-flight RMSE_h (m)")
    ax.set_ylabel("Robustness: mean ΔRMSE_h (m)")
    ax.set_title("HISA vs benchmark optimizers (marker size ∝ measured T_total)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, fig_dir / "comparison.png")


def main(hisa, bench=None):
    hisa = Path(hisa)
    fig_dir = hisa / "figures"
    fig_dir.mkdir(exist_ok=True)
    snap = json.loads((hisa / "config_snapshot.json").read_text())
    levels = {lv: read_level(hisa, lv) for lv in snap["config"]["LEVELS"]}
    cand = pd.read_csv(hisa / "candidates_all.csv.gz")
    stages = pd.read_csv(hisa / "stage_summary.csv")
    score_vs_iteration(levels, fig_dir)
    rho = stage_progression(levels, fig_dir)
    bias_distributions(cand, levels, fig_dir)
    qr_search(levels, fig_dir)
    best_so_far(pd.read_csv(hisa / "best_so_far.csv"), stages, fig_dir)
    ekf_error(pd.read_csv(hisa / "ekf_error_timeseries.csv"), fig_dir)
    (fig_dir / "screening_rank_correlation.json").write_text(json.dumps(rho, indent=2))
    if bench:
        comparison(hisa, Path(bench), fig_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hisa", default="results/HISA")
    parser.add_argument("--benchmark", help="a results/benchmark_<stamp> folder to compare against")
    a = parser.parse_args()
    main(a.hisa, a.benchmark)
