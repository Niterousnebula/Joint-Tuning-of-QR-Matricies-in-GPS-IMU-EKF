"""Benchmark figures and table: accuracy -> robustness -> convergence efficiency -> compute.

Reads results/baseline/runs.csv and, when present, results/<GA|PSO|DE|GWO>/runs.csv and
convergence_seed<seed>.csv, all written by ekf.save_run. Writes to results/:

    results_table.csv, results_table.md   per-optimizer mean (and std) of every dimension
    convergence.png                       best-so-far whole-flight RMSE vs evaluations
    robustness.png                        clean -> perturbed RMSE and Delta RMSE per perturbation
    response_4d.png, response_4d.html     RMSE x robustness x convergence, coloured by compute

    python plot_response.py                        # the default dataset: results/
    python plot_response.py --dataset inavwhu/street00   # results/inavwhu/street00/
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ekf import PERTURBATIONS, results_dir

RESULTS = Path("results")
GROUPS = ["baseline", "GA", "PSO", "DE", "GWO"]
COLORS = {"baseline": "#555555", "GA": "#1f77b4", "PSO": "#ff7f0e", "DE": "#2ca02c", "GWO": "#d62728"}
MARKERS = {"baseline": "*", "GA": "o", "PSO": "s", "DE": "^", "GWO": "D"}

# Table columns in the fixed order: accuracy, robustness, convergence, compute
TABLE = [("rmse_h", "Whole-flight RMSE_h (m)"),
         ("delta_vs_baseline", "vs baseline EKF (m)"),
         ("delta_vs_gps", "vs raw GPS (m)"),
         ("mean_nis", "Mean NIS (expect 6)"),
         ("delta_rmse_mean", "Robustness mean ΔRMSE_h (m)"),
         ("delta_rmse_max", "Robustness max ΔRMSE_h (m)"),
         ("evals_to_converge", "Evals to 99 % of improvement"),
         ("evals_to_best", "Evals to best"),
         ("n_evals", "N_eval"),
         ("t_total", "T_total (s)"),
         ("t_eval", "T_eval (s)")]


def load():
    runs, curves = [], {}
    for name in GROUPS:
        folder = RESULTS / name
        if not (folder / "runs.csv").exists():
            continue
        df = pd.read_csv(folder / "runs.csv")
        df["optimizer"] = name
        runs.append(df)
        curves[name] = {int(s): pd.read_csv(folder / f"convergence_seed{s}.csv")
                        for s in df["seed"] if (folder / f"convergence_seed{s}.csv").exists()}
    if not runs:
        raise SystemExit("no results/*/runs.csv found; run run_baseline.py first")
    return pd.concat(runs, ignore_index=True), curves


def write_table(runs):
    cols = [c for c, _ in TABLE if c in runs]
    grouped = runs.groupby("optimizer", sort=False)[cols]
    mean, std, count = grouped.mean(), grouped.std(), grouped.size()
    out = mean.copy()
    out.insert(0, "n_seeds", count)
    for c in cols:
        out[f"{c}_std"] = std[c]
    out.to_csv(RESULTS / "results_table.csv")

    labels = dict(TABLE)
    lines = ["| Optimizer | Seeds | " + " | ".join(labels[c] for c in cols) + " |",
             "|---|---:|" + "---:|" * len(cols)]
    for name, row in mean.iterrows():
        cells = []
        for c in cols:
            v, s = row[c], std.loc[name, c]
            fmt = "{:.0f}" if c in ("evals_to_converge", "evals_to_best", "n_evals") else "{:.3f}"
            cells.append("–" if pd.isna(v) else fmt.format(v) + ("" if pd.isna(s) else " ± " + fmt.format(s)))
        lines.append(f"| {name} | {count[name]} | " + " | ".join(cells) + " |")
    (RESULTS / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {RESULTS / 'results_table.md'} ({len(mean)} rows)")


def plot_convergence(runs, curves):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, by_seed in curves.items():
        if name == "baseline" or not by_seed:
            continue
        n = max(len(c) for c in by_seed.values())
        stack = np.full((len(by_seed), n), np.nan)
        for i, c in enumerate(by_seed.values()):
            stack[i, :len(c)] = c["best_so_far"].to_numpy()
            ax.plot(c["k"], c["best_so_far"], color=COLORS[name], alpha=0.25, lw=0.8)
        ax.plot(np.arange(1, n + 1), np.nanmedian(stack, axis=0), color=COLORS[name], lw=2.2,
                label=f"{name} (median of {len(by_seed)} seeds)")
    base = runs[runs["optimizer"] == "baseline"]
    if len(base):
        ax.axhline(base["rmse_h"].iloc[0], color=COLORS["baseline"], ls="--", lw=1.2, label="Baseline EKF (THETA0)")
        ax.axhline(base["rmse_gps"].iloc[0], color="k", ls=":", lw=1.2, label="Raw GPS")
    ax.set_xlabel("Fitness evaluations k (full-flight EKF runs)")
    ax.set_ylabel("Best-so-far whole-flight RMSE_h J*(k) (m)")
    ax.set_title("Convergence efficiency")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "convergence.png", dpi=150)
    plt.close(fig)


def plot_robustness(runs):
    names = list(PERTURBATIONS)
    groups = list(runs["optimizer"].unique())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    width = 0.8 / len(groups)
    x = np.arange(len(names) + 1)
    for i, g in enumerate(groups):
        r = runs[runs["optimizer"] == g]
        clean_and_perturbed = [r["rmse_h"].mean()] + [r[f"rmse_{n}"].mean() for n in names]
        deltas = [r[f"delta_{n}"].mean() for n in names]
        ax1.bar(x + i * width, clean_and_perturbed, width, color=COLORS.get(g), label=g)
        ax2.bar(np.arange(len(names)) + i * width, deltas, width, color=COLORS.get(g), label=g)
    ax1.set_xticks(x + 0.4 - width / 2, ["clean"] + names, rotation=25, ha="right")
    ax1.set_ylabel("Whole-flight RMSE_h (m)")
    ax1.set_title("Clean → perturbed RMSE_h")
    lo = min(runs["rmse_h"].min(), *(runs[f"rmse_{n}"].min() for n in names))
    hi = max(runs["rmse_h"].max(), *(runs[f"rmse_{n}"].max() for n in names))
    ax1.set_ylim(lo - 0.1 * (hi - lo + 0.1), hi + 0.1 * (hi - lo + 0.1))
    ax2.set_xticks(np.arange(len(names)) + 0.4 - width / 2, names, rotation=25, ha="right")
    ax2.set_ylabel("ΔRMSE_h = perturbed − clean (m)")
    ax2.set_title("Robustness (lower is better)")
    ax2.axhline(0, color="k", lw=0.8)
    for ax in (ax1, ax2):
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "robustness.png", dpi=150)
    plt.close(fig)


def compute_colour(runs):
    """Compute dimension for the colour scale: N_eval, or T_total when N_eval is equal for all optimizers."""
    opt = runs[runs["optimizer"] != "baseline"]
    if len(opt) and opt["n_evals"].nunique() > 1:
        return "n_evals", "Compute: N_eval"
    n = int(opt["n_evals"].iloc[0]) if len(opt) else 1
    return "t_total", f"Compute: T_total (s), N_eval = {n} for every optimizer"


AXES = {"x": ("rmse_h", "Accuracy: whole-flight RMSE_h (m)"),
        "y": ("delta_rmse_mean", "Robustness: mean ΔRMSE_h (m)"),
        "z": ("evals_to_converge", "Convergence: evals to 99 % of improvement")}
TITLE = "Accuracy × Robustness × Convergence efficiency × Computational cost"


def plot_4d_png(runs):
    c_col, c_label = compute_colour(runs)
    (x, xl), (y, yl), (z, zl) = AXES["x"], AXES["y"], AXES["z"]
    vmin, vmax = runs[c_col].min(), runs[c_col].max()
    if vmax - vmin < 1e-9:
        vmin, vmax = vmin - 1.0, vmax + 1.0
    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(projection="3d")
    sc = None
    for name, g in runs.groupby("optimizer", sort=False):
        sc = ax.scatter(g[x], g[y], g[z], c=g[c_col], cmap="viridis", vmin=vmin, vmax=vmax,
                        marker=MARKERS.get(name, "o"), s=180 if name == "baseline" else 60,
                        edgecolors="k", linewidths=0.6, label=name, depthshade=False)
    ax.set_xlabel(xl, labelpad=10)
    ax.set_ylabel(yl, labelpad=10)
    ax.set_zlabel(zl, labelpad=10)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.12)
    cb.set_label(c_label)
    gps = runs["rmse_gps"].iloc[0]
    ax.set_title(f"{TITLE}\nRaw GPS RMSE_h = {gps:.3f} m", fontsize=11)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS / "response_4d.png", dpi=150)
    plt.close(fig)


def plot_4d_html(runs):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed, skipping response_4d.html")
        return
    c_col, c_label = compute_colour(runs)
    (x, xl), (y, yl), (z, zl) = AXES["x"], AXES["y"], AXES["z"]
    symbols = {"baseline": "diamond-open", "GA": "circle", "PSO": "square", "DE": "cross", "GWO": "diamond"}
    cmin, cmax = runs[c_col].min(), runs[c_col].max()
    fig = go.Figure()
    for i, (name, g) in enumerate(runs.groupby("optimizer", sort=False)):
        text = [f"{name} seed {s}<br>vs GPS {d:+.3f} m<br>N_eval {n}<br>T_total {t:.1f} s"
                for s, d, n, t in zip(g["seed"], g["delta_vs_gps"], g["n_evals"], g["t_total"])]
        fig.add_trace(go.Scatter3d(
            x=g[x], y=g[y], z=g[z], mode="markers", name=name, text=text,
            marker=dict(size=10 if name == "baseline" else 6, symbol=symbols.get(name, "circle"),
                        color=g[c_col], colorscale="Viridis", cmin=cmin, cmax=cmax,
                        showscale=i == 0, colorbar=dict(title=c_label)),
            hovertemplate="%{text}<br>RMSE_h %{x:.3f} m<br>ΔRMSE_h %{y:.3f} m"
                          "<br>Evals to converge %{z}<extra></extra>"))
    fig.update_layout(title=f"{TITLE}<br><sup>Raw GPS RMSE_h = {runs['rmse_gps'].iloc[0]:.3f} m</sup>",
                      scene=dict(xaxis_title=xl, yaxis_title=yl, zaxis_title=zl))
    fig.write_html(RESULTS / "response_4d.html", include_plotlyjs="cdn")


def main():
    global RESULTS
    parser = argparse.ArgumentParser(description="Benchmark figures and table")
    parser.add_argument("--dataset", default="eth", help="processed dataset id whose results to plot")
    RESULTS = results_dir(parser.parse_args().dataset)
    runs, curves = load()
    write_table(runs)
    plot_convergence(runs, curves)
    plot_robustness(runs)
    plot_4d_png(runs)
    plot_4d_html(runs)


if __name__ == "__main__":
    main()
