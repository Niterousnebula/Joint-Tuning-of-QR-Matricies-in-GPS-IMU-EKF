"""Tables, graphs and 4D response surfaces from run_all.py's results.

    python plot_response.py                       # every dataset under results/, then results/summary/
    python plot_response.py --root results_smoke  # the smoke-test results

Per dataset (results/<dataset>/), from runs.csv and convergence_seed<s>.csv of baseline, GA,
PSO, DE and GWO (written by ekf.save_run):
    results_table.md/.csv   mean +- std per optimizer: accuracy, robustness, convergence, compute
    convergence.png         best-so-far RMSE_h against evaluations, every seed and the median
    robustness.png          clean -> perturbed RMSE_h and delta RMSE_h per perturbation

results/summary/:
    results_table.md/.csv            every dataset and optimizer in one table
    best_values.md/.csv              the best seed of each optimizer on each dataset, with its theta
    response_surface_<OPT>.png/.html one 4D response surface per optimizer, one panel per dataset

The 4D response surface of an optimizer on a dataset uses every evaluation of every seed: at each
evaluation k a seed has its best-so-far accuracy A(k) (whole-flight RMSE_h), the robustness R(k)
of that best-so-far theta (mean delta RMSE_h over the perturbations, from incumbents_seed<s>.csv)
and its wall-clock time T(k). Each seed is a path (A, R, k) through accuracy x robustness x
convergence; the surface is spanned by the five paths (seeds ordered by final accuracy and
interpolated between neighbours), and coloured by compute T. The seed paths are drawn on it.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors

from ekf import PERTURBATIONS, THETA_NAMES
from ekf.response import response_path

OPTIMIZERS = ["GA", "PSO", "DE", "GWO"]
GROUPS = ["baseline", *OPTIMIZERS]
COLORS = {"baseline": "#555555", "GA": "#1f77b4", "PSO": "#ff7f0e", "DE": "#2ca02c", "GWO": "#d62728"}

# Table columns in the fixed order: accuracy, robustness, convergence, compute
TABLE = [("rmse_h", "Whole-flight RMSE_h (m)"),
         ("delta_vs_baseline", "vs baseline EKF (m)"),
         ("delta_vs_gps", "vs raw GPS (m)"),
         ("mean_nis", "Mean NIS"),
         ("delta_rmse_mean", "Robustness mean ΔRMSE_h (m)"),
         ("delta_rmse_max", "Robustness max ΔRMSE_h (m)"),
         ("evals_to_converge", "Evals to 99 % of improvement"),
         ("evals_to_best", "Evals to best"),
         ("n_evals", "N_eval"),
         ("t_total", "T_total (s)"),
         ("t_objective_per_eval", "T_objective per eval (s)")]
INTS = {"evals_to_converge", "evals_to_best", "n_evals"}


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def load_runs(out, groups=GROUPS):
    runs, curves = [], {}
    for name in groups:
        folder = out / name
        if not (folder / "runs.csv").exists():
            continue
        df = pd.read_csv(folder / "runs.csv")
        df["optimizer"] = name
        runs.append(df)
        curves[name] = {int(s): pd.read_csv(folder / f"convergence_seed{s}.csv")
                        for s in df["seed"] if (folder / f"convergence_seed{s}.csv").exists()}
    return (pd.concat(runs, ignore_index=True) if runs else pd.DataFrame()), curves


def fmt_cell(v, s, c):
    f = "{:,.0f}" if c in INTS else "{:.3f}"
    if pd.isna(v):
        return "–"
    return f.format(v) + ("" if s is None or pd.isna(s) or s == 0 else " ± " + f.format(s))


def table(runs):
    cols = [c for c, _ in TABLE if c in runs]
    g = runs.groupby("optimizer", sort=False)[cols]
    mean, std, count = g.mean(), g.std(), g.size()
    out = mean.copy()
    out.insert(0, "runs", count)
    for c in cols:
        out[f"{c}_std"] = std[c]
    return out, cols


# ---------------------------------------------------------------------------
# One dataset
# ---------------------------------------------------------------------------
def dataset_figures(out, dataset, optimizers=OPTIMIZERS):
    out = Path(out)
    runs, curves = load_runs(out, ["baseline", *optimizers])
    if runs.empty:
        print(f"{out}: no runs.csv yet")
        return
    tab, cols = table(runs)
    tab.to_csv(out / "results_table.csv")
    dof = int(runs["nis_dof"].iloc[0]) if "nis_dof" in runs else None
    labels = dict(TABLE)
    if dof:
        labels["mean_nis"] = f"Mean NIS (expect {dof})"
    lines = [f"Dataset `{dataset}`. Raw GPS RMSE_h {runs['rmse_gps'].iloc[0]:.3f} m.", "",
             "| Optimizer | Runs | " + " | ".join(labels[c] for c in cols) + " |",
             "|---|---:|" + "---:|" * len(cols)]
    for name, r in tab.iterrows():
        lines.append(f"| {name} | {int(r['runs'])} | " + " | ".join(fmt_cell(r[c], r[f'{c}_std'], c) for c in cols) + " |")
    (out / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out / 'results_table.md'}")
    plot_convergence(out, dataset, runs, curves)
    plot_robustness(out, dataset, runs)


def plot_convergence(out, dataset, runs, curves):
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
    ax.set_title(f"Convergence efficiency: {dataset}")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, out / "convergence.png")


def plot_robustness(out, dataset, runs):
    names = [n for n in PERTURBATIONS if f"rmse_{n}" in runs]
    groups = list(runs["optimizer"].unique())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    width = 0.8 / len(groups)
    x = np.arange(len(names) + 1)
    for i, g in enumerate(groups):
        r = runs[runs["optimizer"] == g]
        ax1.bar(x + i * width, [r["rmse_h"].mean()] + [r[f"rmse_{n}"].mean() for n in names], width,
                color=COLORS.get(g), label=g)
        ax2.bar(np.arange(len(names)) + i * width, [r[f"delta_{n}"].mean() for n in names], width,
                color=COLORS.get(g), label=g)
    ax1.set_xticks(x + 0.4 - width / 2, ["clean"] + names, rotation=25, ha="right")
    ax1.set_ylabel("Whole-flight RMSE_h (m)")
    ax1.set_title(f"Clean → perturbed RMSE_h: {dataset}")
    vals = np.r_[runs["rmse_h"].to_numpy(), *[runs[f"rmse_{n}"].to_numpy() for n in names]]
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    ax1.set_ylim(lo - 0.1 * (hi - lo + 0.1), hi + 0.1 * (hi - lo + 0.1))
    ax2.set_xticks(np.arange(len(names)) + 0.4 - width / 2, names, rotation=25, ha="right")
    ax2.set_ylabel("ΔRMSE_h = perturbed − clean (m)")
    ax2.set_title("Robustness (lower is better)")
    ax2.axhline(0, color="k", lw=0.8)
    for ax in (ax1, ax2):
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    save(fig, out / "robustness.png")


# ---------------------------------------------------------------------------
# 4D response surfaces
# ---------------------------------------------------------------------------
def response_surface(folder, n_cols=120, n_rows=25, smooth=9):
    """Accuracy, robustness, log10(k) and wall-clock grids spanned by the seeds' paths, and the paths."""
    runs = pd.read_csv(folder / "runs.csv")
    paths = {}
    for s in sorted(runs["seed"].astype(int)):
        conv_p, inc_p = folder / f"convergence_seed{s}.csv", folder / f"incumbents_seed{s}.csv"
        if conv_p.exists() and inc_p.exists():
            paths[s] = response_path(pd.read_csv(conv_p), pd.read_csv(inc_p))
    if len(paths) < 2:
        return None
    n = min(len(p) for p in paths.values())
    ks = np.unique(np.round(np.geomspace(1, n, n_cols)).astype(int))
    order = sorted(paths, key=lambda s: paths[s]["accuracy"].iloc[n - 1], reverse=True)   # worst -> best
    rows = {key: np.array([paths[s][key].to_numpy()[ks - 1] for s in order])
            for key in ("accuracy", "robustness", "elapsed")}
    # Robustness jumps whenever the incumbent changes; a moving mean over neighbouring k columns
    # (log-spaced, window `smooth`) turns the steps into a surface. The raw paths are drawn on top.
    pad = smooth // 2
    padded = np.pad(rows["robustness"], ((0, 0), (pad, pad)), mode="edge")
    rows["robustness"] = np.array([np.convolve(r, np.ones(smooth) / smooth, mode="valid") for r in padded])
    u = np.linspace(0, len(order) - 1, n_rows)                  # interpolate between neighbouring seeds
    lo = np.floor(u).astype(int).clip(0, len(order) - 2)
    w = (u - lo)[:, None]
    grid = {key: (1 - w) * v[lo] + w * v[lo + 1] for key, v in rows.items()}
    grid["logk"] = np.tile(np.log10(ks), (n_rows, 1))
    return grid, paths, order


def surface_figures(root, datasets, name):
    panels = []
    for ds in datasets:
        folder = root / ds / name
        if (folder / "runs.csv").exists():
            res = response_surface(folder)
            if res is not None:
                base = root / ds / "baseline" / "runs.csv"
                panels.append((ds, res, pd.read_csv(base).iloc[0] if base.exists() else None))
    if not panels:
        print(f"{name}: no incumbents files yet, no response surface")
        return
    summary_dir = root / "summary"
    fig = plt.figure(figsize=(6.4 * len(panels), 6.2))
    for i, (ds, (g, paths, order), base) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, len(panels), i, projection="3d")
        norm = colors.Normalize(g["elapsed"].min(), g["elapsed"].max())
        ax.plot_surface(g["accuracy"], g["robustness"], g["logk"], facecolors=cm.viridis(norm(g["elapsed"])),
                        rstride=1, cstride=1, shade=False, linewidth=0, antialiased=True, alpha=0.85)
        for s in order:
            p = paths[s]
            ax.plot(p["accuracy"], p["robustness"], np.log10(p["k"]), color="k", lw=0.7, alpha=0.8)
        if base is not None:
            ax.scatter([base["rmse_h"]], [base["delta_rmse_mean"]], [0], marker="*", s=140, color="#555555",
                       edgecolors="k", label="baseline (THETA0)")
            ax.legend(loc="upper left", fontsize=8)
        k_max = int(round(10 ** g["logk"].max()))
        ticks = [t for t in (1, 10, 100, 1000, 10000) if 2.5 * t < k_max] + [k_max]
        ax.set_zticks(np.log10(ticks), [f"{t:,}" for t in ticks])
        ax.set_xlabel("Accuracy: best-so-far RMSE_h (m)", labelpad=8)
        ax.set_ylabel("Robustness: mean ΔRMSE_h (m)", labelpad=8)
        ax.set_zlabel("Convergence: evaluations k", labelpad=8)
        ax.set_title(f"{ds}  ({len(order)} seeds)", fontsize=10)
        cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, shrink=0.55, pad=0.16)
        cb.set_label("Compute: wall-clock time (s)")
    fig.suptitle(f"{name}: 4D response surface: accuracy × robustness × convergence, colour = compute\n"
                 "(every evaluation of every seed; robustness smoothed over neighbouring k; "
                 "black lines: the seeds' raw paths)", fontsize=11)
    save(fig, summary_dir / f"response_surface_{name}.png")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return
    pf = make_subplots(rows=1, cols=len(panels), specs=[[{"type": "scene"}] * len(panels)],
                       subplot_titles=[ds for ds, _, _ in panels])
    for i, (ds, (g, paths, order), base) in enumerate(panels, start=1):
        pf.add_trace(go.Surface(x=g["accuracy"], y=g["robustness"], z=g["logk"], surfacecolor=g["elapsed"],
                                colorscale="Viridis", opacity=0.9, name=ds,
                                colorbar=dict(title="wall clock (s)", x=i / len(panels) - 0.03, len=0.7),
                                hovertemplate="RMSE_h %{x:.3f} m<br>ΔRMSE_h %{y:.3f} m<br>log10 k %{z:.2f}"
                                              "<br>t %{surfacecolor:.0f} s<extra></extra>"), row=1, col=i)
        for s in order:
            p = paths[s]
            pf.add_trace(go.Scatter3d(x=p["accuracy"], y=p["robustness"], z=np.log10(p["k"]), mode="lines",
                                      line=dict(color="black", width=2), name=f"{ds} seed {s}", showlegend=False),
                         row=1, col=i)
    pf.update_scenes(xaxis_title="RMSE_h (m)", yaxis_title="ΔRMSE_h (m)", zaxis_title="log10 evaluations")
    pf.update_layout(title=f"{name}: 4D response surface (accuracy × robustness × convergence, colour = compute)",
                     height=650, width=620 * len(panels))
    pf.write_html(summary_dir / f"response_surface_{name}.html", include_plotlyjs="cdn")
    print(f"wrote {summary_dir / f'response_surface_{name}.html'}")


# ---------------------------------------------------------------------------
# Summary over every dataset
# ---------------------------------------------------------------------------
def summary(root, datasets, optimizers=OPTIMIZERS):
    root = Path(root)
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    rows, best = [], []
    for ds in datasets:
        runs, _ = load_runs(root / ds, ["baseline", *optimizers])
        if runs.empty:
            continue
        tab, _ = table(runs)
        tab.insert(0, "dataset", ds)
        tab["rmse_gps"] = runs["rmse_gps"].iloc[0]
        tab["nis_dof"] = runs["nis_dof"].iloc[0] if "nis_dof" in runs else np.nan
        rows.append(tab)
        for name in optimizers:
            r = runs[runs["optimizer"] == name]
            if len(r):
                b = r.loc[r["rmse_h"].idxmin()]
                best.append({"dataset": ds, "optimizer": name, "seed": int(b["seed"]),
                             **{c: b[c] for c in ("rmse_h", "delta_vs_gps", "mean_nis", "delta_rmse_mean",
                                                  "delta_rmse_max", *THETA_NAMES)}})
    if not rows:
        print(f"{root}: nothing to summarise yet")
        return
    allt = pd.concat(rows)
    allt.to_csv(out / "results_table.csv")
    keep = ["rmse_h", "delta_vs_baseline", "delta_vs_gps", "mean_nis", "delta_rmse_mean", "evals_to_converge", "t_total"]
    lines = ["| Dataset | Method | Runs | RMSE_h (m) | vs baseline (m) | vs raw GPS (m) | Mean NIS (dof) | "
             "Robustness mean ΔRMSE_h (m) | Evals to 99 % | T_total (s) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in allt.iterrows():
        cells = [fmt_cell(r.get(c), r.get(f"{c}_std"), c) for c in keep]
        if not pd.isna(r["nis_dof"]):
            cells[3] = f"{cells[3]} ({int(r['nis_dof'])})"
        lines.append(f"| {r['dataset']} | {name} | {int(r['runs'])} | " + " | ".join(cells) + " |")
    (out / "results_table.md").write_text("Mean ± std over seeds.\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out / 'results_table.md'}")
    if best:
        b = pd.DataFrame(best)
        b.to_csv(out / "best_values.csv", index=False)
        head = ("| Dataset | Optimizer (seed) | RMSE_h (m) | vs GPS (m) | NIS | Robustness mean / max (m) | "
                + " | ".join(f"`{t}`" for t in THETA_NAMES) + " |")
        lines = [head, "|---|---|" + "---:|" * (4 + len(THETA_NAMES))]
        for _, r in b.iterrows():
            lines.append(f"| {r['dataset']} | {r['optimizer']} ({r['seed']}) | {r['rmse_h']:.3f} | {r['delta_vs_gps']:+.3f} | "
                         f"{r['mean_nis']:.2f} | {r['delta_rmse_mean']:+.3f} / {r['delta_rmse_max']:+.3f} | "
                         + " | ".join(f"{r[t]:.2f}" for t in THETA_NAMES) + " |")
        (out / "best_values.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {out / 'best_values.md'}")
    for name in optimizers:
        surface_figures(root, datasets, name)


def build_all(root="results", datasets=None, optimizers=OPTIMIZERS):
    root = Path(root)
    if datasets is None:
        datasets = sorted(str(p.parent.parent.relative_to(root)).replace("\\", "/")
                          for p in root.glob("**/baseline/runs.csv"))
    for ds in datasets:
        if (root / ds).exists():
            dataset_figures(root / ds, ds, optimizers)
    summary(root, datasets, optimizers)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="results")
    ap.add_argument("--datasets", nargs="+", help="default: every dataset folder under --root")
    a = ap.parse_args()
    build_all(a.root, a.datasets)
