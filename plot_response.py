"""Tables and graphs from main.py's results.

    python plot_response.py                       # every dataset under results/, then results/summary/
    python plot_response.py --root results_smoke  # the smoke-test results

Per dataset (results/<dataset>/), from runs.csv and convergence_seed<s>.csv of baseline, GA,
PSO, DE and GWO (written by ekf.save_run):
    results_table.md/.csv   mean +- std per optimizer: accuracy, consistency, convergence, compute
    convergence.png         best-so-far RMSE_h against evaluations, every seed and the median

results/summary/ (one panel per dataset):
    results_table.md/.csv        every dataset and optimizer in one table
    best_values.md/.csv          the best seed of each optimizer on each dataset, with its theta
    convergence.png              each optimizer's median best-so-far RMSE_h, band = best to worst seed
    accuracy_vs_convergence.png  final RMSE_h against evaluations to converge, one dot per seed,
                                 a diamond per optimizer mean, mean NIS in the legend
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ekf import THETA_NAMES

OPTIMIZERS = ["GA", "PSO", "DE", "GWO"]
GROUPS = ["baseline", *OPTIMIZERS]
COLORS = {"baseline": "#555555", "GA": "#1f77b4", "PSO": "#ff7f0e", "DE": "#2ca02c", "GWO": "#d62728"}

# Table columns in the fixed order: accuracy, consistency, convergence, compute
TABLE = [("rmse_h", "Whole-flight RMSE_h (m)"),
         ("delta_vs_baseline", "vs baseline EKF (m)"),
         ("delta_vs_gps", "vs raw GPS (m)"),
         ("mean_nis", "Mean NIS"),
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


# ---------------------------------------------------------------------------
# Summary figures: one panel per dataset
# ---------------------------------------------------------------------------
def summary_convergence(root, datasets, optimizers, out):
    """Best-so-far RMSE_h against evaluations: each optimizer's median over seeds, band = min to max."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.8), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        runs, curves = load_runs(root / ds, ["baseline", *optimizers])
        for name in optimizers:
            by_seed = curves.get(name, {})
            if not by_seed:
                continue
            n = min(len(c) for c in by_seed.values())
            stack = np.array([c["best_so_far"].to_numpy()[:n] for c in by_seed.values()])
            k = np.arange(1, n + 1)
            ax.fill_between(k, stack.min(axis=0), stack.max(axis=0), color=COLORS[name], alpha=0.15, lw=0)
            ax.plot(k, np.median(stack, axis=0), color=COLORS[name], lw=2, label=name)
        base = runs[runs["optimizer"] == "baseline"]
        if len(base):
            ax.axhline(base["rmse_h"].iloc[0], color=COLORS["baseline"], ls="--", lw=1.2, label="baseline (THETA0)")
        final = runs[runs["optimizer"] != "baseline"]["rmse_h"]
        if len(final) and len(base):                 # zoom on the part where the optimizers differ
            lo, hi = final.min(), base["rmse_h"].iloc[0]
            ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.25 * (hi - lo))
        ax.set_title(f"{ds}\nraw GPS RMSE_h {runs['rmse_gps'].iloc[0]:.2f} m", fontsize=10)
        ax.set_xlabel("evaluations k")
        ax.set_ylabel("best-so-far RMSE_h (m)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Convergence: median of 5 seeds, band = best to worst seed", fontsize=11)
    fig.tight_layout()
    save(fig, out / "convergence.png")


def summary_tradeoff(root, datasets, optimizers, out):
    """Final accuracy against convergence speed: one dot per seed, a diamond per optimizer mean."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.8), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        runs, _ = load_runs(root / ds, ["baseline", *optimizers])
        for name in optimizers:
            r = runs[runs["optimizer"] == name]
            if not len(r):
                continue
            ax.scatter(r["rmse_h"], r["evals_to_converge"], s=28, color=COLORS[name], alpha=0.45)
            ax.scatter(r["rmse_h"].mean(), r["evals_to_converge"].mean(), s=150, marker="D", color=COLORS[name],
                       edgecolors="k", label=f"{name}  (mean NIS {r['mean_nis'].mean():.2f})")
        base = runs[runs["optimizer"] == "baseline"]
        if len(base):
            ax.axvline(base["rmse_h"].iloc[0], color=COLORS["baseline"], ls="--", lw=1.2,
                       label=f"baseline {base['rmse_h'].iloc[0]:.2f} m")
        dof = int(runs["nis_dof"].iloc[0]) if "nis_dof" in runs else None
        ax.set_title(f"{ds}" + (f"  (consistent filter: NIS = {dof})" if dof else ""), fontsize=10)
        ax.set_xlabel("final RMSE_h (m)   ← better")
        ax.set_ylabel("evaluations to 99 % of improvement   ← faster")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Accuracy vs convergence speed: dots = seeds, diamonds = optimizer means", fontsize=11)
    fig.tight_layout()
    save(fig, out / "accuracy_vs_convergence.png")


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
                             **{c: b[c] for c in ("rmse_h", "delta_vs_gps", "mean_nis", *THETA_NAMES)}})
    if not rows:
        print(f"{root}: nothing to summarise yet")
        return
    allt = pd.concat(rows)
    allt.to_csv(out / "results_table.csv")
    keep = ["rmse_h", "delta_vs_baseline", "delta_vs_gps", "mean_nis", "evals_to_converge", "t_total"]
    lines = ["| Dataset | Method | Runs | RMSE_h (m) | vs baseline (m) | vs raw GPS (m) | Mean NIS (dof) | "
             "Evals to 99 % | T_total (s) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
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
        head = ("| Dataset | Optimizer (seed) | RMSE_h (m) | vs GPS (m) | NIS | "
                + " | ".join(f"`{t}`" for t in THETA_NAMES) + " |")
        lines = [head, "|---|---|" + "---:|" * (3 + len(THETA_NAMES))]
        for _, r in b.iterrows():
            lines.append(f"| {r['dataset']} | {r['optimizer']} ({r['seed']}) | {r['rmse_h']:.3f} | {r['delta_vs_gps']:+.3f} | "
                         f"{r['mean_nis']:.2f} | "
                         + " | ".join(f"{r[t]:.2f}" for t in THETA_NAMES) + " |")
        (out / "best_values.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {out / 'best_values.md'}")
    shown = [ds for ds in datasets if (root / ds / "baseline" / "runs.csv").exists()]
    summary_convergence(root, shown, optimizers, out)
    summary_tradeoff(root, shown, optimizers, out)


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
