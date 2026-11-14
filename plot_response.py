"""4D response graph: RMSE x robustness x compute, coloured by wall time.

Reads results/baseline/baseline_parameters.csv and, when present,
results/<OPTIMIZER>/optimized_parameters.csv for GA, PSO, DE and GWO, and writes
results/response_4d.html (interactive) and results/response_4d.png.

    python plot_response.py

Each optimizer file holds one row per run with at least these columns:
    optimizer, seed, rmse_h_held_out, robustness_std, evals_to_best, wall_time
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("results")
OPTIMIZERS = ["GA", "PSO", "DE", "GWO"]
COLUMNS = ["optimizer", "seed", "rmse_h_held_out", "robustness_std", "evals_to_best", "wall_time"]
MARKERS = {"baseline": "*", "GA": "o", "PSO": "s", "DE": "^", "GWO": "D"}
AXES = {"rmse_h_held_out": "Held-out RMSE (m)",
        "robustness_std": "Robustness: std of RMSE over datasets (m)",
        "evals_to_best": "Compute: evals to best",
        "wall_time": "Wall time (s)"}


def load_runs():
    frames = [pd.read_csv(RESULTS / "baseline" / "baseline_parameters.csv")[COLUMNS]]
    for name in OPTIMIZERS:
        path = RESULTS / name / "optimized_parameters.csv"
        if path.exists():
            runs = pd.read_csv(path)
            if "bounds" in runs:
                runs = runs[runs["bounds"] == "baseline"]
            frames.append(runs[COLUMNS])
    return pd.concat(frames, ignore_index=True)


def plot_png(runs, path):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    vmin, vmax = runs["wall_time"].min(), runs["wall_time"].max()
    if vmax - vmin < 1e-6:
        vmin, vmax = vmin - 1.0, vmax + 1.0
    sc = None
    for name, group in runs.groupby("optimizer", sort=False):
        sc = ax.scatter(group["rmse_h_held_out"], group["robustness_std"], group["evals_to_best"],
                        c=group["wall_time"], cmap="viridis", vmin=vmin, vmax=vmax,
                        marker=MARKERS.get(name, "o"), s=160 if name == "baseline" else 60,
                        edgecolors="k", linewidths=0.5, label=name)
    ax.set_xlabel(AXES["rmse_h_held_out"])
    ax.set_ylabel(AXES["robustness_std"])
    ax.set_zlabel(AXES["evals_to_best"])
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.12, label=AXES["wall_time"])
    ax.legend(loc="upper left")
    ax.set_title("Optimizer response: RMSE × robustness × compute × time")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_html(runs, path):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed, skipping the interactive graph")
        return
    symbols = {"baseline": "diamond-open", "GA": "circle", "PSO": "square", "DE": "cross", "GWO": "diamond"}
    fig = go.Figure()
    cmin, cmax = runs["wall_time"].min(), runs["wall_time"].max()
    for i, (name, group) in enumerate(runs.groupby("optimizer", sort=False)):
        fig.add_trace(go.Scatter3d(
            x=group["rmse_h_held_out"], y=group["robustness_std"], z=group["evals_to_best"],
            mode="markers", name=name,
            text=[f"{name} seed {s}" for s in group["seed"]],
            marker=dict(size=10 if name == "baseline" else 6, symbol=symbols.get(name, "circle"),
                        color=group["wall_time"], colorscale="Viridis", cmin=cmin, cmax=cmax,
                        showscale=i == 0, colorbar=dict(title=AXES["wall_time"])),
            hovertemplate="%{text}<br>RMSE %{x:.3f} m<br>Robustness %{y:.3f} m"
                          "<br>Evals to best %{z}<br>Time %{marker.color:.1f} s<extra></extra>"))
    fig.update_layout(title="Optimizer response: RMSE × robustness × compute × time",
                      scene=dict(xaxis_title=AXES["rmse_h_held_out"], yaxis_title=AXES["robustness_std"],
                                 zaxis_title=AXES["evals_to_best"]))
    fig.write_html(path, include_plotlyjs="cdn")


def main():
    runs = load_runs()
    summary = runs.groupby("optimizer", sort=False)[COLUMNS[2:]].agg(["mean", "std"])
    summary.to_csv(RESULTS / "response_summary.csv")
    print(summary.round(3).to_string())
    plot_png(runs, RESULTS / "response_4d.png")
    plot_html(runs, RESULTS / "response_4d.html")


if __name__ == "__main__":
    main()
