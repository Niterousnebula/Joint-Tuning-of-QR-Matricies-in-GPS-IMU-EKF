"""How each filter behaves in the degraded GPS sections of the "outages" scenario (main.py).

    python plot_sections.py                                   # every dataset of the outages scenario
    python plot_sections.py --datasets eth_outages_s0

The datasets keep the raw GPS except three sections of 250 ft of travel (make_noisy_gps.py --sections):
about 5 m added error, about 15 m, and a complete outage. For the baseline (THETA0) and the best seed
of every optimizer, rerun on the filterpy reference, per section:

    gps_rmse_h     the GPS fixes' horizontal error in the section (none in an outage)
    rmse_h, max_h  the filter's horizontal error in the section: RMS and largest
    end_h          the error when the section ends (in an outage: the drift on the IMU alone)
    after_rmse_h   RMS error over the AFTER_S seconds after the section
    recover_s      seconds from the section's end until the error is back within the filter's own RMS
                   outside every section (NaN if it never is before the next section)
    mean_nis       mean NIS of the GPS updates in the section (consistency under bad GPS; none in an outage)

plus each run's RMSE_h over the whole drive and outside the sections. Writes
results/<scenario>/<dataset>/sections.csv, sections.md and sections.png (error against time around each
section), and summary/sections.md and summary/sections.png over every dataset.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ekf
from make_noisy_gps import fix_errors
from plot_maps import COLORS, SECTION_COLORS, best_runs, section_label

AFTER_S = 30.0            # window after a section for after_rmse_h
BEFORE_PLOT_S = 30.0      # figure: seconds shown before and after each section
AFTER_PLOT_S = 60.0

_data = None


def _init(data, options):
    global _data
    _data = data
    ekf.filter.OPTIONS.update(options)          # the parent's filter settings (spawned workers start fresh)


def _run(theta):
    """CPU reference run: horizontal error at every ground-truth time and the GPS updates' (t, NIS)."""
    r = ekf.evaluate(_data, theta)
    x = r["x_log"]
    err = ekf.horizontal_errors(_data["t_imu"][:len(x)], x[:, :3], _data["t_true"], _data["p_true"])
    inn = r["innovation_log"]
    return err, inn[:, 0], inn[:, -1], r["rmse_h"]


def sections_of(dataset):
    meta = json.loads((ekf.PROCESSED_DIR / dataset / "meta.json").read_text())
    return meta.get("gps_sections", {}).get("sections", [])


def metrics(data, sections, err, t_upd, nis):
    """One row per section for one run, plus the run's RMSE_h outside the sections."""
    t = data["t_true"]
    outside = np.ones(len(t), bool)
    for s in sections:
        outside &= ~((t >= s["t_start_s"]) & (t <= s["t_end_s"] + AFTER_S))
    clean = ekf.rmse(err[outside])
    fix_err = fix_errors(data["t_gps"], data["p_gps"], data)
    rows = []
    for k, s in enumerate(sections):
        t0, t1 = s["t_start_s"], s["t_end_s"]
        inside = (t >= t0) & (t <= t1)
        after = (t > t1) & (t <= t1 + AFTER_S)
        nxt = sections[k + 1]["t_start_s"] if k + 1 < len(sections) else np.inf
        back = np.flatnonzero((t > t1) & (t < nxt) & (err <= clean))
        fixes = (data["t_gps"] >= t0) & (data["t_gps"] <= t1)
        upd = (t_upd >= t0) & (t_upd <= t1)
        rows.append({"section": k + 1, "kind": s["kind"], "label": section_label(s),
                     "gps_rmse_h": ekf.rmse(fix_err[fixes]) if fixes.any() else np.nan,
                     "rmse_h": ekf.rmse(err[inside]), "max_h": float(err[inside].max()),
                     "end_h": float(err[inside][-1]), "after_rmse_h": ekf.rmse(err[after]),
                     "recover_s": float(t[back[0]] - t1) if len(back) else np.nan,
                     "mean_nis": float(nis[upd].mean()) if upd.any() else np.nan,
                     "rmse_h_outside": clean})
    return rows


def compute(dataset, root):
    """sections.csv for one dataset, and what the figure needs."""
    sections = sections_of(dataset)
    data, runs = best_runs(dataset, root)
    with Pool(len(runs), initializer=_init, initargs=(data, dict(ekf.filter.OPTIONS))) as pool:
        out = pool.map(_run, [theta for _, _, theta in runs])
    rows, curves = [], {}
    for (name, label, _), (err, t_upd, nis, rmse_h) in zip(runs, out):
        for r in metrics(data, sections, err, t_upd, nis):
            rows.append({"dataset": dataset, "run": name, "run_label": label.replace("EKF, ", ""), "rmse_h_total": rmse_h, **r})
        curves[name] = err
    table = pd.DataFrame(rows)
    table.to_csv(Path(root) / dataset / "sections.csv", index=False)
    return data, sections, table, curves


def markdown(table):
    """One table per section: every run's numbers in that section."""
    lines = []
    for (ds, k), g in table.groupby(["dataset", "section"], sort=False):
        first = g.iloc[0]
        gps = "no fixes" if np.isnan(first["gps_rmse_h"]) else f"GPS error {first['gps_rmse_h']:.2f} m"
        lines += [f"**{ds}, section {k}: {first['label']}, {gps}**", "",
                  "| Run | RMSE_h in section (m) | Max (m) | At end (m) | Next 30 s (m) | Recovers in (s) | "
                  "Mean NIS in section | RMSE_h outside sections (m) | RMSE_h whole drive (m) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        f = lambda v, d=2: "–" if np.isnan(v) else f"{v:.{d}f}"
        for _, r in g.iterrows():
            lines.append(f"| {r['run_label']} | {f(r['rmse_h'])} | {f(r['max_h'])} | {f(r['end_h'])} | "
                         f"{f(r['after_rmse_h'])} | {f(r['recover_s'], 0)} | {f(r['mean_nis'])} | "
                         f"{f(r['rmse_h_outside'])} | {f(r['rmse_h_total'], 3)} |")
        lines.append("")
    return "\n".join(lines)


def draw_rows(axes_rows, items):
    """One row of panels per dataset, one panel per section: error against time from the section's start."""
    for axes, (dataset, data, sections, curves) in zip(axes_rows, items):
        t = data["t_true"]
        fix_err = fix_errors(data["t_gps"], data["p_gps"], data)
        for k, (ax, s) in enumerate(zip(axes, sections)):
            t0, t1 = s["t_start_s"], s["t_end_s"]
            lo, hi = t0 - BEFORE_PLOT_S, t1 + AFTER_PLOT_S
            ax.axvspan(0, t1 - t0, color=SECTION_COLORS[k % len(SECTION_COLORS)], alpha=0.18, lw=0)
            f = (data["t_gps"] >= lo) & (data["t_gps"] <= hi)
            ax.plot(data["t_gps"][f] - t0, fix_err[f], ".", ms=2.5, color=COLORS["gps"], alpha=0.6, label="GPS fix error")
            w = (t >= lo) & (t <= hi)
            for name, err in curves.items():
                ax.plot(t[w] - t0, err[w], color=COLORS[name], lw=1.3, ls="--" if name == "baseline" else "-",
                        label="baseline (THETA0)" if name == "baseline" else name)
            top = max(float(np.nanmax([e[w].max() for e in curves.values()])), 1.0)
            if s["kind"] == "noise":
                top = max(top, float(np.percentile(fix_err[f], 95)) if f.any() else 0.0)
            ax.set_ylim(0, 1.08 * top)
            ax.set_xlim(lo - t0, hi - t0)
            ax.set_title(f"{dataset}\n{section_label(s)}", fontsize=9)
            ax.set_xlabel("time from section start (s)")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("horizontal error (m)")
    axes_rows[0][0].legend(fontsize=7, loc="upper left")


def figure(items, path):
    n_sec = max(len(it[2]) for it in items)
    fig, axes = plt.subplots(len(items), n_sec, figsize=(5.2 * n_sec, 3.6 * len(items)), squeeze=False)
    draw_rows(axes, items)
    fig.suptitle("Horizontal error around each degraded GPS section (shaded): baseline and the best seed of each optimizer",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


def main(datasets, root="results"):
    root = Path(root)
    items, tables = [], []
    for ds in datasets:
        if not sections_of(ds) or not (root / ds / "baseline" / "runs.csv").exists():
            continue
        data, sections, table, curves = compute(ds, root)
        (root / ds / "sections.md").write_text(markdown(table), encoding="utf-8")
        figure([(ds, data, sections, curves)], root / ds / "sections.png")
        items.append((ds, data, sections, curves))
        tables.append(table)
    if items:
        out = root / "summary"
        out.mkdir(parents=True, exist_ok=True)
        allt = pd.concat(tables, ignore_index=True)
        allt.to_csv(out / "sections.csv", index=False)
        (out / "sections.md").write_text(markdown(allt), encoding="utf-8")
        print(f"wrote {out / 'sections.md'}")
        figure(items, out / "sections.png")


if __name__ == "__main__":
    import main as runner
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--datasets", nargs="+", default=runner.SCENARIOS["outages"])
    p.add_argument("--root", default=str(runner.results_root("outages")))
    a = p.parse_args()
    main(a.datasets, a.root)
