"""Maps of each run on OpenStreetMap: ground truth, the GPS fixes, the baseline EKF (THETA0) and
the best tuned run of every optimizer.

    python plot_maps.py                                   # every dataset of main.py's scenario
    python plot_maps.py --root results/gps16m --datasets eth_gps16m_nov_s0

Writes results/<scenario>/<dataset>/map.png (static, OSM tiles: one map per filter, baseline, GA, PSO, DE and GWO,
each with the GPS fixes and the ground truth) and map.html (interactive, all tracks in one map:
pan, zoom and toggle tracks; needs a browser with internet). The EKF tracks are rerun on the CPU
reference with the THETA0 and best thetas from <root>/<dataset>/<OPT>/runs.csv. Map tiles come from
tile.openstreetmap.org and are cached in .tile_cache/; without internet the PNG is drawn without them.
"""
import argparse
import io
import json
import math
import urllib.request
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ekf
from data_pipeline.geo import enu_to_latlon

OPTIMIZERS = ["GA", "PSO", "DE", "GWO"]
COLORS = {"truth": "#111111", "gps": "#7f7f7f", "baseline": "#555555",
          "GA": "#1f77b4", "PSO": "#ff7f0e", "DE": "#2ca02c", "GWO": "#d62728"}
SECTION_COLORS = ["#e6ab02", "#e7298a", "#7570b3"]        # degraded GPS sections, in order
TILE_CACHE = Path(".tile_cache")


def section_label(sec):
    """'5 m GPS noise', '15 m GPS noise' or 'GPS outage', with the section's length and duration."""
    what = f"{sec['sigma_h_m']:g} m GPS noise" if sec["kind"] == "noise" else "GPS outage"
    return f"{what} ({sec['length_m']:g} m, {sec['duration_s']:.0f} s)"
MAX_POINTS = 5000            # per track, enough for a map


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------
_data = None


def _init(data, options):
    global _data
    _data = data
    ekf.filter.OPTIONS.update(options)          # the parent's filter settings (spawned workers start fresh)


def _track(theta):
    """CPU reference run: horizontal ENU positions (thinned) and the whole-flight RMSE_h."""
    r = ekf.evaluate(_data, theta)
    x = r["x_log"]
    step = max(1, len(x) // MAX_POINTS)
    return x[::step, :2], r["rmse_h"]


def best_runs(dataset, root):
    """The dataset and [(name, label, theta)] for the baseline and the best seed (lowest objective) of every optimizer."""
    data = ekf.load_data(dataset)
    runs = [("baseline", "EKF, THETA0 (baseline)", data["theta0"])]
    for name in OPTIMIZERS:
        path = Path(root) / dataset / name / "runs.csv"
        if path.exists():
            r = pd.read_csv(path)
            b = r.loc[r["fitness" if "fitness" in r and r.get("nis_weight", pd.Series([0])).iloc[0] else "rmse_h"].idxmin()]
            runs.append((name, f"EKF, {name} best (seed {int(b['seed'])})", b[ekf.THETA_NAMES].to_numpy(dtype=float)))
    return data, runs


def tracks(dataset, root):
    """[(name, label, ENU array, rmse)] for the baseline and the best seed of every optimizer."""
    data, runs = best_runs(dataset, root)
    with Pool(len(runs), initializer=_init, initargs=(data, dict(ekf.filter.OPTIONS))) as pool:
        out = pool.map(_track, [t for _, _, t in runs])
    return data, [(n, lab, xy, rm) for (n, lab, _), (xy, rm) in zip(runs, out)]


# ---------------------------------------------------------------------------
# OSM tiles for the static map (web-mercator, slippy-map tile numbering)
# ---------------------------------------------------------------------------
def _merc(lat, lon):
    """Web-mercator x, y in metres."""
    r = 6378137.0
    return r * np.radians(lon), r * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def _tile(z, x, y):
    path = TILE_CACHE / f"{z}_{x}_{y}.png"
    if not path.exists():
        req = urllib.request.Request(f"https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                                     headers={"User-Agent": "EKF-QR-tuning-maps/1.0"})   # OSM tile policy
        TILE_CACHE.mkdir(exist_ok=True)
        path.write_bytes(urllib.request.urlopen(req, timeout=20).read())
    return plt.imread(io.BytesIO(path.read_bytes()), format="png")


def basemap(ax, lat_min, lat_max, lon_min, lon_max, max_tiles=6):
    """Draw OSM tiles covering the box (web-mercator axes); False if they could not be fetched."""
    def tile_xy(lat, lon, z):
        n = 2 ** z
        return (int((lon + 180) / 360 * n),
                int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n))
    z = 19
    while z > 1:                                                # largest zoom that fits max_tiles across
        x0, y0 = tile_xy(lat_max, lon_min, z)
        x1, y1 = tile_xy(lat_min, lon_max, z)
        if max(x1 - x0, y1 - y0) + 1 <= max_tiles:
            break
        z -= 1
    try:
        rows = [np.concatenate([_tile(z, x, y)[..., :3] for x in range(x0, x1 + 1)], axis=1) for y in range(y0, y1 + 1)]
    except OSError as err:
        print(f"  no OSM tiles ({err}); map drawn without them")
        return False
    n = 2 ** z
    lon_l, lon_r = x0 / n * 360 - 180, (x1 + 1) / n * 360 - 180
    lat_t = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y0 / n))))
    lat_b = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y1 + 1) / n))))
    (xl, yb), (xr, yt) = _merc(lat_b, lon_l), _merc(lat_t, lon_r)
    ax.imshow(np.concatenate(rows, axis=0), extent=(xl, xr, yb, yt), zorder=0, interpolation="bilinear")
    return True


# ---------------------------------------------------------------------------
def draw(dataset, root):
    data, runs = tracks(dataset, root)
    meta = json.loads((ekf.PROCESSED_DIR / dataset / "meta.json").read_text())
    lat0, lon0 = meta["gps_origin_lat_lon_alt"][:2]
    ll = lambda xy: enu_to_latlon(xy[:, 0], xy[:, 1], lat0, lon0)
    truth = ll(data["p_true"][:, :2])
    gps = ll(data["p_gps"][:, :2])
    gps_rmse = ekf.gps_rmse(data)
    sections = meta.get("gps_sections", {}).get("sections", [])
    out = root / dataset

    # static PNG: one map per filter (baseline, GA, PSO, DE, GWO), each with the GPS fixes and the ground
    # truth, framed on the ground truth (the noisy GPS cloud would hide the tracks)
    lat_min, lat_max, lon_min, lon_max = truth[0].min(), truth[0].max(), truth[1].min(), truth[1].max()
    pad_lat, pad_lon = 0.08 * (lat_max - lat_min) + 1e-4, 0.08 * (lon_max - lon_min) + 1e-4
    box = (lat_min - pad_lat, lat_max + pad_lat, lon_min - pad_lon, lon_max + pad_lon)
    (xl, yb), (xr, yt) = _merc(box[0], box[2]), _merc(box[1], box[3])
    cols = 3
    rows = -(-len(runs) // cols)
    panel = 6.0
    fig, axes = plt.subplots(rows, cols, figsize=(cols * panel, rows * (panel * (yt - yb) / (xr - xl) + 0.6) + 0.6), squeeze=False)
    for ax in axes.flat[len(runs):]:
        ax.axis("off")
    gx, gy = _merc(*gps)
    tx, ty = _merc(*truth)
    for ax, (name, label, xy, rm) in zip(axes.flat, runs):
        has_tiles = basemap(ax, *box)
        ax.plot(gx, gy, ".", ms=2, color=COLORS["gps"], alpha=0.5, label=f"GPS fixes ({gps_rmse:.2f} m)", zorder=2)
        ax.plot(tx, ty, color=COLORS["truth"], lw=2.2, label="ground truth", zorder=5)
        for k, sec in enumerate(sections):
            inside = (data["t_true"] >= sec["t_start_s"]) & (data["t_true"] <= sec["t_end_s"])
            sx, sy = _merc(*ll(data["p_true"][inside, :2]))
            ax.plot(sx, sy, color=SECTION_COLORS[k % len(SECTION_COLORS)], lw=7, alpha=0.45,
                    solid_capstyle="butt", label=section_label(sec), zorder=3)
        mx, my = _merc(*ll(xy))
        ax.plot(mx, my, color=COLORS[name], lw=0.9 if gps_rmse > 10 else 1.5, ls="--" if name == "baseline" else "-",
                label=f"EKF ({rm:.2f} m)", zorder=4)
        ax.set_xlim(xl, xr)
        ax.set_ylim(yb, yt)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax.set_title(label.replace("EKF, ", "") + f": RMSE_h {rm:.2f} m", fontsize=10)
        if has_tiles:
            ax.text(0.99, 0.005, "© OpenStreetMap contributors", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=6, bbox=dict(facecolor="white", alpha=0.7, lw=0))
    fig.suptitle(f"{dataset}: GPS fixes, ground truth and each filter on OpenStreetMap", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "map.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out / 'map.png'}")

    # interactive HTML
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scattermap(lat=gps[0], lon=gps[1], mode="markers", marker=dict(size=3, color=COLORS["gps"]),
                                name=f"GPS fixes (RMSE_h {gps_rmse:.2f} m)"))
    fig.add_trace(go.Scattermap(lat=truth[0], lon=truth[1], mode="lines", line=dict(width=3, color=COLORS["truth"]),
                                name="ground truth"))
    for name, label, xy, rm in runs:
        lat, lon = ll(xy)
        fig.add_trace(go.Scattermap(lat=lat, lon=lon, mode="lines", line=dict(width=2, color=COLORS[name]),
                                    name=f"{label} (RMSE_h {rm:.2f} m)"))
    fig.update_layout(map=dict(style="open-street-map", center=dict(lat=float(truth[0].mean()), lon=float(truth[1].mean())),
                               zoom=15 if dataset.startswith("inavwhu") else 14),
                      title=f"{dataset}: tracks on OpenStreetMap", height=800, margin=dict(l=0, r=0, t=40, b=0))
    fig.write_html(out / "map.html", include_plotlyjs="cdn")
    print(f"wrote {out / 'map.html'}")


def main(datasets, root="results"):
    root = Path(root)
    for ds in datasets:
        if (root / ds / "baseline" / "runs.csv").exists():
            draw(ds, root)


if __name__ == "__main__":
    import main as runner
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--datasets", nargs="+", default=runner.DATASETS)
    p.add_argument("--root", default=str(runner.results_root(runner.SCENARIO)))
    a = p.parse_args()
    main(a.datasets, a.root)
