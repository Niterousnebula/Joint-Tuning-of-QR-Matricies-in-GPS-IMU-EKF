"""Maps of each run on OpenStreetMap: ground truth, the GPS fixes, the baseline EKF (THETA0) and
the best tuned run of every optimizer.

    python plot_maps.py                                   # every dataset of main.DATASETS
    python plot_maps.py --datasets eth_gps2m_s0           # one dataset

Writes results/<dataset>/map.png (static: OSM tiles under the tracks) and map.html (interactive:
pan, zoom and toggle tracks; needs a browser with internet). The EKF tracks are rerun on the CPU
reference with the THETA0 and best thetas from results/<dataset>/<OPT>/runs.csv. Map tiles come from
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
TILE_CACHE = Path(".tile_cache")
MAX_POINTS = 5000            # per track, enough for a map


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------
_data = None


def _init(data):
    global _data
    _data = data


def _track(theta):
    """CPU reference run: horizontal ENU positions (thinned) and the whole-flight RMSE_h."""
    r = ekf.evaluate(_data, theta)
    x = r["x_log"]
    step = max(1, len(x) // MAX_POINTS)
    return x[::step, :2], r["rmse_h"]


def tracks(dataset, root):
    """[(name, label, ENU array, rmse)] for the baseline and the best seed of every optimizer."""
    data = ekf.load_data(dataset)
    runs = [("baseline", "EKF, THETA0 (baseline)", data["theta0"])]
    for name in OPTIMIZERS:
        path = root / dataset / name / "runs.csv"
        if path.exists():
            r = pd.read_csv(path)
            b = r.loc[r["rmse_h"].idxmin()]
            runs.append((name, f"EKF, {name} best (seed {int(b['seed'])})", b[ekf.THETA_NAMES].to_numpy(dtype=float)))
    with Pool(len(runs), initializer=_init, initargs=(data,)) as pool:
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
    out = root / dataset

    # static PNG: framed on the ground truth (the noisy GPS cloud would hide the tracks)
    fig, ax = plt.subplots(figsize=(10, 10))
    lat_min, lat_max, lon_min, lon_max = truth[0].min(), truth[0].max(), truth[1].min(), truth[1].max()
    pad_lat, pad_lon = 0.08 * (lat_max - lat_min) + 1e-4, 0.08 * (lon_max - lon_min) + 1e-4
    box = (lat_min - pad_lat, lat_max + pad_lat, lon_min - pad_lon, lon_max + pad_lon)
    has_tiles = basemap(ax, *box)
    mx, my = _merc(*gps)
    ax.plot(mx, my, ".", ms=2, color=COLORS["gps"], alpha=0.5, label=f"GPS fixes (RMSE_h {gps_rmse:.2f} m)", zorder=2)
    mx, my = _merc(*truth)
    ax.plot(mx, my, color=COLORS["truth"], lw=2.2, label="ground truth", zorder=5)
    for name, label, xy, rm in runs:
        mx, my = _merc(*ll(xy))
        noisy = gps_rmse > 10                       # 64 m GPS: tracks zig-zag at every fix, so draw them lighter
        ax.plot(mx, my, color=COLORS[name], lw=0.7 if noisy else 1.3, alpha=0.55 if noisy else 1.0,
                ls="--" if name == "baseline" else "-", label=f"{label} (RMSE_h {rm:.2f} m)", zorder=4)
    (xl, yb), (xr, yt) = _merc(box[0], box[2]), _merc(box[1], box[3])
    ax.set_xlim(xl, xr)
    ax.set_ylim(yb, yt)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title(f"{dataset}: tracks on OpenStreetMap", fontsize=11)
    if has_tiles:
        ax.text(0.99, 0.005, "© OpenStreetMap contributors", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, bbox=dict(facecolor="white", alpha=0.7, lw=0))
    fig.tight_layout()
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
    p.add_argument("--root", default="results")
    a = p.parse_args()
    main(a.datasets, a.root)
