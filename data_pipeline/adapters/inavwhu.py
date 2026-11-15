"""iNav-WHU sequences (raw_data/iNavWHU/<sequence>/): ADIS16465 IMU and F9P GNSS only.

Raw text files, whitespace separated, no header. Time stamps are GNSS seconds of week and
share one base across the files, so no conversion is needed (the pipeline rebases to the
first IMU sample).

    <seq>_ADIS16465_IMU.txt   t, dtheta (xyz) [rad], dvel (xyz) [m/s]   increments per sample
    <seq>_F9P_GNSS.pos        t, lat, lon [deg], alt [m], std lat, lon, alt [m]   1 Hz, SPP, no velocity
    <seq>_groundtruth.nav     t, pos NED [m], vel NED [m/s], roll, pitch, yaw [deg]
The odometers, the MID360 and OEM7 files and trajectory.csv are not used.

Conversions:
  - Increments to rates: gyro = dtheta / dt and accel = dvel / dt, with dt the median sample
    interval (the sample clock is regular, so this avoids timestamp quantisation noise).
  - The IMU is x front, y right, z down: a static IMU reads az = -g, as in the ETH data.
  - Ground truth NED to ENU: e = pos_e, n = pos_n, u = -pos_d.
  - GNSS lat/lon/alt to ENU in the ground-truth frame. The ground-truth frame is local NED
    whose origin is where the vehicle starts, and the truth stays at the origin until the
    vehicle moves. The GNSS origin is therefore the mean of the fixes taken while the truth
    has not yet moved, which uses no later truth. If fewer than 5 such fixes exist, the
    first fix is used and the meta says so.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .. import geo
from ..core import Adapter, Candidate, Recording, register

SUFFIXES = {"imu": "_ADIS16465_IMU.txt", "gnss": "_F9P_GNSS.pos", "gt": "_groundtruth.nav"}
MOVE_THRESHOLD_M = 0.05     # the truth has "moved" once it is this far from its origin
MIN_STATIC_FIXES = 5


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, engine="c")


@register
class INavWHU(Adapter):
    @classmethod
    def discover(cls, raw_root: Path) -> list[Candidate]:
        root = Path(raw_root) / "iNavWHU"
        found = []
        if not root.is_dir():
            return found
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            paths = {k: folder / f"{folder.name}{s}" for k, s in SUFFIXES.items()}
            if all(p.is_file() for p in paths.values()):
                found.append(Candidate(f"inavwhu/{folder.name}",
                                       f"iNav-WHU {folder.name} (ADIS16465 IMU 200 Hz, F9P GNSS 1 Hz)",
                                       cls, paths))
        return found

    @classmethod
    def load(cls, cand: Candidate) -> Recording:
        p = cand.paths
        raw = _read(p["imu"])
        t = raw[0].to_numpy()
        dt = float(np.median(np.diff(t)))
        imu = pd.DataFrame({"t": t,
                            "ax": raw[4] / dt, "ay": raw[5] / dt, "az": raw[6] / dt,
                            "wx": raw[1] / dt, "wy": raw[2] / dt, "wz": raw[3] / dt})

        gt_raw = _read(p["gt"])        # t, n, e, d, vn, ve, vd, roll, pitch, yaw
        gt = pd.DataFrame({"t": gt_raw[0], "e": gt_raw[2], "n": gt_raw[1], "u": -gt_raw[3],
                           "ve": gt_raw[5], "vn": gt_raw[4], "vu": -gt_raw[6]})
        moved = np.flatnonzero(np.linalg.norm(gt_raw[[1, 2, 3]].to_numpy(), axis=1) > MOVE_THRESHOLD_M)
        t_move = float(gt_raw[0].iloc[moved[0]]) if len(moved) else float(gt_raw[0].iloc[-1])

        g = _read(p["gnss"])           # t, lat, lon, alt, std_lat, std_lon, std_alt
        static = (g[0] < t_move) & (g[0] >= gt_raw[0].iloc[0])
        n_static = int(static.sum())
        use = g[static] if n_static >= MIN_STATIC_FIXES else g.iloc[:1]
        lat0, lon0, alt0 = float(use[1].mean()), float(use[2].mean()), float(use[3].mean())
        e, n, u = geo.latlon_to_enu(g[1], g[2], g[3], lat0, lon0, alt0)
        gps = pd.DataFrame({"t": g[0], "e": e, "n": n, "u": u,
                            "std_e": g[5], "std_n": g[4], "std_u": g[6]})   # ve, vn, vu absent

        meta = {
            "source_files": [Path(v).name for v in p.values()],
            "time_base": "GNSS seconds of week, shared by all files",
            "imu_increment_dt_s": dt,
            "imu_frame": "x front, y right, z down (static az = -g)",
            "gnss": "F9P single point positioning, 1 Hz, no velocity (gps ve/vn/vu are NaN); "
                    "std_e/std_n/std_u are the receiver's reported sigmas in metres",
            "gps_origin_lat_lon_alt": [lat0, lon0, alt0],
            "gps_origin_method": (f"mean of {n_static} fixes while the ground truth is still at its origin"
                                  if n_static >= MIN_STATIC_FIXES else "first fix (too few static fixes)"),
            "gt_first_motion_t_source_s": t_move,
            "gt_frame_conversion": "NED -> ENU: e=pos_e, n=pos_n, u=-pos_d, ve=vel_e, vn=vel_n, vu=-vel_d",
            "gt_attitude": "roll, pitch, yaw not carried into the canonical format",
        }
        return Recording(cand.id, cand.label, imu, gps, gt, meta)
