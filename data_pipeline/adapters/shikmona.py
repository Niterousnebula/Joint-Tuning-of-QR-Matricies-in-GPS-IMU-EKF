"""MV Shikmona (raw_data/MV - Shikmona/): smartphone logs with an MRU as the reference.

    smartphone/YYYY-MM-DDHH.MM.SS.csv    time [s since the log started], gF (xyz) [g], w (xyz) [rad/s],
                                         optional Latitude, Longitude [deg], Speed (m/s)
    MRU/M2000279-YYYY-MM-DD-HH-MM-SS.csv GpsTime [GPS seconds of week], Latitude, Longitude, plus its own IMU

Each phone log is one recording: phone IMU, phone GPS, and the MRU position as ground truth.
Known problems with this data (all recorded in meta.json):
  - The phone log is event driven: rows repeat held sensor values, timestamps are out of
    order, and in every file only the first ~30 s (plus a short burst at the end) hold data.
    The longest continuous part is kept.
  - The phone has no absolute time. Its start is taken from the file name (local time,
    UTC+3 for Israel in April 2021, GPS-UTC = 18 s) and compared with the MRU GpsTime.
    File names have 1 s resolution and the MRU is started separately (11 to 32 s later), so the
    alignment is uncertain and could not be confirmed from the gyro or the track.
  - The phone GPS gives a speed, not a velocity vector, so ve/vn/vu are NaN.
  - The MRU has no altitude, so ground-truth u is NaN.
  - The 2021-04-18 files have no MRU, and one has no GPS columns at all.
"""
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .. import geo
from ..core import Adapter, Candidate, Recording, register

G = 9.80665
LOCAL_UTC_OFFSET_H = 3          # Israel Daylight Time, April 2021
GPS_UTC_LEAP_S = 18
GPS_EPOCH = datetime(1980, 1, 6)
MAX_MRU_LAG_S = 120.0           # an MRU file belongs to a phone log if it starts within this after it
PHONE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(\d{2})\.(\d{2})\.(\d{2})\.csv$")
MRU_RE = re.compile(r"^M\d+-(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.csv$")


def _sow(local: datetime) -> float:
    """GPS seconds of week for a local clock time."""
    gps = local - timedelta(hours=LOCAL_UTC_OFFSET_H) + timedelta(seconds=GPS_UTC_LEAP_S)
    return (gps - GPS_EPOCH).total_seconds() % 604800


def _stamp(match) -> datetime:
    return datetime(*(int(g) for g in match.groups()))


@register
class Shikmona(Adapter):
    @classmethod
    def discover(cls, raw_root: Path) -> list[Candidate]:
        root = Path(raw_root) / "MV - Shikmona"
        phones = sorted((root / "smartphone").glob("*.csv")) if (root / "smartphone").is_dir() else []
        mrus = []
        for f in sorted((root / "MRU").glob("*.csv")) if (root / "MRU").is_dir() else []:
            m = MRU_RE.match(f.name)
            if m:
                mrus.append((_stamp(m), f))
        found = []
        for f in phones:
            m = PHONE_RE.match(f.name)
            if not m:
                continue
            start = _stamp(m)
            lag = [(s - start).total_seconds() for s, _ in mrus]
            pair = None
            ok = [i for i, d in enumerate(lag) if 0 <= d <= MAX_MRU_LAG_S]
            if ok:
                pair = mrus[min(ok, key=lambda i: lag[i])][1]
            tag = start.strftime("%Y%m%d-%H%M%S")
            found.append(Candidate(f"mv-shikmona/{tag}",
                                   f"MV Shikmona phone {start:%Y-%m-%d %H:%M:%S}"
                                   + (" with MRU reference" if pair else " (no MRU reference)"),
                                   cls, {"phone": f, "mru": pair, "start": start}))
        return found

    @classmethod
    def load(cls, cand: Candidate) -> Recording:
        raw = pd.read_csv(cand.paths["phone"])
        raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("Unnamed")]
        raw = raw.sort_values("time", kind="stable").reset_index(drop=True)

        imu = pd.DataFrame({"t": raw["time"],
                            "ax": raw["gFx"] * G, "ay": raw["gFy"] * G, "az": raw["gFz"] * G,
                            "wx": raw["wx"], "wy": raw["wy"], "wz": raw["wz"]})
        # The gyro starts reporting a moment after the accelerometer: its leading zeros are not data
        moving = np.flatnonzero((raw[["wx", "wy", "wz"]].abs().sum(axis=1) > 0).to_numpy())
        lead = int(moving[0]) if len(moving) else 0
        imu = imu.iloc[lead:].reset_index(drop=True)

        gps_fixes = None
        if "Latitude" in raw.columns:
            lat, lon = raw["Latitude"].to_numpy(), raw["Longitude"].to_numpy()
            changed = (lat != 0) & np.r_[True, (np.diff(lat) != 0) | (np.diff(lon) != 0)]
            gps_fixes = raw.loc[changed, ["time", "Latitude", "Longitude", "Speed (m/s)"]]
            if gps_fixes.empty:
                gps_fixes = None

        sow0 = _sow(cand.paths["start"])
        mru = None
        if cand.paths["mru"] is not None:
            mru = pd.read_csv(cand.paths["mru"], usecols=["GpsTime", "Latitude", "Longitude"])

        # One ENU origin for GPS and truth: the phone's first fix, else the MRU's first point
        if gps_fixes is not None:
            lat0, lon0 = float(gps_fixes["Latitude"].iloc[0]), float(gps_fixes["Longitude"].iloc[0])
        elif mru is not None:
            lat0, lon0 = float(mru["Latitude"].iloc[0]), float(mru["Longitude"].iloc[0])
        else:
            lat0 = lon0 = None

        gps = None
        if gps_fixes is not None:
            e, n, _ = geo.latlon_to_enu(gps_fixes["Latitude"], gps_fixes["Longitude"], 0.0, lat0, lon0, 0.0)
            gps = pd.DataFrame({"t": gps_fixes["time"].to_numpy(), "e": e, "n": n, "u": np.nan,
                                "speed": gps_fixes["Speed (m/s)"].to_numpy()})
        gt = None
        meta = {
            "source_files": [cand.paths["phone"].name] + ([cand.paths["mru"].name] if cand.paths["mru"] else []),
            "phone_gyro_leading_zero_rows_dropped": lead,
            "phone_accel_scale": "gF (g) x 9.80665; |gF| median is 1.00",
            "gps_origin_lat_lon": None if lat0 is None else [lat0, lon0],
            "gps": "phone fixes (rows where the position changes); speed is a scalar, ve/vn/vu are NaN; no altitude",
            "gt": "MRU latitude/longitude, no altitude" if mru is not None else "none (no MRU for this log)",
            "time_alignment": "phone start from the file name (local time UTC+3, GPS-UTC 18 s) vs MRU GpsTime; "
                              "uncertain, see module notes",
        }
        if mru is not None:
            e, n, _ = geo.latlon_to_enu(mru["Latitude"], mru["Longitude"], 0.0, lat0, lon0, 0.0)
            gt = pd.DataFrame({"t": mru["GpsTime"].to_numpy() - sow0, "e": e, "n": n, "u": np.nan})
            meta["mru_start_after_phone_start_s"] = float(mru["GpsTime"].iloc[0] - sow0)
            meta["mru_end_after_phone_start_s"] = float(mru["GpsTime"].iloc[-1] - sow0)
        # Keep the longest continuous part of the phone log (the first ~30 s)
        return Recording(cand.id, cand.label, imu, gps, gt, meta, gap_split_s=2.0)
