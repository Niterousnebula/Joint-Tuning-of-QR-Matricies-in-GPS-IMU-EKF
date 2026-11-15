"""Adapter for one-file-per-dataset tables (xlsx or csv), described by a TableSpec.

A new dataset whose file has the same kind of table as an existing spec needs only a new
TableSpec entry. Column names, time units, scale factors and known problems are declared
there, not coded.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import geo
from ..core import Adapter, Candidate, Recording, register

ACCEL_KEYS = ("ax", "ay", "az")
GYRO_KEYS = ("wx", "wy", "wz")

# Plausibility limits applied to every table dataset. A value outside them is set to NaN
# (the fix itself is kept, unless the speed gate drops it). Recorded in meta.json.
ALT_RANGE_M = (-500.0, 2000.0)     # receivers log sentinels such as -9999.9 and 33962 m
VEL_MAX_MS = 50.0                  # |v| above this on a land vehicle is a bad fix
SPEED_GATE_MS = 50.0               # a fix implying more than this speed from the previous fix is dropped


@dataclass(frozen=True)
class TableSpec:
    id: str
    label: str
    file: str                          # relative to the raw data root; may be a glob such as "Folder/*.csv"
    time: str                          # source time column
    time_scale: float                  # multiply the time column by this to get seconds
    imu: dict                          # canonical IMU name (ax..wz) -> source column
    accel_scale: float = 1.0           # e.g. 9.80665 when the accelerometer is logged in g
    gyro_scale: float = 1.0            # e.g. pi/180 when the gyro is logged in deg/s
    sheet: str | int | None = None     # xlsx sheet; None = first sheet
    gps: dict | None = None            # lat, lon (degrees); optional alt (m), ve, vn, vu (m/s)
    gps_box: tuple | None = None       # (lat_min, lat_max, lon_min, lon_max): fixes outside are dropped
    gap_split_s: float | None = None   # split the timeline at gaps longer than this, keep the longest part
    notes: str = ""                    # known problems or unverified assumptions, copied into meta.json


def read_table(path: Path, sheet) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        df = pd.read_excel(path, sheet_name=0 if sheet is None else sheet, engine="openpyxl")
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _speed_gate(t, e, n, limit, max_reject_s=5.0):
    """Keep a fix only if it is reachable from the last kept fix at `limit` m/s.

    If fixes have been rejected for more than max_reject_s, the anchor itself is presumed
    bad and the gate re-anchors on the current fix, so a bad start cannot block the track.
    """
    keep = np.zeros(len(t), dtype=bool)
    anchor = None
    reject_since = None
    for i in range(len(t)):
        if anchor is None:
            keep[i] = True
            anchor = i
            continue
        dt = t[i] - t[anchor]
        if dt > 0 and np.hypot(e[i] - e[anchor], n[i] - n[anchor]) <= limit * dt:
            keep[i] = True
            anchor = i
            reject_since = None
        elif reject_since is None:
            reject_since = t[i]
        elif t[i] - reject_since > max_reject_s:
            keep[i] = True
            anchor = i
            reject_since = None
    return keep


class TableAdapter(Adapter):
    """Base for table datasets. Subclasses set `specs` and are registered with @register."""

    specs: tuple[TableSpec, ...] = ()

    @classmethod
    def discover(cls, raw_root: Path) -> list[Candidate]:
        """One candidate per file matching spec.file (a path or a glob). A spec's id and
        label may use {parent} and {stem}, the lower-cased folder and file names."""
        found = []
        for spec in cls.specs:
            for path in sorted(Path(raw_root).glob(spec.file)):
                if not path.is_file():
                    continue
                fields = {"parent": path.parent.name.lower().replace(" ", "-"),
                          "stem": path.stem.lower().replace(" ", "-")}
                found.append(Candidate(spec.id.format(**fields), spec.label.format(**fields),
                                       cls, {"spec": spec, "path": path}))
        return found

    @classmethod
    def load(cls, cand: Candidate) -> Recording:
        spec: TableSpec = cand.paths["spec"]
        path: Path = cand.paths["path"]
        df = read_table(path, spec.sheet)

        needed = [spec.time, *spec.imu.values(), *(spec.gps or {}).values()]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"{cand.id}: columns {missing} not found in {path.name}; "
                             f"available: {list(df.columns)}")

        def num(col):
            return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

        t = num(spec.time) * spec.time_scale
        imu = pd.DataFrame({"t": t})
        for key, col in spec.imu.items():
            scale = spec.accel_scale if key in ACCEL_KEYS else spec.gyro_scale if key in GYRO_KEYS else 1.0
            imu[key] = num(col) * scale

        gps = None
        meta = {
            "source_file": path.name,
            "sheet": spec.sheet if spec.sheet is not None else 0,
            "accel_scale_applied": spec.accel_scale,
            "gyro_scale_applied": spec.gyro_scale,
            "notes": spec.notes,
        }
        if spec.gps:
            fix = pd.DataFrame({"t": t, "lat": num(spec.gps["lat"]), "lon": num(spec.gps["lon"])})
            fix["alt"] = num(spec.gps["alt"]) if "alt" in spec.gps else np.nan
            for key in ("ve", "vn", "vu"):
                fix[key] = num(spec.gps[key]) if key in spec.gps else np.nan
            fix = fix.dropna(subset=["t", "lat", "lon"])

            n_before = len(fix)
            if spec.gps_box is not None:
                lat_min, lat_max, lon_min, lon_max = spec.gps_box
                inside = fix["lat"].between(lat_min, lat_max) & fix["lon"].between(lon_min, lon_max)
                fix = fix[inside].copy()
            meta["gps_fixes_outside_box"] = n_before - len(fix)
            meta["gps_box"] = list(spec.gps_box) if spec.gps_box else None

            alt_bad = ~fix["alt"].between(*ALT_RANGE_M) & fix["alt"].notna()
            fix.loc[alt_bad, "alt"] = np.nan

            # Receivers repeat the last fix between updates: keep only rows where it changed
            key_cols = fix[["lat", "lon", "alt"]].astype(str)
            fix = fix[(key_cols != key_cols.shift()).any(axis=1)].reset_index(drop=True)

            if len(fix):
                lat0, lon0 = float(fix["lat"].iloc[0]), float(fix["lon"].iloc[0])
                alt0 = float(fix["alt"].iloc[0]) if np.isfinite(fix["alt"].iloc[0]) else 0.0
                e, n, u = geo.latlon_to_enu(fix["lat"], fix["lon"], fix["alt"], lat0, lon0, alt0)
                ok = _speed_gate(fix["t"].to_numpy(), e, n, SPEED_GATE_MS)
                meta["gps_fixes_speed_rejected"] = int((~ok).sum())
                meta["gps_origin_lat_lon_alt"] = [lat0, lon0, alt0]
                gps = pd.DataFrame({"t": fix["t"], "e": e, "n": n, "u": u,
                                    "ve": fix["ve"], "vn": fix["vn"], "vu": fix["vu"]})[ok].reset_index(drop=True)
                for key in ("ve", "vn", "vu"):
                    bad = gps[key].abs() > VEL_MAX_MS
                    meta[f"gps_{key}_rejected"] = int(bad.sum())
                    gps.loc[bad, key] = np.nan
            meta["altitude_range_m"] = list(ALT_RANGE_M)
            meta["velocity_limit_ms"] = VEL_MAX_MS
            meta["speed_gate_ms"] = SPEED_GATE_MS

        return Recording(cand.id, cand.label, imu, gps, None, meta, gap_split_s=spec.gap_split_s)
