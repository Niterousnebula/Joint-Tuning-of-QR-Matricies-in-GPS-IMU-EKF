"""Canonical recording format, the adapter registry, and the load / validate / write steps.

Nothing in this module knows about a particular dataset. Each raw layout is handled by an
adapter (see data_pipeline/adapters/) that turns its files into a Recording in the canonical
format below. Adding a dataset in a known layout is a config entry; a new layout is a new
adapter module. This file does not change.

Canonical conventions (every adapter must deliver these):
    time        seconds, rebased so the first valid IMU sample is t = 0
    IMU         ax, ay, az in m/s^2 and wx, wy, wz in rad/s, in the sensor (body) frame as logged;
                the pipeline does not rotate or re-sign the IMU
    GPS, GT     e, n, u (ENU) in metres and ve, vn, vu (ENU) in m/s; velocity or u may be NaN
                when the source has no such channel
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = REPO_ROOT / "raw_data"
DEFAULT_OUT = REPO_ROOT / "processed_data"

IMU_COLUMNS = ("t", "ax", "ay", "az", "wx", "wy", "wz")
POSE_COLUMNS = ("t", "e", "n", "u", "ve", "vn", "vu")
POSE_REQUIRED = ("t", "e", "n")

# Ground truth denser than 1.5x GT_MAX_HZ is decimated to about GT_MAX_HZ: scoring against
# 200 Hz truth adds memory and time to every evaluation and no information.
GT_MAX_HZ = 10.0

# Minimum requirements for a recording to be marked ekf_ready in meta.json. These are
# heuristics chosen here, not properties of the EKF.
MIN_GPS_FIXES = 10
MIN_GT_SAMPLES = 10
MIN_DURATION_S = 60.0
MIN_COVERAGE = 0.5          # share of the IMU window that the GPS and the GT must each span
GRAVITY_RANGE = (8.0, 11.5)  # m/s^2: mean specific force over the first second (static start)


@dataclass
class Candidate:
    """A dataset found on disk. Discovery only lists these; nothing is read yet."""
    id: str
    label: str
    adapter: type
    paths: dict = field(default_factory=dict)


@dataclass
class Recording:
    id: str
    label: str
    imu: pd.DataFrame
    gps: pd.DataFrame | None = None
    gt: pd.DataFrame | None = None
    meta: dict = field(default_factory=dict)
    gap_split_s: float | None = None   # split the IMU timeline at gaps longer than this; keep the longest part


class Adapter(ABC):
    """Knows one raw layout. discover() lists its datasets, load() reads one of them."""

    @classmethod
    @abstractmethod
    def discover(cls, raw_root: Path) -> list[Candidate]:
        ...

    @classmethod
    @abstractmethod
    def load(cls, candidate: Candidate) -> Recording:
        ...


ADAPTERS: list[type[Adapter]] = []


def register(adapter: type[Adapter]) -> type[Adapter]:
    """Class decorator: makes an adapter visible to discover() and load()."""
    if adapter not in ADAPTERS:
        ADAPTERS.append(adapter)
    return adapter


def discover(raw_root: Path = DEFAULT_RAW) -> list[Candidate]:
    found: list[Candidate] = []
    for adapter in ADAPTERS:
        found.extend(adapter.discover(Path(raw_root)))
    ids = [c.id for c in found]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate dataset ids from the adapters: {sorted(dupes)}")
    return sorted(found, key=lambda c: [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", c.id)])


def load(dataset_id: str, raw_root: Path = DEFAULT_RAW) -> Recording:
    """Find a dataset by id, read it with its adapter, and return a validated Recording."""
    for cand in discover(raw_root):
        if cand.id == dataset_id:
            return finalise(cand.adapter.load(cand))
    raise KeyError(f"no dataset with id {dataset_id!r} under {raw_root}")


def _numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _split_segments(imu: pd.DataFrame, gap_s: float):
    """Keep the longest run of IMU samples with no step larger than gap_s; describe all runs."""
    dt = np.diff(imu["t"].to_numpy())
    cuts = np.flatnonzero(dt > gap_s) + 1
    bounds = [0, *cuts.tolist(), len(imu)]
    spans = list(zip(bounds[:-1], bounds[1:]))
    keep = int(np.argmax([b - a for a, b in spans]))
    info = [{"start_s": float(imu["t"].iloc[a]), "end_s": float(imu["t"].iloc[b - 1]),
             "samples": int(b - a), "kept": i == keep} for i, (a, b) in enumerate(spans)]
    a, b = spans[keep]
    kept = imu.iloc[a:b].reset_index(drop=True)
    return kept, info


def _pose(df: pd.DataFrame | None, t0: float, name: str, meta: dict,
          window: tuple | None = None, max_hz: float | None = None) -> pd.DataFrame | None:
    if df is None:
        return None
    n_in = len(df)
    df = df.copy()
    for c in POSE_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    extra = [c for c in df.columns if c not in POSE_COLUMNS]
    df = _numeric(df, POSE_COLUMNS)
    df = df.dropna(subset=list(POSE_REQUIRED)).sort_values("t", kind="stable")
    df = df.drop_duplicates("t", keep="first")
    if window is not None:
        df = df[(df["t"] >= window[0]) & (df["t"] <= window[1])]
    df = df.reset_index(drop=True)
    if max_hz and len(df) > 2:
        rate = 1.0 / float(np.median(np.diff(df["t"].to_numpy())))
        if rate > 1.5 * max_hz:
            step = int(round(rate / max_hz))
            df = df.iloc[::step].reset_index(drop=True)
            meta[f"{name}_decimation_step"] = step
    df["t"] = df["t"] - t0
    df = df[[*POSE_COLUMNS, *extra]]
    meta[f"{name}_samples"] = len(df)
    meta[f"{name}_dropped"] = n_in - len(df)
    if df.empty:
        return None             # nothing usable left: report it as absent, not as an empty table
    meta[f"{name}_start_s"] = float(df["t"].iloc[0])
    meta[f"{name}_end_s"] = float(df["t"].iloc[-1])
    return df


def finalise(rec: Recording) -> Recording:
    """Check the canonical conventions, clean the rows, and rebase time to the first IMU sample."""
    missing = [c for c in IMU_COLUMNS if c not in rec.imu.columns]
    if missing:
        raise ValueError(f"{rec.id}: IMU is missing {missing}")

    meta = dict(rec.meta)
    n_in = len(rec.imu)
    extra = [c for c in rec.imu.columns if c not in IMU_COLUMNS]
    imu = _numeric(rec.imu, IMU_COLUMNS).dropna(subset=list(IMU_COLUMNS))
    imu = imu.sort_values("t", kind="stable").drop_duplicates("t", keep="first").reset_index(drop=True)
    if imu.empty:
        raise ValueError(f"{rec.id}: no valid IMU samples after cleaning")

    n_valid = len(imu)
    window = None
    if rec.gap_split_s:
        imu, segments = _split_segments(imu, rec.gap_split_s)
        meta["segment_gap_split_s"] = rec.gap_split_s
        meta["segments"] = segments
        if len(segments) > 1:
            window = (float(imu["t"].iloc[0]), float(imu["t"].iloc[-1]))

    t0 = float(imu["t"].iloc[0])
    imu["t"] = imu["t"] - t0
    imu = imu[[*IMU_COLUMNS, *extra]]
    dt = np.diff(imu["t"].to_numpy())
    meta.update({
        "t0_source_s": t0,
        "imu_samples": len(imu),
        "imu_dropped_invalid": n_in - n_valid,
        "imu_dropped_outside_kept_segment": n_valid - len(imu),
        "imu_duration_s": float(imu["t"].iloc[-1]),
        "imu_rate_hz": float(1.0 / np.median(dt)) if len(dt) else None,
        "imu_max_gap_s": float(dt.max()) if len(dt) else None,
        "units": {"time": "s", "accel": "m/s^2", "gyro": "rad/s", "position": "m (ENU)",
                  "velocity": "m/s (ENU)"},
        "frames": "IMU in the sensor/body frame as logged; GPS and GT in ENU",
    })

    t_end = t0 + float(imu["t"].iloc[-1])
    gps = _pose(rec.gps, t0, "gps", meta, window)
    # Truth outside the IMU window cannot be scored against an estimate, so it is dropped
    gt = _pose(rec.gt, t0, "gt", meta, (t0, t_end), max_hz=GT_MAX_HZ)
    meta["ekf_ready"] = _assess(imu, gps, gt, meta)
    return Recording(rec.id, rec.label, imu, gps, gt, meta)


def _span(df: pd.DataFrame | None, duration: float) -> float:
    if df is None or len(df) < 2:
        return 0.0
    return max(0.0, float(min(df["t"].iloc[-1], duration) - max(df["t"].iloc[0], 0.0)))


def _assess(imu: pd.DataFrame, gps: pd.DataFrame | None, gt: pd.DataFrame | None, meta: dict) -> dict:
    """Whether the recording can drive a GPS-aided EKF tuned against ground truth, and why not."""
    duration = meta["imu_duration_s"]
    n_gps = 0 if gps is None else len(gps)
    n_gt = 0 if gt is None else len(gt)
    reasons = []
    if n_gps < MIN_GPS_FIXES:
        reasons.append(f"GPS has {n_gps} usable fixes (needs {MIN_GPS_FIXES})")
    if n_gt < MIN_GT_SAMPLES:
        reasons.append(f"ground truth has {n_gt} samples inside the IMU window (needs {MIN_GT_SAMPLES})")
    if duration < MIN_DURATION_S:
        reasons.append(f"IMU lasts {duration:.0f} s (needs {MIN_DURATION_S:.0f})")
    for name, df, n, need in (("GPS", gps, n_gps, MIN_GPS_FIXES), ("ground truth", gt, n_gt, MIN_GT_SAMPLES)):
        if n >= need and duration > 0 and _span(df, duration) / duration < MIN_COVERAGE:
            reasons.append(f"{name} spans {100 * _span(df, duration) / duration:.0f} % of the IMU window "
                           f"(needs {100 * MIN_COVERAGE:.0f} %)")
    start = imu[imu["t"] < 1.0][["ax", "ay", "az"]].mean().to_numpy()
    start_norm = float(np.linalg.norm(start))
    meta["start_specific_force_ms2"] = start_norm
    if not GRAVITY_RANGE[0] <= start_norm <= GRAVITY_RANGE[1]:
        reasons.append(f"mean specific force over the first second is {start_norm:.1f} m/s^2, not about g "
                       "(the EKF initialises attitude from a static start)")
    channels = [] if gps is None else [c for c in ("e", "n", "u", "ve", "vn", "vu") if gps[c].notna().all()]
    return {"ready": not reasons, "reasons": reasons, "gps_channels": channels}


def write(rec: Recording, out_root: Path = DEFAULT_OUT) -> Path:
    """Write imu.csv, gps.csv, gt.csv (when present) and meta.json under out_root/<id>/."""
    out = Path(out_root) / rec.id
    out.mkdir(parents=True, exist_ok=True)
    rec.imu.to_csv(out / "imu.csv", index=False)
    if rec.gps is not None:
        rec.gps.to_csv(out / "gps.csv", index=False)
    if rec.gt is not None:
        rec.gt.to_csv(out / "gt.csv", index=False)
    (out / "meta.json").write_text(
        json.dumps({"id": rec.id, "label": rec.label, **rec.meta}, indent=2, default=str))
    return out


def select(candidates: list[Candidate], choice: str) -> Candidate:
    """Resolve a 1-based list number or a dataset id to one candidate."""
    choice = str(choice).strip()
    if choice.isdigit():
        i = int(choice)
        if not 1 <= i <= len(candidates):
            raise KeyError(f"no dataset number {i} (1 to {len(candidates)})")
        return candidates[i - 1]
    for cand in candidates:
        if cand.id == choice:
            return cand
    raise KeyError(f"no dataset with id {choice!r}")
