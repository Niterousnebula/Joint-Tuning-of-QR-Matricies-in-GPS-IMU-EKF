"""Adapter for the quadrotor sets (raw_data/Quadrotor/...).

Each IMU file is one sensor stream. Files are matched within their folder:
    First_IMU.csv / Second_IMU.csv         single run (Eight Trajectory)
    First_IMU_<k>.csv / Second_IMU_<k>.csv  numbered run k, paired with GT_<k>.csv
Columns are read by position. Some headers repeat a name (for example AccX, AccX, AccX),
so names cannot be used. The column order follows each folder's Description.txt.
GT positions and velocities are NED in the source and are converted to ENU here.
"""
import re
from pathlib import Path

import pandas as pd

from ..core import Adapter, Candidate, Recording, register

IMU_RE = re.compile(r"^(First|Second)_IMU(?:_(\d+))?\.csv$")
GT_RE = re.compile(r"^GT(?:_(\d+))?\.csv$")


@register
class Quadrotor(Adapter):
    @classmethod
    def discover(cls, raw_root: Path) -> list[Candidate]:
        root = Path(raw_root) / "Quadrotor"
        if not root.is_dir():
            return []
        found = []
        for folder in sorted({p.parent for p in root.rglob("*.csv")}):
            files = sorted(folder.glob("*.csv"))
            gts = {(m.group(1) or ""): f for f in files if (m := GT_RE.match(f.name))}
            experiment = folder.relative_to(root).as_posix().lower().replace(" ", "-")
            for f in files:
                m = IMU_RE.match(f.name)
                if not m:
                    continue
                sensor, run = m.group(1), m.group(2) or ""
                run_slug = f"run-{run}" if run else "single"
                label = (f"Quadrotor {folder.relative_to(root).as_posix()}, "
                         f"{'run ' + run if run else 'single run'}, {sensor} IMU")
                found.append(Candidate(f"quadrotor/{experiment}/{run_slug}/{sensor.lower()}",
                                       label, cls, {"imu": f, "gt": gts.get(run)}))
        return found

    @classmethod
    def load(cls, cand: Candidate) -> Recording:
        imu_raw = pd.read_csv(cand.paths["imu"])
        imu = pd.DataFrame({
            "t": imu_raw.iloc[:, 0], "ax": imu_raw.iloc[:, 1], "ay": imu_raw.iloc[:, 2],
            "az": imu_raw.iloc[:, 3], "wx": imu_raw.iloc[:, 4], "wy": imu_raw.iloc[:, 5],
            "wz": imu_raw.iloc[:, 6],
        })

        gt = None
        gt_path = cand.paths["gt"]
        if gt_path is not None:
            g = pd.read_csv(gt_path)
            # columns: Time, PN, PE, PD, VN, VE, VD, Roll, Pitch, Yaw
            gt = pd.DataFrame({
                "t": g.iloc[:, 0], "e": g.iloc[:, 2], "n": g.iloc[:, 1], "u": -g.iloc[:, 3],
                "ve": g.iloc[:, 5], "vn": g.iloc[:, 4], "vu": -g.iloc[:, 6],
            })

        meta = {
            "source_files": [cand.paths["imu"].name] + ([gt_path.name] if gt_path else []),
            "gt_attitude": "Roll, Pitch, Yaw columns not carried into the canonical format",
            "gt_frame_conversion": "NED -> ENU: e=PE, n=PN, u=-PD, ve=VE, vn=VN, vu=-VD",
            "imu_axis_convention": "not verified against the EKF body frame",
        }
        return Recording(cand.id, cand.label, imu, None, gt, meta)
