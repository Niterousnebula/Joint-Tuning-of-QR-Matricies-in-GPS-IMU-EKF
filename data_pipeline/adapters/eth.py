"""Adapter for the ETH Zurich GPS-IMU flight (four CSVs with no extension, in raw_data/ETH)."""
from pathlib import Path

import pandas as pd

from ..core import REPO_ROOT, Adapter, Candidate, Recording, register

FILES = {
    "accel": "RawAccel ETH",
    "gyro": "RawGyro ETH",
    "gps": "OnboardGPS ETH",
    "gt": "GroundTruth ETH",
}


@register
class EthFlight(Adapter):
    @classmethod
    def discover(cls, raw_root: Path) -> list[Candidate]:
        # raw_data/ETH, or the tracked data/ folder of the repo (same files) when raw_data is absent
        for folder in (Path(raw_root) / "ETH", REPO_ROOT / "data"):
            if all((folder / name).is_file() for name in FILES.values()):
                paths = {key: folder / name for key, name in FILES.items()}
                return [Candidate("eth", "ETH Zurich GPS-IMU flight", cls, paths)]
        return []

    @classmethod
    def load(cls, cand: Candidate) -> Recording:
        p = cand.paths
        accel = pd.read_csv(p["accel"])
        gyro = pd.read_csv(p["gyro"])
        imu = accel[["t", "ax", "ay", "az"]].merge(gyro[["t", "wx", "wy", "wz"]], on="t", how="inner")

        gps_raw = pd.read_csv(p["gps"])
        origin = gps_raw.iloc[0]
        gps = gps_raw[["t", "e", "n", "u", "vel_e", "vel_n", "vel_u", "is_jump"]].rename(
            columns={"vel_e": "ve", "vel_n": "vn", "vel_u": "vu"})

        gt = pd.read_csv(p["gt"])[["t", "e", "n", "u", "motion"]]

        meta = {
            "source_files": list(FILES.values()),
            "accel_rows": len(accel), "gyro_rows": len(gyro), "imu_rows_matched": len(imu),
            "gps_origin_lat_lon_alt": [float(origin["lat"]), float(origin["lon"]), float(origin["alt"])],
            "gps_origin_note": "ENU origin is the first GPS fix (already applied in the source files)",
            "gt_velocity": "not provided",
        }
        return Recording(cand.id, cand.label, imu, gps, gt, meta)
