"""Dataset pipeline: raw files in raw_data/ -> one canonical Recording -> processed_data/.

    python -m data_pipeline              # list datasets and choose one
    python -m data_pipeline --list
    python -m data_pipeline --select quadrotor/eight-trajectory/single/first

    import data_pipeline as dp
    rec = dp.load("eth")                 # rec.imu, rec.gps, rec.gt, rec.meta
"""
from . import adapters  # noqa: F401  (registers every adapter)
from .core import (ADAPTERS, DEFAULT_OUT, DEFAULT_RAW, IMU_COLUMNS, POSE_COLUMNS, Adapter,
                   Candidate, Recording, discover, finalise, load, register, select, write)

__all__ = ["ADAPTERS", "Adapter", "Candidate", "DEFAULT_OUT", "DEFAULT_RAW", "IMU_COLUMNS",
           "POSE_COLUMNS", "Recording", "discover", "finalise", "load", "register", "select", "write"]
