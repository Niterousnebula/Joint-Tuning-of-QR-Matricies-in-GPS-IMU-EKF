"""Remote-control-car recordings (raw_data/Remote control car/<course>/<n>.csv).

A Samsung Galaxy S8 on top of a toy car, 100 Hz accelerometer and gyroscope, x axis along
the driving direction. There is no GPS and no ground-truth trajectory: only the final
position of each course is known (readme.txt), in the start frame with x along the first
driving direction. So these recordings test IMU-only dead reckoning, not a GPS-aided EKF,
and are marked not ekf_ready by the pipeline.
"""
from ..core import register
from .table import TableAdapter, TableSpec

FOLDER = "Remote control car"
IMU = {"ax": "acc_x", "ay": "acc_y", "az": "acc_z", "wx": "gyro_x", "wy": "gyro_y", "wz": "gyro_z"}

FINAL_POSITION = {
    "line": "(6.3, 0, 0) m: the 6.3 m straight line",
    "circle": "(0, 0, 0) m: back at the start of the 1.98 m diameter circle",
    "square": "(0, 0, 0) m: back at the start of the 2.25 m square",
}

SPECS = tuple(
    TableSpec(
        id=f"remote-control-car/{course}/run-{{stem}}",
        label=f"Remote control car: {course} run {{stem}} (100 Hz, IMU only)",
        file=f"{FOLDER}/{course.capitalize()}/*.csv",
        time="seconds_elapsed", time_scale=1.0, imu=IMU,
        notes=f"No GPS. Final position {FINAL_POSITION[course]}, in the start frame (x along the first "
              "driving direction). Gyro assumed rad/s (Android); accelerometer is in m/s^2.",
    )
    for course in ("line", "circle", "square")
)


@register
class RemoteControlCar(TableAdapter):
    specs = SPECS
