"""Importing this package registers every adapter. Add a new layout by adding a module here
(or, for a table in a known layout, a TableSpec in land_vehicle.py or a new TableAdapter)."""
from . import eth, inavwhu, land_vehicle, quadrotor, remote_control_car, shikmona  # noqa: F401
