"""Land-vehicle datasets (raw_data/Land vehicle/). Each one is a TableSpec entry.

Units, checked against the data (not only the Description.txt):
  - MRU accelerometer is in g (stationary Acc_Z reads about 1): scaled to m/s^2.
  - MRU gyro is in deg/s, not the rad/s stated in the description: Gyro_Z correlates
    -0.99 with the heading rate, with slope -pi/180. Scaled to rad/s. The Stationary file
    has no heading change to test, so its gyro scale is inferred from the same device.
  - Smartphone logs m/s^2 and rad/s already.
Gyro sign: Gyro_Z is opposite to the heading change. The pipeline does not re-sign it.

Gaps and bad fixes (see the spec fields in table.py):
  - KRAYOT MRU has one 104,418 s time jump; the smartphone has a block of rows 6.7 days
    away and two pauses. Each file keeps its longest continuous part.
  - ISFIYA logs about 37 % of its fixes outside the Haifa area (lat up to 38.5, lon down
    to -33); those are dropped by gps_box. Its altitude stays suspect (see processed
    meta.json and the notes in the summary).
"""
from ..core import register
from .table import TableAdapter, TableSpec

G = 9.80665
DEG2RAD = 3.141592653589793 / 180.0
FOLDER = "Land vehicle"
HAIFA_BOX = (32.0, 33.5, 34.0, 36.0)          # lat_min, lat_max, lon_min, lon_max (degrees)
GAP_S = 5.0

MRU_IMU = {"ax": "Acc_X", "ay": "Acc_Y", "az": "Acc_Z", "wx": "Gyro_X", "wy": "Gyro_Y", "wz": "Gyro_Z"}
MRU_GPS = {"lat": "Lat", "lon": "Long", "alt": "Altitude", "ve": "V_East", "vn": "V_North", "vu": "V_Up"}

SPECS = (
    TableSpec(
        id="land-vehicle/krayot-mru", label="Land vehicle: KRAYOT MRU 2021 (100 Hz)",
        file=f"{FOLDER}/KRAYOT_MRU_2021.xlsx", time="Time", time_scale=1e-3,
        imu=MRU_IMU, accel_scale=G, gyro_scale=DEG2RAD, gps=MRU_GPS,
        gps_box=HAIFA_BOX, gap_split_s=GAP_S,
        notes=("Velocity channels do not match the GPS-implied motion (correlation about 0), "
               "so treat ve/vn/vu as unverified. Gyro scale checked against Heading."),
    ),
    TableSpec(
        id="land-vehicle/isfiya-mru", label="Land vehicle: ISFIYA MRU 2021 (100 Hz)",
        file=f"{FOLDER}/ISFIYA_MRU2021.xlsx", time="ms_gps", time_scale=1e-3,
        imu=MRU_IMU, accel_scale=G, gyro_scale=DEG2RAD,
        gps={"lat": "Latitude", "lon": "Longitude", "alt": "Altitude",
             "ve": "V_East", "vn": "V_North", "vu": "V_Up"},
        gps_box=HAIFA_BOX, gap_split_s=GAP_S,
        notes=("About 37 % of fixes are outside the Haifa box and are dropped. Altitude is "
               "implausible in the raw file (-10000 to 33962 m); values outside the altitude range "
               "are set to NaN, but the rest is still suspect. About 100 implied speeds stay above "
               "50 m/s after the speed gate, so the track is only partly clean. Velocity is "
               "unverified. Gyro scale checked against Heading."),
    ),
    TableSpec(
        id="land-vehicle/stationary-mru", label="Land vehicle: stationary MRU 2021 (100 Hz)",
        file=f"{FOLDER}/Stationary_MRU_2021.xlsx", time="Time", time_scale=1e-3,
        imu=MRU_IMU, accel_scale=G, gyro_scale=DEG2RAD,
        gps={"lat": "Lat", "lon": "Long", "alt": "Altitude"},
        gps_box=HAIFA_BOX, gap_split_s=GAP_S,
        notes=("Stationary: no heading change to check the gyro against, so the deg/s scale "
               "is inferred from the same MRU unit. One row has a '=#REF!' formula and is dropped."),
    ),
    TableSpec(
        id="land-vehicle/krayot-smartphone", label="Land vehicle: KRAYOT smartphone (250 Hz)",
        file=f"{FOLDER}/KRAYOT_Smartphone.xlsx", time="Timestamp", time_scale=1e-3,
        imu={"ax": "accX", "ay": "accY", "az": "accZ", "wx": "GyrX", "wy": "GyrY", "wz": "GyrZ"},
        gps={"lat": "lat", "lon": "long"},
        gps_box=HAIFA_BOX, gap_split_s=GAP_S,
        notes=("Timestamps have a block of 250 rows about 6.7 days away and two pauses; "
               "the longest of the four continuous parts is kept. No altitude or velocity channel."),
    ),
)


@register
class LandVehicle(TableAdapter):
    specs = SPECS
