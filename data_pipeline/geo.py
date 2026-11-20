"""Latitude / longitude / altitude to a local ENU frame around a chosen origin."""
import numpy as np

_A = 6378137.0            # WGS84 semi-major axis, m
_E2 = 0.00669437999014    # WGS84 first eccentricity squared


def latlon_to_enu(lat, lon, alt, lat0, lon0, alt0):
    """Tangent-plane ENU offsets from the origin (lat0, lon0, alt0), in metres.

    Uses the WGS84 radii of curvature at the origin latitude. Good for the extent of a
    single recording; it is not a full geodetic transform for very large distances.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    alt = np.asarray(alt, dtype=float)
    k = np.pi / 180.0
    s = np.sin(lat0 * k) ** 2
    meridional = _A * (1 - _E2) / (1 - _E2 * s) ** 1.5
    prime_vertical = _A / np.sqrt(1 - _E2 * s)
    e = (lon - lon0) * k * prime_vertical * np.cos(lat0 * k)
    n = (lat - lat0) * k * meridional
    u = alt - alt0
    return e, n, u


def enu_to_latlon(e, n, lat0, lon0):
    """Inverse of latlon_to_enu (horizontal only): ENU metres back to latitude and longitude."""
    k = np.pi / 180.0
    s = np.sin(lat0 * k) ** 2
    meridional = _A * (1 - _E2) / (1 - _E2 * s) ** 1.5
    prime_vertical = _A / np.sqrt(1 - _E2 * s)
    lat = lat0 + np.asarray(n, dtype=float) / (k * meridional)
    lon = lon0 + np.asarray(e, dtype=float) / (k * prime_vertical * np.cos(lat0 * k))
    return lat, lon
