"""Wind-field divergence utilities."""

from __future__ import annotations

import numpy as np


def compute_divergence(u, v, lat, lon):
    """Compute horizontal divergence on a general curvilinear lat/lon grid."""
    r_earth = 6371000.0  # Earth radius in meters

    u = np.asarray(u)
    v = np.asarray(v)
    lat = np.asarray(lat)
    lon = np.asarray(lon)

    if u.shape != v.shape:
        raise ValueError(f"u and v must have the same shape, got {u.shape} and {v.shape}")

    if lat.ndim == 1 and lon.ndim == 1:
        lon_2d, lat_2d = np.meshgrid(lon, lat)
    elif lat.ndim == 2 and lon.ndim == 2:
        if lat.shape != lon.shape:
            raise ValueError(
                f"lat and lon 2D arrays must have same shape, got {lat.shape} and {lon.shape}"
            )
        lat_2d, lon_2d = lat, lon
    else:
        raise ValueError("lat and lon must both be 1D arrays or both be 2D arrays")

    if u.shape != lat_2d.shape:
        raise ValueError(
            f"u/v shape {u.shape} is incompatible with lat/lon shape {lat_2d.shape}"
        )

    # Convert to radians; unwrap longitude to avoid artificial jumps at the dateline.
    lat_rad = np.radians(lat_2d)
    lon_rad = np.unwrap(np.unwrap(np.radians(lon_2d), axis=1), axis=0)

    # Derivatives in index space (y=row, x=column).
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)
    dlam_dy, dlam_dx = np.gradient(lon_rad)
    dphi_dy, dphi_dx = np.gradient(lat_rad)

    # Invert local Jacobian to map index-space derivatives to (lambda, phi) derivatives.
    det = dlam_dx * dphi_dy - dlam_dy * dphi_dx
    det = np.where(np.abs(det) < 1e-14, np.nan, det)

    du_dlam = (du_dx * dphi_dy - du_dy * dphi_dx) / det
    dv_dphi = (dlam_dx * dv_dy - dlam_dy * dv_dx) / det

    cos_lat = np.cos(lat_rad)
    cos_lat = np.where(np.abs(cos_lat) < 1e-12, np.nan, cos_lat)

    divergence = du_dlam / (r_earth * cos_lat) + dv_dphi / r_earth

    print(f"Divergence shape: {divergence.shape}")
    print(f"Divergence range: [{np.nanmin(divergence):.2e}, {np.nanmax(divergence):.2e}]")
    return divergence
