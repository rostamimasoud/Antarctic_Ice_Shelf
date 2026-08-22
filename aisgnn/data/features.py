"""Derived fields for the emulator.

Turns the raw archived variables into the canonical node features.  Three points
where the archives and the model conventions disagree are handled here, and each
would otherwise corrupt the target silently:

* **Sign of depths.** ``corrected_isfdraft`` and ``corrected_isf_bathy`` are
  stored as positive magnitudes, whereas the freezing-point relation expects a
  negative-downward coordinate.  Feeding the positive values in would *raise*
  the freezing point with depth and invert the sign of thermal driving.
* **Grid extent.** Geometry is archived on a 1334x1334 grid and the masks on
  1200x1200.  Both are the same 5 km polar-stereographic grid, aligned on the
  same offsets, so the reconciliation is an exact coordinate selection rather
  than an interpolation.
* **Slopes.** Archived as separate longitudinal and latitudinal components; the
  emulator uses the magnitude.
"""

from __future__ import annotations

import numpy as np

from ..config import CONST, NODE_FEATURES


def coriolis(latitude: np.ndarray) -> np.ndarray:
    """Coriolis parameter from latitude in degrees."""
    return 2.0 * CONST.omega * np.sin(np.deg2rad(np.asarray(latitude, float)))


def slope_magnitude(d_lon: np.ndarray, d_lat: np.ndarray) -> np.ndarray:
    """Magnitude of a gradient archived as two components."""
    return np.hypot(np.asarray(d_lon, float), np.asarray(d_lat, float))


def as_depth(positive_magnitude: np.ndarray) -> np.ndarray:
    """Convert an archived positive depth magnitude to a negative-down coordinate."""
    return -np.abs(np.asarray(positive_magnitude, float))


def water_column(bathy: np.ndarray, draft: np.ndarray) -> np.ndarray:
    """Cavity water-column thickness from positive-magnitude bathymetry and draft."""
    return np.abs(np.asarray(bathy, float)) - np.abs(np.asarray(draft, float))


def align_to(source, target_x: np.ndarray, target_y: np.ndarray,
             tolerance: float = 1.0):
    """Select ``source`` onto the target grid coordinates.

    The geometry and mask grids share a projection and spacing but not extent, so
    an exact selection is both correct and cheap.  A tolerance of one metre
    guards against floating-point noise in the stored coordinates while still
    failing loudly if the grids are genuinely different.
    """
    return source.reindex(x=target_x, y=target_y, method="nearest",
                          tolerance=tolerance)


def clean(field: np.ndarray, mask: np.ndarray, name: str,
          fill: str = "median") -> np.ndarray:
    """Replace non-finite values inside the mask.

    Small numbers of missing cells occur at cavity edges where the archived
    fields disagree slightly about the mask.  Dropping those nodes would change
    the graph topology between features, so they are filled instead, and the
    fraction filled is returned to the caller's log rather than hidden.

    Raises
    ------
    ValueError
        If more than half the masked cells are non-finite, which means the wrong
        field or the wrong grid, not an edge effect.
    """
    field = np.asarray(field, float).copy()
    bad = ~np.isfinite(field) & mask
    n_bad = int(bad.sum())
    n_tot = int(mask.sum())
    if n_tot and n_bad / n_tot > 0.5:
        raise ValueError(f"{name}: {100 * n_bad / n_tot:.0f}% of masked cells "
                         f"are non-finite; wrong field or wrong grid")
    if n_bad:
        good = field[mask & np.isfinite(field)]
        value = float(np.median(good)) if good.size and fill == "median" else 0.0
        field[bad] = value
    return field


def assemble(raw: dict, mask: np.ndarray, latitude: np.ndarray
             ) -> tuple[dict, dict]:
    """Build the canonical node features from raw archived arrays.

    Parameters
    ----------
    raw
        Arrays keyed by archived name: ``theta_in``, ``salinity_in``,
        ``thermal_forcing``, ``corrected_isfdraft``, ``corrected_isf_bathy``,
        ``slope_ice_lon``/``lat``, ``slope_bed_lon``/``lat``, ``dGL``, ``dIF``,
        ``entry_depth_max``.
    mask
        Cavity cells for the shelf being assembled.

    Returns
    -------
    fields, report
        ``fields`` maps every name in :data:`NODE_FEATURES` to a 2-D array;
        ``report`` gives the fraction of masked cells filled per field.
    """
    draft = np.asarray(raw["corrected_isfdraft"], float)
    bathy = np.asarray(raw["corrected_isf_bathy"], float)

    fields = {
        "T": raw["theta_in"],
        "S": raw["salinity_in"],
        "thermal_driving": raw["thermal_forcing"],
        "ice_draft": as_depth(draft),
        "bed_depth": as_depth(bathy),
        "water_column": water_column(bathy, draft),
        "slope_ice": slope_magnitude(raw["slope_ice_lon"], raw["slope_ice_lat"]),
        "slope_bed": slope_magnitude(raw["slope_bed_lon"], raw["slope_bed_lat"]),
        "dist_gl": raw["dGL"],
        "dist_front": raw["dIF"],
        "coriolis": coriolis(latitude),
        "entry_depth": as_depth(raw["entry_depth_max"]),
    }

    missing = [f for f in NODE_FEATURES if f not in fields]
    if missing:
        raise KeyError(f"assembled fields are missing {missing}")

    report = {}
    for name in NODE_FEATURES:
        arr = np.asarray(fields[name], float)
        n_tot = int(mask.sum())
        report[name] = (float((~np.isfinite(arr) & mask).sum()) / n_tot
                        if n_tot else 0.0)
        fields[name] = clean(arr, mask, name)

    return fields, report


def brunt_vaisala(temperature: np.ndarray, salinity: np.ndarray,
                  depth: np.ndarray) -> np.ndarray:
    """Squared buoyancy frequency from a vertical profile.

    Computed with the same linear equation of state used by the box model, so
    the stratification the emulator sees and the stratification the box model
    responds to are defined consistently.
    """
    rho = CONST.rho_sw * (1.0 - CONST.alpha_T * (np.asarray(temperature, float) - CONST.T0)
                          + CONST.beta_S * (np.asarray(salinity, float) - CONST.S0))
    z = np.abs(np.asarray(depth, float))
    drho_dz = np.gradient(rho, z, edge_order=1)
    return CONST.g / CONST.rho_sw * drho_dz
