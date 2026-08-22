"""Loaders for the Burgard et al. NEMO ensembles.

The published archives spread one cavity-year across several files -- ice-shelf
masks and distances, bedrock and draft geometry, temperature and salinity at the
ice draft, and the melt rate itself -- and the variable names differ between the
2022 and 2023 releases and between file types.  Rather than hard-code names that
would break silently on the wrong file, every canonical field is resolved
through a list of candidate names, and failure raises with the variables the
file actually contains.

Layout of the extracted archives::

    <raw>/burgard2023/interim/ANTARCTICA_IS_MASKS/<SIM>/nemo_5km_isf_masks_...nc
    <raw>/burgard2023/interim/ANTARCTICA_IS_MASKS/<SIM>/nemo_5km_slope_info_...nc
    <raw>/burgard2023/interim/T_S_PROF/<SIM>/T_S_2D_fields_isf_draft_...nc
    <raw>/burgard2022/processed/PROCESSED_nemo_5km_<SIM>/...
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import RAW_DIR, SIMULATIONS

# --------------------------------------------------------------------------- #
# Variable-name resolution
# --------------------------------------------------------------------------- #

#: Canonical field -> candidate variable names, most specific first.
#:
#: Names confirmed against the extracted archives.  Note that the pieces are
#: spread across three different files and, for the geometry, a different grid:
#: masks and distances are 1200x1200 on the 5 km stereographic grid, whereas
#: ``corrected_draft_bathy_isf.nc`` is 1334x1334 and must be regridded before
#: use (see :func:`load_geometry`).
CANDIDATES: dict[str, tuple[str, ...]] = {
    # From processed/MELT_RATE/.../melt_rates_2D_*.nc
    "melt_rate": ("melt_m_ice_per_y", "melt_cavity", "melt_rate", "meltrate",
                  "fwfisf", "sowflisf", "melt"),
    # From interim/T_S_PROF/.../T_S_2D_fields_isf_draft_*.nc
    "T": ("theta_in", "temperature_in", "thetao", "votemper", "theta_ocean"),
    "S": ("salinity_in", "so", "vosaline", "salinity_ocean", "salinity"),
    "thermal_driving": ("thermal_forcing", "thermal_driving"),
    "freezing_T": ("freezing_T", "T_freeze"),
    "depth_of_int": ("depth_of_int", "depth_of_integration"),
    # From interim/.../corrected_draft_bathy_isf.nc  (1334x1334 grid)
    "ice_draft": ("corrected_isfdraft", "ice_draft", "isfdraft", "draft"),
    "bed_depth": ("corrected_isf_bathy", "bathy_metry", "bedrock_topography",
                  "bed", "bathymetry"),
    "water_column": ("water_column_thickness", "thickness_cavity", "wct"),
    # From interim/ANTARCTICA_IS_MASKS/.../nemo_5km_slope_info_*.nc
    "slope_ice_lon": ("slope_ice_lon",),
    "slope_ice_lat": ("slope_ice_lat",),
    "slope_bed_lon": ("slope_bed_lon",),
    "slope_bed_lat": ("slope_bed_lat",),
    "entry_depth": ("entry_depth_max",),
    # From interim/ANTARCTICA_IS_MASKS/.../nemo_5km_isf_masks_*.nc
    "u_mag": ("u_mag", "speed", "current_speed", "uo"),
    "dist_gl": ("dGL", "dist_from_grounding_line", "distance_gl"),
    "dist_front": ("dIF", "dist_from_front", "distance_front"),
    "isf_mask": ("ISF_mask", "mask_isf", "isf_mask"),
    "ground_mask": ("ground_mask",),
    "front_mask": ("IF_mask",),
    "gl_mask": ("GL_mask",),
    # Per-shelf metadata, dimension Nisf
    "isf_name": ("isf_name",),
    "isf_melt_obs": ("isf_melt",),
    "isf_area": ("isf_area_rignot", "isf_area_here"),
    "gl_flux": ("GL_flux",),
    "latitude": ("latitude", "lat", "nav_lat"),
    "longitude": ("longitude", "lon", "nav_lon"),
    "x": ("x", "X", "nav_x"),
    "y": ("y", "Y", "nav_y"),
}


class VariableNotFound(KeyError):
    """Raised when no candidate name for a canonical field is present."""


def resolve(ds, field: str, extra: tuple[str, ...] = ()) -> str:
    """Return the variable name in ``ds`` corresponding to a canonical field.

    Raises
    ------
    VariableNotFound
        Listing what the dataset does contain, which is the only useful thing to
        report when an archive layout turns out to differ from the expected one.
    """
    names = tuple(extra) + CANDIDATES.get(field, ())
    available = set(ds.variables) | set(ds.coords)
    for name in names:
        if name in available:
            return name
    # Fall back to a case-insensitive match before giving up.
    lower = {n.lower(): n for n in available}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    raise VariableNotFound(
        f"none of {names} found for field {field!r}; "
        f"available: {sorted(available)}")


def get(ds, field: str, extra: tuple[str, ...] = ()) -> np.ndarray:
    """Fetch a canonical field as a squeezed NumPy array."""
    return np.asarray(ds[resolve(ds, field, extra)].squeeze().values)


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class NemoFiles:
    """The files describing one simulation-year."""

    simulation: str
    year: int
    masks: Path | None = None
    geometry: Path | None = None
    ts_fields: Path | None = None
    melt: Path | None = None

    def complete(self) -> bool:
        return all(p is not None for p in (self.masks, self.geometry, self.ts_fields))

    def missing(self) -> list[str]:
        return [k for k in ("masks", "geometry", "ts_fields", "melt")
                if getattr(self, k) is None]


_YEAR = re.compile(r"_(\d{4})(?:_|\.)")


def _year_of(path: Path) -> int | None:
    m = _YEAR.search(path.name)
    return int(m.group(1)) if m else None


def discover(simulation: str, root: Path | None = None,
             release: str = "burgard2023") -> dict[int, NemoFiles]:
    """Index the available files for one simulation, keyed by year.

    Parameters
    ----------
    simulation
        Run identifier, e.g. ``OPM021`` or ``bi646``.
    root
        Raw data directory; defaults to the configured one.
    release
        Which extracted archive to search.
    """
    root = Path(root) if root is not None else RAW_DIR
    base = root / release / "interim"
    if not base.is_dir():
        raise FileNotFoundError(f"{base} does not exist; run scripts/00_download_data.py")

    found: dict[int, dict[str, Path]] = {}

    patterns = {
        "masks": (base / "ANTARCTICA_IS_MASKS" / simulation,
                  "*isf_masks_and_info_and_distance*.nc"),
        "geometry": (base / "ANTARCTICA_IS_MASKS" / simulation,
                     "*slope_info_bedrock_draft*.nc"),
        "ts_fields": (base / "T_S_PROF" / simulation,
                      "T_S_2D_fields_isf_draft*.nc"),
        "melt": (base / "MELT_RATE" / simulation, "*melt*.nc"),
    }

    for kind, (directory, glob) in patterns.items():
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob(glob)):
            year = _year_of(path)
            if year is None:
                continue
            found.setdefault(year, {})[kind] = path

    return {year: NemoFiles(simulation=simulation, year=year, **kinds)
            for year, kinds in sorted(found.items())}


def available_simulations(root: Path | None = None,
                          release: str = "burgard2023") -> list[str]:
    """Simulation identifiers present in an extracted archive."""
    root = Path(root) if root is not None else RAW_DIR
    base = root / release / "interim" / "ANTARCTICA_IS_MASKS"
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def find_geometry(simulation: str, root: Path | None = None) -> Path | None:
    """Locate the corrected draft and bathymetry file for a simulation.

    These live under the 2022 release in a per-simulation directory named after
    the full NEMO configuration, and on a different grid from everything else
    (1334x1334 rather than 1200x1200), so they must be regridded onto the mask
    grid before the fields can be combined.
    """
    root = Path(root) if root is not None else RAW_DIR
    base = root / "burgard2022" / "interim"
    if not base.is_dir():
        return None
    for directory in sorted(base.glob(f"NEMO_*{simulation}*")):
        candidate = directory / "corrected_draft_bathy_isf.nc"
        if candidate.is_file():
            return candidate
    return None


def find_melt(simulation: str, root: Path | None = None) -> Path | None:
    """Locate the 2-D reference melt field for a simulation."""
    root = Path(root) if root is not None else RAW_DIR
    for base, pattern in (
        (root / "burgard2022" / "processed" / "MELT_RATE",
         f"nemo_5km_{simulation}/melt_rates_2D_*_timmean_oneFRIS.nc"),
        (root / "burgard2023" / "processed" / "MELT_RATE",
         f"{simulation}/melt_rates_2D_*.nc"),
    ):
        if base.is_dir():
            hits = sorted(base.glob(pattern))
            if hits:
                return hits[0]
    return None


def shelf_index(masks_path: Path) -> dict[str, int]:
    """Map ice-shelf name to its integer id in ``ISF_mask``.

    The mask file carries a 136-element table of shelf names and ids; matching
    on name rather than on a hard-coded id is necessary because the numbering
    is not stable between the two releases.
    """
    with open_dataset(masks_path) as ds:
        names = ds[resolve(ds, "isf_name")].values
        ids = ds["Nisf"].values
    out: dict[str, int] = {}
    for name, ident in zip(names, ids):
        label = (name.decode() if isinstance(name, bytes) else str(name)).strip()
        if label and label.lower() != "nan":
            out[label] = int(ident)
    return out


def match_shelf(index: dict[str, int], shelf) -> int | None:
    """Resolve one of our :class:`IceShelf` entries against the archive's names."""
    wanted = [shelf.name] + list(getattr(shelf, "aliases", ()))
    normalised = {k.lower().replace("_", " ").replace("-", " "): v
                  for k, v in index.items()}
    for candidate in wanted:
        key = candidate.lower().replace("_", " ").replace("-", " ")
        if key in normalised:
            return normalised[key]
    # Fall back to a containment match, which catches e.g. "Ross_West".
    for candidate in wanted:
        key = candidate.lower().replace("_", " ").replace("-", " ")
        for name, ident in normalised.items():
            if key in name or name in key:
                return ident
    return None


# --------------------------------------------------------------------------- #
# Field assembly
# --------------------------------------------------------------------------- #

def open_dataset(path: Path):
    """Open a netCDF file with xarray, decoding times lazily."""
    import xarray as xr

    return xr.open_dataset(path, decode_times=False, mask_and_scale=True)


def inspect(path: Path) -> dict[str, tuple]:
    """Return ``{variable: shape}`` for a file.

    Used by the preprocessing script to report what an archive actually holds
    before anything is derived from it.
    """
    with open_dataset(path) as ds:
        return {str(k): tuple(v.shape) for k, v in ds.variables.items()}


def load_shelf_fields(files: NemoFiles, shelf_id: int,
                      scenario: str = "present_day") -> dict:
    """Assemble the gridded fields for one ice shelf from one simulation-year.

    Returns a dict with ``fields`` (canonical name -> 2-D array), ``mask``
    (boolean, the cavity cells of this shelf), ``x``/``y`` (projected metres),
    ``target`` (melt rate, m/yr) and provenance.  Derived quantities are added by
    :mod:`aisgnn.data.features`.
    """
    if not files.complete():
        raise FileNotFoundError(
            f"{files.simulation} {files.year}: missing {files.missing()}")

    with open_dataset(files.masks) as dm, \
         open_dataset(files.geometry) as dg, \
         open_dataset(files.ts_fields) as dt:

        isf = get(dm, "isf_mask")
        mask = isf == shelf_id
        if not mask.any():
            raise ValueError(f"shelf id {shelf_id} absent from {files.masks.name}")

        draft = get(dg, "ice_draft")
        bed = get(dg, "bed_depth")

        fields = {
            "T": get(dt, "T"),
            "S": get(dt, "S"),
            "ice_draft": draft,
            "bed_depth": bed,
            "water_column": np.abs(bed) - np.abs(draft),
            "dist_gl": get(dm, "dist_gl"),
            "dist_front": get(dm, "dist_front"),
        }

        lat = get(dm, "latitude")
        lon = get(dm, "longitude")

    return {"fields": fields, "mask": mask, "lat": lat, "lon": lon,
            "simulation": files.simulation, "year": files.year,
            "scenario": scenario, "shelf_id": int(shelf_id)}


def scenario_of(simulation: str) -> str:
    """Map a simulation identifier to its climate scenario."""
    sim = SIMULATIONS.get(simulation)
    return sim.scenario if sim else "unknown"
