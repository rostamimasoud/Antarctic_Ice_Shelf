"""Configuration: filesystem layout, dataset registry, physical constants.

All paths are resolved from the ``AISGNN_ROOT`` environment variable so that the
same code runs unchanged on a laptop and on the HPC.  On the HPC set::

    export AISGNN_ROOT=/p/projects/climber3/rostami/Antarctic_Ice_Shelf
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #

ROOT = Path(os.environ.get("AISGNN_ROOT", Path.cwd())).expanduser().resolve()

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
GRAPH_DIR = DATA_DIR / "graphs"

RUN_DIR = ROOT / "runs"
BOXMODEL_DIR = RUN_DIR / "boxmodel"
TRAIN_DIR = RUN_DIR / "train"
SWEEP_DIR = RUN_DIR / "sweeps"
ENSEMBLE_DIR = RUN_DIR / "ensembles"

FIGURE_DIR = ROOT / "figures"
LOG_DIR = ROOT / "logs"

_ALL_DIRS = (
    RAW_DIR, INTERIM_DIR, PROCESSED_DIR, GRAPH_DIR,
    BOXMODEL_DIR, TRAIN_DIR, SWEEP_DIR, ENSEMBLE_DIR,
    FIGURE_DIR, LOG_DIR,
)


def ensure_dirs() -> None:
    """Create the full directory tree if it does not already exist."""
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Physical constants (SI unless stated)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Constants:
    # Seawater freezing point, Jenkins (1991) linearised form:
    #   T_f = lam1 * S + lam2 - lam3 * |z|,  with z negative downward
    lam1: float = -5.73e-2      # degC per g/kg
    lam2: float = 8.32e-2       # degC
    lam3: float = 7.61e-4       # degC per m; enters as lam3 * z with z < 0,
    #                             so the freezing point falls with depth

    rho_sw: float = 1028.0      # seawater density, kg/m3
    rho_i: float = 917.0        # ice density, kg/m3
    rho_fw: float = 1000.0      # freshwater density, kg/m3
    c_pw: float = 3974.0        # seawater heat capacity, J/(kg K)
    c_pi: float = 2009.0        # ice heat capacity, J/(kg K)
    L_f: float = 3.34e5         # latent heat of fusion, J/kg

    # Linear equation of state about (T0, S0)
    alpha_T: float = 3.87e-5    # thermal expansion, 1/degC
    beta_S: float = 7.86e-4     # haline contraction, 1/(g/kg)
    T0: float = -1.0            # reference temperature, degC
    S0: float = 34.5            # reference salinity, g/kg

    g: float = 9.81             # gravity, m/s2
    omega: float = 7.2921e-5    # Earth rotation rate, rad/s

    sec_per_year: float = 365.25 * 86400.0


CONST = Constants()


# --------------------------------------------------------------------------- #
# Dataset registry
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ZenodoRecord:
    """A Zenodo record, optionally restricted to a subset of its files."""

    record_id: str
    label: str
    description: str
    files: tuple[str, ...] = ()          # empty -> download every file
    subdir: str = ""                     # placed under data/raw/<subdir>

    @property
    def api_url(self) -> str:
        return f"https://zenodo.org/api/records/{self.record_id}"


#: Burgard et al. (2023) emulator dataset -- primary training data.
BURGARD2023 = ZenodoRecord(
    record_id="10149919",
    label="burgard2023",
    description="Emulating present and future melt rates at the base of Antarctic ice shelves",
    files=(
        "PROCESSED.zip",
        "INTERIM_INPUT_DATA.zip",
        "INTERIM_ANTARCTICA_IS_MASKS.zip",
        "INTERIM_T_S_PROF.zip",
        "INTERIM_SMITH_bf663.zip",     # REPEAT1970
        "INTERIM_SMITH_bi646.zip",     # 4xCO2
        "INTERIM_BOXES.zip",
        "INTERIM_PLUMES.zip",
        "RAW.zip",
    ),
    subdir="burgard2023",
)

#: Burgard et al. (2022) parameterisation assessment -- 5 km NEMO fields + geometry.
BURGARD2022 = ZenodoRecord(
    record_id="7308352",
    label="burgard2022",
    description="An assessment of basal melt parameterisations for Antarctic ice shelves",
    files=(
        "PROCESSED_nemo_5km_OPM006.zip",   # HIGHGETZ
        "PROCESSED_nemo_5km_OPM016.zip",   # WARMROSS
        "PROCESSED_nemo_5km_OPM018.zip",   # COLDAMU
        "PROCESSED_nemo_5km_OPM021.zip",   # REALISTIC
        "INTERIM_geometry_interp_OPM006.zip",
        "INTERIM_geometry_interp_OPM016.zip",
        "INTERIM_geometry_interp_OPM018.zip",
        "INTERIM_geometry_interp_OPM021.zip",
        "INTERIM_T_S_PROF.zip",
        "INTERIM_SIMPLE.zip",
        "PROCESSED_BedMachine_for_comparison.zip",
    ),
    subdir="burgard2022",
)

#: MISOMIP2 phase-1 Amundsen hindcasts -- three models, identical protocol.
MISOMIP2_ROMS = ZenodoRecord(
    record_id="21728621",
    label="misomip2_roms_utas",
    description="MISOMIP2 OceanA-hind, ROMS-UTAS v1.0 (~2 km Amundsen, 31 s-levels)",
    subdir="misomip2/roms_utas",
)

MISOMIP2_NEMO4 = ZenodoRecord(
    record_id="21514655",
    label="misomip2_nemo4",
    description="MISOMIP2 Ocean-hind, IGE-CNRS-UGA NEMO4.0",
    subdir="misomip2/nemo4",
)

MISOMIP2_MITGCM = ZenodoRecord(
    record_id="21626519",
    label="misomip2_mitgcm",
    description="MISOMIP2 Ocean-hind, UCLA-UMD MITgcm",
    subdir="misomip2/mitgcm",
)

MISOMIP2_MIPKIT_A = ZenodoRecord(
    record_id="21679622",
    label="misomip2_mipkit_a",
    description="MISOMIP2 MIPkit-A observational dataset (Amundsen)",
    subdir="misomip2/mipkit_a",
)

RECORDS: dict[str, ZenodoRecord] = {
    r.label: r
    for r in (
        BURGARD2023, BURGARD2022,
        MISOMIP2_ROMS, MISOMIP2_NEMO4, MISOMIP2_MITGCM, MISOMIP2_MIPKIT_A,
    )
}


# --------------------------------------------------------------------------- #
# Simulation / scenario registry
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Simulation:
    key: str
    name: str
    scenario: str            # present_day | repeat1970 | 4xCO2 | hindcast
    source: str              # nemo | nemo4 | roms | mitgcm
    note: str = ""


#: NEMO training runs of Burgard et al. (2022, 2023).
NEMO_TRAIN = (
    Simulation("OPM006", "HIGHGETZ", "present_day", "nemo", "warm-biased Getz"),
    Simulation("OPM016", "WARMROSS", "present_day", "nemo", "warm-biased Ross"),
    Simulation("OPM018", "COLDAMU", "present_day", "nemo", "cold-biased Amundsen"),
    Simulation("OPM021", "REALISTIC", "present_day", "nemo", "best present-day estimate"),
)

#: Smith et al. (2021) HadGEM3-forced NEMO runs used as the independent test set.
NEMO_TEST = (
    Simulation("bf663", "REPEAT1970", "repeat1970", "nemo", "pre-industrial-like forcing"),
    Simulation("bi646", "4xCO2", "4xCO2", "nemo", "abrupt quadrupled CO2"),
)

SIMULATIONS = {s.key: s for s in NEMO_TRAIN + NEMO_TEST}

SCENARIO_ORDER = ("repeat1970", "present_day", "4xCO2")


# --------------------------------------------------------------------------- #
# Ice shelves of interest
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class IceShelf:
    name: str
    sector: str
    regime: str              # cold | warm | intermediate
    aliases: tuple[str, ...] = field(default_factory=tuple)


#: The nine cavities analysed with the low-dimensional model, plus the two
#: large cold-cavity systems that dominate the tipping-point literature.
TARGET_SHELVES = (
    IceShelf("Filchner-Ronne", "Weddell", "cold", ("Filchner_Ronne", "FRIS", "Ronne")),
    IceShelf("Ross", "Ross", "cold", ("Ross_West", "Ross_East", "RIS")),
    IceShelf("Amery", "East Antarctica", "cold", ("Amery",)),
    IceShelf("Fimbul", "East Antarctica", "cold", ("Fimbulisen", "Fimbul")),
    IceShelf("Larsen C", "Weddell", "cold", ("LarsenC", "Larsen_C")),
    IceShelf("Riiser-Larsen", "East Antarctica", "cold", ("RiiserLarsen",)),
    IceShelf("Shackleton", "East Antarctica", "intermediate", ("Shackleton",)),
    IceShelf("Totten", "East Antarctica", "warm", ("Totten",)),
    IceShelf("Getz", "Amundsen", "warm", ("Getz",)),
    IceShelf("Pine Island", "Amundsen", "warm", ("PineIsland", "Pine_Island", "PIG")),
    IceShelf("Thwaites", "Amundsen", "warm", ("Thwaites",)),
)

SHELVES = {s.name: s for s in TARGET_SHELVES}

#: Subset used for the Stage-1 smoke tests (spans cold, intermediate and warm).
SMOKE_TEST_SHELVES = ("Filchner-Ronne", "Ross", "Getz", "Pine Island", "Thwaites")


# --------------------------------------------------------------------------- #
# Feature definitions
# --------------------------------------------------------------------------- #

#: Node features fed to every emulator, in a fixed order.
#:
#: This list is what the published archives actually contain, which is less than
#: the research plan assumed.  The NEMO ensembles distribute temperature,
#: salinity, thermal forcing and geometry at the ice draft, but **no horizontal
#: velocity, mixed-layer depth or in-cavity stratification**.  Any hypothesis
#: about a shift from thermal-driving control to circulation control therefore
#: cannot be tested on these fields directly; stratification has to come from
#: the separately archived continental-shelf T/S profiles (see
#: :data:`PROFILE_FEATURES`), and velocity is simply unavailable.
NODE_FEATURES = (
    "T",              # potential temperature at the ice draft, degC
    "S",              # practical salinity at the ice draft, g/kg
    "thermal_driving",  # T - T_f, degC, supplied directly as thermal_forcing
    "ice_draft",      # depth of the ice base, m (negative down)
    "water_column",   # cavity water-column thickness, m
    "bed_depth",      # bedrock depth, m (negative down)
    "slope_ice",      # magnitude of the ice-draft gradient, dimensionless
    "slope_bed",      # magnitude of the bedrock gradient, dimensionless
    "dist_gl",        # distance to the grounding line, m
    "dist_front",     # distance to the ice-shelf front, m
    "coriolis",       # f = 2 Omega sin(lat), 1/s
    "entry_depth",    # deepest depth at which water can enter the cavity, m
)

#: Derived from the archived continental-shelf profiles, appended when available.
PROFILE_FEATURES = (
    "N2_shelf",       # squared Brunt-Vaisala frequency on the shelf, 1/s2
    "T_gradient",     # vertical temperature gradient at the draft depth, degC/m
)

#: Edge features for the edge-conditioned network.
EDGE_FEATURES = (
    "distance",        # great-circle / projected separation, m
    "d_bed",           # bedrock depth difference, m
    "d_draft",         # ice-draft difference, m
    "bearing_sin",     # sin of the edge bearing
    "bearing_cos",     # cos of the edge bearing
    "along_contour",   # alignment with the local water-column-thickness contour
)

TARGET = "melt_rate"   # basal melt rate, m of ice per year
