#!/usr/bin/env python3
"""Build per-shelf, per-year cavity graphs from the NEMO archives.

Everything for one graph comes from a single simulation and a single year:
masks, distances and slope geometry, temperature and salinity at the ice draft,
the time-varying draft and bathymetry, and the reference melt.  Mixing sources
would be easy here and quietly wrong -- an earlier version paired melt from one
run with temperature from another, which is not a prediction problem the ocean
model ever poses.

Two archive conventions are reconciled:

* melt is stored as a mass flux in kg/m2/s for the SMITH runs and as m/yr for
  the OPM runs, so it is converted explicitly rather than assumed;
* melt and geometry are on a 1334-square grid while masks and T/S are on a
  1200-square grid, both the same 5 km polar-stereographic grid aligned on the
  same offsets, so the two are reconciled by exact coordinate selection.

Examples
--------
    python scripts/02_build_graphs.py --dry-run
    python scripts/02_build_graphs.py --simulation SMITH_bf663 --years 1990 2000
    python scripts/02_build_graphs.py --all-scenarios --every 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aisgnn.config import (                                        # noqa: E402
    GRAPH_DIR,
    NODE_FEATURES,
    RAW_DIR,
    SHELVES,
    SMOKE_TEST_SHELVES,
    ensure_dirs,
)
from aisgnn.data.features import align_to, assemble                # noqa: E402
from aisgnn.data.graph import build_graph                          # noqa: E402
from aisgnn.data.nemo import (                                     # noqa: E402
    discover,
    first_year,
    find_geometry,
    find_melt,
    get,
    match_shelf,
    check_melt_sign,
    melt_to_m_per_year,
    open_dataset,
    resolve,
    scenario_of,
    select_year,
    shelf_index,
)

DEFAULT_RADIUS = 12_000.0        # 5 km grid, so roughly two cells each way
CELL_AREA = 25.0e6               # m2

#: The two SMITH runs give the scenario contrast the analysis needs.
SCENARIO_RUNS = ("SMITH_bf663", "SMITH_bi646")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_year(simulation: str, year: int, shelves: list[str], radius: float,
               root: Path, outdir: Path, dry_run: bool,
               handles: dict) -> dict:
    """Build every requested shelf for one simulation-year."""
    files = handles["index"][year]
    with open_dataset(files.masks) as dm, open_dataset(files.ts_fields) as dts:

        target_x, target_y = dm.x.values, dm.y.values
        xg, yg = np.meshgrid(target_x, target_y)

        isf = get(dm, "isf_mask")
        lat = get(dm, "latitude")

        base = handles["base_year"]
        geom = align_to(select_year(handles["geom"], year, base_year=base),
                        target_x, target_y)
        melt_ds = select_year(handles["melt"], year, base_year=base)
        melt_name = resolve(melt_ds, "melt_rate")
        melt_units = getattr(melt_ds[melt_name], "units", "")
        melt = align_to(melt_ds, target_x, target_y)[melt_name].values
        melt = melt_to_m_per_year(np.squeeze(melt), melt_units)

        raw = {
            "theta_in": get(dts, "T"),
            "salinity_in": get(dts, "S"),
            "thermal_forcing": get(dts, "thermal_driving"),
            "corrected_isfdraft": np.squeeze(geom["corrected_isfdraft"].values),
            "corrected_isf_bathy": np.squeeze(geom["corrected_isf_bathy"].values),
            "dGL": get(dm, "dist_gl"),
            "dIF": get(dm, "dist_front"),
        }

        names = shelf_index(files.masks)
        scenario = scenario_of(simulation)
        out: dict[str, dict] = {}

        for shelf_name in shelves:
            ident = match_shelf(names, SHELVES[shelf_name])
            if ident is None:
                continue
            mask = (isf == ident) & np.isfinite(melt)
            if mask.sum() < 20:
                continue
            check_melt_sign(melt, mask, f"{simulation} {shelf_name} {year}")

            if dry_run:
                out[shelf_name] = {"cells": int(mask.sum()),
                                   "melt_min": float(np.nanmin(melt[mask])),
                                   "melt_max": float(np.nanmax(melt[mask]))}
                continue

            # One unusable shelf must not discard the rest of the year.  Small
            # cavities occasionally fall outside the valid region of the
            # temperature field, and aborting here would silently drop every
            # shelf processed after it.
            try:
                fields, report = assemble(raw, mask, lat)
                graph = build_graph(fields, mask, xg, yg, melt, radius=radius,
                                    cell_area=CELL_AREA, shelf=shelf_name,
                                    scenario=scenario, simulation=simulation,
                                    node_features=NODE_FEATURES)
            except (ValueError, KeyError) as exc:
                log(f"    {shelf_name}: skipped -- {exc}")
                continue

            slug = shelf_name.replace(" ", "_")
            path = outdir / f"{simulation}_{slug}_{year}.npz"
            graph.save(path)

            out[shelf_name] = {
                "nodes": graph.n_nodes, "edges": graph.n_edges,
                "melt_mean": float(graph.y.mean()),
                "melt_min": float(graph.y.min()), "melt_max": float(graph.y.max()),
                "path": str(path),
                "filled": {k: v for k, v in report.items() if v > 0},
            }

    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--simulation", default="SMITH_bf663")
    p.add_argument("--all-scenarios", action="store_true",
                   help=f"build both of {SCENARIO_RUNS}")
    p.add_argument("--years", nargs="*", type=int, default=None,
                   help="explicit years (default: every --every-th available year)")
    p.add_argument("--every", type=int, default=10,
                   help="stride through the available years (default: 10)")
    p.add_argument("--shelves", nargs="*", default=list(SMOKE_TEST_SHELVES))
    p.add_argument("--all-shelves", action="store_true")
    p.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    p.add_argument("--root", type=Path, default=RAW_DIR)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    ensure_dirs()
    outdir = args.outdir or GRAPH_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    shelves = sorted(SHELVES) if args.all_shelves else args.shelves
    unknown = [s for s in shelves if s not in SHELVES]
    if unknown:
        raise SystemExit(f"unknown shelves {unknown}")

    runs = list(SCENARIO_RUNS) if args.all_scenarios else [args.simulation]
    manifest: dict[str, dict] = {}

    for simulation in runs:
        index = discover(simulation, root=args.root)
        complete = sorted(y for y, f in index.items() if f.complete())
        if not complete:
            log(f"{simulation}: no complete years, skipping")
            continue

        melt_path = find_melt(simulation, root=args.root)
        geom_path = find_geometry(simulation, root=args.root)
        if melt_path is None or geom_path is None:
            log(f"{simulation}: melt={melt_path} geometry={geom_path}, skipping")
            continue

        years = args.years or complete[::args.every]
        years = [y for y in years if y in index]
        log(f"{simulation} ({scenario_of(simulation)}): {len(years)} years "
            f"{years[0]}-{years[-1]}, {len(shelves)} shelves")
        log(f"  melt     {melt_path.name}")
        log(f"  geometry {geom_path.name}")

        with open_dataset(melt_path) as dmelt, open_dataset(geom_path) as dgeom:
            base = first_year(dgeom) or 1970
            handles = {"index": index, "melt": dmelt, "geom": dgeom,
                       "base_year": base}
            log(f"  years anchored at {base}")
            for year in years:
                try:
                    res = build_year(simulation, year, shelves, args.radius,
                                     args.root, outdir, args.dry_run, handles)
                except (KeyError, ValueError, OSError) as exc:
                    log(f"  {year}: failed -- {exc}")
                    continue
                manifest[f"{simulation}_{year}"] = res
                total = sum(v.get("nodes", v.get("cells", 0)) for v in res.values())
                log(f"  {year}: {len(res)} shelves, {total} nodes")

    if not args.dry_run and manifest:
        (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        n = sum(len(v) for v in manifest.values())
        log(f"wrote {n} graphs to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
