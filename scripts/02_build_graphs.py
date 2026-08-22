#!/usr/bin/env python3
"""Build per-shelf cavity graphs from the NEMO archives.

Combines the four archived pieces -- masks and distances, slope geometry,
temperature and salinity at the ice draft, and the reference NEMO melt -- into
one graph per ice shelf, and writes them to ``data/graphs``.

The reference melt is ``melt_rates_2D_NEMO*.nc``, not the ``melt_rates_2D_boxes``
or ``_plumes`` files in the same directory: those hold parameterised melt from
PICO-style schemes, and training an emulator on them would reproduce a
parameterisation rather than the ocean model.

Examples
--------
    python scripts/02_build_graphs.py --dry-run
    python scripts/02_build_graphs.py --shelves Filchner-Ronne Ross
    python scripts/02_build_graphs.py --simulation OPM021 --radius 12000
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
    find_geometry,
    find_melt,
    get,
    match_shelf,
    open_dataset,
    resolve,
    shelf_index,
)

#: 5 km grid, so this reaches roughly two cells in each direction.
DEFAULT_RADIUS = 12_000.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_for_simulation(simulation: str, mask_simulation: str, year: int,
                         shelves: list[str], radius: float, root: Path,
                         outdir: Path, dry_run: bool) -> dict:
    """Build graphs for every requested shelf of one simulation."""
    index = discover(mask_simulation, root=root)
    if year not in index:
        raise SystemExit(f"{mask_simulation}: year {year} not indexed; "
                         f"available {sorted(index)[:5]}...")
    files = index[year]

    geom_path = find_geometry(simulation, root=root)
    melt_path = find_melt(simulation, root=root)
    if geom_path is None or melt_path is None:
        raise SystemExit(f"{simulation}: geometry={geom_path} melt={melt_path}")

    log(f"masks    {files.masks.name}")
    log(f"slopes   {files.geometry.name}")
    log(f"T/S      {files.ts_fields.name}")
    log(f"geometry {geom_path.parent.name}/{geom_path.name}")
    log(f"melt     {melt_path.parent.name}/{melt_path.name}")

    summary: dict[str, dict] = {}

    with open_dataset(files.masks) as dm, \
         open_dataset(files.geometry) as dslope, \
         open_dataset(files.ts_fields) as dts, \
         open_dataset(geom_path) as dgeom, \
         open_dataset(melt_path) as dmelt:

        target_x, target_y = dm.x.values, dm.y.values
        geom = align_to(dgeom, target_x, target_y)

        isf = get(dm, "isf_mask")
        lat = get(dm, "latitude")
        xg, yg = np.meshgrid(target_x, target_y)

        melt = np.asarray(dmelt[resolve(dmelt, "melt_rate")].values)
        while melt.ndim > 2:                 # collapse time / parameter axes
            melt = np.nanmean(melt, axis=0)

        raw = {
            "theta_in": get(dts, "T"),
            "salinity_in": get(dts, "S"),
            "thermal_forcing": get(dts, "thermal_driving"),
            "corrected_isfdraft": np.asarray(geom["corrected_isfdraft"].values),
            "corrected_isf_bathy": np.asarray(geom["corrected_isf_bathy"].values),
            "slope_ice_lon": get(dslope, "slope_ice_lon"),
            "slope_ice_lat": get(dslope, "slope_ice_lat"),
            "slope_bed_lon": get(dslope, "slope_bed_lon"),
            "slope_bed_lat": get(dslope, "slope_bed_lat"),
            "dGL": get(dm, "dist_gl"),
            "dIF": get(dm, "dist_front"),
            "entry_depth_max": get(dslope, "entry_depth"),
        }

        names = shelf_index(files.masks)

        for shelf_name in shelves:
            shelf = SHELVES[shelf_name]
            ident = match_shelf(names, shelf)
            if ident is None:
                log(f"  {shelf_name:16s} not present in this archive")
                continue

            mask = (isf == ident) & np.isfinite(melt)
            if mask.sum() < 20:
                log(f"  {shelf_name:16s} only {int(mask.sum())} valid cells, skipping")
                continue

            fields, report = assemble(raw, mask, lat)
            filled = {k: v for k, v in report.items() if v > 0.0}

            if dry_run:
                log(f"  {shelf_name:16s} id={ident:3d} cells={int(mask.sum()):6d} "
                    f"melt {np.nanmin(melt[mask]):6.2f} to {np.nanmax(melt[mask]):6.2f} m/yr"
                    + (f"  filled={filled}" if filled else ""))
                summary[shelf_name] = {"id": ident, "cells": int(mask.sum())}
                continue

            graph = build_graph(
                fields, mask, xg, yg, melt, radius=radius,
                cell_area=25.0e6, shelf=shelf_name,
                scenario="present_day", simulation=simulation,
                node_features=NODE_FEATURES)

            path = outdir / f"{simulation}_{shelf_name.replace(' ', '_')}.npz"
            graph.save(path)
            log(f"  {graph.summary()}"
                + (f"  filled={filled}" if filled else ""))

            summary[shelf_name] = {
                "id": ident, "nodes": graph.n_nodes, "edges": graph.n_edges,
                "melt_min": float(graph.y.min()), "melt_max": float(graph.y.max()),
                "melt_mean": float(graph.y.mean()), "path": str(path),
                "filled_fraction": filled,
            }

    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--simulation", default="OPM021",
                   help="NEMO run supplying geometry and melt (default: OPM021)")
    p.add_argument("--mask-simulation", default="SMITH_bf663",
                   help="run supplying masks, slopes and T/S")
    p.add_argument("--year", type=int, default=2006)
    p.add_argument("--shelves", nargs="*", default=list(SMOKE_TEST_SHELVES))
    p.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    p.add_argument("--root", type=Path, default=RAW_DIR)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="report cell counts and melt ranges without writing graphs")
    args = p.parse_args()

    ensure_dirs()
    outdir = args.outdir or GRAPH_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    unknown = [s for s in args.shelves if s not in SHELVES]
    if unknown:
        raise SystemExit(f"unknown shelves {unknown}; known: {sorted(SHELVES)}")

    log(f"building graphs for {args.simulation} ({len(args.shelves)} shelves, "
        f"radius {args.radius / 1000:.0f} km)")
    summary = build_for_simulation(
        args.simulation, args.mask_simulation, args.year, args.shelves,
        args.radius, args.root, outdir, args.dry_run)

    if not args.dry_run and summary:
        (outdir / f"summary_{args.simulation}.json").write_text(
            json.dumps(summary, indent=2))
        log(f"wrote {len(summary)} graphs to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
