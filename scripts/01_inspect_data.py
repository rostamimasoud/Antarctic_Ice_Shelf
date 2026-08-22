#!/usr/bin/env python3
"""Report what the downloaded archives actually contain.

Run this before any preprocessing.  It lists the simulations found, the years
available for each, and the variables and shapes in a representative file of
each type, together with which canonical fields could be resolved.  Archive
layouts and variable names differ between releases, and a name that silently
fails to resolve is far more expensive to discover halfway through a training
run than here.

Examples
--------
    python scripts/01_inspect_data.py
    python scripts/01_inspect_data.py --release burgard2022 --simulation OPM021
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aisgnn.config import RAW_DIR                                  # noqa: E402
from aisgnn.data.nemo import (                                     # noqa: E402
    CANDIDATES,
    VariableNotFound,
    available_simulations,
    discover,
    open_dataset,
    resolve,
)


def describe(path: Path, label: str, max_vars: int = 40) -> None:
    print(f"\n=== {label}: {path.name}")
    try:
        with open_dataset(path) as ds:
            items = list(ds.variables.items())
            for name, var in items[:max_vars]:
                print(f"    {name:38s} {tuple(var.shape)}")
            if len(items) > max_vars:
                print(f"    ... and {len(items) - max_vars} more")

            print("  canonical fields:")
            for field in CANDIDATES:
                try:
                    print(f"    {field:16s} -> {resolve(ds, field)}")
                except VariableNotFound:
                    pass
    except Exception as exc:                                       # noqa: BLE001
        print(f"    could not open: {exc}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=RAW_DIR)
    p.add_argument("--release", default="burgard2023")
    p.add_argument("--simulation", default=None,
                   help="restrict to one simulation (default: the first found)")
    p.add_argument("--year", type=int, default=None)
    args = p.parse_args()

    sims = available_simulations(args.root, args.release)
    print(f"release {args.release} under {args.root}")
    print(f"simulations: {sims or 'none found'}")
    if not sims:
        return 1

    chosen = args.simulation or sims[0]
    if chosen not in sims:
        raise SystemExit(f"{chosen!r} not among {sims}")

    index = discover(chosen, root=args.root, release=args.release)
    years = sorted(index)
    print(f"\n{chosen}: {len(years)} years, {years[0]}-{years[-1]}" if years
          else f"\n{chosen}: no indexed years")
    if not years:
        return 1

    complete = [y for y in years if index[y].complete()]
    print(f"complete years: {len(complete)}")
    if complete:
        print(f"missing pieces in {len(years) - len(complete)} year(s); "
              f"example gaps: {index[years[0]].missing()}")

    year = args.year or (complete[0] if complete else years[0])
    files = index[year]
    print(f"\ninspecting {chosen} {year}")
    for kind in ("masks", "geometry", "ts_fields", "melt"):
        path = getattr(files, kind)
        if path is not None:
            describe(path, kind)
        else:
            print(f"\n=== {kind}: not present")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
