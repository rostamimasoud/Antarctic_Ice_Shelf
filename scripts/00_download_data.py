#!/usr/bin/env python3
"""Fetch the public datasets underpinning the study.

Examples
--------
    python scripts/00_download_data.py --list
    python scripts/00_download_data.py --dry-run
    python scripts/00_download_data.py --records burgard2022 burgard2023
    python scripts/00_download_data.py                     # everything
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aisgnn.config import RAW_DIR, RECORDS, ensure_dirs   # noqa: E402
from aisgnn.data.download import download_all             # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", nargs="*", default=None,
                   help="record labels to fetch (default: all)")
    p.add_argument("--root", type=Path, default=RAW_DIR,
                   help=f"destination directory (default: {RAW_DIR})")
    p.add_argument("--no-extract", action="store_true",
                   help="download archives but do not expand them")
    p.add_argument("--cleanup", action="store_true",
                   help="delete zip archives after successful extraction")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be fetched and exit")
    p.add_argument("--list", action="store_true",
                   help="list the registered records and exit")
    args = p.parse_args()

    if args.list:
        for label, rec in sorted(RECORDS.items()):
            nfiles = len(rec.files) if rec.files else "all"
            print(f"{label:24s}  zenodo:{rec.record_id:<10s}  files={nfiles:<4}  {rec.description}")
        return 0

    ensure_dirs()
    args.root.mkdir(parents=True, exist_ok=True)
    download_all(labels=args.records, root=args.root,
                 extract=not args.no_extract, cleanup=args.cleanup,
                 dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
