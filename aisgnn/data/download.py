"""Zenodo download utilities.

Deliberately restricted to the Python standard library so that data can be
fetched before any conda environment exists.  Downloads are resumable, verified
against the MD5 checksums published in the Zenodo record metadata, and archives
are expanded in place.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from contextlib import contextmanager
from pathlib import Path

from ..config import RAW_DIR, RECORDS, ZenodoRecord

CHUNK = 8 << 20            # 8 MiB
MAX_RETRIES = 5
PROGRESS_INTERVAL = 30.0   # seconds between progress lines when not on a tty
USER_AGENT = "aisgnn-data-fetch/1.0"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def _open(url: str, headers: dict[str, str] | None = None):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(req, timeout=120)


@contextmanager
def exclusive_lock(path: Path):
    """Refuse to run if another fetch already holds the lock.

    Concurrent fetches would append to the same ``.part`` file and silently
    corrupt it, so this is enforced rather than advisory.  A lock whose
    recorded PID is no longer alive is treated as stale and reclaimed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        try:
            pid = int(path.read_text().split()[0])
            os.kill(pid, 0)
        except (ValueError, IndexError, ProcessLookupError, OSError):
            _log(f"reclaiming stale lock {path}")
            path.unlink(missing_ok=True)
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        else:
            raise SystemExit(
                f"another download is running (pid {pid}); lock: {path}\n"
                f"kill it or remove the lock file before retrying"
            )
    os.write(fd, f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n".encode())
    os.close(fd)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


class _Progress:
    """Progress reporter: a live bar on a tty, periodic lines in a log file."""

    def __init__(self, name: str, total: int, offset: int = 0):
        self.name, self.total, self.offset = name, total, offset
        self.tty = sys.stdout.isatty()
        self.t0 = self.last = time.time()

    def update(self, done: int) -> None:
        now = time.time()
        if not self.tty and now - self.last < PROGRESS_INTERVAL:
            return
        self.last = now
        rate = (done - self.offset) / max(now - self.t0, 1e-6)
        pct = 100.0 * done / self.total if self.total else 0.0
        line = (f"  {self.name}: {pct:5.1f}%  {_human(done)}/{_human(self.total)}  "
                f"{_human(rate)}/s")
        if self.tty:
            sys.stdout.write("\r" + line + "   ")
            sys.stdout.flush()
        else:
            _log(line)

    def close(self) -> None:
        if self.tty:
            sys.stdout.write("\n")
            sys.stdout.flush()


def fetch_metadata(record: ZenodoRecord) -> dict:
    """Return the parsed Zenodo API record."""
    with _open(record.api_url) as resp:
        return json.load(resp)


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #

def download_file(url: str, dest: Path, size: int | None = None,
                  checksum: str | None = None) -> Path:
    """Download ``url`` to ``dest``, resuming a partial transfer if present.

    ``checksum`` is the Zenodo ``md5:...`` string; when supplied the finished
    file is verified and re-downloaded once on mismatch.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    if dest.exists() and (size is None or dest.stat().st_size == size):
        if checksum is None:
            _log(f"  exists, skipping: {dest.name}")
            return dest
        if md5sum(dest) == checksum.split(":")[-1]:
            _log(f"  verified, skipping: {dest.name}")
            return dest
        _log(f"  checksum mismatch, refetching: {dest.name}")
        dest.unlink()

    for attempt in range(1, MAX_RETRIES + 1):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with _open(url, headers) as resp, part.open("ab" if offset else "wb") as out:
                total = offset + int(resp.headers.get("Content-Length", 0) or 0)
                done = offset
                bar = _Progress(dest.name, total, offset)
                while True:
                    block = resp.read(CHUNK)
                    if not block:
                        break
                    out.write(block)
                    done += len(block)
                    bar.update(done)
                bar.close()
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            wait = min(60, 2 ** attempt)
            _log(f"  attempt {attempt}/{MAX_RETRIES} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    else:
        raise RuntimeError(f"failed to download {url} after {MAX_RETRIES} attempts")

    # Tolerate the case where a concurrent or interrupted run already promoted
    # the partial file: the checksum below is what actually decides validity.
    if part.exists():
        part.replace(dest)
    elif not dest.exists():
        raise RuntimeError(f"neither {part.name} nor {dest.name} present after download")

    if checksum is not None:
        got = md5sum(dest)
        want = checksum.split(":")[-1]
        if got != want:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"checksum mismatch for {dest.name}: {got} != {want}; "
                f"the corrupt file has been removed, re-run to fetch it again"
            )
        _log(f"  checksum ok: {dest.name}")

    return dest


def extract_archive(archive: Path, target: Path, cleanup: bool = False) -> Path:
    """Expand a zip archive, skipping the work if a marker file already exists."""
    marker = target / f".extracted_{archive.stem}"
    if marker.exists():
        _log(f"  already extracted: {archive.name}")
        return target

    target.mkdir(parents=True, exist_ok=True)
    _log(f"  extracting {archive.name} -> {target}")
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            # Guard against path traversal in untrusted archives.
            resolved = (target / member).resolve()
            if not str(resolved).startswith(str(target.resolve())):
                raise RuntimeError(f"unsafe path in archive: {member}")
        zf.extractall(target)

    marker.touch()
    if cleanup:
        archive.unlink()
    return target


# --------------------------------------------------------------------------- #
# Record-level driver
# --------------------------------------------------------------------------- #

def download_record(record: ZenodoRecord, root: Path = RAW_DIR,
                    extract: bool = True, cleanup: bool = False,
                    dry_run: bool = False) -> list[Path]:
    """Download (a subset of) a Zenodo record and expand any zip archives."""
    meta = fetch_metadata(record)
    target = root / (record.subdir or record.label)
    target.mkdir(parents=True, exist_ok=True)

    entries = meta.get("files", [])
    if record.files:
        wanted = set(record.files)
        entries = [f for f in entries if f["key"] in wanted]
        missing = wanted - {f["key"] for f in entries}
        if missing:
            raise RuntimeError(f"{record.label}: files not in record: {sorted(missing)}")

    total = sum(f["size"] for f in entries)
    _log(f"{record.label}: {len(entries)} file(s), {_human(total)} -> {target}")

    if dry_run:
        for f in entries:
            _log(f"  would fetch {f['key']} ({_human(f['size'])})")
        return []

    written: list[Path] = []
    manifest = target / "manifest.tsv"
    with manifest.open("w") as mf:
        mf.write("file\tsize_bytes\tmd5\n")
        for f in entries:
            url = f["links"]["self"]
            dest = target / f["key"]
            download_file(url, dest, size=f["size"], checksum=f.get("checksum"))
            mf.write(f"{f['key']}\t{f['size']}\t{f.get('checksum', '')}\n")
            written.append(dest)
            if extract and dest.suffix == ".zip":
                extract_archive(dest, target, cleanup=cleanup)

    _log(f"{record.label}: done, manifest at {manifest}")
    return written


def download_all(labels: list[str] | None = None, root: Path = RAW_DIR,
                 extract: bool = True, cleanup: bool = False,
                 dry_run: bool = False) -> None:
    """Download every registered record, or the subset named in ``labels``."""
    labels = labels or list(RECORDS)
    unknown = [x for x in labels if x not in RECORDS]
    if unknown:
        raise SystemExit(f"unknown record label(s): {unknown}; known: {sorted(RECORDS)}")

    if dry_run:
        for label in labels:
            download_record(RECORDS[label], root=root, dry_run=True)
        return

    root.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(root / ".download.lock"):
        free = shutil.disk_usage(root).free
        _log(f"free space at destination: {_human(free)}")

        failures: list[tuple[str, str]] = []
        for label in labels:
            try:
                download_record(RECORDS[label], root=root, extract=extract,
                                cleanup=cleanup, dry_run=False)
            except Exception as exc:                      # noqa: BLE001
                # One bad record must not strand the remaining ~30 GB.
                _log(f"{label}: FAILED -- {exc}")
                failures.append((label, str(exc)))

    if failures:
        _log(f"completed with {len(failures)} failed record(s):")
        for label, msg in failures:
            _log(f"  {label}: {msg}")
        raise SystemExit(1)

    _log("all requested records complete")
