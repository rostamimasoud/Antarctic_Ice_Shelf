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
from pathlib import Path, PurePosixPath

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
                  checksum: str | None = None, attempts: int = 3) -> Path:
    """Download ``url`` to ``dest``, resuming a partial transfer if present.

    A completed file is verified against the Zenodo MD5 and re-fetched from
    scratch on mismatch: transient corruption over a multi-gigabyte transfer is
    common enough that failing the whole record for it wastes hours.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if _already_valid(dest, size, checksum):
        return dest

    last_error = ""
    for attempt in range(1, attempts + 1):
        # Resume only on the first attempt.  Every observed failure mode here --
        # a partial file that ends up larger than the target, or one that
        # vanishes between transfer and rename -- comes from appending to a
        # response that did not honour the Range request, so a retry that
        # resumed again would reproduce it exactly.
        try:
            _fetch_once(url, dest, size, resume=(attempt == 1))
        except RuntimeError as exc:
            last_error = str(exc)
            _log(f"  {dest.name}: {exc}; attempt {attempt}/{attempts}, "
                 f"refetching from scratch")
            dest.unlink(missing_ok=True)
            dest.with_suffix(dest.suffix + ".part").unlink(missing_ok=True)
            continue

        if checksum is None:
            return dest

        got = md5sum(dest)
        want = checksum.split(":")[-1]
        if got == want:
            _log(f"  checksum ok: {dest.name}")
            return dest

        last_error = f"checksum {got} != {want}"
        _log(f"  checksum mismatch for {dest.name}; "
             f"attempt {attempt}/{attempts}, refetching in full")
        dest.unlink(missing_ok=True)

    raise RuntimeError(f"could not fetch {dest.name} after "
                       f"{attempts} attempts: {last_error}")


def _already_valid(dest: Path, size: int | None, checksum: str | None) -> bool:
    """True when ``dest`` is present and verifiably complete."""
    if not dest.exists():
        return False
    if size is not None and dest.stat().st_size != size:
        return False
    if checksum is None:
        _log(f"  exists, skipping: {dest.name}")
        return True
    if md5sum(dest) == checksum.split(":")[-1]:
        _log(f"  verified, skipping: {dest.name}")
        return True
    _log(f"  checksum mismatch, refetching: {dest.name}")
    dest.unlink(missing_ok=True)
    return False


def _fetch_once(url: str, dest: Path, size: int | None,
                resume: bool = True) -> None:
    """Transfer ``url`` into ``dest``, optionally resuming a partial file."""
    part = dest.with_suffix(dest.suffix + ".part")

    if not resume:
        part.unlink(missing_ok=True)

    # A partial file at or beyond the target size means an earlier resume
    # appended to a response that ignored the Range header; start again rather
    # than promoting a file that can only fail its checksum.
    if part.exists() and size is not None and part.stat().st_size >= size:
        if part.stat().st_size > size:
            _log(f"  discarding oversized partial: {part.name}")
            part.unlink()

    for attempt in range(1, MAX_RETRIES + 1):
        offset = part.stat().st_size if part.exists() else 0
        if size is not None and offset == size:
            break
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with _open(url, headers) as resp, part.open("ab" if offset else "wb") as out:
                # A server that ignores Range replies 200 with the whole body;
                # appending it to the existing bytes would corrupt the file.
                if offset and resp.status != 206:
                    out.close()
                    part.unlink(missing_ok=True)
                    _log(f"  range not honoured for {dest.name}; restarting")
                    continue

                total = offset + int(resp.headers.get("Content-Length", 0) or 0)
                done = offset
                bar = _Progress(dest.name, total or (size or 0), offset)
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

    if part.exists():
        part.replace(dest)
    elif not dest.exists():
        raise RuntimeError(f"neither {part.name} nor {dest.name} present after download")

    if size is not None and dest.stat().st_size != size:
        raise RuntimeError(f"{dest.name}: got {dest.stat().st_size} bytes, expected {size}")


def _safe_member_path(member: str) -> Path | None:
    """Normalise an archive member to a path that cannot escape the target.

    Several of these archives were zipped from a nested working directory and
    legitimately carry ``../../interim/...`` prefixes.  Refusing them outright
    would reject real data, so the leading parent references and any absolute
    prefix are stripped instead; the remainder is guaranteed to stay inside the
    extraction directory.  Returns ``None`` for members that normalise away to
    nothing.
    """
    parts = [p for p in PurePosixPath(member).parts
             if p not in ("..", ".", "/") and not p.endswith(":")]
    return Path(*parts) if parts else None


def extract_archive(archive: Path, target: Path, cleanup: bool = False) -> Path:
    """Expand a zip archive, skipping the work if a marker file already exists."""
    marker = target / f".extracted_{archive.stem}"
    if marker.exists():
        _log(f"  already extracted: {archive.name}")
        return target

    target.mkdir(parents=True, exist_ok=True)
    _log(f"  extracting {archive.name} -> {target}")

    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            rel = _safe_member_path(info.filename)
            if rel is None:
                continue
            dest = target / rel
            if info.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out, CHUNK)

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
