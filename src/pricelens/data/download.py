"""Fetch Inside Airbnb snapshots, checksum, and write a manifest."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import requests

from pricelens.config import CityConfig

log = logging.getLogger(__name__)

# Relative to a pinned snapshot's base_url. Deliberately excludes
# calendar.csv.gz -- it carries per-date price, i.e. the target. See
# TECHNICAL_DESIGN.md §2.
_FILES = {
    "listings": "data/listings.csv.gz",
    "reviews": "data/reviews.csv.gz",
    "neighbourhoods": "visualisations/neighbourhoods.geojson",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_count(path: Path) -> int | None:
    """Data rows (header excluded). None for non-tabular files.

    Counts CSV records, not newlines: free-text fields like `description`
    contain literal embedded newlines inside quoted cells, so a naive
    line-count overstates the row count substantially.
    """
    if path.suffix == ".geojson":
        return None
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as fh:
        return sum(1 for _ in csv.reader(fh)) - 1


def _download_one(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)


def _verify(manifest: dict, snapshot_dir: Path) -> bool:
    for entry in manifest["files"].values():
        path = snapshot_dir / entry["filename"]
        if not path.exists() or _sha256(path) != entry["sha256"]:
            return False
    return True


def download_snapshot(cfg: CityConfig, raw_dir: Path) -> Path:
    """Fetch listings/reviews/neighbourhoods for the pinned snapshot.

    Idempotent: if a manifest already exists and every file's checksum still
    matches it, no network request is made -- re-running only re-verifies.
    Raises if an existing manifest's checksums don't match what's on disk,
    since silently re-downloading over a corrupted or hand-edited file would
    hide the problem instead of surfacing it.
    """
    snapshot_dir = raw_dir / cfg.city / cfg.scrape_date
    manifest_path = snapshot_dir / "manifest.json"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _verify(manifest, snapshot_dir):
            log.info("snapshot already present and verified: %s", snapshot_dir)
            return snapshot_dir
        raise RuntimeError(
            f"existing snapshot at {snapshot_dir} failed checksum verification; "
            "remove it manually before re-downloading"
        )

    entries = {}
    for name, rel_path in _FILES.items():
        url = f"{cfg.base_url}/{rel_path}"
        dest = snapshot_dir / Path(rel_path).name
        log.info("downloading %s -> %s", url, dest)
        _download_one(url, dest)
        entries[name] = {
            "url": url,
            "filename": dest.name,
            "sha256": _sha256(dest),
            "bytes": dest.stat().st_size,
            "rows": _row_count(dest),
        }
        log.info(
            "%s: %d bytes, %s rows, sha256=%s",
            name,
            entries[name]["bytes"],
            entries[name]["rows"],
            entries[name]["sha256"][:12],
        )

    manifest = {
        "city": cfg.city,
        "scrape_date": cfg.scrape_date,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "files": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return snapshot_dir
