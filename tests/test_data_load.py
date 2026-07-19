from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest

from pricelens.config import load_city_config
from pricelens.data.load import REQUIRED_COLUMNS, SchemaError, load_listings

CFG = load_city_config(Path("configs/city/nyc.yaml"))


def _write_gz_csv(frame: pd.DataFrame, path: Path) -> Path:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        frame.to_csv(fh, index=False)
    return path


def _detailed_frame(n_filler: int = 40) -> pd.DataFrame:
    """A frame shaped like the real detailed listings.csv.gz: every required
    column, every kill-list column, plus enough filler columns to clear the
    "is this actually the detailed file" threshold even after the kill-list
    is dropped.
    """
    data = {col: ["placeholder"] for col in REQUIRED_COLUMNS}
    data["id"] = [1]
    data["host_id"] = [10]
    data["price"] = ["$100.00"]
    data["latitude"] = [40.7]
    data["longitude"] = [-73.9]
    data["accommodates"] = [2]
    data["minimum_nights"] = [1]
    for col in CFG.drop_columns:
        data[col] = ["leak"]
    for i in range(n_filler):
        data[f"filler_{i}"] = ["x"]
    return pd.DataFrame(data)


def test_load_listings_drops_kill_list_columns(tmp_path: Path) -> None:
    path = _write_gz_csv(_detailed_frame(), tmp_path / "listings.csv.gz")

    listings = load_listings(path, CFG)

    for col in CFG.drop_columns:
        assert col not in listings.columns
    assert set(listings.columns) >= REQUIRED_COLUMNS


def test_load_listings_raises_on_missing_required_column(tmp_path: Path) -> None:
    frame = _detailed_frame().drop(columns=["room_type"])
    path = _write_gz_csv(frame, tmp_path / "listings.csv.gz")

    with pytest.raises(SchemaError, match="room_type"):
        load_listings(path, CFG)


def test_load_listings_rejects_summary_shaped_file(tmp_path: Path) -> None:
    """A ~18-column summary file has every required column (they overlap)
    but nothing else -- the file-identity guard must still catch it.
    """
    frame = _detailed_frame(n_filler=0).drop(columns=CFG.drop_columns, errors="ignore")
    path = _write_gz_csv(frame, tmp_path / "listings.csv.gz")

    with pytest.raises(SchemaError, match="summary"):
        load_listings(path, CFG)
