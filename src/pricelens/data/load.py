"""Dtype-safe, schema-guarded readers for listings/reviews/neighbourhoods."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from pricelens.config import CityConfig

log = logging.getLogger(__name__)

# The columns every downstream phase (cleaning, features, validation grouping)
# depends on existing. If Inside Airbnb renames one of these, we want a loud
# failure here, not a silently-NaN column three phases later.
REQUIRED_COLUMNS = {
    "id",
    "host_id",
    "price",
    "latitude",
    "longitude",
    "accommodates",
    "room_type",
    "property_type",
    "description",
    "amenities",
    "neighbourhood_cleansed",
    "bathrooms_text",
    "minimum_nights",
}

# The detailed listings file has ~75 columns; the summary file Inside Airbnb
# also publishes has only 18 and no text fields. Below this threshold (even
# after the kill-list drop) we've very likely been pointed at the wrong file.
_MIN_DETAILED_COLUMNS = 40


class SchemaError(RuntimeError):
    """Raised when a snapshot doesn't match the expected data contract."""


def load_listings(path: Path, cfg: CityConfig) -> pd.DataFrame:
    """Load the detailed listings.csv.gz, enforcing the schema contract.

    Raises `SchemaError` if required columns are missing (schema drift) or if
    the frame looks like the 18-column summary file instead of the detailed
    one. Drops every column in `cfg.drop_columns` before returning -- this is
    where the leakage kill-list is enforced, at load time, before anything
    else can touch the frame.
    """
    listings = pd.read_csv(path, compression="gzip", low_memory=False)

    missing = REQUIRED_COLUMNS - set(listings.columns)
    if missing:
        raise SchemaError(f"snapshot is missing required columns: {sorted(missing)}")

    leaks = set(cfg.drop_columns) & set(listings.columns)
    listings = listings.drop(columns=list(leaks))
    log.info("dropped %d kill-list columns: %s", len(leaks), sorted(leaks))

    if listings.shape[1] <= _MIN_DETAILED_COLUMNS:
        raise SchemaError(
            f"only {listings.shape[1]} columns remain after the kill-list drop -- "
            "this looks like the summary listings.csv, not the detailed file"
        )

    return listings


def load_reviews(path: Path) -> pd.DataFrame:
    """Load the detailed reviews.csv.gz (one row per review, with text)."""
    return pd.read_csv(path, compression="gzip", low_memory=False)


def load_neighbourhoods(path: Path) -> gpd.GeoDataFrame:
    """Load the neighbourhood polygons used for spatial joins/choropleths."""
    return gpd.read_file(path)
