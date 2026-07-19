"""pydantic-settings schema for the city/feature/model YAML configs."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class BBox(BaseModel):
    """Geographic bounding box used to validate listing coordinates."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class CityConfig(BaseModel):
    """The data contract for one city: source location, scrape pin, and the
    kill-list of columns that must never reach a feature matrix.
    """

    city: str
    country: str
    region: str
    scrape_date: str
    currency: str
    bbox: BBox
    price_floor: float
    drop_columns: list[str] = Field(default_factory=list)

    @property
    def base_url(self) -> str:
        """Root of the pinned Inside Airbnb snapshot for this city."""
        return f"https://data.insideairbnb.com/{self.country}/{self.region}/{self.city}/{self.scrape_date}"


def load_city_config(path: Path) -> CityConfig:
    """Load and validate a `configs/city/<city>.yaml` data contract."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return CityConfig.model_validate(raw)
