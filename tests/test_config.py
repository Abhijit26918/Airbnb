from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from pricelens.config import CityConfig, load_city_config


def test_nyc_config_loads_and_validates() -> None:
    cfg = load_city_config(Path("configs/city/nyc.yaml"))

    assert cfg.city == "new-york-city"
    assert cfg.currency == "USD"
    assert cfg.bbox.lat_min < cfg.bbox.lat_max
    assert cfg.bbox.lon_min < cfg.bbox.lon_max
    assert "estimated_revenue_l365d" in cfg.drop_columns


def test_base_url_is_built_from_config_fields() -> None:
    cfg = load_city_config(Path("configs/city/nyc.yaml"))

    assert cfg.base_url == (
        f"https://data.insideairbnb.com/{cfg.country}/{cfg.region}/{cfg.city}/{cfg.scrape_date}"
    )


def test_missing_required_field_raises() -> None:
    with pytest.raises(ValidationError):
        CityConfig.model_validate({"city": "nowhere"})
