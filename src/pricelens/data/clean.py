"""Price parsing, filter policy, dedup, bathrooms_text parsing."""

from __future__ import annotations

import pandas as pd


def parse_price(s: pd.Series) -> pd.Series:
    """Parse Inside Airbnb's `"$1,234.00"`-style price strings to floats.

    The `$` is a formatting artefact, not a currency claim -- non-USD cities
    use the same format with their own local currency (see the city config's
    `currency` field). Empty strings and nulls become `pd.NA`, not 0 or NaN
    silently coerced by a plain float cast.
    """
    return (
        s.astype("string")
        .str.replace(r"[^\d.]", "", regex=True)
        .replace("", pd.NA)
        .astype("Float64")
    )
