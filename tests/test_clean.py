from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pricelens.data.clean import parse_price


def test_parse_price_strips_dollar_and_thousands_separator() -> None:
    result = parse_price(pd.Series(["$1,234.00", "$80.06"]))

    assert result.tolist() == [1234.00, 80.06]


def test_parse_price_empty_string_is_missing() -> None:
    result = parse_price(pd.Series(["$100.00", ""]))

    assert result.iloc[0] == 100.00
    assert pd.isna(result.iloc[1])


def test_parse_price_null_is_missing() -> None:
    result = parse_price(pd.Series(["$100.00", None]))

    assert pd.isna(result.iloc[1])


def test_parse_price_returns_nullable_float_dtype() -> None:
    result = parse_price(pd.Series(["$100.00"]))

    assert result.dtype == "Float64"


@given(st.floats(min_value=0, max_value=1_000_000, allow_nan=False, allow_infinity=False))
def test_parse_price_round_trips_formatted_currency(value: float) -> None:
    formatted = f"${value:,.2f}"

    result = parse_price(pd.Series([formatted]))

    assert result.iloc[0] == pytest.approx(round(value, 2), abs=0.01)
