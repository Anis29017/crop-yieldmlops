"""Tests for the yield-data validation (src/validate_yields.py).

Each test takes real rows from the yield workbook, breaks one thing on purpose,
and checks the validator reports it. The real data itself must pass.
"""

import pandas as pd
import pytest

from src.build_yield_labels import load_yields
from src.validate_yields import (
    POINTS_PATH,
    YIELD_SHEET,
    YIELD_XLSX,
    validate_or_stop,
    validate_yield_table,
)


@pytest.fixture(scope="module")
def real():
    df = pd.read_excel(YIELD_XLSX, sheet_name=YIELD_SHEET)
    return df, pd.read_csv(POINTS_PATH)


def _rows(real):
    """Real rows: Faisalabad (Punjab) and Hyderabad (Sindh), all crops and years."""
    df, points = real
    return df[df["district"].isin(["Faisalabad", "Hyderabad"])].reset_index(drop=True).copy(), points


def _errors(df, points):
    return {p["check"] for p in validate_yield_table(df, points)["errors"]}


def test_full_real_workbook_passes(real):
    df, points = real
    report = validate_yield_table(df, points)
    assert report["errors"] == []
    assert report["rows"] == len(df)


def test_missing_column_is_an_error(real):
    df, points = _rows(real)
    assert "E1" in _errors(df.drop(columns="source"), points)


def test_missing_value_is_an_error(real):
    df, points = _rows(real)
    df.loc[0, "yield_t_ha"] = None
    assert "E2" in _errors(df, points)


def test_bad_year_is_an_error(real):
    df, points = _rows(real)
    df.loc[0, "year"] = 1850
    assert "E3" in _errors(df, points)


def test_unknown_crop_is_an_error(real):
    df, points = _rows(real)
    df.loc[0, "crop"] = "maize"
    assert "E4" in _errors(df, points)


def test_negative_area_is_an_error(real):
    df, points = _rows(real)
    df.loc[0, "area_ha"] = -5
    assert "E5" in _errors(df, points)


def test_yield_not_matching_production_over_area_is_an_error(real):
    df, points = _rows(real)
    df.loc[0, "yield_t_ha"] = df.loc[0, "yield_t_ha"] * 10  # e.g. a misplaced decimal point
    assert "E6" in _errors(df, points)


def test_duplicate_row_is_an_error(real):
    df, points = _rows(real)
    assert "E7" in _errors(pd.concat([df, df.head(1)]), points)


def test_unknown_district_is_an_error(real):
    df, points = _rows(real)
    df.loc[0, "district"] = "Faisalabadd"  # a new spelling
    assert "E8" in _errors(df, points)


def test_missing_source_is_an_error(real):
    df, points = _rows(real)
    df.loc[0, "source"] = " "
    assert "E9" in _errors(df, points)


def test_big_jump_is_only_a_warning(real):
    df, points = _rows(real)
    i = df.index[(df["crop"] == "wheat") & (df["year"] == 2020)][0]
    df.loc[i, ["production_t", "yield_t_ha"]] = df.loc[i, ["production_t", "yield_t_ha"]] * 4
    report = validate_yield_table(df, points)
    assert report["errors"] == []
    assert "W2" in {p["check"] for p in report["warnings"]}


def test_validate_or_stop_raises_on_errors(real):
    df, points = _rows(real)
    df.loc[0, "crop"] = "maize"
    with pytest.raises(ValueError):
        validate_or_stop(df, points)


def test_load_yields_accepts_an_already_read_table(real):
    df, _ = _rows(real)
    out = load_yields(df=df)
    assert len(out) == len(df)
    assert out["year"].dtype.kind == "i"
