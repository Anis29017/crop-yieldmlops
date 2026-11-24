"""Tests for the MNFSR book reader (src/dataops/read_mnfsr.py).

The rows below are copied from the real book (MNFSR CAP 2024-25, pages 20-21).
The full-book test runs only if that PDF is available (set MNFSR_TEST_PDF to its path).
"""

import os

import pandas as pd
import pytest

from src.dataops import read_mnfsr as r

EDITION = ("2024", "25")
ALIASES = pd.read_csv(r.ALIASES).query("source == 'mnfsr'")

# Sindh wheat, page 21: area 23-24, area 24-25, production 23-24, production 24-25 ('000 ha / '000 t)
KHAIRPUR = {"crop": "wheat", "book_province": "Sindh", "book_name": "Khairpur", "page": 21,
            "area_prev": 111.59, "area": 111.97, "prod_prev": 419.13, "prod": 333.23}
TMKHAN = {"crop": "wheat", "book_province": "Sindh", "book_name": "T.M.Khan", "page": 21,
          "area_prev": 18.80, "area": 18.31, "prod_prev": 59.46, "prod": 52.88}
# Sindh cotton, page 42 ('000 bales)
GHOTKI_COTTON = {"crop": "cotton", "book_province": "Sindh", "book_name": "Ghotki", "page": 42,
                 "area_prev": 84.97, "area": 82.56, "prod_prev": 479.08, "prod": 381.31}


def _rows(*book_rows):
    out, _ = r.to_yield_rows(list(book_rows), EDITION, "test book", ALIASES)
    return {(x["district"], x["crop"], x["year"]): x for x in out}


def test_year_rule():
    assert r.years(EDITION, "wheat") == (2025, 2024)  # harvest year
    assert r.years(EDITION, "cotton") == (2024, 2023)  # sowing year


def test_units_and_yield():
    rows = _rows(KHAIRPUR, GHOTKI_COTTON)
    wheat = rows[("Khairpur", "wheat", 2025)]
    assert (wheat["area_ha"], wheat["production_t"], wheat["yield_t_ha"]) == (111970, 333230.0, 2.976)
    cotton = rows[("Ghotki", "cotton", 2024)]
    assert cotton["production_t"] == round(381.31 * 170.09)  # bales of lint -> tonnes
    assert cotton["production_unit_printed"] == "000 bales (lint)"


def test_both_years_are_read():
    rows = _rows(KHAIRPUR)
    assert rows[("Khairpur", "wheat", 2024)]["area_ha"] == 111590  # last year's column


def test_book_spelling_becomes_our_name():
    rows = _rows(TMKHAN)
    assert ("Tando Muhammad Khan", "wheat", 2025) in rows


def test_zero_area_rows_are_left_out():
    zero = {**KHAIRPUR, "book_name": "Karachi", "area": 0.0, "prod": 0.0, "area_prev": 0.0, "prod_prev": 0.0}
    assert _rows(zero) == {}


def test_province_total_check():
    rows = [KHAIRPUR, TMKHAN]
    good = {"crop": "wheat", "province": "Sindh", "page": 21,
            "printed": [111.59 + 18.80, 111.97 + 18.31, 419.13 + 59.46, 333.23 + 52.88]}
    assert all(t["ok"] for t in r.check_totals(rows, [good]))
    bad = {**good, "printed": [500.0, 500.0, 500.0, 500.0]}  # a district missing or misread
    assert not any(t["ok"] for t in r.check_totals(rows, [bad]))


def test_every_alias_points_to_a_district_in_our_data():
    wb = pd.read_excel(r.WORKBOOK, sheet_name="All Data")
    known = set(zip(wb["province"], wb["district"]))
    missing = [(a.province, a.district) for a in ALIASES.itertuples() if (a.province, a.district) not in known]
    assert missing == []


@pytest.mark.skipif(not os.environ.get("MNFSR_TEST_PDF"), reason="set MNFSR_TEST_PDF to the CAP 2024-25 PDF")
def test_full_book_matches_the_hand_checked_m1_rows():
    result = r.read_book(os.environ["MNFSR_TEST_PDF"])
    assert all(t["ok"] for t in result["totals"])
    assert result["unknown_names"] == []
    comp = pd.DataFrame(result["comparison"])
    mnfsr_rows = comp[(comp["column"] == "this") & (comp["province"] != "Punjab")]  # M1 used MNFSR for these
    assert (mnfsr_rows["result"] == "same").all()
