"""Tests for the Punjab CRS book reader (src/dataops/read_punjab.py).

Numbers are copied from the real books (Rabi 2024-25 page 1, Final Kharif Estimates 2024-25 page 9).
The full-book tests run only when the PDFs are available:
  PUNJAB_TEST_RABI=<Rabi_2024-25.pdf>  PUNJAB_TEST_KHARIF=<Final Kharif Estimates 2025-26.pdf>
"""

import os

import pandas as pd
import pytest

from src.dataops import read_mnfsr as common
from src.dataops import read_punjab as rp


def line(name, kind, area, prod):
    return {"crop": "wheat", "name": name, "kind": kind, "page": 1, "area": area, "prod": prod,
            "area_prev": None, "prod_prev": None}


# Rabi 2024-25: Bahawalpur and Sargodha divisions, wheat ('000 ha, '000 t); Islamabad is listed last
BAHAWALPUR_DIV = [line("Bahawalpur Divn.", "division", 1070.0, 3869.0), line("Bahawalpur", "district", 304.0, 1101.0),
                  line("R.Y. Khan", "district", 331.0, 1211.0), line("Bahawalnagar", "district", 435.0, 1557.0)]
SARGODHA_DIV = [line("Sargodha Divn.", "division", 685.0, 1998.0), line("Sargodha", "district", 212.0, 669.0),
                line("Khushab", "district", 110.0, 249.0), line("Mianwali", "district", 186.0, 568.0),
                line("Bhakkar", "district", 177.0, 512.0), line("Islamabad", "district", 19.0, 32.0)]


def test_kind_of_name():
    assert rp.kind_of_name("SARGODHA DIV:") == "division"
    assert rp.kind_of_name("Bahawalpur Divn.") == "division"
    assert rp.kind_of_name("RAWALPINDI DV:") == "division"
    assert rp.kind_of_name("The Punjab") == "punjab"
    assert rp.kind_of_name("SOUTH PUNJAB") == "region"
    assert rp.kind_of_name("D.G. Khan") == "district"


def test_districts_add_up_to_their_division_islamabad_not_counted():
    ordered = BAHAWALPUR_DIV + SARGODHA_DIV
    rows = [r for r in ordered if r["kind"] == "district"]
    totals = [r for r in ordered if r["kind"] != "district"]
    checks = rp.check_totals(rows, totals, ordered)
    assert len(checks) == 4 and all(c["ok"] for c in checks)


def test_a_wrong_number_fails_the_division_check():
    ordered = [dict(r) for r in BAHAWALPUR_DIV]
    ordered[2]["area"] = 3310.0  # 331.0 misread
    rows = [r for r in ordered if r["kind"] == "district"]
    checks = rp.check_totals(rows, [ordered[0]], ordered)
    assert not all(c["ok"] for c in checks)


def test_row_total_check_catches_the_wazirabad_typo():
    # Final Kharif Estimates 2024-25, rice production page 9: BAS 43.6 + NON-BAS 114.2 printed as TOTAL 1156.7
    line_ = {"page": 9, "this": 1156.7, "last": 161.2, "this_parts": [43.6, 114.2], "last_parts": [74.6, 86.6]}
    row = {"crop": "rice", "name": "Wazirabad", "lines": {"production": line_}}
    checks = {c["what"]: c["ok"] for c in rp.check_row_totals([row])}
    assert checks == {"Wazirabad: production": False, "Wazirabad: production last year": True}


def test_book_spellings_become_our_names():
    workbook = pd.read_excel(common.WORKBOOK, sheet_name="All Data")
    names = rp.name_table(["T.T. Singh", "Rajan Pur", "Wazieabad", "Muzzafargargh", "Islamabad", "Okara"],
                          workbook, common.ALIASES)
    got = dict(zip(names["book_name"], zip(names["province"], names["district"])))
    assert got == {"T.T. Singh": ("Punjab", "Toba Tek Singh"), "Rajan Pur": ("Punjab", "Rajanpur"),
                   "Wazieabad": ("Punjab", "Wazirabad"), "Muzzafargargh": ("Punjab", "Muzaffargarh"),
                   "Islamabad": ("Islamabad", "Islamabad"), "Okara": ("Punjab", "Okara")}


@pytest.mark.skipif(not os.environ.get("PUNJAB_TEST_RABI"), reason="set PUNJAB_TEST_RABI to the Rabi 2024-25 PDF")
def test_full_rabi_book_matches_the_hand_checked_m1_rows():
    result = rp.read_book(os.environ["PUNJAB_TEST_RABI"])
    assert all(t["ok"] for t in result["totals"]) and result["unknown_names"] == []
    assert {c["result"] for c in result["comparison"]} == {"same"}
    assert len(result["rows"]) == 42


@pytest.mark.skipif(not os.environ.get("PUNJAB_TEST_KHARIF"),
                    reason="set PUNJAB_TEST_KHARIF to the Final Kharif Estimates 2025-26 PDF")
def test_full_kharif_book_adds_up():
    result = rp.read_book(os.environ["PUNJAB_TEST_KHARIF"])
    assert all(t["ok"] for t in result["totals"]) and result["unknown_names"] == []
    # its last-year column (2024) is the M1 data, except Chiniot cotton (book says 1 thousand bales, M1 2)
    last = [c for c in result["comparison"] if c["column"] == "last"]
    assert sum(c["result"] != "same" for c in last) == 1
