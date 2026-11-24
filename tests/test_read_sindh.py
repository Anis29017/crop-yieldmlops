"""Tests for the Sindh Statistics reader (src/dataops/read_sindh.py).

The title texts below are what OCR returned for the real book (Sindh Statistics 2025, pages 86 and 92).
The full-book test needs the PDF and the tesseract program: set SINDH_TEST_PDF to the PDF.
"""

import os
import shutil

import pytest

from src.dataops import read_sindh as rs


def test_title_with_years_split_by_ocr():
    # page 86 at low resolution: "2018-" and "-19" separated by the table number
    text = ("Sindh Statisties 2025 Area and Production of Cotton in Sindh by District, 2018- Table 2.20 -19 to "
            "2023-24 AGRICULTURE Province/District 2018-19 2019-20 2020-21 2021-22 2022-23 2023-24 Area")
    assert rs.title_info(text) == {"crop": "cotton", "first": 2018, "last": 2023}


def test_title_with_table_number_in_the_middle():
    text = ("Sindh Statistics 2025, Area and Production of Wheat in Sindh by District, Table 2.26 2018-19 to "
            "2023-24 AGRICULTURE 2018-19 2019-20 2020-21 2021-22 2022-23 2023-24 Province/District")
    assert rs.title_info(text) == {"crop": "wheat", "first": 2018, "last": 2023}


def test_not_a_district_table():
    assert rs.title_info("Area, Production and Yield of Major Crops in Sindh: Kharif, 2004-05 to 2023-24") is None


def test_area_and_production_lines_paired_by_name():
    # wheat page 92, first two districts, 2018-19 and 2019-20 (ha / tonnes)
    blocks = {"area": {"rows": [("Badin", [19770, 26730]), ("Dadu", [74860, 75978])]},
              "prod": {"rows": [("Dadu", [259596, 239035]), ("Badin", [70349, 82539])]}}
    assert rs._pair(blocks) == [("Badin", [19770, 26730], [70349, 82539]), ("Dadu", [74860, 75978], [259596, 239035])]


def test_lines_paired_by_order_when_ocr_lost_the_production_names():
    blocks = {"area": {"rows": [("Badin", [19770]), ("Dadu", [74860])]},
              "prod": {"rows": [("", [70349]), ("", [259596])]}}
    assert rs._pair(blocks) == [("Badin", [19770], [70349]), ("Dadu", [74860], [259596])]


@pytest.mark.skipif(not os.environ.get("SINDH_TEST_PDF") or not shutil.which("tesseract"),
                    reason="set SINDH_TEST_PDF to Sindh Statistics 2025 and install tesseract")
def test_full_book():
    result = rs.read_book(os.environ["SINDH_TEST_PDF"])
    failed = {t["what"].split(":")[0] + " " + t["crop"] for t in result["totals"] if not t["ok"]}
    assert failed <= {"2022-23 cotton"}  # that column is misread by OCR and left out
    assert result["unknown_names"] == []
    # rows we took from Sindh Statistics 2024 in M1 are read identically
    m1 = [c for c in result["comparison"] if "Sindh Statistics" in str(c["our_source"])]
    assert m1 and all(c["result"] == "same" for c in m1)
