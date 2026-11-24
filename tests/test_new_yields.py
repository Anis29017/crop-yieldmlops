"""Tests for adding rows from new books to the workbook rows (build_yield_labels.combine_with_new).

Rows are copied from real books: Punjab CRS Final Kharif Estimates 2025-26 and MNFSR CAP 2024-25.
"""

import pandas as pd

from src import build_yield_labels as byl

COLS = ["province", "district", "crop", "year", "area_ha", "production_t", "yield_t_ha", "source"]
CRS = "Punjab CRS Final Estimates of Major Kharif Crops 2025-26 [Final Kharif Estimates 2025-26.pdf]"
FAISALABAD_COTTON_2025 = ["Punjab", "Faisalabad", "cotton", 2025, 15000, 9865.0, 0.658, CRS]
BAHAWALNAGAR_RICE_2025 = ["Punjab", "Bahawalnagar", "rice", 2025, 142000, 308000.0, 2.169, CRS]
BADIN_RICE_2024 = ["Sindh", "Badin", "rice", 2024, 125000, 364550.0, 2.916, "MNFSR CAP 2024-25"]

WORKBOOK = pd.read_excel(byl.YIELD_XLSX, sheet_name=byl.YIELD_SHEET)


def write(folder, name, rows):
    pd.DataFrame(rows, columns=COLS).to_csv(folder / name, index=False)


def key(df, province, district, crop, year):
    return df[(df.province == province) & (df.district == district) & (df.crop == crop) & (df.year == year)]


def test_no_new_files_gives_the_workbook(tmp_path):
    out = byl.combine_with_new(WORKBOOK, str(tmp_path))
    assert len(out) == len(WORKBOOK)


def test_new_year_rows_are_added(tmp_path):
    write(tmp_path, "punjab_crs__Final Kharif Estimates 2025-26.csv", [FAISALABAD_COTTON_2025, BAHAWALNAGAR_RICE_2025])
    out = byl.combine_with_new(WORKBOOK, str(tmp_path))
    assert len(out) == len(WORKBOOK) + 2
    assert key(out, "Punjab", "Faisalabad", "cotton", 2025)["area_ha"].item() == 15000


def test_the_workbook_wins_over_a_new_book(tmp_path):
    ours = key(WORKBOOK, "Sindh", "Badin", "rice", 2024)
    changed = BADIN_RICE_2024[:4] + [1, 1.0, 1.0, "a different book"]
    write(tmp_path, "mnfsr__other.csv", [changed])
    out = byl.combine_with_new(WORKBOOK, str(tmp_path))
    assert key(out, "Sindh", "Badin", "rice", 2024)["area_ha"].item() == ours["area_ha"].item()


def test_for_punjab_crs_wins_over_mnfsr(tmp_path):
    mnfsr_row = FAISALABAD_COTTON_2025[:4] + [16000, 10000.0, 0.625, "MNFSR"]
    write(tmp_path, "mnfsr__CAP 2025-26.csv", [mnfsr_row])
    write(tmp_path, "punjab_crs__Final Kharif Estimates 2025-26.csv", [FAISALABAD_COTTON_2025])
    out = byl.combine_with_new(WORKBOOK, str(tmp_path))
    assert key(out, "Punjab", "Faisalabad", "cotton", 2025)["source"].item() == CRS


def test_outside_punjab_mnfsr_wins_over_sindh_statistics(tmp_path):
    sindh = ["Sindh", "Badin", "rice", 2025, 120000, 350000.0, 2.917, "Sindh Statistics"]
    mnfsr = ["Sindh", "Badin", "rice", 2025, 125000, 364550.0, 2.916, "MNFSR"]
    write(tmp_path, "sindh_bos__Sindh Statistics 2026.csv", [sindh])
    write(tmp_path, "mnfsr__CAP 2025-26.csv", [mnfsr])
    out = byl.combine_with_new(WORKBOOK, str(tmp_path))
    assert key(out, "Sindh", "Badin", "rice", 2025)["source"].item() == "MNFSR"
