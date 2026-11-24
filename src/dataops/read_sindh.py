"""DataOps - read the district crop tables of a "Sindh Statistics" book (Sindh Bureau of Statistics).

The book is scanned: every page is a picture, so the text is read with OCR (tesseract).
Tables used (Sindh Statistics 2025, checked): "Area and Production of <Crop> in Sindh by District,
2018-19 to 2023-24" for Cotton, Rice, Sugarcane and Wheat. Each has an Area block (hectares) and a
Production block (tonnes; bales for cotton), a SINDH total line, one line per district, and one
column per year.

Because OCR can misread a number, every year column must add up EXACTLY to the SINDH total
printed above it. A column that does not add up is reported and none of its rows are written.
Empty cells are printed as "-" in the book; OCR turns some of them into stray marks, which are
ignored (only numbers sitting under a year column count).

Needs the tesseract program: Mac `brew install tesseract`, Ubuntu `apt-get install tesseract-ocr`.

Usage:
  python -m src.dataops.read_sindh <book.pdf> --out data/sources/new/sindh_statistics_2025.xlsx
"""

import argparse
import os
import re

import pandas as pd
import pdfplumber
import pytesseract

from src.dataops import read_mnfsr as common

CROPS = {"cotton": "cotton", "rice": "rice", "sugarcane": "sugarcane", "wheat": "wheat"}
TITLE = re.compile(r"Area\s+and\s+Production\s+of\s+(\w+)\s+in\s+Sindh\s+by\s+District", re.I)
SEASON = re.compile(r"\b((?:19|20)\d{2})\s*-\s*(\d{2})\b")
NUMBER = re.compile(r"^\d{1,3}(?:,\d{3})+$|^\d+$")
SCAN_DPI, TABLE_DPI = 110, 260
OCR_DPIS = (260, 330, 200)  # tried in this order until every year column adds up
MIN_CONFIDENCE = 50  # OCR confidence (0-100) below which a number is not trusted


def title_info(text):
    """Crop and year range of a district table, from the OCR text of the top of its page."""
    m = TITLE.search(text)
    if not m:
        return None
    # the year columns (2018-19 ... 2023-24) are printed under the title
    firsts = sorted({int(a) for a, b in SEASON.findall(text) if (int(a) + 1) % 100 == int(b)})
    if not firsts:
        return None
    return {"crop": m.group(1).lower(), "first": firsts[0], "last": firsts[-1]}


def find_tables(pdf, wanted=tuple(CROPS)):
    """Page number of each crop's district table, found by OCR of the page titles."""
    found = {}
    for page_no, page in enumerate(pdf.pages, 1):
        top = page.crop((0, 0, page.width, page.height * 0.25)).to_image(resolution=SCAN_DPI).original
        text = " ".join(pytesseract.image_to_string(top).split())
        info = title_info(text)
        if info and info["crop"] in wanted and info["crop"] not in found:
            found[info["crop"]] = {"page": page_no, "first": info["first"], "last": info["last"]}
            if len(found) == len(wanted):
                break
    return found


def ocr_lines(page, dpi=TABLE_DPI):
    """OCR words of a page grouped into lines; each word has its text, position and confidence."""
    image = page.to_image(resolution=dpi).original
    d = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config="--psm 6")
    words = [{"text": d["text"][i].strip(), "left": d["left"][i], "right": d["left"][i] + d["width"][i],
              "top": d["top"][i], "conf": float(d["conf"][i])}
             for i in range(len(d["text"])) if d["text"][i].strip()]
    words.sort(key=lambda w: (w["top"], w["left"]))
    lines = []
    for w in words:
        if lines and abs(lines[-1][0] - w["top"]) <= 15 * dpi / TABLE_DPI:
            lines[-1][1].append(w)
        else:
            lines.append([w["top"], [w]])
    return [sorted(ws, key=lambda w: w["left"]) for _, ws in lines]


def _is_number(w):
    return NUMBER.match(w["text"].rstrip(".,;:")) is not None


def _value(w):
    return int(w["text"].rstrip(".,;:").replace(",", ""))


def read_table(page, n_years, dpi=TABLE_DPI):
    """{"area"/"prod": {"total": [...], "rows": [(name, [...])]}} plus the production unit.

    The block headings ("A r e a (In Hectares)", "P r o d u c t i o n") are letter-spaced and OCR
    often loses them, so the blocks are found by their SINDH total lines: the first one starts the
    Area block, the second one the Production block.
    """
    lines = ocr_lines(page, dpi)
    page_text = " ".join(w["text"] for ws in lines for w in ws).lower()
    unit = "bales" if "bale" in page_text else "tonnes"
    blocks, order, current, columns, skipped = {}, ["area", "prod"], None, None, []
    for ws in lines:
        nums = [w for w in ws if _is_number(w)]
        name = " ".join(w["text"] for w in ws if not _is_number(w) and w["right"] < (nums[0]["left"] if nums else 1e9))
        name = re.sub(r"^[^A-Za-z]+|[^A-Za-z.]+$", "", name)
        is_total = len(nums) == n_years and (name.upper().replace(" ", "") == "SINDH" or
                                             (not name and current is not None and blocks[current]["rows"]))
        if is_total and len(blocks) < len(order):
            current = order[len(blocks)]
            columns = [w["right"] for w in nums]
            blocks[current] = {"total": [_value(w) for w in nums], "rows": []}
            continue
        if current is None or (not nums and len(key(name)) < 3):
            continue  # nothing to read, or the page number
        if name.lower().startswith(("note", "source")):
            continue
        values = [None] * n_years
        for w in nums:
            col = min(range(n_years), key=lambda i: abs(columns[i] - w["right"]))
            if abs(columns[col] - w["right"]) > 40 * dpi / TABLE_DPI or w["conf"] < MIN_CONFIDENCE:
                skipped.append({"name": name, "reason": "number not under a year column or read with low confidence",
                                "numbers": f"{w['text']} (confidence {w['conf']:.0f})"})
                continue
            values[col] = _value(w)
        blocks[current]["rows"].append((name, values))
    return blocks, unit, skipped


def key(name):
    return re.sub(r"[^a-z]", "", name.lower())


def _pair(blocks):
    """Area and production lines of the same district: by name, or by order when OCR lost the names."""
    area, prod = blocks["area"]["rows"], blocks["prod"]["rows"]
    prod_named = {key(n): v for n, v in prod if key(n)}
    if all(key(n) in prod_named for n, _ in area if key(n)):
        return [(n, a, prod_named.get(key(n))) for n, a in area]
    if len(area) == len(prod):
        return [(n, a, p) for (n, a), (_, p) in zip(area, prod)]
    return None


def read_crop_table(page, years_label, crop, page_no):
    """Read the table at several OCR resolutions; keep, for each year, a reading whose columns add up."""
    n = len(years_label)
    chosen, checks, skipped, unit = {}, [], [], None
    readings = []
    for dpi in OCR_DPIS:
        blocks, unit_here, bad = read_table(page, n, dpi)
        if "area" not in blocks or "prod" not in blocks:
            continue
        pairs = _pair(blocks)
        if pairs is None:
            continue
        unit = unit or unit_here
        readings.append((dpi, blocks, pairs, bad))
        for i in range(n):
            if i in chosen:
                continue
            sums = {what: sum((a if what == "area" else p or [None] * n)[i] or 0 for _, a, p in pairs)
                    for what in ("area", "prod")}
            if sums["area"] == blocks["area"]["total"][i] and sums["prod"] == blocks["prod"]["total"][i]:
                chosen[i] = (dpi, pairs, blocks)
        if len(chosen) == n:
            break
    if not readings:
        skipped.append({"crop": crop, "name": "(whole table)", "page": page_no,
                        "reason": "area and production blocks could not be read", "numbers": ""})
        return {}, unit, checks, skipped
    table = {}
    for i, first in enumerate(years_label):
        label = f"{first}-{str(first + 1)[2:]}"
        dpi, pairs, blocks = chosen.get(i, (readings[0][0], readings[0][2], readings[0][1]))
        for what, blk_name in (("area", "area"), ("production", "prod")):
            s = sum(((a if blk_name == "area" else p or [None] * n)[i] or 0) for _, a, p in pairs)
            checks.append({"check": "Sindh total (exact)", "crop": crop, "province": "Sindh",
                           "what": f"{label}: {what} (OCR at {dpi} dpi)", "districts_sum": s,
                           "book_total": blocks[blk_name]["total"][i], "page": page_no, "ok": i in chosen})
        if i not in chosen:
            skipped.append({"crop": crop, "name": f"(column {label})", "page": page_no,
                            "reason": f"does not add up to the SINDH total at any OCR resolution {OCR_DPIS}: "
                                      "a misread is likely, so no rows are written for this year", "numbers": ""})
            continue
        for name, a, p in pairs:
            table.setdefault(name, {})[i] = (a[i], (p or [None] * n)[i])
    used = {d for d, _, _ in chosen.values()}
    for dpi, _, _, bad in readings:
        if dpi in used:
            skipped += [{**b, "crop": crop, "page": page_no, "reason": f"{b['reason']} ({dpi} dpi)"} for b in bad]
    return table, unit, checks, skipped


def read_book(pdf_path, aliases_path=common.ALIASES, workbook_path=common.WORKBOOK, tables=None):
    workbook = pd.read_excel(workbook_path, sheet_name="All Data")
    ours = {key(d): d for p, d in zip(workbook["province"], workbook["district"]) if p == "Sindh"}
    aliases = pd.read_csv(aliases_path)
    alias = {key(a.book_name): a.district for a in aliases.itertuples() if a.source in ("sindh_bos", "mnfsr")
             and a.province == "Sindh"}
    book = os.path.basename(pdf_path)
    rows, checks, skipped = [], [], []
    with pdfplumber.open(pdf_path) as pdf:
        found = tables or find_tables(pdf)
        missing = [c for c in CROPS if c not in found]
        for c in missing:
            skipped.append({"crop": c, "name": "(whole table)", "page": "", "reason": "table not found in the book",
                            "numbers": ""})
        for crop, info in found.items():
            page = pdf.pages[info["page"] - 1]
            years_label = list(range(info["first"], info["last"] + 1))  # first year of each season (2018 = 2018-19)
            table, unit, table_checks, table_skipped = read_crop_table(page, years_label, crop, info["page"])
            checks += table_checks
            skipped += table_skipped
            title = f"Sindh Statistics (Bureau of Statistics Sindh), {crop} by district [{book}]"
            for a_name, by_year in table.items():
                district = ours.get(key(a_name)) or alias.get(key(a_name)) or a_name
                for i, (area, prod) in by_year.items():
                    if not area:
                        continue
                    if prod is None:
                        skipped.append({"crop": crop, "name": a_name, "page": info["page"],
                                        "reason": "area printed but production empty", "numbers": str(area)})
                        continue
                    edition = (str(years_label[i]), str(years_label[i] + 1)[2:])
                    book_row = {"crop": crop, "book_province": "Sindh", "book_name": a_name, "page": info["page"],
                                "area": area / 1000, "prod": prod / 1000, "area_prev": None, "prod_prev": None}
                    names = pd.DataFrame([{"book_province": "Sindh", "book_name": a_name, "province": "Sindh",
                                           "district": district}])
                    new, _ = common.to_yield_rows([book_row], edition, title, names)
                    for r in new:
                        r["production_as_printed_000"] = prod / 1000
                        r["production_unit_printed"] = "bales (lint)" if unit == "bales" else "tonnes"
                    rows += new
    known = set(zip(workbook["province"], workbook["district"]))
    unknown = sorted({(r["province"], r["district"], r["book_name"]) for r in rows
                      if (r["province"], r["district"]) not in known})
    return {"edition": "", "title": f"Sindh Statistics district crop tables [{book}]", "rows": rows,
            "totals": checks, "skipped": skipped, "unknown_names": unknown,
            "comparison": common.compare_with_workbook(rows, workbook)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read the district crop tables of a scanned Sindh Statistics book")
    parser.add_argument("pdf")
    parser.add_argument("--out", required=True, help="Excel file to write, e.g. data/sources/new/sindh_statistics_2025.xlsx")
    args = parser.parse_args(argv)
    result = read_book(args.pdf)
    common.report(result, common.write_excel(result, args.out), args.out)


if __name__ == "__main__":
    main()
