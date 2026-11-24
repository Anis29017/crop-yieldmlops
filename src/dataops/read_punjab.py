"""DataOps - read a Punjab Crop Reporting Service (CRS) book into yield rows.

Books from https://agripunjab.punjab.gov.pk/index.php/node/8157 come in two layouts:

  "tables" (Final Kharif / Rabi Estimates books, e.g. Kharif 2023-24 to 2025-26, Rabi 2023-24):
    one table per crop and unit,
    FINAL ESTIMATE OF COTTON CROP / PUNJAB 2024-25 / (Area in '000 hect)
    columns: this year (AMER, DESI, TOTAL), last year (same), % change
    We read the TOTAL columns of the "'000 hect" table and the "Production in '000" table
    of cotton, rice and sugarcane (Kharif) or wheat (Rabi). Empty cells are not printed, so
    numbers are matched to their column by their position on the page, not by counting them.

  "summary" (Rabi books, and the one-page Kharif summaries): one table,
    FINAL ESTIMATES OF MAJOR RABI (or KHARIF) CROPS IN THE PUNJAB: 2024-25
    with an (Area '000 Hect., Production '000 Tonnes / Bales) column pair per crop.
    We read the Wheat pair (Rabi) or the Rice, Cotton and Sugarcane pairs (Kharif).

Checks: districts add up to their division and to the Punjab total; in the "tables" layout
each TOTAL also equals its split columns (AMER + DESI ...). Islamabad is listed with Punjab
(inside Rawalpindi division in some books, after the divisions in others) and is written under
Islamabad, as in our data.
A table the book prints empty (Final Kharif Estimates 2025-26: sugarcane '000 hect) is reported
and its crop left out: nothing is converted or guessed.
Everything else (units, year rule, comparison with our data, the Excel file) is shared with the
MNFSR reader.

Usage:
  python -m src.dataops.read_punjab <book.pdf> --out data/sources/new/punjab_kharif_2024-25.xlsx
"""

import argparse
import os
import re

import pandas as pd
import pdfplumber

from src.dataops import read_mnfsr as common

NUM = re.compile(r"^-?\d+(?:\.\d+)?$")
EDITION = re.compile(r"((?:19|20)\d{2})\s*-\s*(\d{2})\b")
TABLE_CROPS = {"COTTON": "cotton", "RICE": "rice", "SUGARCANE": "sugarcane", "WHEAT": "wheat"}
SPLIT_HEADERS = {"AMER", "DESI", "TOTAL", "BAS", "NON-BAS", "IRRI", "UN-IRRI"}
PUNJAB_TOTALS = {"punjab", "the punjab"}
REGION_TOTALS = {"north punjab", "south punjab"}  # groups of divisions: not checked
TOLERANCE = 0.01  # 1%: every district is rounded in the book


def page_lines(page, tol=3):
    """Words of a page grouped into lines (top to bottom), each line sorted left to right.

    Some books wrap a long name onto the next line ("Nankana" / "Sahib", "Bahawalpur" / "Divn."),
    printed a fraction of a point off that line's own text, or in bold next to a regular name.
    Such a piece is given back to the line above, so it does not get mixed into the next
    district's name.
    """
    words = sorted(page.extract_words(y_tolerance=0.2, extra_attrs=["fontname"]),
                   key=lambda w: (round(w["top"]), w["x0"]))
    lines = []
    for w in words:
        if lines and abs(lines[-1][0] - w["top"]) <= tol:
            lines[-1][1].append(w)
        else:
            lines.append([w["top"], [w]])
    lines = [sorted(ws, key=lambda w: w["x0"]) for _, ws in lines]

    def is_num(w):
        return NUM.match(w["text"].replace(",", ""))

    for i in range(1, len(lines)):
        nums = [w for w in lines[i] if is_num(w)]
        if not nums or not any(is_num(w) for w in lines[i - 1]) or lines[i][0]["text"].upper().startswith("SOURCE"):
            continue
        own = sorted(w["top"] for w in nums)[len(nums) // 2]
        first_x = min(w["x0"] for w in nums)
        names = [w for w in lines[i] if not is_num(w) and w["x1"] <= first_x]  # the name column only
        regular = [w for w in names if not _bold(w)]
        # a wrapped piece sits slightly off this line's own text, or is the bold end of a bold
        # division name on a line whose own name is regular
        wrapped = [w for w in names if abs(own - w["top"]) > 0.25 or (_bold(w) and regular)]
        if wrapped and len(wrapped) < len(names):
            lines[i] = [w for w in lines[i] if w not in wrapped]
            first_num = next((j for j, w in enumerate(lines[i - 1]) if is_num(w)), len(lines[i - 1]))
            lines[i - 1] = lines[i - 1][:first_num] + wrapped + lines[i - 1][first_num:]
    return lines


def all_lines(pdf_path):
    """(page number, words) for every line of the book, in reading order."""
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, 1):
            for ws in page_lines(page):
                yield page_no, ws


def split_line(ws):
    """Name words and number words of a table line."""
    name, nums = [], []
    for w in ws:
        (nums if NUM.match(w["text"].replace(",", "")) else name).append(w)
    return " ".join(w["text"] for w in name).strip(), nums


def _bold(w):
    return "bold" in (w.get("fontname") or "").lower()


def centre(w):
    return (w["x0"] + w["x1"]) / 2


def column_of(word, headers):
    """Index of the header whose centre is nearest to the number's centre."""
    c = centre(word)
    return min(range(len(headers)), key=lambda i: abs(centre(headers[i]) - c))


def kind_of_name(name):
    low = name.lower().rstrip(":. ")
    if low in PUNJAB_TOTALS:
        return "punjab"
    if low in REGION_TOTALS:
        return "region"
    if re.search(r"\bdivn?\b|\bdv\b|div[n.:]", low):
        return "division"
    return "district"


# ---------- Kharif book ----------

def read_kharif(pdf_path):
    tables = {}  # (crop, "area"|"prod") -> list of lines
    crop = what = headers = None
    edition = unit = None
    for page_no, ws in all_lines(pdf_path):
        text = " ".join(w["text"] for w in ws)
        upper = text.upper()
        if upper.startswith("SOURCE"):
            continue
        m = re.search(r"FINAL ESTIMATE OF (.+?)(?: CROPS?)?$", upper)
        if m:
            first = m.group(1).split()[0]
            crop = TABLE_CROPS.get(first)
            what = headers = None
            continue
        if crop is None:
            continue
        if edition is None and upper.startswith("PUNJAB") and EDITION.search(text) and len(ws) <= 3:
            edition = EDITION.search(text).groups()
            continue
        if "AREA IN '000 HECT" in upper:
            what, headers = "area", None
            continue
        if "PRODUCTION IN '000" in upper:
            what, headers = "prod", None
            unit = "000 bales (lint)" if "BALES" in upper else "000 tonnes"
            continue
        if "(AREA IN" in upper or "YIELD" in upper:  # acres or yield tables: not used
            what = headers = None
            continue
        if what is None:
            continue
        if {w["text"].upper() for w in ws} & SPLIT_HEADERS and sum(w["text"].upper() == "TOTAL" for w in ws) >= 2:
            headers = [w for w in ws if w["text"].upper() in SPLIT_HEADERS]
            continue
        if headers is not None and all(w["text"].upper() in SPLIT_HEADERS for w in ws):
            # header printed on two lines (e.g. NON-BAS under the BAS / TOTAL line)
            headers = sorted(headers + ws, key=lambda w: w["x0"])
            continue
        if headers is None:
            continue
        name, nums = split_line(ws)
        if not name or name.upper().startswith(("DIVISIONS", "DISTRICTS")):
            continue
        totals_at = [i for i, h in enumerate(headers) if h["text"].upper() == "TOTAL"][:2]
        values = {}
        for w in nums:
            values.setdefault(column_of(w, headers), float(w["text"].replace(",", "")))
        tables.setdefault((crop, what), []).append({
            "name": name, "page": page_no, "kind": kind_of_name(name),
            "this": values.get(totals_at[0]), "last": values.get(totals_at[1]),
            # the split columns (e.g. AMER + DESI) printed before each TOTAL, to check the TOTAL itself
            "this_parts": [values[i] for i in range(totals_at[0]) if i in values],
            "last_parts": [values[i] for i in range(totals_at[0] + 1, totals_at[1]) if i in values],
        })
    if edition is None:
        raise ValueError("Could not find the edition (e.g. PUNJAB 2024-25) in the book")
    return tables, edition, unit


def kharif_rows(tables):
    """Join each crop's area and production tables district by district."""
    rows, totals, skipped = [], [], []
    for crop in TABLE_CROPS.values():
        area, prod = tables.get((crop, "area"), []), tables.get((crop, "prod"), [])
        if not area or not prod:
            continue
        for table, label in ((area, "'000 hect"), (prod, "production")):
            if all(x["this"] is None and x["last"] is None for x in table):
                skipped.append({"crop": crop, "name": "(whole table)", "page": table[0]["page"],
                                "reason": f"the {label} table is printed empty in the book: no {crop} rows",
                                "numbers": ""})
        if any(all(x["this"] is None and x["last"] is None for x in table) for table in (area, prod)):
            continue
        prod_by_name = {}
        for p in prod:
            prod_by_name.setdefault((p["kind"], p["name"].lower()), p)
        for a in area:
            p = prod_by_name.get((a["kind"], a["name"].lower()))
            if p is None:
                skipped.append({"crop": crop, "name": a["name"], "page": a["page"],
                                "reason": "no matching line in the production table", "numbers": ""})
                continue
            rec = {"crop": crop, "name": a["name"], "kind": a["kind"], "page": a["page"],
                   "area": a["this"], "area_prev": a["last"], "prod": p["this"], "prod_prev": p["last"],
                   "lines": {"area": a, "production": p}}
            if a["kind"] == "district":
                for area_key, prod_key, label in (("area", "prod", "this year"), ("area_prev", "prod_prev", "last year")):
                    area_v, prod_v = rec[area_key], rec[prod_key]
                    if (area_v or 0) > 0 and prod_v is None or (prod_v or 0) > 0 and area_v is None:
                        skipped.append({"crop": crop, "name": a["name"], "page": a["page"],
                                        "reason": f"{label}: area printed but production empty (or the reverse); "
                                                  f"that year left out",
                                        "numbers": f"area {area_v}, production {prod_v}"})
                        rec[area_key] = rec[prod_key] = None
                rows.append(rec)
            else:
                totals.append(rec)
    return rows, totals, skipped


# ---------- one-page summary book ----------

SUMMARY_CROPS = {"WHEAT": "wheat", "RICE": "rice", "COTTON": "cotton", "SUGARCANE": "sugarcane"}


def _is_area(w):
    return w["text"].upper().startswith("AREA")


def _is_production(w):
    return w["text"].upper().endswith("RODUCTION")  # also "P roduction", printed split in some books


def read_summary(pdf_path):
    """One table, one (Area, Production) column pair per crop. District, division and Punjab lines, in order."""
    ordered, skipped = [], []
    edition = crop_words = headers = None
    pairs = {}
    for page_no, ws in all_lines(pdf_path):
        text = " ".join(w["text"] for w in ws)
        upper = text.upper()
        if edition is None and "PUNJAB" in upper and EDITION.search(text):
            edition = EDITION.search(text).groups()
            continue
        if crop_words is None:
            found = [w for w in ws if w["text"].upper() in SUMMARY_CROPS]
            if found:
                crop_words = found
            continue
        if headers is None:
            labels = [w for w in ws if _is_area(w) or _is_production(w)]
            if len(labels) >= 2:
                headers = labels
                areas = [i for i, h in enumerate(headers) if _is_area(h)]
                for cw in crop_words:
                    candidates = [(i, next((j for j in range(i + 1, len(headers)) if _is_production(headers[j])), None))
                                  for i in areas]
                    candidates = [(i, j) for i, j in candidates if j is not None]
                    i, j = min(candidates, key=lambda ij: abs((centre(headers[ij[0]]) + centre(headers[ij[1]])) / 2
                                                             - centre(cw)))
                    pairs[SUMMARY_CROPS[cw["text"].upper()]] = (i, j)
            continue
        if upper.startswith("SOURCE"):
            break
        name, nums = split_line(ws)
        if not name or not nums or name.startswith("("):
            continue
        values = {}
        for w in nums:
            values.setdefault(column_of(w, headers), float(w["text"].replace(",", "")))
        for crop, (i, j) in pairs.items():
            rec = {"crop": crop, "name": name, "kind": kind_of_name(name), "page": page_no,
                   "area": values.get(i), "prod": values.get(j), "area_prev": None, "prod_prev": None}
            if rec["area"] is None or rec["prod"] is None:
                skipped.append({"crop": crop, "name": name, "page": page_no,
                                "reason": "area or production missing", "numbers": " ".join(w["text"] for w in nums)})
                continue
            ordered.append(rec)
    if edition is None or not pairs:
        raise ValueError("Could not find the edition or the crop Area / Production columns in the book")
    rows = [r for r in ordered if r["kind"] == "district"]
    totals = [r for r in ordered if r["kind"] != "district"]
    return rows, totals, skipped, edition, ordered


# ---------- checks ----------

NOT_IN_A_DIVISION = {"islamabad"}  # listed with Punjab; some books count it in Rawalpindi division, some not


def check_row_totals(rows):
    """Kharif: each district's TOTAL equals its split columns (AMER + DESI, BAS + NON-BAS, IRRI + UN-IRRI)."""
    out = []
    for r in rows:
        for what, line in r.get("lines", {}).items():
            for col, label in (("this", what), ("last", f"{what} last year")):
                total, parts = line[col], line[f"{col}_parts"]
                if total is None or not parts:
                    continue
                s = sum(parts)
                step = 1.0 if all(float(x).is_integer() for x in parts + [total]) else 0.1
                out.append({"check": "row total = its parts", "crop": r["crop"], "province": "Punjab",
                            "what": f"{r['name']}: {label}", "districts_sum": round(s, 2), "book_total": total,
                            "page": line["page"], "ok": abs(s - total) <= step * len(parts)})
    return out

def check_totals(rows, totals, ordered):
    """Districts add up to their division (the lines after a division row) and to the Punjab total."""
    out = []
    division_of = {}
    current = {}  # crop -> its current division line
    for rec in ordered:
        if rec["kind"] == "division":
            current[rec["crop"]] = rec
        elif rec["kind"] in ("punjab", "region"):
            current.pop(rec["crop"], None)
        elif rec["crop"] in current:
            division_of[id(rec)] = id(current[rec["crop"]])
    for t in totals:
        if t["kind"] == "region":
            continue
        if t["kind"] == "punjab":
            part = [r for r in rows if r["crop"] == t["crop"]]
            label = "Punjab total"
        else:
            part = [r for r in rows if r["crop"] == t["crop"] and division_of.get(id(r)) == id(t)]
            label = t["name"]
        for key, what in (("area", "area"), ("prod", "production"), ("area_prev", "area last year"),
                          ("prod_prev", "production last year")):
            printed = t.get(key)
            if printed is None:
                continue
            def close(s, n):
                gap = abs(s - printed)
                if float(printed).is_integer():
                    return gap <= max(TOLERANCE * abs(printed), 0.5 * max(n, 1) + 0.5)
                return gap <= max(TOLERANCE * abs(printed), 0.05 * n + 0.05)

            s = sum(r[key] or 0 for r in part)
            ok, note = close(s, len(part)), ""
            outside = [r for r in part if r["name"].lower() in NOT_IN_A_DIVISION]
            if t["kind"] == "division" and outside and not ok:
                # some books count Islamabad in its division, others list it after the divisions
                s = s - sum(r[key] or 0 for r in outside)
                ok, note = close(s, len(part) - len(outside)), " (without Islamabad)"
            out.append({"check": "division total" if t["kind"] == "division" else "Punjab total",
                        "crop": t["crop"], "province": "Punjab", "what": f"{label}: {what}{note}",
                        "districts_sum": round(s, 2), "book_total": printed, "page": t["page"], "ok": ok})
    return out


# ---------- names ----------

def key(name):
    return re.sub(r"[^a-z]", "", name.lower())


def name_table(book_names, workbook, aliases_path):
    """book name -> (province, district): our own spelling, else the alias file, else unknown (left as is)."""
    ours = {key(d): (p, d) for p, d in zip(workbook["province"], workbook["district"]) if p == "Punjab"}
    aliases = pd.read_csv(aliases_path)
    alias = {key(a.book_name): (a.province, a.district) for a in aliases.itertuples() if a.source == "punjab_crs"}
    out = []
    for n in sorted(set(book_names)):
        province, district = ours.get(key(n)) or alias.get(key(n)) or ("Punjab", n)
        out.append({"book_province": "Punjab", "book_name": n, "province": province, "district": district})
    return pd.DataFrame(out, columns=["book_province", "book_name", "province", "district"])


# ---------- whole book ----------

def book_type(pdf_path):
    """'tables': one table per crop and unit (Kharif books); 'summary': one page, all crops side by side."""
    with pdfplumber.open(pdf_path) as pdf:
        first = " ".join((p.extract_text() or "") for p in pdf.pages[:3]).upper()
    if re.search(r"FINAL ESTIMATES? OF MAJOR (KHARIF|RABI) CROPS", first):
        return "summary"
    if re.search(r"FINAL ESTIMATE OF (COTTON|WHEAT) CROP", first):
        return "tables"
    raise ValueError("Not a Punjab CRS book I can read (no 'FINAL ESTIMATE OF COTTON/WHEAT CROP' or "
                     "'FINAL ESTIMATES OF MAJOR KHARIF/RABI CROPS' found)")


def read_book(pdf_path, aliases_path=common.ALIASES, workbook_path=common.WORKBOOK):
    kind = book_type(pdf_path)
    if kind == "tables":
        tables, edition, _ = read_kharif(pdf_path)
        rows, totals, skipped = kharif_rows(tables)
        ordered = _in_book_order(tables, rows, totals)
    else:
        rows, totals, skipped, edition, ordered = read_summary(pdf_path)
    ed = f"{edition[0]}-{edition[1]}"
    label = "Final Estimates of Major Rabi Crops" if rows and rows[0]["crop"] == "wheat" else \
        "Final Estimates of Major Kharif Crops"
    title = f"Punjab CRS {label} {ed} [{os.path.basename(pdf_path)}]"
    workbook = pd.read_excel(workbook_path, sheet_name="All Data")
    names = name_table([r["name"] for r in rows], workbook, aliases_path)
    book_rows = [{**r, "book_province": "Punjab", "book_name": r["name"]} for r in rows]
    new, _ = common.to_yield_rows(book_rows, edition, title, names)
    known = set(zip(workbook["province"], workbook["district"]))
    unknown = sorted({(r["province"], r["district"], r["book_name"]) for r in new
                      if (r["province"], r["district"]) not in known})
    checks = check_totals(rows, totals, ordered) + check_row_totals(rows)
    return {"edition": ed, "title": title, "rows": new, "totals": checks,
            "skipped": skipped, "unknown_names": unknown, "comparison": common.compare_with_workbook(new, workbook)}


def _in_book_order(tables, rows, totals):
    """Rows and totals in the order of the area tables (division row first, then its districts)."""
    by_key = {(r["crop"], r["kind"], r["name"].lower()): r for r in rows + totals}
    ordered = []
    for (crop, what), lines in tables.items():
        if what != "area":
            continue
        for line in lines:
            rec = by_key.get((crop, line["kind"], line["name"].lower()))
            if rec is not None:
                ordered.append(rec)
    return ordered


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read a Punjab CRS Kharif or Rabi book into yield rows")
    parser.add_argument("pdf")
    parser.add_argument("--out", required=True, help="Excel file to write, e.g. data/sources/new/punjab_kharif_2024-25.xlsx")
    args = parser.parse_args(argv)
    result = read_book(args.pdf)
    common.report(result, common.write_excel(result, args.out), args.out)


if __name__ == "__main__":
    main()
