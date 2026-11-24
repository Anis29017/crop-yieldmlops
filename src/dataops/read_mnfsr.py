"""DataOps - read an MNFSR "Crops Area & Production (District Wise)" book into yield rows.

Book layout (checked on CAP 2024-25): one table per crop ("DISTRICT-WISE AREA,
PRODUCTION & PERCENTAGE OF WHEAT CROP"), provinces as headings (PUNJAB, SINDH,
KPK, BALOCHISTAN), one row per district with 8 numbers:
  area last year, % share, area this year, % share,
  production last year, % share, production this year, % share
and a "Total" row per province. Areas in '000 hectares, production in '000 tonnes
('000 bales of lint for cotton).

Steps:
  1. read every district row and province total of the 4 crops
  2. district spellings -> our names through data/raw/district_aliases.csv
     (a spelling not in our data and not in the alias file is REPORTED, never guessed)
  3. units -> hectares / tonnes (1 bale = 170.09 kg lint), yield = production / area
  4. year rule: wheat -> harvest year (2024-25 -> 2025), Kharif crops -> sowing year (2024)
  5. checks: districts add up to the book's province totals; values compared with the
     rows already in our workbook (same district, crop, year); the 6A validation gate
  6. write an Excel file: "All Data" (same columns as the master workbook) + "Checks"

Usage:
  python -m src.dataops.read_mnfsr <book.pdf> --out data/sources/new/mnfsr_2024-25.xlsx
"""

import argparse
import os
import re

import pandas as pd
import pdfplumber

CROPS = ("wheat", "rice", "sugarcane", "cotton")
PROVINCES = {"PUNJAB": "Punjab", "SINDH": "Sindh", "KPK": "Khyber Pakhtunkhwa", "KHYBER": "Khyber Pakhtunkhwa",
             "BALOCHISTAN": "Balochistan", "BALUCHISTAN": "Balochistan"}
NUM = r"-?\d+(?:\.\d+)?"
ROW = re.compile(rf"^([A-Za-z][A-Za-z .'\-()]*?)\s+((?:{NUM}\s*)+)$")
NAME_ONLY = re.compile(r"^[A-Za-z][A-Za-z .'\-()]*$")
NUMBERS_ONLY = re.compile(rf"^(?:{NUM}\s*)+$")
EDITION = re.compile(r"((?:19|20)\d{2})\s*-\s*(\d{2})\b")
NOT_DISTRICTS = {"pakistan", "table no.", "district", "province/"}
KG_PER_BALE = 170.09
ALIASES = "data/raw/district_aliases.csv"
WORKBOOK = "data/raw/Pakistan_Master_Yield_1981-2025.xlsx"
TOTAL_TOLERANCE = 0.005  # 0.5%: the book rounds every district
COMPARE_TOLERANCE = 0.02  # 2%: differences with our existing rows that are reported
COLUMNS = ["year", "crop", "district", "province", "area_ha", "production_t", "yield_t_ha", "note", "correction",
           "source", "pdf_page", "production_as_printed_000", "production_unit_printed"]


def read_tables(pdf_path):
    """Raw district rows and province totals, exactly as printed (thousands)."""
    rows, totals, skipped = [], [], []
    crop = province = None
    edition = None
    pending_name = wrapped_row = None
    title_open = False
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, 1):
            for line in (page.extract_text() or "").splitlines():
                text = line.strip()
                upper = text.upper()
                m = re.search(r"PERCENTAGE OF ([A-Z ]+?) CROP", upper)
                if not m and title_open:
                    # title printed over two lines: "... & PERCENTAGE" / "OF MAIZE CROP" (or "... OF" / "BAJRA CROP")
                    m = re.match(r"^(?:OF )?([A-Z ]+?) CROP$", upper)
                if m:
                    name = m.group(1).strip().lower()
                    crop = name if name in CROPS else None
                    title_open = False
                    continue
                if "DISTRICT-WISE AREA" in upper:
                    crop, title_open = None, True  # a new table starts; its crop is on the next line
                    continue
                title_open = False
                if crop is None:
                    continue
                if upper.startswith("PROVINCE/") and edition is None:
                    found = EDITION.findall(text)
                    if found:
                        edition = found[-1]  # the newer of the two years in the header
                words = upper.split()
                if words and words[0] in PROVINCES and (len(words) == 1 or "HECTAR" in upper):
                    province = PROVINCES[words[0]]
                    continue
                if province is None:
                    continue
                # Some books wrap a long name around its numbers:
                #   "Shaheed Benazir" / "90.30 7.6 84.98 ..." / "Abad"
                if NAME_ONLY.match(text) and not upper.startswith(("TOTAL", "HECTARES", "AREA", "TABLE", "DISTRICT")):
                    if wrapped_row is not None:
                        wrapped_row["book_name"] = f"{wrapped_row['book_name']} {text}"
                        wrapped_row = None
                    else:
                        pending_name = text
                    continue
                wrapped_row = None
                m = ROW.match(text)
                if not m and NUMBERS_ONLY.match(text) and pending_name:
                    m = ROW.match(f"{pending_name} {text}")
                    joined = True
                else:
                    joined = False
                pending_name = None
                if not m:
                    continue
                name, nums = m.group(1).strip(), [float(x) for x in re.findall(NUM, m.group(2))]
                if name.lower() in NOT_DISTRICTS:
                    continue
                if name.lower().startswith("total"):
                    totals.append({"crop": crop, "province": province, "page": page_no, "printed": nums[:4]})
                elif len(nums) == 8:
                    rows.append({"crop": crop, "book_province": province, "book_name": name, "page": page_no,
                                 "area_prev": nums[0], "area": nums[2], "prod_prev": nums[4], "prod": nums[6]})
                    wrapped_row = rows[-1] if joined else None
                else:
                    skipped.append({"crop": crop, "province": province, "name": name, "page": page_no,
                                    "reason": f"{len(nums)} numbers instead of 8", "numbers": " ".join(map(str, nums))})
    if edition is None:
        raise ValueError("Could not find the edition (e.g. 2024-25) in the table headers")
    return rows, totals, skipped, edition


def years(edition, crop):
    """(this year, last year) in our convention for a book edition like ('2024', '25')."""
    first = int(edition[0])
    this = first + 1 if crop == "wheat" else first
    return this, this - 1


def check_totals(rows, totals):
    """Do the districts add up to the province totals printed in the book?"""
    out = []
    for t in totals:
        part = [r for r in rows if r["crop"] == t["crop"] and r["book_province"] == t["province"]]
        sums = [sum(r[k] for r in part) for k in ("area_prev", "area", "prod_prev", "prod")]
        for label, s, printed in zip(("area last year", "area", "production last year", "production"), sums, t["printed"]):
            gap = abs(s - printed) / printed if printed else abs(s)
            out.append({"check": "province total", "crop": t["crop"], "province": t["province"], "what": label,
                        "districts_sum": round(s, 2), "book_total": printed, "page": t["page"],
                        "ok": gap <= TOTAL_TOLERANCE or abs(s - printed) < 0.1})
    return out


def to_yield_rows(rows, edition, book_title, aliases):
    """Book rows -> rows in the master workbook's columns (this year and last year)."""
    alias = {(a.book_province, a.book_name): (a.province, a.district) for a in aliases.itertuples()}
    out, unknown = [], []
    for r in rows:
        province, district = alias.get((r["book_province"], r["book_name"]), (r["book_province"], r["book_name"]))
        this, last = years(edition, r["crop"])
        for which, year, area, prod in (("this", this, r["area"], r["prod"]), ("last", last, r["area_prev"], r["prod_prev"])):
            if area is None or area <= 0:
                continue  # area empty or 0 = crop not grown there (the M1 rule)
            area_ha = round(area * 1000)
            production_t = float(round(prod * (KG_PER_BALE if r["crop"] == "cotton" else 1000)))  # whole tonnes, as in M1
            out.append({
                "year": year, "crop": r["crop"], "district": district, "province": province,
                "area_ha": area_ha, "production_t": production_t,
                "yield_t_ha": round(production_t / area_ha, 3) if production_t > 0 else None,
                "note": "Area under 1,000 ha: yield imprecise due to rounding" if area_ha < 1000 else "",
                "correction": "", "source": book_title, "pdf_page": r["page"],
                "production_as_printed_000": prod,
                "production_unit_printed": "000 bales (lint)" if r["crop"] == "cotton" else "000 tonnes",
                "column": which, "book_name": r["book_name"],
            })
    return out, unknown


def compare_with_workbook(new, workbook):
    """Rows of the book that differ from what our workbook already has (same district, crop, year)."""
    have = workbook.set_index(["province", "district", "crop", "year"])[["area_ha", "production_t", "source"]]
    out = []
    for r in new:
        key = (r["province"], r["district"], r["crop"], r["year"])
        if key not in have.index:
            out.append({**{k: r[k] for k in ("province", "district", "crop", "year", "column")},
                        "result": "new (not in our data)", "book_area_ha": r["area_ha"],
                        "book_production_t": r["production_t"], "our_area_ha": None, "our_production_t": None,
                        "our_source": ""})
            continue
        old = have.loc[key]
        old = old.iloc[0] if isinstance(old, pd.DataFrame) else old
        diff = max(abs(r["area_ha"] - old.area_ha) / max(old.area_ha, 1),
                   abs(r["production_t"] - old.production_t) / max(old.production_t, 1))
        out.append({**{k: r[k] for k in ("province", "district", "crop", "year", "column")},
                    "result": "same" if diff <= COMPARE_TOLERANCE else f"differs by {diff:.0%}",
                    "book_area_ha": r["area_ha"], "book_production_t": r["production_t"],
                    "our_area_ha": old.area_ha, "our_production_t": old.production_t, "our_source": old.source})
    return out


def read_book(pdf_path, aliases_path=ALIASES, workbook_path=WORKBOOK):
    rows, totals, skipped, edition = read_tables(pdf_path)
    ed = f"{edition[0]}-{edition[1]}"
    title = f"MNFSR Crops Area & Production (District Wise) {ed} [{os.path.basename(pdf_path)}]"
    aliases = pd.read_csv(aliases_path)
    aliases = aliases[aliases["source"] == "mnfsr"]
    new, _ = to_yield_rows(rows, edition, title, aliases)
    workbook = pd.read_excel(workbook_path, sheet_name="All Data")
    known = set(zip(workbook["province"], workbook["district"]))
    unknown = sorted({(r["province"], r["district"], r["book_name"]) for r in new
                      if (r["province"], r["district"]) not in known})
    return {"edition": ed, "title": title, "rows": new, "totals": check_totals(rows, totals), "skipped": skipped,
            "unknown_names": unknown, "comparison": compare_with_workbook(new, workbook)}


def write_excel(result, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    data = pd.DataFrame(result["rows"])
    this_year = data[data["column"] == "this"]
    unknown = {(p, d) for p, d, _ in result["unknown_names"]}
    keep = this_year[[(p, d) not in unknown for p, d in zip(this_year["province"], this_year["district"])]]
    summary = [
        {"item": "book", "value": result["title"]},
        {"item": "rows for the new year (written to All Data)", "value": len(keep)},
        {"item": "totals checked / failed", "value":
            f"{len(result['totals'])} / {sum(not t['ok'] for t in result['totals'])}"},
        {"item": "district names not in our data (rows left out)", "value": len(result["unknown_names"])},
        {"item": "lines skipped (see Check - skipped lines)", "value": len(result["skipped"])},
    ]
    with pd.ExcelWriter(out_path) as xl:
        keep[COLUMNS].sort_values(["province", "crop", "district"]).to_excel(xl, sheet_name="All Data", index=False)
        pd.DataFrame(summary).to_excel(xl, sheet_name="Summary", index=False)
        pd.DataFrame(result["totals"]).to_excel(xl, sheet_name="Check - province totals", index=False)
        pd.DataFrame(result["comparison"]).to_excel(xl, sheet_name="Check - vs our data", index=False)
        pd.DataFrame(result["unknown_names"], columns=["province", "district", "book_name"]).to_excel(
            xl, sheet_name="Check - unknown names", index=False)
        pd.DataFrame(result["skipped"]).to_excel(xl, sheet_name="Check - skipped lines", index=False)
    return keep


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read an MNFSR district-wise book into yield rows")
    parser.add_argument("pdf")
    parser.add_argument("--out", required=True, help="Excel file to write, e.g. data/sources/new/mnfsr_2024-25.xlsx")
    args = parser.parse_args(argv)
    result = read_book(args.pdf)
    report(result, write_excel(result, args.out), args.out)


def report(result, keep, out_path):
    """Print what the reader found and what the checks say."""
    bad_totals = [t for t in result["totals"] if not t["ok"]]
    comp = pd.DataFrame(result["comparison"])
    this = comp[comp["column"] == "this"] if len(comp) else comp
    print(f"{result['title']}")
    print(f"  rows for the new year written: {len(keep)}")
    print(f"  totals: {len(result['totals'])} checked, {len(bad_totals)} do not add up")
    for t in bad_totals:
        print(f"    {t['crop']} {t['province']} {t['what']}: districts {t['districts_sum']} vs book {t['book_total']} (p.{t['page']})")
    print(f"  compared with our data (new year): {this['result'].value_counts().to_dict() if len(this) else {}}")
    print(f"  district names not in our data: {len(result['unknown_names'])} {result['unknown_names'][:5]}")
    print(f"  lines skipped: {len(result['skipped'])}")
    print(f"  -> {out_path}")

if __name__ == "__main__":
    main()
