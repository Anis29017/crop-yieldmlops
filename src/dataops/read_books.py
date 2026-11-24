"""DataOps - read every downloaded book with the reader for its source, one Excel file per book.

Runs after the fetcher (src/dataops/fetch.py), on the same folder:
  <folder>/<source>/<book>.pdf   ->   <folder>/excel/<source>__<book>.xlsx
and writes <folder>/reading.md (also added to report.md, the pull request text).

With --accept <dir> (the monthly workflow uses data/sources/new_yields), the rows of every
book that passed ALL its checks (no failed total, no unknown district name) are also written
to <dir>/<source>__<book>.csv. Those files go into the pull request; once a person reviews
and merges it, build_yield_labels adds them to the training data. A book with any failed
check gets no CSV: its Excel shows what failed, and it is fixed in the reader or entered by
hand.

Reader per source:
  mnfsr       -> read_mnfsr   (MNFSR Crops Area & Production, district wise)
  punjab_crs  -> read_punjab  (Punjab CRS Kharif / Rabi final estimates)
  sindh_bos   -> read_sindh   (Sindh Statistics, scanned: OCR)

A book that cannot be read is reported, never skipped silently, and does not stop the others.

Usage:
  python -m src.dataops.read_books /tmp/crop-yield-sources
  python -m src.dataops.read_books /tmp/crop-yield-sources --accept data/sources/new_yields
"""

import argparse
import glob
import os
import sys

import pandas as pd

from src.dataops import read_mnfsr, read_punjab, read_sindh

READERS = {"mnfsr": read_mnfsr, "punjab_crs": read_punjab, "sindh_bos": read_sindh}


def read_folder(folder, accept_dir=None):
    results = []
    for source, reader in READERS.items():
        for pdf in sorted(glob.glob(os.path.join(folder, source, "*.pdf"))):
            name = os.path.splitext(os.path.basename(pdf))[0]
            out = os.path.join(folder, "excel", f"{source}__{name}.xlsx")
            entry = {"source": source, "book": os.path.basename(pdf), "excel": os.path.relpath(out, folder)}
            try:
                result = reader.read_book(pdf)
                kept = read_mnfsr.write_excel(result, out)  # same Excel layout for every source
                comp = pd.DataFrame(result["comparison"])
                this = comp[comp["column"] == "this"] if len(comp) else comp
                entry.update({
                    "status": "read",
                    "rows": len(kept),
                    "checks": len(result["totals"]),
                    "checks_failed": sum(not t["ok"] for t in result["totals"]),
                    "unknown_names": len(result["unknown_names"]),
                    "skipped": len(result["skipped"]),
                    "same_as_our_data": int((this["result"] == "same").sum()) if len(this) else 0,
                    "new_rows": int((this["result"] == "new (not in our data)").sum()) if len(this) else 0,
                    "differ_from_our_data": int((~this["result"].isin(["same", "new (not in our data)"])).sum())
                    if len(this) else 0,
                })
                entry["accepted"] = ""
                if accept_dir:
                    if entry["checks_failed"] == 0 and entry["unknown_names"] == 0 and len(kept):
                        csv = os.path.join(accept_dir, f"{source}__{name}.csv")
                        os.makedirs(accept_dir, exist_ok=True)
                        kept[read_mnfsr.COLUMNS].sort_values(["province", "crop", "district", "year"]).to_csv(
                            csv, index=False)
                        entry["accepted"] = csv
                    else:
                        entry["accepted"] = "no: a check failed or a name is unknown"
            except Exception as e:  # report and carry on with the other books
                entry.update({"status": f"could not read: {type(e).__name__}: {e}"})
            results.append(entry)
    return results


def summarise(results):
    if not results:
        return "## Reading the books\n\nNo downloaded books to read.\n"
    lines = ["## Reading the books", "",
             "| Source | Book | Result | Rows | Checks failed | Unknown names | Skipped lines | "
             "Same / new / differ vs our data | Added to this pull request |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if r["status"] == "read":
            lines.append(f"| {r['source']} | {r['book']} | read | {r['rows']} | {r['checks_failed']} of {r['checks']} | "
                         f"{r['unknown_names']} | {r['skipped']} | {r['same_as_our_data']} / {r['new_rows']} / "
                         f"{r['differ_from_our_data']} | {r.get('accepted', '')} |")
        else:
            lines.append(f"| {r['source']} | {r['book']} | {r['status']} | | | | | | no |")
    lines += ["", "Each book's Excel file (in the run's artifact, folder `excel/`) has the rows and every check.",
              "Before merging: open the Excel of each added book and look at 'Check - vs our data' and the rows."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read every downloaded book into an Excel file")
    parser.add_argument("folder", help="the fetcher's download folder")
    parser.add_argument("--accept", help="folder for the CSV rows of books that passed every check")
    args = parser.parse_args(argv)
    results = read_folder(args.folder, args.accept)
    text = summarise(results)
    print(text)
    with open(os.path.join(args.folder, "reading.md"), "w") as f:
        f.write(text)
    report = os.path.join(args.folder, "report.md")
    if os.path.exists(report):
        with open(report, "a") as f:
            f.write("\n" + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
