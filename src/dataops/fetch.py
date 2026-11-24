"""DataOps - watch the official source pages and download new yield books.

For each source in src/dataops/sources.py:
  1. read the page that lists its books
  2. keep the links that are yield books (the source's rule)
  3. compare them with data/sources/manifest.csv (every book already known)
  4. download only the NEW books, check each is a complete PDF, record its
     SHA-256 checksum and size in the manifest

Nothing is guessed or filled in: a page that cannot be reached, a page with no
links, or a broken download is reported as such and retried on the next run.

The PDFs are large, so they are saved OUTSIDE the repository (DOWNLOAD_DIR);
only the manifest is committed. The GitHub workflow uploads them as an artifact.

Usage:
  python -m src.dataops.fetch          # find and download new books
  python -m src.dataops.fetch --init   # first run: record every book listed today as
                                       # 'known' without downloading (already used in M1)
  python -m src.dataops.fetch --latest --dest ~/Downloads/crop-sources
                                       # test: download only the NEWEST book of each source;
                                       # the manifest is not read or changed
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin

import requests

from src.dataops.sources import SOURCES, _name, edition, season

MANIFEST = "data/sources/manifest.csv"
DOWNLOAD_DIR = os.environ.get("SOURCES_DOWNLOAD_DIR", os.path.join(tempfile.gettempdir(), "crop-yield-sources"))
FIELDS = ["source", "edition", "season", "title", "url", "status", "sha256", "bytes", "first_seen", "note"]
# Statuses: known = listed when the manifest was started (not downloaded)
#           new = downloaded and checked, waiting to be read into the yield data
#           failed / invalid = download error / not a complete PDF (tried again next run)
SKIP = {"known", "new", "processed"}
MIN_PDF_BYTES = 20_000
HEADERS = {"User-Agent": "crop-yield-mlops-dataops/1.0 (+https://github.com/Anis29017/crop-yieldmlops)"}
TIMEOUT = (20, 180)  # seconds: connect, read
RETRIES = 2


class _Links(HTMLParser):
    """Collect every <a href> with its text and its context.

    context = the text between the previous PDF link and this link (at most 200
    characters), e.g. a book title above a "Download" button. It stops at the
    previous PDF link so one book never borrows the title of the book before it.
    """

    def __init__(self, base):
        super().__init__()
        self.base, self.links, self._since_pdf, self._open = base, [], "", None

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            a = dict(attrs)
            if a.get("href"):
                self._open = {"url": urljoin(self.base, a["href"].strip()), "text": a.get("title") or "",
                              "context": self._since_pdf.strip()[-200:]}

    def handle_data(self, data):
        data = " ".join(data.split())
        if not data:
            return
        if self._open is not None:
            self._open["text"] = f"{self._open['text']} {data}".strip()
        self._since_pdf = f"{self._since_pdf} {data}"[-400:]

    def handle_endtag(self, tag):
        if tag == "a" and self._open is not None:
            self.links.append(self._open)
            if _name(self._open["url"]).lower().endswith(".pdf"):
                self._since_pdf = ""
            self._open = None


def find_links(html, base):
    parser = _Links(base)
    parser.feed(html)
    parser.close()
    return parser.links


def find_books(source, html):
    """The yield books on a source page, one per URL."""
    books, seen = [], set()
    for link in find_links(html, source["page"]):
        url, text, context = link["url"], link["text"], link["context"]
        if url in seen or not source["is_book"](url, text, context):
            continue
        seen.add(url)
        name = _name(url)
        title = text if len(text) > 8 else f"{context[-80:]} {text}".strip()
        from_title = edition(text) or edition(context[-80:])
        books.append({
            "source": source["id"],
            "edition": (from_title or edition(name)) if source.get("edition_from") == "title"
            else (edition(name) or from_title),
            "season": season(f"{name} {text}"),
            "title": re.sub(r"\s+", " ", title)[:150] or name,
            "url": url,
        })
    return books


def edition_key(book):
    """Sort key for 'newest': (first year, second year) of the edition, e.g. '2024-25' -> (2024, 2025)."""
    years = re.findall(r"\d{2,4}", book["edition"])
    if not years:
        return (0, 0)
    first = int(years[0])
    second = int(years[1]) if len(years) > 1 else first
    if second < 100:  # '24-25' style: '25' -> 2025
        second += first // 100 * 100
    return (first, second)


def latest(books):
    """The newest book on a page (by edition; on a tie, the one listed first)."""
    return [max(books, key=edition_key)] if books else []


def check_pdf(data):
    """'' if the bytes look like a complete PDF, otherwise the reason it is not."""
    if not data.startswith(b"%PDF"):
        return "not a PDF (the site returned something else, e.g. an error page)"
    if len(data) < MIN_PDF_BYTES:
        return f"too small ({len(data)} bytes)"
    if b"%%EOF" not in data[-2048:]:
        return "incomplete PDF (no end-of-file marker; download cut off?)"
    return ""


def _get(url, session):
    last = None
    for attempt in range(RETRIES + 1):
        try:
            r = session.get(url, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last = e
            time.sleep(3 * (attempt + 1))
    raise last


def load_manifest(path=MANIFEST):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_manifest(rows, path=MANIFEST):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["source"], r["edition"], r["season"], r["url"]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def download(book, session, folder=DOWNLOAD_DIR):
    """Download one book; returns the manifest fields for it."""
    try:
        data = _get(book["url"], session).content
    except requests.RequestException as e:
        return {"status": "failed", "note": f"download error: {type(e).__name__}"}
    problem = check_pdf(data)
    if problem:
        return {"status": "invalid", "bytes": str(len(data)), "note": problem}
    os.makedirs(os.path.join(folder, book["source"]), exist_ok=True)
    path = os.path.join(folder, book["source"], _name(book["url"]))
    with open(path, "wb") as f:
        f.write(data)
    return {"status": "new", "sha256": hashlib.sha256(data).hexdigest(), "bytes": str(len(data)), "note": ""}


def run(init=False, sources=SOURCES, manifest_path=MANIFEST, folder=DOWNLOAD_DIR, session=None, newest_only=False):
    """Check every source. newest_only=True is the test mode: download the newest book of
    each source whether or not it is known, and leave the manifest untouched."""
    session = session or requests.Session()
    rows = load_manifest(manifest_path)
    by_url = {r["url"]: r for r in rows}
    changed = False
    today = datetime.now(timezone.utc).date().isoformat()
    report = []

    for source in sources:
        entry = {"source": source["id"], "page": source["page"], "status": "", "books": 0, "new": [], "problems": []}
        report.append(entry)
        try:
            html = _get(source["page"], session).text
        except requests.RequestException as e:
            entry["status"] = f"unreachable ({type(e).__name__})"
            continue
        if not find_links(html, source["page"]):
            entry["status"] = "no links in the page (list probably loaded by JavaScript)"
            continue
        books = find_books(source, html)
        entry["books"] = len(books)
        entry["status"] = "ok" if books else "page read, but no yield books recognised (did the page change?)"

        if newest_only:
            for book in latest(books):
                row = {**book, **download(book, session, folder)}
                (entry["new"] if row["status"] == "new" else entry["problems"]).append(
                    {"edition": row["edition"], "season": row["season"], "title": row["title"], "url": row["url"],
                     "status": row["status"], "note": row.get("note", ""), "bytes": row.get("bytes", "")})
            continue

        for book in books:
            old = by_url.get(book["url"])
            if old and old["status"] in SKIP:
                continue
            row = {**book, "first_seen": old["first_seen"] if old else today}
            if init:
                row.update({"status": "known", "note": "listed when the manifest was started"})
            else:
                row.update(download(book, session, folder))
                (entry["new"] if row["status"] == "new" else entry["problems"]).append(
                    {"edition": row["edition"], "season": row["season"], "title": row["title"],
                     "url": row["url"], "status": row["status"], "note": row.get("note", "")})
            if old:
                rows.remove(old)
            rows.append(row)
            by_url[row["url"]] = row
            changed = True

    if changed and not newest_only:  # only touch the file when something changed (no empty pull requests)
        save_manifest(rows, manifest_path)
    return report


def summarise(report):
    lines = ["## Source check", "", "| Source | Status | Yield books listed | New | Problems |", "|---|---|---|---|---|"]
    for e in report:
        lines.append(f"| {e['source']} | {e['status']} | {e['books']} | {len(e['new'])} | {len(e['problems'])} |")
    new = [b for e in report for b in e["new"]]
    if new:
        lines += ["", "### New books downloaded", ""]
        lines += [f"- **{b['title']}** ({b['edition']} {b['season']}): {b['url']}" for b in new]
    problems = [b for e in report for b in e["problems"]]
    if problems:
        lines += ["", "### Problems (tried again next run)", ""]
        lines += [f"- {b['title']}: {b['status']} - {b['note']} ({b['url']})" for b in problems]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--init", action="store_true", help="record every listed book as known, download nothing")
    parser.add_argument("--latest", action="store_true",
                        help="test: download only the newest book of each source; manifest not changed")
    parser.add_argument("--dest", default=DOWNLOAD_DIR, help=f"where to save the PDFs (default: {DOWNLOAD_DIR})")
    args = parser.parse_args(argv)
    folder = os.path.expanduser(args.dest)
    report = run(init=args.init, folder=folder, newest_only=args.latest)
    text = summarise(report)
    print(text)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(folder, "report.md"), "w") as f:
        f.write(text + "\n")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text + "\n")
    new = sum(len(e["new"]) for e in report)
    print(f"\n{new} book(s) downloaded. Saved in {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
