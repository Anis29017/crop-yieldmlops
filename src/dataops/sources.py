"""DataOps - the official sources of district crop yields, and how to recognise their books.

Each source is one web page that lists the yearly books. For every link on that
page the fetcher passes the link's address (url), its text and the text just
before it (context, e.g. a book title above a "Download" button) to the source's
`is_book` rule. Only links that pass are treated as yield books.

To add a source: add one entry to SOURCES. The fetcher, the manifest and the
GitHub Actions workflow need no change.
"""

import re
from urllib.parse import unquote

YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?:\s*[-_/]\s*((?:19|20)?\d{2}))?(?!\d)")


def _name(url):
    """The decoded file name of a link, e.g. 'Final Kharif Estimates 2024-25.pdf'."""
    return unquote(url.split("?")[0].rsplit("/", 1)[-1])


def edition(text):
    """The edition a book covers, as written ('2024-25', '2020-2021', '2025'), or ''."""
    m = YEAR.search(text)
    if not m:
        return ""
    return f"{m.group(1)}-{m.group(2)}" if m.group(2) else m.group(1)


def season(text):
    """'kharif', 'rabi', 'kharif+rabi' or '' (all crops)."""
    t = text.lower()
    k, r = "kharif" in t, "rabi" in t
    return "kharif+rabi" if k and r else "kharif" if k else "rabi" if r else ""


def _pdf(url):
    return _name(url).lower().endswith(".pdf")


# ---------- the rules, one per source ----------


def _punjab_crs(url, text, context):
    # Final Kharif / Rabi Estimates books; every file name contains kharif or rabi and a year.
    name = _name(url)
    return _pdf(url) and season(name) != "" and edition(name) != ""


def _mnfsr(url, text, context):
    # File names are arbitrary (CAP.pdf, Free.pdf, cropsarea.pdf): recognise the book by its title.
    words = f"{text} {context}".lower()
    return _pdf(url) and "crops area" in words and "production" in words


def _sindh(url, text, context):
    # Development Statistics of Sindh / Sindh Statistics, one PDF per year.
    name = _name(url).lower()
    return _pdf(url) and "/files/sbos/" in url.lower() and ("statistic" in name or name.startswith("ds-"))


SOURCES = [
    {
        "id": "punjab_crs",
        "name": "Crop Reporting Service Punjab - Final Kharif / Rabi Estimates",
        "provinces": ["Punjab"],
        "page": "https://agripunjab.punjab.gov.pk/index.php/node/8157",
        "is_book": _punjab_crs,
    },
    {
        "id": "mnfsr",
        "name": "MNFSR - Crops Area and Production (district wise)",
        "provinces": ["Sindh", "Khyber Pakhtunkhwa", "Balochistan", "Islamabad"],
        "page": "https://mnfsr.gov.pk/Publications",
        "is_book": _mnfsr,
        # File names can carry the wrong year (Cap_2024 ... is the 2023-24 book): use the title.
        "edition_from": "title",
    },
    {
        "id": "sindh_bos",
        "name": "Sindh Bureau of Statistics - Development Statistics of Sindh",
        "provinces": ["Sindh"],
        "page": "https://sbos.sindh.gov.pk/development-statistics-of-sindh",
        "is_book": _sindh,
    },
]
# Balochistan: not a separate source. Its Bureau of Statistics page loads its list with JavaScript
# (no links in the page), and the MNFSR book already covers Balochistan's new years.
