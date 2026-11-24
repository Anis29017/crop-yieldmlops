"""Tests for the source fetcher (src/dataops/fetch.py), run offline.

The pages are small HTML snippets built from the REAL link addresses and titles
listed on each source page (checked November 2026), so the tests check that the
fetcher recognises the real books and ignores everything else. No network is used.
"""

import pytest
import requests

from src.dataops import fetch
from src.dataops.sources import SOURCES, edition, season

SRC = {s["id"]: s for s in SOURCES}

PUNJAB = """
<h3>2024-25</h3><a href="/system/files/Final%20Kharif%20Estimates%202024-25.pdf">Final Kharif Crops Estimates</a>
Final Rabi Crops Estimates
<h3>2023-24</h3><a href="/system/files/Final%20Kharif%20Estimates%202023-24.pdf">Final Kharif Crops Estimates</a>
<a href="/system/files/Final%20Rabi%20Estimates%202023-24.pdf">Final Rabi Crops Estimates</a>
<a href="/system/files/Final%20Rabi%20%20Estimates%202019-20.pdf">Final Rabi Crops Estimates</a>
<a href="/system/files/Kharif%20Estimates%202018-19_0.pdf">Final Kharif Crops Estimates</a>
<a href="/system/files/Rabi%20Kharif%202008-09.pdf">Final Kharif &amp; Rabi Estimates</a>
<a href="/index.php/system/files?file=KISSAN%20CARD%202ND%20BALLOTING%20WINNERS%20LIST.pdf">Kissan Card 2nd Balloting Result</a>
<a href="/index.php/node/123">Crop Details</a>
"""

MNFSR = """
<div><h4>Crops Area &amp; Production 2024-25</h4>
<a href="/PublicationDetail/ZmI0ZWVh">Details</a> <a href="/SiteImage/Publication/CAP%202024-25.pdf">Download</a></div>
<div><h4>Crops Area &amp; Production (Area Wise) 2023-2024</h4>
<a href="/SiteImage/Publication/Cap_2024%20updated%20Final-1.pdf">Download</a></div>
<div><h4>Crops Area and Production by 2022-23</h4>
<a href="/SiteImage/Publication/cropsarea.pdf">Download</a></div>
<div><h4>Agricultural Statistics of Pakistan 2024-25</h4>
<a href="/SiteImage/Publication/ASP%202024-25%20FINAL.pdf">Download</a></div>
<div><h4>Crops Area and Production 2020-21</h4><a href="https://mnfsr.gov.pk/">Download</a></div>
"""

SINDH = """
<a href="https://sbos.sindh.gov.pk/files/SBOS/Sindh%20Statistics/SINDH%20STATISTIC%202025.pdf">Year 2025</a>
<a href="https://sbos.sindh.gov.pk/files/SBOS/Development%20Statistics/Development%20Statistics%20-%202022.pdf">Year 2022</a>
<a href="https://sbos.sindh.gov.pk/files/SBOS/Development%20Statistics/DS-2020%20(25-01-2022).pdf">Year 2020</a>
<a href="https://sbos.sindh.gov.pk/sindh-statistics-2022-excel-formats">Excel format</a>
<a href="https://sbos.sindh.gov.pk/introduction">Introduction</a>
"""


def _books(source_id, html):
    return {b["url"].rsplit("/", 1)[-1]: b for b in fetch.find_books(SRC[source_id], html)}


# ---------- recognising the books ----------


def test_punjab_keeps_final_estimates_only():
    books = _books("punjab_crs", PUNJAB)
    assert len(books) == 6
    assert not any("KISSAN" in u for u in books)
    b = books["Final%20Kharif%20Estimates%202024-25.pdf"]
    assert (b["edition"], b["season"]) == ("2024-25", "kharif")
    assert books["Rabi%20Kharif%202008-09.pdf"]["season"] == "kharif+rabi"
    assert books["Kharif%20Estimates%202018-19_0.pdf"]["edition"] == "2018-19"


def test_punjab_absolute_links():
    url = next(iter(fetch.find_books(SRC["punjab_crs"], PUNJAB)))["url"]
    assert url.startswith("https://agripunjab.punjab.gov.pk/system/files/")


def test_mnfsr_recognised_by_title_not_file_name():
    books = _books("mnfsr", MNFSR)
    assert set(books) == {"CAP%202024-25.pdf", "Cap_2024%20updated%20Final-1.pdf", "cropsarea.pdf"}
    # the file name says 2024, the title says 2023-2024: the title wins
    assert books["Cap_2024%20updated%20Final-1.pdf"]["edition"] == "2023-2024"
    assert books["cropsarea.pdf"]["edition"] == "2022-23"


def test_mnfsr_ignores_other_publications_and_missing_files():
    books = _books("mnfsr", MNFSR)
    assert "ASP%202024-25%20FINAL.pdf" not in books  # a different publication
    assert all(u.endswith(".pdf") for u in books)  # the 2020-21 entry links to the home page


def test_sindh_yearly_books():
    books = _books("sindh_bos", SINDH)
    assert len(books) == 3
    assert books["SINDH%20STATISTIC%202025.pdf"]["edition"] == "2025"
    assert books["DS-2020%20(25-01-2022).pdf"]["edition"] == "2020"


def test_edition_and_season_helpers():
    assert edition("Final Kharif Estimates 2020-2021.pdf") == "2020-2021"
    assert edition("FINAL RABI 2016-17.pdf") == "2016-17"
    assert edition("no year here") == ""
    assert season("RABI BOOK 2015-16") == "rabi"
    assert season("Statistics 2022") == ""


# ---------- checking downloads ----------


def test_check_pdf():
    good = b"%PDF-1.7\n" + b"x" * 30_000 + b"\n%%EOF\n"
    assert fetch.check_pdf(good) == ""
    assert "not a PDF" in fetch.check_pdf(b"<html>Error 404</html>" + b"x" * 30_000)
    assert "too small" in fetch.check_pdf(b"%PDF-1.7 %%EOF")
    assert "incomplete" in fetch.check_pdf(b"%PDF-1.7\n" + b"x" * 30_000)


# ---------- the run: manifest, new books, unreachable pages ----------


class FakeResponse:
    def __init__(self, text="", content=b""):
        self.text, self.content = text, content

    def raise_for_status(self):
        pass


class FakeSession:
    """Serves the snippets above instead of the internet; records what was requested."""

    def __init__(self, pages, files, down=()):
        self.pages, self.files, self.down, self.requested = pages, files, set(down), []

    def get(self, url, **kwargs):
        self.requested.append(url)
        if url in self.down:
            raise requests.ConnectionError("unreachable")
        if url in self.pages:
            return FakeResponse(text=self.pages[url])
        return FakeResponse(content=self.files.get(url, b"<html>not found</html>"))


PDF = b"%PDF-1.7\n" + b"x" * 30_000 + b"\n%%EOF\n"


@pytest.fixture
def no_wait(monkeypatch):
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)


def _punjab_session(**kw):
    page = SRC["punjab_crs"]["page"]
    books = fetch.find_books(SRC["punjab_crs"], PUNJAB)
    return FakeSession({page: PUNJAB}, {b["url"]: PDF for b in books}, **kw)


def test_init_records_books_without_downloading(tmp_path, no_wait):
    session = _punjab_session()
    manifest = tmp_path / "manifest.csv"
    fetch.run(init=True, sources=[SRC["punjab_crs"]], manifest_path=str(manifest), folder=str(tmp_path), session=session)
    rows = fetch.load_manifest(str(manifest))
    assert len(rows) == 6 and {r["status"] for r in rows} == {"known"}
    assert session.requested == [SRC["punjab_crs"]["page"]]  # only the page, no PDFs


def test_only_new_books_are_downloaded(tmp_path, no_wait):
    manifest = tmp_path / "manifest.csv"
    session = _punjab_session()
    fetch.run(init=True, sources=[SRC["punjab_crs"]], manifest_path=str(manifest), folder=str(tmp_path), session=session)
    # the site publishes the 2024-25 Rabi book
    page = SRC["punjab_crs"]["page"]
    new_url = "https://agripunjab.punjab.gov.pk/system/files/Final%20Rabi%20Estimates%202024-25.pdf"
    session.pages[page] = PUNJAB + f'<a href="{new_url}">Final Rabi Crops Estimates</a>'
    session.files[new_url] = PDF
    session.requested = []
    report = fetch.run(sources=[SRC["punjab_crs"]], manifest_path=str(manifest), folder=str(tmp_path), session=session)
    assert session.requested == [page, new_url]
    assert [b["edition"] for b in report[0]["new"]] == ["2024-25"]
    row = next(r for r in fetch.load_manifest(str(manifest)) if r["url"] == new_url)
    assert row["status"] == "new" and len(row["sha256"]) == 64
    assert (tmp_path / "punjab_crs" / "Final Rabi Estimates 2024-25.pdf").exists()


def test_broken_download_is_reported_and_retried(tmp_path, no_wait):
    manifest = tmp_path / "manifest.csv"
    page = SRC["punjab_crs"]["page"]
    url = "https://agripunjab.punjab.gov.pk/system/files/Final%20Kharif%20Estimates%202024-25.pdf"
    session = FakeSession({page: f'<a href="{url}">Final Kharif Crops Estimates</a>'}, {url: b"<html>error</html>"})
    report = fetch.run(sources=[SRC["punjab_crs"]], manifest_path=str(manifest), folder=str(tmp_path), session=session)
    assert report[0]["new"] == [] and report[0]["problems"][0]["status"] == "invalid"
    session.files[url] = PDF  # fixed on the site
    report = fetch.run(sources=[SRC["punjab_crs"]], manifest_path=str(manifest), folder=str(tmp_path), session=session)
    assert len(report[0]["new"]) == 1
    assert len(fetch.load_manifest(str(manifest))) == 1  # updated, not duplicated


def test_unreachable_and_javascript_pages_are_reported(tmp_path, no_wait):
    session = FakeSession({SRC["sindh_bos"]["page"]: "<div>Loading...</div>"}, {}, down=[SRC["punjab_crs"]["page"]])
    report = fetch.run(sources=[SRC["punjab_crs"], SRC["sindh_bos"]], manifest_path=str(tmp_path / "m.csv"),
                       folder=str(tmp_path), session=session)
    assert report[0]["status"].startswith("unreachable")
    assert "JavaScript" in report[1]["status"]
    assert "unreachable" in fetch.summarise(report)


def test_manifest_untouched_when_nothing_changes(tmp_path, no_wait):
    manifest = tmp_path / "m.csv"
    session = FakeSession({}, {}, down=[SRC["punjab_crs"]["page"]])
    fetch.run(sources=[SRC["punjab_crs"]], manifest_path=str(manifest), folder=str(tmp_path), session=session)
    assert not manifest.exists()  # nothing found -> no file, so no pull request


def test_latest_mode_downloads_only_the_newest_book_and_keeps_the_manifest(tmp_path, no_wait):
    manifest = tmp_path / "m.csv"
    session = _punjab_session()
    fetch.run(init=True, sources=[SRC["punjab_crs"]], manifest_path=str(manifest), folder=str(tmp_path), session=session)
    before = manifest.read_text()
    session.requested = []
    report = fetch.run(sources=[SRC["punjab_crs"]], manifest_path=str(manifest), folder=str(tmp_path),
                       session=session, newest_only=True)
    assert [b["edition"] for b in report[0]["new"]] == ["2024-25"]  # newest, even though it is 'known'
    assert len(session.requested) == 2  # the page + one PDF
    assert manifest.read_text() == before  # test mode never changes the manifest


def test_edition_key_orders_editions():
    def key(e):
        return fetch.edition_key({"edition": e})

    assert key("2024-25") > key("2023-2024") > key("2022-23") > key("2022")
    assert key("2025") > key("2024-25")
    assert key("") == (0, 0)
