"""Tests for running the readers over the fetcher's download folder (src/dataops/read_books.py)."""

from src.dataops import read_books


def test_a_book_that_cannot_be_read_is_reported_and_the_others_continue(tmp_path):
    (tmp_path / "mnfsr").mkdir()
    (tmp_path / "punjab_crs").mkdir()
    (tmp_path / "mnfsr" / "broken.pdf").write_bytes(b"not a pdf")
    (tmp_path / "punjab_crs" / "also broken.pdf").write_bytes(b"not a pdf")
    results = read_books.read_folder(str(tmp_path))
    assert [r["source"] for r in results] == ["mnfsr", "punjab_crs"]
    assert all(r["status"].startswith("could not read") for r in results)
    assert "could not read" in read_books.summarise(results)


def test_report_gets_the_reading_table(tmp_path):
    (tmp_path / "report.md").write_text("# Source check\n")
    read_books.main([str(tmp_path)])
    assert "No downloaded books to read" in (tmp_path / "report.md").read_text()
    assert (tmp_path / "reading.md").exists()
