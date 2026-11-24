"""Tests for data versioning (src/data_version.py) on the real data/processed.csv."""

import shutil
from datetime import date

import pandas as pd

from src import data_version as dv


def test_same_data_same_version_any_line_endings(tmp_path):
    unix = tmp_path / "processed.csv"
    shutil.copyfile(dv.PROCESSED, unix)
    windows = tmp_path / "processed_windows.csv"
    windows.write_bytes(unix.read_bytes().replace(b"\n", b"\r\n"))
    assert dv.describe(str(unix), [])["data_version"] == dv.describe(str(windows), [])["data_version"]


def test_any_change_gives_a_new_version(tmp_path):
    df = pd.read_csv(dv.PROCESSED)
    changed = tmp_path / "processed.csv"
    df.loc[0, "yield_t_ha"] = df.loc[0, "yield_t_ha"] + 0.001  # one number in one row
    df.to_csv(changed, index=False)
    assert dv.describe(str(changed), [])["data_version"] != dv.describe(dv.PROCESSED, [])["data_version"]


def test_describe_lists_every_input_file():
    info = dv.describe()
    assert "data/raw/Pakistan_Master_Yield_1981-2025.xlsx" in info["inputs"]
    assert "data/interim/climate_features.csv" in info["inputs"]
    assert info["rows"] == len(pd.read_csv(dv.PROCESSED))
    assert info["newest_year"]["wheat"] >= 2025


def test_record_adds_a_line_only_when_the_data_changes(tmp_path):
    versions = tmp_path / "DATA_VERSIONS.csv"
    first = dv.record("first", versions=str(versions), patterns=[], today=date(2026, 11, 24))
    again = dv.record("nothing changed", versions=str(versions), patterns=[], today=date(2026, 11, 25))
    assert first is not None and again is None
    assert len(pd.read_csv(versions)) == 1

    df = pd.read_csv(dv.PROCESSED).iloc[:-1]  # one row fewer: new data
    changed = tmp_path / "processed.csv"
    df.to_csv(changed, index=False)
    second = dv.record("changed", processed=str(changed), versions=str(versions), patterns=[])
    table = pd.read_csv(versions)
    assert second is not None and len(table) == 2 and table.iloc[1]["rows"] == len(df)


def test_the_committed_version_list_matches_the_committed_data():
    table = pd.read_csv(dv.VERSIONS)
    assert table.iloc[-1]["data_version"] == dv.describe()["data_version"]
