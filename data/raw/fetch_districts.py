"""Step 1 - District boundaries and query points, matched to the yield data.

Uses the OCHA / HDX Pakistan administrative boundaries (2022):
  https://data.humdata.org/dataset/cod-ab-pak
  adm2 = districts, adm3 = tehsils.

Every district in the yield data gets exactly one shape, built from the
units listed in data/raw/district_name_table.csv:
  - same name or different spelling  -> one HDX district
  - districts the yield data reports as one (e.g. Karachi, Chitral) -> HDX districts joined
  - old districts / frontier regions / 2024 Punjab districts -> the matching HDX tehsil

Writes (same columns as before, so the weather and soil scripts work unchanged):
  data/interim/districts.gpkg         polygons (for CHIRPS + SoilGrids)
  data/interim/district_points.csv    one point per district (for NASA POWER)

The 'district' and 'province' columns are spelled exactly as in the yield
data, so prepare_data.py can join on (province, district).

Usage:  python -m src.fetch_districts
"""

import os
import re

import geopandas as gpd
import pandas as pd

HDX_DIR = "data/raw/hdx/pak_admin_boundaries"
NAME_TABLE = "data/raw/district_name_table.csv"
OUT_POLYGONS = "data/interim/districts.gpkg"
OUT_POINTS = "data/interim/district_points.csv"

PROVINCE_CODE = {
    "Punjab": "PB",
    "Sindh": "SD",
    "Khyber Pakhtunkhwa": "KP",
    "Balochistan": "BL",
    "Islamabad": "IS",
}


def load_units(hdx_dir=HDX_DIR):
    """All HDX districts (adm2) and tehsils (adm3), indexed by their pcode."""
    adm2 = gpd.read_file(os.path.join(hdx_dir, "pak_admin2.shp"))[
        ["adm2_pcode", "geometry"]
    ]
    adm3 = gpd.read_file(os.path.join(hdx_dir, "pak_admin3.shp"))[
        ["adm3_pcode", "geometry"]
    ]
    units = pd.concat(
        [
            adm2.rename(columns={"adm2_pcode": "pcode"}),
            adm3.rename(columns={"adm3_pcode": "pcode"}),
        ],
        ignore_index=True,
    )
    units = gpd.GeoDataFrame(units, geometry="geometry", crs=adm2.crs).to_crs(epsg=4326)
    return units.set_index("pcode")


def make_id(province, district):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", district).strip("_")
    return f"{PROVINCE_CODE[province]}_{slug}"


def build_districts(table, units):
    """One (multi)polygon per yield district: the union of its listed HDX units."""
    rows = []
    for r in table.itertuples(index=False):
        codes = r.hdx_pcodes.split(";")
        missing = [c for c in codes if c not in units.index]
        if missing:
            raise ValueError(
                f"{r.yield_district}: pcodes not found in HDX files: {missing}"
            )
        geom = units.loc[codes, "geometry"].union_all()
        rows.append(
            {
                "district_id": make_id(r.province, r.yield_district),
                "province": r.province,
                "district": r.yield_district,
                "geometry": geom,
            }
        )
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    if gdf.district_id.duplicated().any():
        raise ValueError(
            f"Duplicate district ids: {gdf[gdf.district_id.duplicated()].district_id.tolist()}"
        )
    return gdf


def query_points(gdf):
    """One point per district that is guaranteed to lie inside the polygon.

    representative_point() is used instead of centroid() because the centroid
    of a curved, concave or multi-part district can fall outside it.
    """
    pts = gdf.geometry.representative_point()
    out = gdf[["district_id", "province", "district"]].copy()
    out["lat"] = pts.y.round(4)
    out["lon"] = pts.x.round(4)
    inside = gdf.geometry.contains(pts)
    if not inside.all():
        raise ValueError(
            f"Points outside their district: {gdf[~inside].district.tolist()}"
        )
    return out


def main():
    table = pd.read_csv(NAME_TABLE)
    units = load_units()
    gdf = build_districts(table, units)
    print(
        f"{len(gdf)} districts; per province: {gdf.province.value_counts().to_dict()}"
    )

    os.makedirs("data/interim", exist_ok=True)
    gdf.to_file(OUT_POLYGONS, driver="GPKG")
    query_points(gdf).to_csv(OUT_POINTS, index=False)
    print(f"Wrote {OUT_POLYGONS} and {OUT_POINTS}")


if __name__ == "__main__":
    main()
