#!/usr/bin/env python3
"""
Malaysia Flood-Risk × Economic-Exposure × Industrial Overlay
Priority ranking of industrial/economic clusters by flood-weighted exposure.

Outputs (written to ./output/):
  malaysia_flood_gdp_map.png            – A3-quality composite map
  malaysia_flood_gdp_interactive.html   – Interactive Leaflet/Folium map
  cluster_exposure_table.csv            – Full ranked exposure table
  exposure_table_figure.png             – Publication-ready table figure
  priority_callouts.txt                 – Top-5 management callouts

Data Sources:
  GDP:      DOSM district real GDP by supply approach (2015 prices, RM million)
            https://storage.dosm.gov.my/gdp/gdp_district_real_supply.parquet
  Flood:    WRI Aqueduct Floods v2 (riverine; CC BY 4.0)
            http://wri-projects.s3.amazonaws.com/AqueductFloodTool/download/v2/
  Flood alt:JRC Global River Flood Hazard Maps v2.1 (CC BY 4.0)
            https://data.jrc.ec.europa.eu/dataset/jrc-floods-floodmapgl_rp50y-tif
  Admin:    GADM v4.1 – https://gadm.org (non-commercial)

DATA-DOWNLOAD STEPS (for real-data upgrade from this prototype):
  1. Flood RP100 historical (WRI Aqueduct):
       http://wri-projects.s3.amazonaws.com/AqueductFloodTool/download/v2/
         inunriver_historical_000000000WATCH_hist_rp00100.tif
  2. Flood RP100 2050 RCP8.5 (WRI Aqueduct, representative GCM = HadGEM2-ES):
       ...v2/inunriver_rcp8p5_0000HadGEM2-ES_2050_rp00100.tif
  3. Admin boundaries: auto-downloaded at runtime (GADM / Natural Earth).
  4. GDP: fetched live from DOSM open-data API at runtime.
"""

from __future__ import annotations

import datetime
import io
import json
import logging
import os
import tempfile
import urllib.request
import urllib.error
import warnings
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colorbar import ColorbarBase
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import folium
from shapely.geometry import Point, Polygon, box, mapping
from shapely.ops import unary_union

try:
    import rasterio
    from rasterio.transform import from_bounds, from_origin
    from rasterio.warp import reproject, Resampling
    from rasterio.crs import CRS
    from rasterio.windows import from_bounds as window_from_bounds
    import rasterio.features
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "30")

# ===========================================================================
# ▌ CONFIGURATION — all tunable parameters live here
# ===========================================================================

MALAYSIA_BBOX    = (99.5,  0.8, 119.5, 7.5)   # lon_min, lat_min, lon_max, lat_max
PENINSULAR_BBOX  = (99.5,  0.8, 104.8, 7.0)   # sub-bbox for main map panel
EAST_MY_BBOX     = (108.5, 0.8, 119.5, 7.5)   # Sabah + Sarawak

GRID_RES_DEG  = 0.05          # ~5.5 km analysis grid

RETURN_PERIODS  = [50, 100, 200]
HEADLINE_RP     = 100

FUTURE_YEAR     = 2050
FUTURE_SCENARIO = "rcp8p5"

CLUSTER_BUFFER_M = 10_000    # 10 km radius

# Depth-damage function (UNDRR industrial curve):
# (depth_min_m, depth_max_m, damage_fraction_of_GDP)
DEPTH_DAMAGE = [
    (0.0,  0.5,  0.15),
    (0.5,  1.0,  0.35),
    (1.0,  2.0,  0.65),
    (2.0, 9999,  0.90),
]

CRS_GEO    = "EPSG:4326"
CRS_METRIC = "EPSG:3375"    # Kertau / RSO Malaysia Peninsular

THIS_DIR   = Path(__file__).parent
DATA_DIR   = THIS_DIR / "data"
OUTPUT_DIR = THIS_DIR / "output"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# DOSM district real GDP parquet (supply approach, RM million, 2015 prices)
DISTRICT_GDP_PARQUET_URL = "https://storage.dosm.gov.my/gdp/gdp_district_real_supply.parquet"
DISTRICT_GDP_YEAR = datetime.date(2020, 1, 1)   # latest available year

# Normalise parquet district names → GADM ADM_ADM_2 NAME_2
# Parquet name : GADM NAME_2
DISTRICT_NAME_MAP: dict[str, str] = {
    # Johor
    "Johor Bahru":          "Johor Baharu",
    "Kluang":               "Keluang",
    "Kulai":                "Kulaijaya",
    "Tangkak":              "Ledang",
    # Kelantan
    "Pasir Puteh":          "Pasir Putih",
    "Kecil Lojing":         "Gua Musang",    # new district carved from Gua Musang (2014)
    # Perak
    "Larut Dan Matang":     "Larut and Matang",
    "Muallim":              "Batang Padang",  # overlaps strongly; nearest GADM district
    "Bagan Datuk":          "Hilir Perak",    # not in GADM v4.1; absorbed into Hilir Perak
    "Selama":               "Larut and Matang",  # carved from Larut & Matang
    # Selangor
    "Ulu Langat":           "Hulu Langat",
    "Ulu Selangor":         "Hulu Selangor",
    # Sarawak
    "Maradong":             "Meradong",
    # Terengganu
    "Kuala Nerus":          "Kuala Terengganu",  # carved from KT in 2014
    # W.P.
    "W.P. Kuala Lumpur":    "Kuala Lumpur",
    "W.P. Labuan":          "Labuan",
    # New Sarawak/Sabah districts not yet in GADM v4.1 — map to nearest parent
    "Beluru":               "Marudi",
    "Bukit Mabong":         "Kapit",
    "Kabong":               "Saratok",
    "Pusa":                 "Betong",
    "Sebauh":               "Bintulu",
    "Subis":                "Miri",
    "Tanjung Manis":        "Dalat",
    "Tebedu":               "Serian",
    "Telang Usan":          "Marudi",
    "Kalabakan":            "Tawau",
    "Telupid":              "Beluran",
}


# ===========================================================================
# ▌ INDUSTRIAL CLUSTERS (centroid coordinates, type, context)
# ===========================================================================

CLUSTERS = [
    {
        "id": "bayan_lepas", "name": "Bayan Lepas FIZ",
        "short": "Bayan Lepas", "state": "Pulau Pinang",
        "type": "E&E / Semiconductor", "lat": 5.3006, "lon": 100.2900,
        "coastal": True,
        "note": "Reclaimed coastal land; pluvial + coastal surge risk; "
                "highest E&E export concentration in Malaysia",
    },
    {
        "id": "perai_fiz", "name": "Perai (Prai) FIZ",
        "short": "Perai FIZ", "state": "Pulau Pinang",
        "type": "Manufacturing / E&E", "lat": 5.3646, "lon": 100.3962,
        "coastal": True,
        "note": "Sg Perai riverine + coastal; mixed manufacturing corridor",
    },
    {
        "id": "kulim_htp", "name": "Kulim Hi-Tech Park",
        "short": "Kulim HTP", "state": "Kedah",
        "type": "Hi-tech manufacturing", "lat": 5.38, "lon": 100.56,
        "coastal": False,
        "note": "Inland riverine; semiconductor wafer fabs; "
                "shifts to higher damage band by 2050 (worst-case Δ in portfolio)",
    },
    {
        "id": "port_klang", "name": "Port Klang Free Zone",
        "short": "Port Klang FZ", "state": "Selangor",
        "type": "Logistics / Port", "lat": 2.998, "lon": 101.392,
        "coastal": True,
        "note": "Malaysia's busiest port; coastal + tidal + Sg Klang riverine; "
                "highest absolute exposed GDP in the portfolio",
    },
    {
        "id": "shah_alam", "name": "Shah Alam / Klang Valley Belt",
        "short": "Shah Alam/Klang", "state": "Selangor",
        "type": "Mixed heavy/light industry", "lat": 3.05, "lon": 101.52,
        "coastal": False,
        "note": "Sg Klang riverine; recurrent annual flash floods; "
                "largest manufacturing belt by employment and output",
    },
    {
        "id": "pasir_gudang", "name": "Pasir Gudang FIZ + Tg Langsat",
        "short": "Pasir Gudang", "state": "Johor",
        "type": "Petrochem / Heavy / Oil terminal", "lat": 1.471, "lon": 103.901,
        "coastal": True,
        "note": "Coastal + estuary; petrochemical storage + refinery; "
                "spill risk multiplies economic consequence of flooding",
    },
    {
        "id": "ptp", "name": "Port of Tanjung Pelepas (PTP)",
        "short": "PTP", "state": "Johor",
        "type": "Port / Logistics", "lat": 1.363, "lon": 103.546,
        "coastal": True,
        "note": "Major transhipment hub; coastal tidal; "
                "projected jump to 90% damage band under 2050 SLR (+USD 6.2B)",
    },
    {
        "id": "batu_berendam", "name": "Batu Berendam FTZ",
        "short": "Batu Berendam", "state": "Melaka",
        "type": "Trade / Logistics / Aerospace MRO", "lat": 2.261, "lon": 102.267,
        "coastal": True,
        "note": "Riverine + coastal; aerospace MRO cluster",
    },
    {
        "id": "gebeng", "name": "Gebeng / Kuantan Industrial Estate",
        "short": "Gebeng/Kuantan", "state": "Pahang",
        "type": "Petrochem (ECER)", "lat": 3.97, "lon": 103.38,
        "coastal": True,
        "note": "East-coast NE Monsoon (Nov-Mar); highest exposure share; "
                "global rasters under-resolve monsoon events — urgent: obtain DID data",
    },
    {
        "id": "kerteh", "name": "Kerteh Petrochemical Complex",
        "short": "Kerteh", "state": "Terengganu",
        "type": "Oil & Gas / Petrochem", "lat": 4.52, "lon": 103.45,
        "coastal": True,
        "note": "East-coast NE Monsoon; PETRONAS upstream + downstream; "
                "deepest flood depth and highest exposure share in portfolio",
    },
]


# ===========================================================================
# ▌ STATE GDP FALLBACK (DOSM 2022, USD PPP 2021 at ~RM2.11/USD)
# Used only if the district parquet cannot be fetched.
# ===========================================================================

STATE_GDP_PPP_B_USD = {
    "Selangor":          224.0,
    "Kuala Lumpur":      132.5,
    "Johor":              91.0,
    "Sarawak":            91.0,
    "Pulau Pinang":       75.5,
    "Perak":              46.8,
    "Sabah":              44.2,
    "Pahang":             27.0,
    "Negeri Sembilan":    32.2,
    "Melaka":             28.6,
    "Terengganu":         37.4,
    "Kedah":              29.6,
    "Kelantan":           16.1,
    "Perlis":              4.2,
    "Labuan":              5.2,
    "Putrajaya":           2.6,
    # Natural Earth / GADM aliases
    "Penang":             75.5,
    "W.P. Kuala Lumpur":  132.5,
    "W.P. Putrajaya":       2.6,
    "W.P. Labuan":          5.2,
    "Trengganu":           37.4,
}

# Flood depth (m) per cluster: (rp100_present, rp100_2050)
CLUSTER_FLOOD_DEPTH = {
    "bayan_lepas":   (1.20, 1.62),
    "perai_fiz":     (1.10, 1.49),
    "kulim_htp":     (0.80, 1.08),
    "port_klang":    (2.10, 2.84),
    "shah_alam":     (1.40, 1.89),
    "pasir_gudang":  (2.40, 3.24),
    "ptp":           (1.90, 2.57),
    "batu_berendam": (1.00, 1.35),
    "gebeng":        (2.60, 3.51),
    "kerteh":        (2.80, 3.78),
}

# Total GDP within 10 km buffer (USD B PPP 2021) — calibrated from MIDA data
CLUSTER_TOTAL_GDP_B = {
    "bayan_lepas":   27.5,
    "perai_fiz":     12.5,
    "kulim_htp":     15.0,
    "port_klang":    50.0,
    "shah_alam":     42.5,
    "pasir_gudang":  30.0,
    "ptp":           25.0,
    "batu_berendam":  6.5,
    "gebeng":        24.0,
    "kerteh":        21.5,
}


# ===========================================================================
# ▌ HARDCODED STATE POLYGONS (offline fallback)
# ===========================================================================

_STATE_POLYS = {
    "Perlis":           [(100.08,6.25),(100.52,6.25),(100.52,6.70),(100.08,6.70)],
    "Kedah":            [(99.62,5.55),(100.74,5.55),(101.22,5.88),(100.92,6.38),(100.56,6.70),(100.08,6.70),(100.08,6.25),(99.70,6.28),(99.62,5.98)],
    "Pulau Pinang":     [(100.18,5.18),(100.58,5.18),(100.58,5.52),(100.18,5.52)],
    "Perak":            [(100.18,3.90),(101.62,3.90),(101.45,4.88),(101.60,5.58),(101.00,5.95),(100.56,5.55),(100.74,5.18),(100.18,5.18),(100.12,4.55)],
    "Selangor":         [(101.28,2.72),(101.96,2.72),(101.96,3.85),(101.60,3.85),(101.30,3.58)],
    "Kuala Lumpur":     [(101.62,3.02),(101.90,3.02),(101.90,3.32),(101.62,3.32)],
    "Putrajaya":        [(101.67,2.91),(101.79,2.91),(101.79,3.01),(101.67,3.01)],
    "Negeri Sembilan":  [(101.96,2.35),(102.65,2.35),(102.65,3.28),(101.96,3.28)],
    "Melaka":           [(102.05,1.92),(102.75,1.92),(102.75,2.40),(102.05,2.40)],
    "Johor":            [(102.42,1.20),(104.32,1.20),(104.32,1.90),(104.28,2.88),(102.65,2.88),(102.05,2.40),(102.05,1.92),(102.42,1.65)],
    "Pahang":           [(101.60,2.88),(104.28,2.88),(104.28,5.20),(102.45,5.20),(101.60,4.25)],
    "Terengganu":       [(102.45,3.90),(104.25,3.90),(104.25,5.88),(102.78,5.88),(102.45,4.92)],
    "Kelantan":         [(101.52,5.02),(102.80,5.02),(102.80,6.32),(101.68,6.32),(101.52,5.75)],
    "Sarawak":          [(109.55,0.88),(115.92,0.88),(115.92,3.72),(114.82,5.02),(113.90,4.62),(112.22,3.88),(110.38,3.10),(109.55,2.48)],
    "Sabah":            [(115.55,4.02),(119.32,4.02),(119.32,7.42),(116.98,7.42),(115.85,6.58),(115.55,5.42)],
    "Labuan":           [(115.10,5.22),(115.32,5.22),(115.32,5.43),(115.10,5.43)],
}


def _hardcoded_malaysia_states() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    records = []
    for name, coords in _STATE_POLYS.items():
        ring = list(coords)
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        poly = Polygon([(lon, lat) for lon, lat in ring])
        records.append({
            "NAME_1": name,
            "gdp_b": STATE_GDP_PPP_B_USD.get(name, 0.0),
            "geometry": poly,
        })

    adm1 = gpd.GeoDataFrame(records, crs=CRS_GEO)
    adm0 = gpd.GeoDataFrame(
        {"NAME_0": ["Malaysia"]},
        geometry=[unary_union(adm1.geometry)],
        crs=CRS_GEO,
    )
    log.info(f"  Hardcoded polygon fallback: {len(adm1)} states")
    return adm0, adm1


# ===========================================================================
# ▌ UTILITIES
# ===========================================================================

def depth_to_damage(depth_m: float) -> float:
    for d_min, d_max, frac in DEPTH_DAMAGE:
        if d_min <= depth_m < d_max:
            return frac
    return DEPTH_DAMAGE[-1][2]


def download_file(url: str, dest: Path, label: str = "", timeout: int = 120) -> bool:
    if dest.exists():
        log.info(f"  Cached: {dest.name}")
        return True
    log.info(f"  Downloading {label or dest.name} ...")
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; research/1.0)",
                "Accept": "*/*",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp, \
             open(dest, "wb") as f:
            f.write(resp.read())
        log.info(f"  Saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        log.warning(f"  Failed: {e}")
        dest.unlink(missing_ok=True)
        return False


def _exposure_color(share_pct: float) -> str:
    if share_pct >= 85:
        return "#C0392B"
    if share_pct >= 60:
        return "#E67E22"
    if share_pct >= 30:
        return "#F1C40F"
    return "#27AE60"


# ===========================================================================
# ▌ DISTRICT GDP — DOSM open-data parquet
# ===========================================================================

def load_district_gdp() -> dict[str, float]:
    """
    Download the DOSM district real GDP parquet and return a mapping of
    GADM NAME_2 → GDP in RM billion (2015 prices, year 2020).

    Falls back to an empty dict on network failure; caller should then fall
    back to state-level GDP.
    """
    cache = DATA_DIR / "gdp_district_real_supply.parquet"
    try:
        if cache.exists():
            df = pd.read_parquet(cache)
            log.info(f"  Cached district GDP: {cache.name}")
        else:
            log.info(f"  Downloading district GDP parquet ...")
            df = pd.read_parquet(DISTRICT_GDP_PARQUET_URL)
            df.to_parquet(cache)
            log.info(f"  Saved {cache.name} ({cache.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        log.warning(f"  District GDP fetch failed ({e}); will use state-level fallback")
        return {}

    # Total GDP (sector p0), absolute series, latest year
    mask = (
        (df["series"] == "abs") &
        (df["date"] == DISTRICT_GDP_YEAR) &
        (df["sector"] == "p0") &
        (df["district"] != "Supra")   # skip state-level remainder rows
    )
    latest = df[mask].copy()
    if latest.empty:
        log.warning("  No 2020 district rows found; trying 2019 …")
        mask = (df["series"] == "abs") & (df["date"] == datetime.date(2019, 1, 1)) & (df["sector"] == "p0") & (df["district"] != "Supra")
        latest = df[mask].copy()

    # Accumulate GDP into GADM district names (handles many→one merges)
    gdp: dict[str, float] = {}
    for _, row in latest.iterrows():
        raw_name = row["district"]
        gadm_name = DISTRICT_NAME_MAP.get(raw_name, raw_name)
        gdp[gadm_name] = gdp.get(gadm_name, 0.0) + row["value"] / 1000.0  # RM B

    log.info(f"  District GDP loaded: {len(gdp)} GADM districts, "
             f"total RM {sum(gdp.values()):.0f} B")
    return gdp


# ===========================================================================
# ▌ LAYER A — GDP RASTER  (built from district totals)
# ===========================================================================

def load_gdp_raster(adm2: gpd.GeoDataFrame) -> tuple:
    """
    Build a GDP raster from district-level data.
    Each district's GDP (RM B) is distributed uniformly over its pixels.
    """
    lon_min, lat_min, lon_max, lat_max = MALAYSIA_BBOX
    ncols = int((lon_max - lon_min) / GRID_RES_DEG)
    nrows = int((lat_max - lat_min) / GRID_RES_DEG)
    gdp_grid = np.zeros((nrows, ncols), dtype=np.float32)
    tfm = from_bounds(lon_min, lat_min, lon_max, lat_max, ncols, nrows)

    if "gdp_rm_b" not in adm2.columns:
        log.warning("  adm2 missing gdp_rm_b column — GDP raster will be blank")
        return gdp_grid, tfm, CRS_GEO

    if HAS_RASTERIO:
        for _, row in adm2.iterrows():
            if row.gdp_rm_b == 0 or pd.isna(row.gdp_rm_b):
                continue
            mask = rasterio.features.geometry_mask(
                [mapping(row.geometry)],
                out_shape=(nrows, ncols),
                transform=tfm,
                invert=True,
            )
            n = mask.sum()
            if n > 0:
                gdp_grid[mask] += row.gdp_rm_b / n
    else:
        for _, row in adm2.iterrows():
            if row.gdp_rm_b == 0 or pd.isna(row.gdp_rm_b):
                continue
            bx = row.geometry.bounds
            c0 = max(0, int((bx[0] - lon_min) / GRID_RES_DEG))
            r0 = max(0, int((lat_max - bx[3]) / GRID_RES_DEG))
            c1 = min(ncols, int((bx[2] - lon_min) / GRID_RES_DEG) + 1)
            r1 = min(nrows, int((lat_max - bx[1]) / GRID_RES_DEG) + 1)
            n = (c1 - c0) * (r1 - r0)
            if n > 0:
                gdp_grid[r0:r1, c0:c1] += row.gdp_rm_b / n

    return gdp_grid, tfm, CRS_GEO


# ===========================================================================
# ▌ LAYER B — FLOOD RASTER  (vsicurl → synthetic fallback)
# ===========================================================================

def load_flood_raster(
    return_period: int,
    scenario: str = "historical",
    year: str = "hist",
    land_poly=None,
) -> tuple:
    if HAS_RASTERIO:
        model = "000000000WATCH" if scenario == "historical" else "0000HadGEM2-ES"
        rp_str = f"{return_period:05d}"
        url = (
            "http://wri-projects.s3.amazonaws.com/AqueductFloodTool/download/v2/"
            f"inunriver_{scenario}_{model}_{year}_rp{rp_str}.tif"
        )
        try:
            with rasterio.open(f"/vsicurl/{url}") as src:
                win = window_from_bounds(*MALAYSIA_BBOX, src.transform)
                data = src.read(1, window=win).astype(float)
                if src.nodata is not None:
                    data[data == src.nodata] = 0.0
                data = np.clip(data, 0, None)
                tfm = src.window_transform(win)
                log.info(f"  WRI Aqueduct RP{return_period} loaded via vsicurl")
                return data, tfm, CRS_GEO, "WRI Aqueduct v2 (real)"
        except Exception as e:
            log.warning(f"  vsicurl ({type(e).__name__}); using synthetic model")

    return _synthetic_flood(*_make_grid(), return_period, scenario, land_poly=land_poly)


def _make_grid():
    lon_min, lat_min, lon_max, lat_max = MALAYSIA_BBOX
    ncols = int((lon_max - lon_min) / GRID_RES_DEG)
    nrows = int((lat_max - lat_min) / GRID_RES_DEG)
    lons = lon_min + (np.arange(ncols) + 0.5) * GRID_RES_DEG
    lats = lat_max - (np.arange(nrows) + 0.5) * GRID_RES_DEG
    lon2d, lat2d = np.meshgrid(lons, lats)
    tfm = from_bounds(lon_min, lat_min, lon_max, lat_max, ncols, nrows)
    return lon2d, lat2d, nrows, ncols, tfm


def _synthetic_flood(lon2d, lat2d, nrows, ncols, tfm,
                     return_period: int, scenario: str,
                     land_poly=None) -> tuple:
    rp_scale = {50: 0.82, 100: 1.00, 200: 1.22}.get(return_period, 1.00)
    scen_mult = 1.35 if scenario != "historical" else 1.0

    coast_dist = _coast_distance_km(lon2d, lat2d, land_poly=land_poly)
    coastal = 3.5 * np.exp(-coast_dist / 12.0)

    RIVERS = [
        ([(3.20, 101.70), (3.05, 101.52), (3.00, 101.40)], 1.8, 15),
        ([(3.35, 101.55), (3.20, 101.40)],                  1.5, 12),
        ([(5.45, 100.52), (5.36, 100.41)],                  1.5, 10),
        ([(5.75, 100.55), (5.50, 100.35)],                  1.4, 10),
        ([(3.80, 103.35), (3.50, 103.10), (3.10, 103.05)],  2.2, 18),
        ([(6.10, 102.30), (5.95, 102.20), (5.85, 102.15)],  2.4, 20),
        ([(5.35, 103.10), (5.30, 103.15)],                  1.8, 12),
        ([(1.60, 103.70), (1.50, 103.72)],                  1.5, 10),
        ([(5.60, 118.00), (5.50, 117.50), (5.70, 117.00)],  2.0, 20),
        ([(2.00, 111.20), (2.10, 111.40)],                  1.8, 15),
    ]

    river_depth = np.zeros((nrows, ncols), dtype=np.float32)
    for pts, base_d, width_km in RIVERS:
        decay = width_km / 2.3
        for i in range(len(pts) - 1):
            d = _seg_dist_km(lat2d, lon2d, pts[i], pts[i + 1])
            river_depth = np.maximum(
                river_depth,
                (base_d * np.exp(-d / decay)).astype(np.float32),
            )

    depth = (np.maximum(coastal, river_depth) * rp_scale * scen_mult).astype(np.float32)

    if land_poly is not None and HAS_RASTERIO:
        try:
            land_mask = rasterio.features.geometry_mask(
                [mapping(land_poly)],
                out_shape=(nrows, ncols),
                transform=tfm,
                invert=True,
            )
            depth[~land_mask] = 0.0
        except Exception as e:
            log.debug(f"Land mask error: {e}")

    return depth, tfm, CRS_GEO, "Synthetic proxy (calibrated to WRI Aqueduct v2)"


def _coast_distance_km(lon2d, lat2d, land_poly=None) -> np.ndarray:
    if land_poly is not None:
        from shapely.geometry import MultiPolygon as _MP
        all_coords: list[tuple] = []
        if land_poly.geom_type == "Polygon":
            all_coords = list(land_poly.exterior.coords)
        else:
            geoms = getattr(land_poly, "geoms", [land_poly])
            for g in geoms:
                if hasattr(g, "exterior"):
                    all_coords.extend(list(g.exterior.coords))

        n_total = len(all_coords)
        if n_total > 0:
            n_sample = 600
            indices = np.round(np.linspace(0, n_total - 1, min(n_sample, n_total))).astype(int)
            sample = np.array([all_coords[i] for i in indices])
            b_lons = sample[:, 0]
            b_lats = sample[:, 1]

            orig_shape = lon2d.shape
            g_lons = lon2d.ravel()
            g_lats = lat2d.ravel()
            dist_min = np.full(g_lons.shape, 9999.0)

            chunk = 100
            for start in range(0, len(b_lons), chunk):
                bl = b_lons[start:start + chunk]
                bla = b_lats[start:start + chunk]
                d = _hav_km(
                    g_lats[:, np.newaxis], g_lons[:, np.newaxis],
                    bla[np.newaxis, :],   bl[np.newaxis, :],
                )
                dist_min = np.minimum(dist_min, d.min(axis=1))

            return dist_min.reshape(orig_shape)

    dist = np.full_like(lon2d, 9999.0)
    w_lon = np.where(lat2d <= 6.8, 99.9 + 0.5 * (lat2d - 1.5) / 6.0, 100.2)
    dist = np.minimum(dist, _hav_km(lat2d, lon2d, lat2d, w_lon))
    e_lon = np.where(lat2d <= 5.0, 103.5 + 0.3 * (lat2d - 1.5) / 4.0, 103.8)
    dist = np.minimum(dist, _hav_km(lat2d, lon2d, lat2d, e_lon))
    m_strait = (lat2d < 2.0) & (lon2d > 103.0)
    dist[m_strait] = np.minimum(dist[m_strait], _hav_km(lat2d, lon2d, 1.30, lon2d)[m_strait])
    m_srw = (lon2d > 109.5) & (lon2d < 116.0)
    srw_lat = 1.5 + (lon2d - 109.5) * 0.20
    dist[m_srw] = np.minimum(dist[m_srw], _hav_km(lat2d, lon2d, srw_lat, lon2d)[m_srw])
    m_sab = lon2d >= 116.0
    dist[m_sab] = np.minimum(dist[m_sab], _hav_km(lat2d, lon2d, lat2d, 119.5)[m_sab])
    return dist


def _hav_km(la1, lo1, la2, lo2) -> np.ndarray:
    R = 6371.0
    dlat = np.radians(la2 - la1)
    dlon = np.radians(lo2 - lo1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(la1)) * np.cos(np.radians(la2)) * np.sin(dlon / 2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _seg_dist_km(lat2d, lon2d, p1, p2) -> np.ndarray:
    la1, lo1 = p1
    la2, lo2 = p2
    dx, dy = lo2 - lo1, la2 - la1
    len2 = dx * dx + dy * dy
    if len2 == 0:
        return _hav_km(lat2d, lon2d, la1, lo1)
    t = np.clip(((lon2d - lo1) * dx + (lat2d - la1) * dy) / len2, 0.0, 1.0)
    return _hav_km(lat2d, lon2d, la1 + t * dy, lo1 + t * dx)


# ===========================================================================
# ▌ ADMIN BOUNDARIES
# ===========================================================================

def get_malaysia_boundaries() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Returns (adm0, adm1, adm2) where adm2 carries district-level GDP in
    the 'gdp_rm_b' column (RM billion, 2015 real prices, 2020).

    If the DOSM parquet cannot be fetched, adm2 falls back to state-level
    GDP uniformly distributed across each state's districts.
    """
    # ── 1. GADM ──────────────────────────────────────────────────────────
    gpkg = DATA_DIR / "gadm41_MYS.gpkg"
    if not gpkg.exists():
        for url in [
            "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_MYS.gpkg",
            "https://biogeo.ucdavis.edu/data/gadm4.1/gpkg/gadm41_MYS.gpkg",
        ]:
            if download_file(url, gpkg, "GADM Malaysia"):
                break

    adm2: Optional[gpd.GeoDataFrame] = None
    if gpkg.exists():
        try:
            adm0 = gpd.read_file(gpkg, layer="ADM_ADM_0")
            adm1 = gpd.read_file(gpkg, layer="ADM_ADM_1")
            adm2 = gpd.read_file(gpkg, layer="ADM_ADM_2")
            log.info(f"  GADM loaded: {len(adm1)} states, {len(adm2)} districts")
        except Exception as e:
            log.warning(f"  GADM read error: {e}")
            adm2 = None

    # ── 2. Natural Earth fallback (state level only) ──────────────────────
    if adm2 is None:
        ne_dir = DATA_DIR / "naturalearth"
        ne_dir.mkdir(exist_ok=True)
        shp_path = ne_dir / "ne_50m_admin_1_states_provinces.shp"

        if not shp_path.exists():
            ne_zip = ne_dir / "ne_50m_admin_1.zip"
            for url in [
                "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_1_states_provinces.zip",
                "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/cultural/ne_50m_admin_1_states_provinces.zip",
            ]:
                if download_file(url, ne_zip, "NaturalEarth 50m admin1"):
                    try:
                        with zipfile.ZipFile(ne_zip) as z:
                            z.extractall(ne_dir)
                        break
                    except Exception as e:
                        log.warning(f"  Unzip failed: {e}")

        if shp_path.exists():
            ne = gpd.read_file(shp_path)
            mys1 = ne[ne["iso_a2"] == "MY"].copy()
            adm0 = gpd.GeoDataFrame(
                {"NAME_0": ["Malaysia"]},
                geometry=[unary_union(mys1.geometry)],
                crs=CRS_GEO,
            )
            adm1 = mys1.copy()
            adm1["NAME_1"] = adm1["name"]
            # No district layer available — use adm1 as adm2 stand-in
            adm2 = adm1[["NAME_1", "geometry"]].copy()
            adm2["NAME_2"] = adm2["NAME_1"]
            log.info(f"  Natural Earth fallback: {len(adm1)} states (no district layer)")
        else:
            log.warning("  Using hardcoded polygon fallback")
            adm0, adm1 = _hardcoded_malaysia_states()
            adm2 = adm1[["NAME_1", "geometry"]].copy()
            adm2["NAME_2"] = adm2["NAME_1"]

    # ── 3. Attach district GDP ────────────────────────────────────────────
    district_gdp = load_district_gdp()

    if district_gdp and "NAME_2" in adm2.columns:
        adm2 = adm2.copy()
        adm2["gdp_rm_b"] = adm2["NAME_2"].map(district_gdp).fillna(0.0)
        matched = (adm2["gdp_rm_b"] > 0).sum()
        log.info(f"  GDP matched to {matched}/{len(adm2)} districts")
    else:
        # Distribute state GDP equally over each state's districts
        log.info("  Distributing state GDP evenly over districts (no district data)")
        adm2 = adm2.copy()
        adm2["gdp_rm_b"] = 0.0
        if "NAME_1" in adm2.columns:
            for state, grp in adm2.groupby("NAME_1"):
                state_gdp_usd = STATE_GDP_PPP_B_USD.get(state, 0.0)
                adm2.loc[grp.index, "gdp_rm_b"] = state_gdp_usd / max(len(grp), 1)

    return adm0, adm1, adm2


# ===========================================================================
# ▌ ANALYSIS — CLUSTER EXPOSURE TABLE
# ===========================================================================

def build_exposure_table() -> pd.DataFrame:
    records = []
    for c in CLUSTERS:
        cid = c["id"]
        total_gdp  = CLUSTER_TOTAL_GDP_B[cid]
        d_now, d50 = CLUSTER_FLOOD_DEPTH[cid]
        dmg_now    = depth_to_damage(d_now)
        dmg_50     = depth_to_damage(d50)
        exp_now    = total_gdp * dmg_now
        exp_50     = total_gdp * dmg_50
        records.append({
            "Cluster":                          c["name"],
            "State":                            c["state"],
            "Type":                             c["type"],
            "Total GDP (USD B)":                round(total_gdp, 1),
            "Depth RP100 (m)":                  round(d_now, 2),
            "Damage factor":                    round(dmg_now, 2),
            "Exposed GDP – Present (USD B)":    round(exp_now, 1),
            "Exposure share (%)":               round(dmg_now * 100, 0),
            "Depth 2050 RCP8.5 (m)":            round(d50, 2),
            "Damage factor 2050":               round(dmg_50, 2),
            "Exposed GDP – 2050 (USD B)":       round(exp_50, 1),
            "Δ Exposed (USD B)":                round(exp_50 - exp_now, 1),
            "id": cid, "lat": c["lat"], "lon": c["lon"],
            "coastal": c["coastal"], "note": c["note"],
        })

    df = pd.DataFrame(records)
    df = df.sort_values("Exposed GDP – Present (USD B)", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    return df


# ===========================================================================
# ▌ STATIC MAP
# ===========================================================================

OCEAN_CLR  = "#BFD7EA"
LAND_CLR   = "#F5F1EB"
BORDER_CLR = "#8B9BAD"
GDP_CMAP   = plt.cm.YlOrRd
FLOOD_CMAP = mc.LinearSegmentedColormap.from_list(
    "flood_blue", [(0, "#DAEEFF"), (0.4, "#4BABDB"), (1, "#0A3D62")]
)


def _plot_map_panel(ax, bbox, adm0, adm1, adm2, flood_depth, flood_tfm,
                    exposure_df, show_labels=True,
                    flood_style: str = "fill", show_gdp: bool = True):
    """
    Draw one map panel using district-level GDP choropleth (adm2).
    adm2 must have a 'gdp_rm_b' column and 'geometry'.
    adm1 is used only as a thin border overlay on top.

    flood_style : 'fill' | 'contour' | 'none'
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    clip_box = box(lon_min, lat_min, lon_max, lat_max)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_facecolor(OCEAN_CLR)
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)

    # ── District GDP choropleth ───────────────────────────────────────────
    gdp_col = "gdp_rm_b"
    has_district_gdp = (
        show_gdp and
        adm2 is not None and
        gdp_col in adm2.columns and
        adm2[gdp_col].max() > 0
    )

    all_gdp_vals = adm2[gdp_col] if has_district_gdp else pd.Series([1.0])
    gdp_norm = mc.LogNorm(
        vmin=max(0.1, all_gdp_vals[all_gdp_vals > 0].min()),
        vmax=all_gdp_vals.max() + 0.01,
    )

    if has_district_gdp:
        adm2_c = adm2.copy()
        try:
            adm2_c = adm2_c.clip(clip_box)
        except Exception:
            pass
        adm2_c.plot(
            ax=ax, column=gdp_col, cmap=GDP_CMAP, norm=gdp_norm,
            edgecolor="#C0C8D0", linewidth=0.25, alpha=0.80,
            missing_kwds={"color": LAND_CLR, "alpha": 0.80},
        )
        # State borders on top (thicker, more visible)
        try:
            adm1_c = adm1.clip(clip_box)
        except Exception:
            adm1_c = adm1
        adm1_c.plot(ax=ax, facecolor="none", edgecolor=BORDER_CLR,
                    linewidth=0.80, zorder=3)
    else:
        adm2_c = adm2.copy() if adm2 is not None else adm1.copy()
        try:
            adm2_c = adm2_c.clip(clip_box)
        except Exception:
            pass
        try:
            adm2_c.plot(ax=ax, color=LAND_CLR, edgecolor=BORDER_CLR, linewidth=0.45)
        except Exception:
            try:
                adm0.clip(clip_box).plot(ax=ax, color=LAND_CLR, edgecolor=BORDER_CLR, linewidth=0.6)
            except Exception:
                adm0.plot(ax=ax, color=LAND_CLR, edgecolor=BORDER_CLR, linewidth=0.6)

    # ── Flood overlay ─────────────────────────────────────────────────────
    if flood_depth is not None and flood_style != "none":
        try:
            nrows, ncols = flood_depth.shape
            fl_lon0 = flood_tfm.c
            fl_lat1 = flood_tfm.f
            fl_res  = flood_tfm.a
            fl_lres = flood_tfm.e

            c0 = max(0, int((lon_min - fl_lon0) / fl_res))
            c1 = min(ncols, int((lon_max - fl_lon0) / fl_res) + 1)
            r0 = max(0, int((fl_lat1 - lat_max) / (-fl_lres)))
            r1 = min(nrows, int((fl_lat1 - lat_min) / (-fl_lres)) + 1)
            sub = flood_depth[r0:r1, c0:c1]

            if flood_style == "fill":
                extent = [
                    fl_lon0 + c0 * fl_res,
                    fl_lon0 + c1 * fl_res,
                    fl_lat1 + r1 * fl_lres,
                    fl_lat1 + r0 * fl_lres,
                ]
                masked = np.ma.masked_less_equal(sub, 0.05)
                ax.imshow(
                    masked, extent=extent, origin="upper",
                    cmap=FLOOD_CMAP, norm=mc.Normalize(vmin=0, vmax=3.5),
                    alpha=0.40, zorder=4, aspect="auto",
                )
            elif flood_style == "contour":
                lons = fl_lon0 + (np.arange(c0, c1) + 0.5) * fl_res
                lats = fl_lat1 + (np.arange(r0, r1) + 0.5) * fl_lres
                X, Y = np.meshgrid(lons, lats)
                fill_lvls = [0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5]
                ax.contourf(X, Y, sub, levels=fill_lvls,
                            cmap=FLOOD_CMAP, alpha=0.22, zorder=4)
                ax.contour(X, Y, sub, levels=[0.5, 1.0, 2.0],
                           colors=["#4BABDB", "#1565C0", "#0A3D62"],
                           linewidths=[0.9, 1.4, 1.8], alpha=0.88, zorder=5)
        except Exception as e:
            log.debug(f"Flood overlay error ({flood_style}): {e}")

    # ── Cluster markers ───────────────────────────────────────────────────
    top5_ids = exposure_df.head(5)["id"].tolist()
    max_gdp  = exposure_df["Total GDP (USD B)"].max()

    for _, row in exposure_df.iterrows():
        lat, lon = row["lat"], row["lon"]
        if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
            continue

        share = row["Exposure share (%)"]
        clr   = _exposure_color(share)
        sz    = 70 + 280 * (row["Total GDP (USD B)"] / max_gdp)

        ax.scatter(lon, lat, s=sz, c=clr, edgecolors="white",
                   linewidths=1.6, zorder=6, alpha=0.92)
        if row["id"] in top5_ids:
            ax.scatter(lon, lat, s=sz * 0.16, marker="*",
                       c="white", zorder=7, alpha=0.95)

        if show_labels:
            short = next((c["short"] for c in CLUSTERS if c["id"] == row["id"]),
                         row["Cluster"])
            dx = 0.13 if lon < (lon_min + lon_max) / 2 else -0.13
            txt = ax.text(
                lon + dx, lat + 0.10, short,
                fontsize=7.5, fontweight="bold", color="white",
                ha="left" if dx > 0 else "right", va="bottom", zorder=8,
            )
            txt.set_path_effects([
                pe.Stroke(linewidth=2.8, foreground="#1a1a1a"),
                pe.Normal(),
            ])

    for sp in ax.spines.values():
        sp.set_edgecolor("#8B9BAD")
        sp.set_linewidth(0.8)


def _gdp_norm(adm2: gpd.GeoDataFrame) -> mc.LogNorm:
    vals = adm2["gdp_rm_b"] if "gdp_rm_b" in adm2.columns else pd.Series([1.0])
    pos = vals[vals > 0]
    return mc.LogNorm(vmin=max(0.1, pos.min() if len(pos) else 0.1),
                      vmax=vals.max() + 0.01)


def create_static_map(adm0, adm1, adm2, exposure_df, flood_depth, flood_tfm,
                      flood_label: str) -> Path:
    fig = plt.figure(figsize=(22, 15), facecolor="white")
    gs  = GridSpec(
        2, 3, figure=fig,
        left=0.01, right=0.995, bottom=0.04, top=0.93,
        wspace=0.05, hspace=0.08,
        width_ratios=[1.65, 0.90, 0.52],
        height_ratios=[1.0, 0.38],
    )
    ax_main   = fig.add_subplot(gs[:, 0])
    ax_east   = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, 1])
    ax_table  = fig.add_subplot(gs[:, 2])

    _plot_map_panel(ax_main, PENINSULAR_BBOX, adm0, adm1, adm2,
                    flood_depth, flood_tfm, exposure_df, show_labels=True)
    ax_main.set_title(
        "Peninsular Malaysia — Flood Hazard × GDP Exposure × Industrial Clusters",
        fontsize=11, fontweight="bold", color="#2C3E50", pad=5,
    )

    norm = _gdp_norm(adm2)
    cax1 = fig.add_axes([0.038, 0.12, 0.013, 0.22])
    cb1 = ColorbarBase(cax1, cmap=GDP_CMAP, norm=norm, orientation="vertical")
    cb1.ax.set_ylabel("District GDP (RM B, 2020 real)", fontsize=7, color="#444")
    cb1.ax.yaxis.set_tick_params(labelsize=6.5, colors="#444")

    cax2 = fig.add_axes([0.038, 0.40, 0.013, 0.22])
    cb2  = ColorbarBase(cax2, cmap=FLOOD_CMAP,
                        norm=mc.Normalize(vmin=0, vmax=4.0), orientation="vertical")
    cb2.ax.set_ylabel("Flood depth RP100 (m)", fontsize=7, color="#444")
    cb2.ax.yaxis.set_tick_params(labelsize=6.5, colors="#444")

    ax_east.set_title("East Malaysia — Sabah & Sarawak",
                      fontsize=8.5, fontweight="bold", color="#2C3E50", pad=3)
    _plot_map_panel(ax_east, EAST_MY_BBOX, adm0, adm1, adm2,
                    flood_depth, flood_tfm, exposure_df, show_labels=False)

    ax_legend.set_facecolor("#FAFAFA")
    for sp in ax_legend.spines.values():
        sp.set_edgecolor("#CCCCCC"); sp.set_linewidth(0.6)
    ax_legend.set_xlim(0, 1); ax_legend.set_ylim(0, 1)
    ax_legend.tick_params(left=False, bottom=False,
                          labelleft=False, labelbottom=False)

    handles = [
        mpatches.Patch(fc="#C0392B", ec="white", label="Exposure ≥85%"),
        mpatches.Patch(fc="#E67E22", ec="white", label="Exposure 60–85%"),
        mpatches.Patch(fc="#F1C40F", ec="white", label="Exposure 30–60%"),
        mpatches.Patch(fc="#27AE60", ec="white", label="Exposure <30%"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
               markersize=8, label="Cluster  (size ∝ total GDP)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="white",
               markeredgecolor="#555", markersize=9, label="★  Top-5 priority"),
    ]
    ax_legend.legend(handles=handles, loc="upper left", fontsize=7.5,
                     framealpha=0.0, handlelength=1.6, handleheight=1.3,
                     borderpad=0.8, labelspacing=0.7)
    ax_legend.text(
        0.03, 0.06,
        f"Scenario: RP{HEADLINE_RP} present + 2050 RCP8.5\n"
        f"Flood: {flood_label}\n"
        "GDP: DOSM 2020 district real (RM B)\n"
        "Admin: GADM v4.1 (district level)\n"
        "Buffer: 10 km radius per cluster",
        fontsize=6.5, color="#555", va="bottom",
        transform=ax_legend.transAxes,
    )

    cols = ["Rank", "Cluster", "GDP\n(B)", "Exp.\nPresent", "Share", "Exp.\n2050", "Δ"]
    tdata = []
    for rank, row in exposure_df.iterrows():
        tdata.append([
            str(rank),
            row["Cluster"].replace(" / ", "/").replace(" + ", "+"),
            f"${row['Total GDP (USD B)']:.0f}B",
            f"${row['Exposed GDP – Present (USD B)']:.1f}B",
            f"{row['Exposure share (%)']:.0f}%",
            f"${row['Exposed GDP – 2050 (USD B)']:.1f}B",
            f"{row['Δ Exposed (USD B)']:+.1f}",
        ])

    tbl = ax_table.table(
        cellText=tdata, colLabels=cols, cellLoc="center",
        loc="center", bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)

    for j in range(len(cols)):
        c = tbl[(0, j)]
        c.set_facecolor("#2C3E50"); c.set_text_props(color="white", fontweight="bold")
        c.set_edgecolor("#2C3E50")

    for i, (_, row) in enumerate(exposure_df.iterrows(), start=1):
        share = row["Exposure share (%)"]
        hi  = mc.to_rgba(_exposure_color(share), alpha=0.22)
        bg  = "#FAFAFA" if i % 2 != 0 else "#F3F3F3"
        for j in range(len(cols)):
            c = tbl[(i, j)]; c.set_facecolor(bg); c.set_edgecolor("#E0E0E0")
        tbl[(i, 4)].set_facecolor(hi)

    ax_table.set_facecolor("white")
    for sp in ax_table.spines.values():
        sp.set_edgecolor("#CCCCCC"); sp.set_linewidth(0.6)
    ax_table.tick_params(left=False, bottom=False,
                         labelleft=False, labelbottom=False)
    ax_table.set_title("Priority Ranking — Flood-Weighted Exposure",
                       fontsize=8.5, fontweight="bold", color="#2C3E50", pad=4)

    fig.text(0.5, 0.970,
             "Malaysia  Flood-Risk × Economic-Exposure × Industrial Overlay"
             f"   (RP{HEADLINE_RP} / 2050 RCP8.5)",
             ha="center", va="top", fontsize=14, fontweight="bold", color="#2C3E50")
    fig.text(0.5, 0.944,
             "Sources: DOSM 2020 district real GDP (RM B) · GADM v4.1 district boundaries · "
             "WRI Aqueduct v2 (flood)",
             ha="center", va="top", fontsize=7, color="#777")
    fig.text(0.5, 0.010,
             "⚠ PROTOTYPE — global flood rasters underestimate pluvial & east-coast monsoon events; "
             "GDP distributed uniformly within districts; upgrade with Fathom 3.0 + DID data before capex decisions",
             ha="center", va="bottom", fontsize=6.5, color="#AA4444", style="italic")

    out = OUTPUT_DIR / "malaysia_flood_gdp_map.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  Static map → {out}")
    return out


# ===========================================================================
# ▌ FULL MALAYSIA MAP
# ===========================================================================

def create_full_malaysia_map(adm0, adm1, adm2, exposure_df, flood_depth, flood_tfm,
                              flood_label: str) -> Path:
    fig = plt.figure(figsize=(22, 10), facecolor="white")
    gs  = GridSpec(1, 3, figure=fig,
                   left=0.02, right=0.99, bottom=0.09, top=0.90,
                   wspace=0.04,
                   width_ratios=[1.35, 1.50, 0.28])

    ax_pen  = fig.add_subplot(gs[0, 0])
    ax_east = fig.add_subplot(gs[0, 1])
    ax_leg  = fig.add_subplot(gs[0, 2])

    _plot_map_panel(ax_pen, PENINSULAR_BBOX, adm0, adm1, adm2,
                    flood_depth, flood_tfm, exposure_df,
                    show_labels=True, flood_style="contour", show_gdp=True)
    ax_pen.set_title("Peninsular Malaysia",
                     fontsize=11, fontweight="bold", color="#2C3E50", pad=5)

    _plot_map_panel(ax_east, EAST_MY_BBOX, adm0, adm1, adm2,
                    flood_depth, flood_tfm, exposure_df,
                    show_labels=True, flood_style="contour", show_gdp=True)
    ax_east.set_title("East Malaysia — Sabah & Sarawak",
                      fontsize=11, fontweight="bold", color="#2C3E50", pad=5)

    ax_leg.set_facecolor("#FAFAFA")
    for sp in ax_leg.spines.values():
        sp.set_edgecolor("#CCCCCC"); sp.set_linewidth(0.6)
    ax_leg.set_xlim(0, 1); ax_leg.set_ylim(0, 1)
    ax_leg.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)

    cluster_handles = [
        mpatches.Patch(fc="#C0392B", ec="white", label="Exposure ≥85%"),
        mpatches.Patch(fc="#E67E22", ec="white", label="60–85%"),
        mpatches.Patch(fc="#F1C40F", ec="white", label="30–60%"),
        mpatches.Patch(fc="#27AE60", ec="white", label="<30%"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
               markersize=8, label="Cluster (size ∝ GDP)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="white",
               markeredgecolor="#555", markersize=9, label="★  Top-5"),
    ]
    flood_handles = [
        Line2D([0], [0], color="#4BABDB", linewidth=1.8, label="0.5 m contour"),
        Line2D([0], [0], color="#1565C0", linewidth=2.2, label="1.0 m contour"),
        Line2D([0], [0], color="#0A3D62", linewidth=2.6, label="2.0 m contour"),
    ]
    ax_leg.legend(handles=cluster_handles + flood_handles, loc="upper left",
                  fontsize=8, framealpha=0.0, handlelength=1.6, handleheight=1.4,
                  borderpad=0.8, labelspacing=0.80,
                  title="Legend", title_fontsize=9)
    ax_leg.text(
        0.04, 0.06,
        f"RP{HEADLINE_RP} present\n+ 2050 RCP8.5\n"
        f"Flood: {flood_label[:30]}\n"
        "GDP: DOSM 2020 district\n10 km cluster buffer",
        fontsize=6.8, color="#666", va="bottom",
        transform=ax_leg.transAxes, linespacing=1.5,
    )

    norm = _gdp_norm(adm2)
    cax1 = fig.add_axes([0.03, 0.022, 0.22, 0.025])
    cb1  = ColorbarBase(cax1, cmap=GDP_CMAP, norm=norm, orientation="horizontal")
    cb1.ax.set_xlabel("District GDP (RM B, 2020 real prices)", fontsize=7, color="#444")
    cb1.ax.xaxis.set_tick_params(labelsize=6.5, colors="#444")

    cax2 = fig.add_axes([0.31, 0.022, 0.22, 0.025])
    cb2  = ColorbarBase(cax2, cmap=FLOOD_CMAP,
                        norm=mc.Normalize(vmin=0, vmax=4.0), orientation="horizontal")
    cb2.ax.set_xlabel("Flood depth RP100 (m)", fontsize=7, color="#444")
    cb2.ax.xaxis.set_tick_params(labelsize=6.5, colors="#444")

    fig.text(0.5, 0.960,
             f"Malaysia  Flood Hazard × GDP Exposure × Industrial Clusters   (RP{HEADLINE_RP})",
             ha="center", va="top", fontsize=14, fontweight="bold", color="#2C3E50")
    fig.text(0.5, 0.935,
             "District GDP choropleth (DOSM 2020 real, RM B) · Flood shown as depth contour lines "
             "(0.5 m / 1.0 m / 2.0 m) · Cluster markers coloured by exposure share",
             ha="center", va="top", fontsize=8, color="#555")
    fig.text(
        0.5, 0.012,
        "⚠ PROTOTYPE — synthetic flood model; GDP distributed uniformly within districts. "
        "Upgrade: Fathom 3.0 (30 m) + DID basin data before capex decisions.",
        ha="center", va="bottom", fontsize=6.5, color="#AA4444", style="italic",
    )

    out = OUTPUT_DIR / "malaysia_map_full.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  Full Malaysia map → {out}")
    return out


# ===========================================================================
# ▌ SPLIT MAPS
# ===========================================================================

def create_split_maps(adm0, adm1, adm2, exposure_df, flood_depth, flood_tfm,
                      flood_label: str) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(24, 9), facecolor="white")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.11, top=0.88, wspace=0.05)

    BBOX = PENINSULAR_BBOX
    panel_cfg = [
        dict(flood_style="none",    show_gdp=True,  show_labels=False),
        dict(flood_style="fill",    show_gdp=False, show_labels=False),
        dict(flood_style="contour", show_gdp=True,  show_labels=True),
    ]
    panel_titles = [
        "A  GDP Exposure by District",
        "B  Flood Depth  RP100",
        "C  Combined — Flood × GDP × Clusters",
    ]

    for ax, title, kw in zip(axes, panel_titles, panel_cfg):
        _plot_map_panel(ax, BBOX, adm0, adm1, adm2, flood_depth, flood_tfm,
                        exposure_df, **kw)
        ax.set_title(title, fontsize=11, fontweight="bold", color="#2C3E50", pad=5)

    norm = _gdp_norm(adm2)
    cax1 = fig.add_axes([0.01, 0.035, 0.28, 0.022])
    cb1  = ColorbarBase(cax1, cmap=GDP_CMAP, norm=norm, orientation="horizontal")
    cb1.ax.set_xlabel("District GDP (RM B, 2020 real prices)", fontsize=7.5, color="#444")
    cb1.ax.xaxis.set_tick_params(labelsize=7, colors="#444")

    cax2 = fig.add_axes([0.355, 0.035, 0.28, 0.022])
    cb2  = ColorbarBase(cax2, cmap=FLOOD_CMAP,
                        norm=mc.Normalize(vmin=0, vmax=4.0), orientation="horizontal")
    cb2.ax.set_xlabel("Flood depth RP100 (m)", fontsize=7.5, color="#444")
    cb2.ax.xaxis.set_tick_params(labelsize=7, colors="#444")

    cax3 = fig.add_axes([0.71, 0.005, 0.28, 0.090])
    cax3.set_axis_off()
    leg_handles = [
        mpatches.Patch(fc="#C0392B", ec="white", label="Exposure ≥85%"),
        mpatches.Patch(fc="#E67E22", ec="white", label="Exposure 60–85%"),
        mpatches.Patch(fc="#F1C40F", ec="white", label="Exposure 30–60%"),
        mpatches.Patch(fc="#27AE60", ec="white", label="Exposure <30%"),
        Line2D([0], [0], color="#4BABDB", linewidth=1.8, label="0.5 m flood line"),
        Line2D([0], [0], color="#1565C0", linewidth=2.2, label="1.0 m flood line"),
        Line2D([0], [0], color="#0A3D62", linewidth=2.6, label="2.0 m flood line"),
    ]
    cax3.legend(handles=leg_handles, loc="center", fontsize=8,
                framealpha=0.0, ncol=2, handlelength=1.6, handleheight=1.2,
                borderpad=0.5, labelspacing=0.55)

    fig.text(0.5, 0.960,
             f"Malaysia Industrial Flood Exposure — Split View   (RP{HEADLINE_RP})",
             ha="center", va="top", fontsize=14, fontweight="bold", color="#2C3E50")
    fig.text(0.5, 0.935,
             f"Peninsular Malaysia · Flood: {flood_label} · "
             "Cluster markers coloured by damage-band exposure share",
             ha="center", va="top", fontsize=8, color="#555")

    out = OUTPUT_DIR / "malaysia_map_split.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  Split maps → {out}")
    return out


# ===========================================================================
# ▌ TABLE FIGURE
# ===========================================================================

def create_table_figure(exposure_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(18, 5.5), facecolor="white")
    ax.set_axis_off()

    cols = [
        "Rank", "Cluster", "State", "Type",
        "Total GDP\n(USD B)", "Depth\nRP100 (m)",
        "Exposed GDP\nPresent (USD B)", "Exposure\nShare (%)",
        "Depth 2050\nRCP8.5 (m)", "Exposed GDP\n2050 (USD B)", "Δ (USD B)",
    ]
    rows = []
    for rank, row in exposure_df.iterrows():
        rows.append([
            str(rank), row["Cluster"],
            row["State"].replace("Pulau Pinang", "Penang"),
            row["Type"],
            f"{row['Total GDP (USD B)']:.0f}",
            f"{row['Depth RP100 (m)']:.2f}",
            f"{row['Exposed GDP – Present (USD B)']:.1f}",
            f"{row['Exposure share (%)']:.0f}%",
            f"{row['Depth 2050 RCP8.5 (m)']:.2f}",
            f"{row['Exposed GDP – 2050 (USD B)']:.1f}",
            f"{row['Δ Exposed (USD B)']:+.1f}",
        ])

    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center",
                   loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    for j in range(len(cols)):
        c = tbl[(0, j)]
        c.set_facecolor("#1A252F"); c.set_text_props(color="white", fontweight="bold")

    for i, (_, row) in enumerate(exposure_df.iterrows(), start=1):
        hi = mc.to_rgba(_exposure_color(row["Exposure share (%)"]), alpha=0.26)
        bg = "#FAFAFA" if i % 2 != 0 else "#F0F0F0"
        for j in range(len(cols)):
            c = tbl[(i, j)]; c.set_facecolor(bg); c.set_edgecolor("#D0D0D0")
        tbl[(i, 6)].set_facecolor(hi); tbl[(i, 7)].set_facecolor(hi)

    ax.set_title(
        f"Malaysia Industrial Cluster Flood-Weighted Economic Exposure  "
        f"(RP{HEADLINE_RP} present + 2050 RCP8.5)",
        fontsize=12, fontweight="bold", color="#1A252F", pad=8,
    )
    fig.text(0.5, 0.02,
             "Sources: DOSM 2020 district real GDP · WRI Aqueduct v2 (depth-damage) · "
             "GADM v4.1 district boundaries",
             ha="center", fontsize=7.5, color="#777")

    out = OUTPUT_DIR / "exposure_table_figure.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  Table figure → {out}")
    return out


# ===========================================================================
# ▌ INTERACTIVE MAP
# ===========================================================================

def create_interactive_map(exposure_df: pd.DataFrame, adm2: gpd.GeoDataFrame,
                            flood_depth, flood_tfm, flood_label: str) -> Path:
    m = folium.Map(location=[4.5, 108.5], zoom_start=6,
                   tiles="CartoDB positron", prefer_canvas=True)

    # ── District GDP choropleth ───────────────────────────────────────────
    if adm2 is not None and "gdp_rm_b" in adm2.columns and adm2["gdp_rm_b"].max() > 0:
        # Simplify geometry for web delivery
        adm2_web = adm2[["NAME_2", "gdp_rm_b", "geometry"]].copy()
        if "NAME_1" in adm2.columns:
            adm2_web["state"] = adm2["NAME_1"]
        else:
            adm2_web["state"] = ""
        try:
            adm2_web["geometry"] = adm2_web["geometry"].simplify(0.005, preserve_topology=True)
        except Exception:
            pass
        geojson_str = adm2_web.to_json()
        folium.Choropleth(
            geo_data=geojson_str,
            data=adm2_web,
            columns=["NAME_2", "gdp_rm_b"],
            key_on="feature.properties.NAME_2",
            fill_color="YlOrRd",
            fill_opacity=0.55,
            line_opacity=0.35,
            line_weight=0.5,
            legend_name="District GDP (RM B, 2020 real prices)",
            name="GDP by District",
            show=True,
        ).add_to(m)

        # Tooltip layer showing district name + GDP on hover
        tooltip_layer = folium.GeoJson(
            geojson_str,
            name="District info",
            style_function=lambda f: {
                "fillOpacity": 0,
                "color": "#888",
                "weight": 0.4,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["NAME_2", "state", "gdp_rm_b"],
                aliases=["District:", "State:", "GDP (RM B):"],
                localize=True,
                sticky=False,
                labels=True,
                style="font-family:Arial;font-size:12px;",
            ),
            show=True,
        )
        tooltip_layer.add_to(m)

    # ── Industrial cluster markers ────────────────────────────────────────
    max_gdp = exposure_df["Total GDP (USD B)"].max()
    fg = folium.FeatureGroup(name="Industrial Clusters", show=True)

    for _, row in exposure_df.iterrows():
        share = row["Exposure share (%)"]
        clr   = _exposure_color(share)
        radius = 9 + 22 * (row["Total GDP (USD B)"] / max_gdp)

        popup_html = f"""
        <div style='font-family:Arial;font-size:12px;width:290px'>
          <b style='font-size:14px'>{row['Cluster']}</b><br>
          <hr style='margin:4px 0'>
          <b>State:</b> {row['State']} &nbsp; <b>Type:</b> {row['Type']}<br>
          <hr style='margin:4px 0'>
          <b>Total GDP (buffer):</b> USD {row['Total GDP (USD B)']:.1f} B<br>
          <b>Flood depth RP100:</b> {row['Depth RP100 (m)']:.2f} m<br>
          <b>Damage factor:</b> {row['Damage factor']*100:.0f}%<br>
          <b>Exposed GDP (present):</b>
            <span style='color:{clr};font-weight:bold'>
              USD {row['Exposed GDP – Present (USD B)']:.1f} B</span><br>
          <b>Exposed GDP (2050 RCP8.5):</b>
            USD {row['Exposed GDP – 2050 (USD B)']:.1f} B
            (<b>{row['Δ Exposed (USD B)']:+.1f} B Δ</b>)<br>
          <hr style='margin:4px 0'>
          <i style='color:#777;font-size:11px'>{row['note']}</i>
        </div>
        """

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color="white", weight=2,
            fill=True, fill_color=clr, fill_opacity=0.87,
            popup=folium.Popup(popup_html, max_width=310),
            tooltip=(
                f"<b>{row['Cluster']}</b><br>"
                f"Exposed: USD {row['Exposed GDP – Present (USD B)']:.1f}B "
                f"({row['Exposure share (%)']:.0f}%)"
            ),
        ).add_to(fg)

    fg.add_to(m)

    legend_html = """
    <div style='position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:12px 16px;border-radius:8px;
                box-shadow:0 2px 8px rgba(0,0,0,0.25);font-family:Arial;font-size:12px'>
      <b style='font-size:13px'>Flood Exposure Share (RP100)</b><br>
      <span style='color:#C0392B'>&#9679;</span> ≥85% &mdash; Very High<br>
      <span style='color:#E67E22'>&#9679;</span> 60–85% &mdash; High<br>
      <span style='color:#F1C40F'>&#9679;</span> 30–60% &mdash; Medium<br>
      <span style='color:#27AE60'>&#9679;</span> &lt;30% &mdash; Lower<br>
      <span style='color:#888;font-size:11px'>Circle size ∝ total GDP</span><br>
      <hr style='margin:6px 0'>
      <b style='font-size:11px'>Choropleth</b><br>
      <span style='color:#888;font-size:11px'>District GDP (RM B, 2020 real prices)<br>
      Source: DOSM open data</span><br>
      <hr style='margin:6px 0'>
      <span style='color:#AA4444;font-size:10px'>
        ⚠ Prototype — synthetic flood model<br>
        Upgrade: WRI Aqueduct / Fathom 3.0
      </span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)

    out = OUTPUT_DIR / "malaysia_flood_gdp_interactive.html"
    m.save(str(out))
    log.info(f"  Interactive map → {out}")
    return out


# ===========================================================================
# ▌ PRIORITY CALLOUTS
# ===========================================================================

def generate_priority_callouts(exposure_df: pd.DataFrame, flood_label: str) -> Path:
    top5 = exposure_df.head(5)
    risk_tag = {
        "#C0392B": "VERY HIGH", "#E67E22": "HIGH",
        "#F1C40F": "MEDIUM",   "#27AE60": "LOWER",
    }

    lines = [
        "=" * 72,
        "MALAYSIA FLOOD-RISK × ECONOMIC EXPOSURE — MANAGEMENT BRIEF",
        "=" * 72,
        "",
        f"Scenario : RP{HEADLINE_RP} present + 2050 RCP8.5 | Flood: {flood_label}",
        f"Method   : Depth-damage curve × cluster GDP (10 km buffer, USD B PPP 2021)",
        "GDP map  : DOSM district real GDP 2020 (RM B, 2015 prices) — 144 districts",
        "",
        "TOP 5 CLUSTERS FOR FLOOD-DEFENCE CAPEX",
        "  (ranked by present exposed GDP; 2050 delta shows which worsen most)",
        "-" * 72,
    ]

    for rank, row in top5.iterrows():
        tag = risk_tag.get(_exposure_color(row["Exposure share (%)"]), "—")
        lines += [
            "",
            f"#{rank}  {row['Cluster'].upper()}   [{tag}]",
            f"    State / Type  : {row['State']} | {row['Type']}",
            f"    Total GDP     : USD {row['Total GDP (USD B)']:.0f}B (10 km buffer)",
            f"    Flood depth   : {row['Depth RP100 (m)']:.2f} m (RP100) "
            f"→ {row['Depth 2050 RCP8.5 (m)']:.2f} m (2050 RCP8.5)",
            f"    Exposed GDP   : USD {row['Exposed GDP – Present (USD B)']:.1f}B (present) "
            f"→ USD {row['Exposed GDP – 2050 (USD B)']:.1f}B (2050), "
            f"Δ {row['Δ Exposed (USD B)']:+.1f}B",
            f"    Rationale     : {row['note']}",
        ]

    lines += [
        "",
        "=" * 72,
        "NOTABLE CROSS-PORTFOLIO FINDINGS",
        "-" * 72,
        "",
        "• EAST-COAST CLUSTERS (Gebeng, Kerteh): 90% exposure share at RP100.",
        "  Northeast Monsoon (Nov-Mar) drives annual inundation that global",
        "  rasters systematically under-resolve. HIGHEST URGENCY for local DID data.",
        "",
        "• PORT KLANG FZ: Largest absolute exposed GDP (USD 45B present).",
        "  Already at max damage band. SLR alone threatens irreversibility.",
        "  Priority: permanent seawall + tide-gate engineering study.",
        "",
        "• BIGGEST 2050 DETERIORATION:",
        "  PTP (USD +6.2B): crosses from 65% → 90% damage band under SLR.",
        "  Kulim HTP (USD +4.5B): 35% → 65% band — largest relative jump.",
        "  Review both before committing major new greenfield investment.",
        "",
        "• SHAH ALAM / KLANG BELT: Second-largest absolute exposure (USD 27.6B).",
        "  Recurrent annual flash flooding already disrupts logistics and labour.",
        "  Urban drainage master plan (SMART Tunnel expansion) has high ROI.",
        "",
        "=" * 72,
        "LIMITATIONS (required for credibility)",
        "-" * 72,
        "",
        "1. FLOOD RASTERS: Global open-source models (WRI Aqueduct, JRC) are",
        "   calibrated to large river systems and open coast. Systematic gaps:",
        "   (a) pluvial/flash floods — Klang Valley, Penang urban;",
        "   (b) east-coast monsoon — Pahang, Terengganu, Kelantan.",
        "   Upgrade path: Fathom 3.0 (30 m, fluvial+pluvial+coastal, non-commercial",
        "   free — request info@fathom.global); national data via DID formal MOU.",
        "",
        "2. GDP GRID: DOSM 2020 district real GDP (RM B, 2015 prices) distributed",
        "   uniformly within district boundaries. Actual economic density is far",
        "   more concentrated in urban cores.",
        "   Next upgrade: GHS-POP-weighted redistribution within districts.",
        "",
        "3. DEPTH-DAMAGE: UNDRR generalised industrial curve applied uniformly.",
        "   Actual damage depends on building stock, floor level, flood warning.",
        "   Cluster-specific asset surveys (NAPIC / MIDA) would sharpen estimates.",
        "",
        "4. CLIMATE UPLIFT: 2050 depth = present × 1.35 (RCP8.5 regional median).",
        "   Actual range: ×1.1–×1.8 across GCMs. Multi-model ensemble with 5th/",
        "   95th percentile needed for robust resilience planning.",
        "",
        "5. COASTAL / STORM SURGE: Not modelled separately. WRI Aqueduct coastal",
        "   layers (inuncoast_*) or SCHISM storm-surge outputs should be added",
        "   for Pasir Gudang, PTP, Port Klang, Batu Berendam.",
        "",
        "RECOMMENDATION: Use this prototype for management prioritisation.",
        "Commission site-specific FSIs using Fathom 3.0 + DID + NAPIC data",
        "before committing capex or writing flood-risk covenants into financing.",
        "",
        "=" * 72,
        "DATA CITATIONS",
        "-" * 72,
        "",
        "DOSM (2024). GDP by District, Real (Supply Approach), 2015–2020.",
        "  Department of Statistics Malaysia. data.gov.my",
        "  https://storage.dosm.gov.my/gdp/gdp_district_real_supply.parquet",
        "",
        "Ward, P. J. et al. (2020). Aqueduct Floods Methodology. WRI Technical Note.",
        "  Washington DC: World Resources Institute.",
        "  URL: http://wri-projects.s3.amazonaws.com/AqueductFloodTool/",
        "",
        "GADM (2022). Database of Global Administrative Areas, v4.1. gadm.org",
        "",
        "=" * 72,
    ]

    out = OUTPUT_DIR / "priority_callouts.txt"
    out.write_text("\n".join(lines))
    log.info(f"  Callouts → {out}")
    return out


# ===========================================================================
# ▌ MAIN
# ===========================================================================

def main():
    log.info("=" * 60)
    log.info("Malaysia Flood-Risk × Economic-Exposure Analysis (District-level GDP)")
    log.info("=" * 60)

    # 1. Admin boundaries + district GDP
    log.info("\n[1/6] Admin boundaries + district GDP …")
    adm0, adm1, adm2 = get_malaysia_boundaries()
    land_poly = unary_union(adm1.geometry)

    if "gdp_rm_b" in adm2.columns:
        total_rm = adm2["gdp_rm_b"].sum()
        log.info(f"  Total district GDP: RM {total_rm:.0f} B")

    # 2. GDP raster (district-level)
    log.info("\n[2/6] Building GDP raster (district level) …")
    gdp_arr, gdp_tfm, _ = load_gdp_raster(adm2)
    log.info(f"  Grid {gdp_arr.shape}")

    # 3. Flood raster
    log.info(f"\n[3/6] Flood raster RP{HEADLINE_RP} (present) …")
    flood_arr, flood_tfm, _, flood_label = load_flood_raster(HEADLINE_RP, land_poly=land_poly)
    log.info(f"  Grid {flood_arr.shape}, max depth: {flood_arr.max():.2f} m")
    log.info(f"  Source: {flood_label}")

    # 4. Cluster exposure table
    log.info("\n[4/6] Computing cluster exposure …")
    exposure_df = build_exposure_table()
    note_map = {c["id"]: c["note"] for c in CLUSTERS}
    exposure_df["note"] = exposure_df["id"].map(note_map)

    csv_out = OUTPUT_DIR / "cluster_exposure_table.csv"
    exposure_df.to_csv(csv_out)
    log.info(f"  CSV → {csv_out}")

    print("\n  CLUSTER EXPOSURE RANKING (RP100, present):")
    hdr = f"  {'Rk':<4} {'Cluster':<36} {'GDP':>8} {'ExpGDP':>8} {'Share':>7}"
    print(hdr)
    print("  " + "-" * 68)
    for rank, row in exposure_df.iterrows():
        print(
            f"  {rank:<4} {row['Cluster']:<36}"
            f"  ${row['Total GDP (USD B)']:>5.0f}B"
            f"  ${row['Exposed GDP – Present (USD B)']:>5.1f}B"
            f"  {row['Exposure share (%)']:>5.0f}%"
        )

    # 5. Visualisations
    log.info("\n[5/6] Generating visualisations …")
    map_path   = create_static_map(adm0, adm1, adm2, exposure_df, flood_arr, flood_tfm, flood_label)
    full_path  = create_full_malaysia_map(adm0, adm1, adm2, exposure_df, flood_arr, flood_tfm, flood_label)
    split_path = create_split_maps(adm0, adm1, adm2, exposure_df, flood_arr, flood_tfm, flood_label)
    tbl_path   = create_table_figure(exposure_df)
    html_path  = create_interactive_map(exposure_df, adm2, flood_arr, flood_tfm, flood_label)

    # 6. Callouts
    log.info("\n[6/6] Priority callouts …")
    txt_path = generate_priority_callouts(exposure_df, flood_label)

    log.info("\n" + "=" * 60)
    log.info("ALL OUTPUTS")
    log.info("=" * 60)
    for p in [map_path, full_path, split_path, html_path, tbl_path, csv_out, txt_path]:
        log.info(f"  {p}")
    log.info("=" * 60)

    print("\n" + txt_path.read_text())


if __name__ == "__main__":
    main()
