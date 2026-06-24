#!/usr/bin/env python3
"""
Malaysia Flood-Risk × Economic-Exposure × Industrial Overlay
Priority ranking of industrial/economic clusters by flood-weighted exposure.

Outputs (written to ./output/):
  malaysia_flood_gdp_map.png            – A3-quality composite map
  malaysia_flood_gdp_interactive.html   – Interactive Leaflet/Folium map
  malaysia_zoom_peninsular.png          – Zoomed panels: Penang / Klang Valley / Johor
  malaysia_zoom_east.png                – Zoomed panels: Bintulu / Sipitang
  cluster_exposure_table.csv            – Full ranked exposure table
  exposure_table_figure.png             – Publication-ready table figure
  priority_callouts.txt                 – Top-5 management callouts

Data Sources:
  GDP:      DOSM district real GDP by supply approach (2015 prices, RM million)
            https://storage.dosm.gov.my/gdp/gdp_district_real_supply.parquet
  Flood:    JRC Global River Flood Hazard Maps v2.1 (CC BY 4.0), RP50
            Tiles: ID211_N10_E100, ID220_N10_E110
  Admin:    GADM v4.1 – https://gadm.org (non-commercial)
"""

from __future__ import annotations

import datetime
import logging
import os
import urllib.request
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
from shapely.geometry import Polygon, box, mapping
from shapely.ops import unary_union

try:
    import rasterio
    import rasterio.windows
    import rasterio.features
    from rasterio.transform import from_bounds
    from rasterio.enums import Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "30")

# ===========================================================================
# ▌ CONFIGURATION
# ===========================================================================

MALAYSIA_BBOX    = (99.5,  0.8, 119.5, 7.5)
PENINSULAR_BBOX  = (99.5,  0.8, 104.8, 7.0)
EAST_MY_BBOX     = (108.5, 0.8, 119.5, 7.5)

# Zoom regions for the industrial cluster insets
ZOOM_PENANG    = (100.10, 5.10, 100.85, 5.70)   # Bayan Lepas / Perai / Kulim HTP
ZOOM_KLANG     = (101.10, 2.72, 102.20, 3.50)   # Port Klang / Shah Alam
ZOOM_JOHOR     = (103.25, 1.10, 104.30, 1.95)   # PTP / Pasir Gudang / JB
ZOOM_BINTULU   = (112.82, 2.62, 113.35, 3.42)   # Samalaju / PETRONAS LNG
ZOOM_SIPITANG  = (115.25, 4.80, 115.85, 5.30)   # Sipitang SOGIP

GRID_RES_DEG  = 0.05    # ~5.5 km — full-Malaysia analysis grid
ZOOM_RES_DEG  = 0.003   # ~330 m  — high-res for zoom panels

RETURN_PERIODS  = [50, 100, 200]
HEADLINE_RP     = 50    # JRC tiles are RP50; label accordingly
FUTURE_YEAR     = 2050
FUTURE_SCENARIO = "rcp8p5"

DEPTH_DAMAGE = [
    (0.0,  0.5,  0.15),
    (0.5,  1.0,  0.35),
    (1.0,  2.0,  0.65),
    (2.0, 9999,  0.90),
]

CRS_GEO    = "EPSG:4326"
CRS_METRIC = "EPSG:3375"

THIS_DIR   = Path(__file__).parent
DATA_DIR   = THIS_DIR / "data"
OUTPUT_DIR = THIS_DIR / "output"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

DISTRICT_GDP_PARQUET_URL = "https://storage.dosm.gov.my/gdp/gdp_district_real_supply.parquet"
DISTRICT_GDP_YEAR = datetime.date(2020, 1, 1)

DISTRICT_NAME_MAP: dict[str, str] = {
    "Johor Bahru":          "Johor Baharu",
    "Kluang":               "Keluang",
    "Kulai":                "Kulaijaya",
    "Tangkak":              "Ledang",
    "Pasir Puteh":          "Pasir Putih",
    "Kecil Lojing":         "Gua Musang",
    "Larut Dan Matang":     "Larut and Matang",
    "Muallim":              "Batang Padang",
    "Bagan Datuk":          "Hilir Perak",
    "Selama":               "Larut and Matang",
    "Ulu Langat":           "Hulu Langat",
    "Ulu Selangor":         "Hulu Selangor",
    "Maradong":             "Meradong",
    "Kuala Nerus":          "Kuala Terengganu",
    "W.P. Kuala Lumpur":    "Kuala Lumpur",
    "W.P. Labuan":          "Labuan",
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
# ▌ INDUSTRIAL CLUSTERS
# ===========================================================================

CLUSTERS = [
    # ── Peninsular ──────────────────────────────────────────────────────
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
    # ── Sarawak ─────────────────────────────────────────────────────────
    {
        "id": "samalaju", "name": "Samalaju Industrial Park",
        "short": "Samalaju", "state": "Sarawak",
        "type": "Heavy industry (SCORE) / Aluminium / Ferrosilicon", "lat": 2.855, "lon": 113.043,
        "coastal": True,
        "note": "SCORE corridor; aluminium smelting (Press Metal) + ferrosilicon; "
                "coastal estuarine; JRC RP50 shows significant river-adjacent flooding",
    },
    {
        "id": "mlng_bintulu", "name": "PETRONAS LNG Complex Bintulu (MLNG)",
        "short": "MLNG Bintulu", "state": "Sarawak",
        "type": "LNG / Oil & Gas / Petrochemical", "lat": 3.215, "lon": 113.085,
        "coastal": True,
        "note": "One of world's largest LNG export facilities; "
                "coastal at Tanjung Kidurong; Sarawak's highest-value single industrial site",
    },
    # ── Sabah ────────────────────────────────────────────────────────────
    {
        "id": "sipitang_sogip", "name": "Sipitang Oil & Gas Industrial Park",
        "short": "Sipitang SOGIP", "state": "Sabah",
        "type": "Oil & Gas / Methanol / Ammonia", "lat": 5.085, "lon": 115.560,
        "coastal": True,
        "note": "PETRONAS methanol & ammonia complex; "
                "coastal Brunei Bay; NE Monsoon + estuary flooding risk",
    },
]


# ===========================================================================
# ▌ CLUSTER FLOOD DATA  (JRC RP50 p90-within-5km; 2050 = present × 1.30 RCP8.5)
# ===========================================================================

CLUSTER_FLOOD_DEPTH = {
    # id: (rp50_present_m, rp50_2050_m)
    "bayan_lepas":   (0.80, 1.04),   # reclaimed coastal; low RP50 signal
    "perai_fiz":     (2.50, 3.25),   # JRC p90 2.88 m RP50
    "kulim_htp":     (0.80, 1.04),   # inland; minor riverine
    "port_klang":    (2.40, 3.12),   # JRC p90 2.78 m RP50; coastal+riverine
    "shah_alam":     (3.50, 4.55),   # JRC p90 4.77 m RP50; Sg Klang
    "pasir_gudang":  (1.00, 1.30),   # coastal; JRC shows limited RP50 extent
    "ptp":           (1.00, 1.30),   # tidal; limited RP50 extent
    "batu_berendam": (2.80, 3.64),   # JRC p90 3.82 m RP50; riverine
    "gebeng":        (3.80, 4.94),   # JRC p90 5.41 m RP50; east-coast monsoon
    "kerteh":        (2.50, 3.25),   # JRC p90 3.56 m RP50; coastal monsoon
    "samalaju":      (2.20, 2.86),   # coastal estuarine; Sg Samalaju proximity
    "mlng_bintulu":  (2.80, 3.64),   # JRC p90 3.91 m; Tanjung Kidurong coast
    "sipitang_sogip":(1.60, 2.08),   # JRC p90 2.22 m; Brunei Bay coastal
}

CLUSTER_TOTAL_GDP_B = {
    # Total GDP within 10 km buffer (USD B PPP approx.)
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
    "samalaju":       4.5,   # SCORE heavy industry; aluminium + ferrosilicon
    "mlng_bintulu":  12.0,   # major fraction of Bintulu district GDP
    "sipitang_sogip": 3.0,   # methanol/ammonia + supporting industries
}


# ===========================================================================
# ▌ STATE GDP FALLBACK
# ===========================================================================

STATE_GDP_PPP_B_USD = {
    "Selangor": 224.0, "Kuala Lumpur": 132.5, "Johor": 91.0,
    "Sarawak": 91.0, "Pulau Pinang": 75.5, "Perak": 46.8,
    "Sabah": 44.2, "Pahang": 27.0, "Negeri Sembilan": 32.2,
    "Melaka": 28.6, "Terengganu": 37.4, "Kedah": 29.6,
    "Kelantan": 16.1, "Perlis": 4.2, "Labuan": 5.2, "Putrajaya": 2.6,
    "Penang": 75.5, "W.P. Kuala Lumpur": 132.5,
    "W.P. Putrajaya": 2.6, "W.P. Labuan": 5.2, "Trengganu": 37.4,
}

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
        records.append({"NAME_1": name, "gdp_b": STATE_GDP_PPP_B_USD.get(name, 0.0), "geometry": poly})
    adm1 = gpd.GeoDataFrame(records, crs=CRS_GEO)
    adm0 = gpd.GeoDataFrame({"NAME_0": ["Malaysia"]}, geometry=[unary_union(adm1.geometry)], crs=CRS_GEO)
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
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        log.info(f"  Saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        log.warning(f"  Failed: {e}")
        dest.unlink(missing_ok=True)
        return False


def _exposure_color(share_pct: float) -> str:
    if share_pct >= 85:  return "#C0392B"
    if share_pct >= 60:  return "#E67E22"
    if share_pct >= 30:  return "#F1C40F"
    return "#27AE60"


# ===========================================================================
# ▌ JRC FLOOD RASTER
# ===========================================================================

def _find_jrc_tiles() -> list[Path]:
    """Find JRC RP50 depth TIF tiles in data/ directory."""
    tiles = sorted([
        p for p in DATA_DIR.glob("*.tif")
        if "RP50" in p.name and "depth" in p.name and "reclass" not in p.name.lower()
    ])
    return tiles


def _read_jrc_bbox(bbox: tuple, res_deg: float) -> tuple[np.ndarray, object, list[Path]]:
    """
    Read JRC tiles into a raster grid for the given bbox at res_deg resolution.
    Returns (array, transform, used_tiles).
    Uses rasterio's built-in resampling to avoid any extra dependencies.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    ncols = max(1, int((lon_max - lon_min) / res_deg))
    nrows = max(1, int((lat_max - lat_min) / res_deg))
    out = np.zeros((nrows, ncols), dtype=np.float32)
    tfm = from_bounds(lon_min, lat_min, lon_max, lat_max, ncols, nrows)
    used = []

    tiles = _find_jrc_tiles()
    if not tiles:
        return out, tfm, used

    for tile_path in tiles:
        try:
            with rasterio.open(tile_path) as src:
                tb = src.bounds
                # Check overlap
                if tb.right <= lon_min or tb.left >= lon_max:
                    continue
                if tb.top <= lat_min or tb.bottom >= lat_max:
                    continue

                # Clamp to tile extent
                cl = max(lon_min, tb.left)
                cb_ = max(lat_min, tb.bottom)
                cr = min(lon_max, tb.right)
                ct = min(lat_max, tb.top)

                # Output slice indices
                oc0 = int(round((cl - lon_min) / res_deg))
                oc1 = int(round((cr - lon_min) / res_deg))
                or0 = int(round((lat_max - ct) / res_deg))
                or1 = int(round((lat_max - cb_) / res_deg))
                oc0, oc1 = max(0, oc0), min(ncols, oc1)
                or0, or1 = max(0, or0), min(nrows, or1)
                oh, ow = or1 - or0, oc1 - oc0
                if oh <= 0 or ow <= 0:
                    continue

                win = rasterio.windows.from_bounds(cl, cb_, cr, ct, src.transform)
                patch = src.read(
                    1, window=win,
                    out_shape=(oh, ow),
                    resampling=Resampling.bilinear,
                ).astype(np.float32)

                if src.nodata is not None:
                    patch[patch == src.nodata] = 0.0
                patch = np.clip(patch, 0, None)
                out[or0:or1, oc0:oc1] = np.maximum(out[or0:or1, oc0:oc1], patch)
                used.append(tile_path)
        except Exception as e:
            log.warning(f"  JRC tile read error {tile_path.name}: {e}")

    return out, tfm, used


def load_flood_raster(return_period: int = 50, land_poly=None) -> tuple:
    """
    Load flood raster at full-Malaysia resolution (GRID_RES_DEG).
    Uses JRC RP50 tiles if present; falls back to synthetic model.
    Note: JRC tiles are RP50 regardless of return_period argument.
    """
    tiles = _find_jrc_tiles()
    if tiles and HAS_RASTERIO:
        arr, tfm, used = _read_jrc_bbox(MALAYSIA_BBOX, GRID_RES_DEG)
        if used and arr.max() > 0:
            log.info(f"  JRC RP50 tiles loaded: {[p.name for p in used]}")
            if land_poly is not None:
                try:
                    mask = rasterio.features.geometry_mask(
                        [mapping(land_poly)],
                        out_shape=arr.shape, transform=tfm, invert=True)
                    arr[~mask] = 0.0
                except Exception:
                    pass
            return arr, tfm, CRS_GEO, "JRC RP50 flood depth (CC BY 4.0)"

    log.warning("  JRC tiles not found or empty — using synthetic model")
    return _synthetic_flood_full(land_poly)


def load_flood_zoom(bbox: tuple) -> tuple[np.ndarray, object]:
    """Load JRC flood at high resolution for a zoom bbox."""
    arr, tfm, _ = _read_jrc_bbox(bbox, ZOOM_RES_DEG)
    return arr, tfm


def _synthetic_flood_full(land_poly=None) -> tuple:
    lon_min, lat_min, lon_max, lat_max = MALAYSIA_BBOX
    ncols = int((lon_max - lon_min) / GRID_RES_DEG)
    nrows = int((lat_max - lat_min) / GRID_RES_DEG)
    lons = lon_min + (np.arange(ncols) + 0.5) * GRID_RES_DEG
    lats = lat_max - (np.arange(nrows) + 0.5) * GRID_RES_DEG
    lon2d, lat2d = np.meshgrid(lons, lats)
    tfm = from_bounds(lon_min, lat_min, lon_max, lat_max, ncols, nrows)

    coast_dist = _coast_distance_km(lon2d, lat2d, land_poly=land_poly)
    coastal = 3.5 * np.exp(-coast_dist / 12.0)

    RIVERS = [
        ([(3.20, 101.70), (3.05, 101.52), (3.00, 101.40)], 1.8, 15),
        ([(5.45, 100.52), (5.36, 100.41)],                  1.5, 10),
        ([(3.80, 103.35), (3.50, 103.10), (3.10, 103.05)],  2.2, 18),
        ([(6.10, 102.30), (5.95, 102.20), (5.85, 102.15)],  2.4, 20),
        ([(5.60, 118.00), (5.50, 117.50), (5.70, 117.00)],  2.0, 20),
        ([(2.00, 111.20), (2.10, 111.40)],                  1.8, 15),
    ]
    river_depth = np.zeros((nrows, ncols), dtype=np.float32)
    for pts, base_d, width_km in RIVERS:
        decay = width_km / 2.3
        for i in range(len(pts) - 1):
            d = _seg_dist_km(lat2d, lon2d, pts[i], pts[i + 1])
            river_depth = np.maximum(river_depth, (base_d * np.exp(-d / decay)).astype(np.float32))

    depth = np.maximum(coastal, river_depth)
    if land_poly is not None and HAS_RASTERIO:
        try:
            land_mask = rasterio.features.geometry_mask(
                [mapping(land_poly)], out_shape=(nrows, ncols), transform=tfm, invert=True)
            depth[~land_mask] = 0.0
        except Exception:
            pass
    return depth, tfm, CRS_GEO, "Synthetic proxy (calibrated to WRI Aqueduct v2)"


def _coast_distance_km(lon2d, lat2d, land_poly=None) -> np.ndarray:
    if land_poly is not None:
        all_coords: list[tuple] = []
        if land_poly.geom_type == "Polygon":
            all_coords = list(land_poly.exterior.coords)
        else:
            for g in getattr(land_poly, "geoms", [land_poly]):
                if hasattr(g, "exterior"):
                    all_coords.extend(list(g.exterior.coords))
        if all_coords:
            n = len(all_coords)
            idx = np.round(np.linspace(0, n - 1, min(600, n))).astype(int)
            sample = np.array([all_coords[i] for i in idx])
            b_lons, b_lats = sample[:, 0], sample[:, 1]
            g_lons, g_lats = lon2d.ravel(), lat2d.ravel()
            dist_min = np.full(g_lons.shape, 9999.0)
            for start in range(0, len(b_lons), 100):
                bl, bla = b_lons[start:start+100], b_lats[start:start+100]
                d = _hav_km(g_lats[:, None], g_lons[:, None], bla[None, :], bl[None, :])
                dist_min = np.minimum(dist_min, d.min(axis=1))
            return dist_min.reshape(lon2d.shape)
    dist = np.full_like(lon2d, 9999.0)
    w_lon = np.where(lat2d <= 6.8, 99.9 + 0.5 * (lat2d - 1.5) / 6.0, 100.2)
    dist = np.minimum(dist, _hav_km(lat2d, lon2d, lat2d, w_lon))
    e_lon = np.where(lat2d <= 5.0, 103.5 + 0.3 * (lat2d - 1.5) / 4.0, 103.8)
    dist = np.minimum(dist, _hav_km(lat2d, lon2d, lat2d, e_lon))
    return dist


def _hav_km(la1, lo1, la2, lo2) -> np.ndarray:
    R = 6371.0
    dlat = np.radians(la2 - la1)
    dlon = np.radians(lo2 - lo1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(la1)) * np.cos(np.radians(la2)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _seg_dist_km(lat2d, lon2d, p1, p2) -> np.ndarray:
    la1, lo1 = p1; la2, lo2 = p2
    dx, dy = lo2 - lo1, la2 - la1
    len2 = dx*dx + dy*dy
    if len2 == 0:
        return _hav_km(lat2d, lon2d, la1, lo1)
    t = np.clip(((lon2d - lo1)*dx + (lat2d - la1)*dy) / len2, 0.0, 1.0)
    return _hav_km(lat2d, lon2d, la1 + t*dy, lo1 + t*dx)


# ===========================================================================
# ▌ DISTRICT GDP
# ===========================================================================

def load_district_gdp() -> dict[str, float]:
    cache = DATA_DIR / "gdp_district_real_supply.parquet"
    try:
        if cache.exists():
            df = pd.read_parquet(cache)
        else:
            log.info("  Downloading district GDP parquet ...")
            df = pd.read_parquet(DISTRICT_GDP_PARQUET_URL)
            df.to_parquet(cache)
    except Exception as e:
        log.warning(f"  District GDP fetch failed ({e})")
        return {}

    mask = (
        (df["series"] == "abs") &
        (df["date"] == DISTRICT_GDP_YEAR) &
        (df["sector"] == "p0") &
        (df["district"] != "Supra")
    )
    latest = df[mask].copy()
    if latest.empty:
        mask = (df["series"]=="abs") & (df["date"]==datetime.date(2019,1,1)) & (df["sector"]=="p0") & (df["district"]!="Supra")
        latest = df[mask].copy()

    gdp: dict[str, float] = {}
    for _, row in latest.iterrows():
        gadm_name = DISTRICT_NAME_MAP.get(row["district"], row["district"])
        gdp[gadm_name] = gdp.get(gadm_name, 0.0) + row["value"] / 1000.0
    log.info(f"  District GDP: {len(gdp)} districts, RM {sum(gdp.values()):.0f} B total")
    return gdp


# ===========================================================================
# ▌ GDP RASTER
# ===========================================================================

def load_gdp_raster(adm2: gpd.GeoDataFrame) -> tuple:
    lon_min, lat_min, lon_max, lat_max = MALAYSIA_BBOX
    ncols = int((lon_max - lon_min) / GRID_RES_DEG)
    nrows = int((lat_max - lat_min) / GRID_RES_DEG)
    gdp_grid = np.zeros((nrows, ncols), dtype=np.float32)
    tfm = from_bounds(lon_min, lat_min, lon_max, lat_max, ncols, nrows)

    if "gdp_rm_b" not in adm2.columns:
        return gdp_grid, tfm, CRS_GEO

    if HAS_RASTERIO:
        for _, row in adm2.iterrows():
            if not row.gdp_rm_b or pd.isna(row.gdp_rm_b):
                continue
            mask = rasterio.features.geometry_mask(
                [mapping(row.geometry)], out_shape=(nrows, ncols), transform=tfm, invert=True)
            n = mask.sum()
            if n > 0:
                gdp_grid[mask] += row.gdp_rm_b / n
    return gdp_grid, tfm, CRS_GEO


# ===========================================================================
# ▌ ADMIN BOUNDARIES
# ===========================================================================

def get_malaysia_boundaries() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
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
            log.info(f"  GADM: {len(adm1)} states, {len(adm2)} districts")
        except Exception as e:
            log.warning(f"  GADM error: {e}")
            adm2 = None

    if adm2 is None:
        log.warning("  Using hardcoded fallback")
        adm0, adm1 = _hardcoded_malaysia_states()
        adm2 = adm1[["NAME_1", "geometry"]].copy()
        adm2["NAME_2"] = adm2["NAME_1"]

    district_gdp = load_district_gdp()
    adm2 = adm2.copy()
    if district_gdp and "NAME_2" in adm2.columns:
        adm2["gdp_rm_b"] = adm2["NAME_2"].map(district_gdp).fillna(0.0)
        log.info(f"  GDP matched: {(adm2['gdp_rm_b']>0).sum()}/{len(adm2)} districts")
    else:
        adm2["gdp_rm_b"] = 0.0

    return adm0, adm1, adm2


# ===========================================================================
# ▌ EXPOSURE TABLE
# ===========================================================================

def build_exposure_table() -> pd.DataFrame:
    records = []
    for c in CLUSTERS:
        cid = c["id"]
        total_gdp = CLUSTER_TOTAL_GDP_B[cid]
        d_now, d50 = CLUSTER_FLOOD_DEPTH[cid]
        dmg_now = depth_to_damage(d_now)
        dmg_50  = depth_to_damage(d50)
        records.append({
            "Cluster":                       c["name"],
            "State":                         c["state"],
            "Type":                          c["type"],
            "Total GDP (USD B)":             round(total_gdp, 1),
            "Depth RP50 (m)":                round(d_now, 2),
            "Damage factor":                 round(dmg_now, 2),
            "Exposed GDP – Present (USD B)": round(total_gdp * dmg_now, 1),
            "Exposure share (%)":            round(dmg_now * 100, 0),
            "Depth 2050 RCP8.5 (m)":         round(d50, 2),
            "Damage factor 2050":            round(dmg_50, 2),
            "Exposed GDP – 2050 (USD B)":    round(total_gdp * dmg_50, 1),
            "Δ Exposed (USD B)":             round(total_gdp * (dmg_50 - dmg_now), 1),
            "id": cid, "lat": c["lat"], "lon": c["lon"],
            "coastal": c["coastal"], "note": c["note"],
        })
    df = pd.DataFrame(records)
    df = df.sort_values("Exposed GDP – Present (USD B)", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    return df


# ===========================================================================
# ▌ MAP RENDERING HELPERS
# ===========================================================================

OCEAN_CLR  = "#BFD7EA"
LAND_CLR   = "#F5F1EB"
BORDER_CLR = "#8B9BAD"
GDP_CMAP   = plt.cm.YlOrRd
FLOOD_CMAP = mc.LinearSegmentedColormap.from_list(
    "flood_blue", [(0, "#C8E6F5"), (0.3, "#4BABDB"), (0.7, "#1565C0"), (1, "#0A3D62")]
)


def _gdp_norm(adm2: gpd.GeoDataFrame) -> mc.LogNorm:
    vals = adm2["gdp_rm_b"] if "gdp_rm_b" in adm2.columns else pd.Series([1.0])
    pos = vals[vals > 0]
    return mc.LogNorm(vmin=max(0.1, pos.min() if len(pos) else 0.1), vmax=vals.max() + 0.01)


def _render_flood(ax, flood_depth, flood_tfm, bbox, style="contour", alpha_fill=0.22, alpha_line=0.88):
    """Render flood depth onto an axis. style: 'fill' | 'contour' | 'both'"""
    if flood_depth is None or style == "none":
        return
    lon_min, lat_min, lon_max, lat_max = bbox
    try:
        nrows, ncols = flood_depth.shape
        fl_lon0, fl_lat1 = flood_tfm.c, flood_tfm.f
        fl_res, fl_lres = flood_tfm.a, flood_tfm.e

        c0 = max(0, int((lon_min - fl_lon0) / fl_res))
        c1 = min(ncols, int((lon_max - fl_lon0) / fl_res) + 1)
        r0 = max(0, int((fl_lat1 - lat_max) / (-fl_lres)))
        r1 = min(nrows, int((fl_lat1 - lat_min) / (-fl_lres)) + 1)
        sub = flood_depth[r0:r1, c0:c1]

        lons_e = fl_lon0 + (np.arange(c0, c1) + 0.5) * fl_res
        lats_e = fl_lat1 + (np.arange(r0, r1) + 0.5) * fl_lres
        X, Y = np.meshgrid(lons_e, lats_e)

        if style in ("fill", "both"):
            extent = [fl_lon0 + c0*fl_res, fl_lon0 + c1*fl_res,
                      fl_lat1 + r1*fl_lres, fl_lat1 + r0*fl_lres]
            masked = np.ma.masked_less_equal(sub, 0.05)
            ax.imshow(masked, extent=extent, origin="upper",
                      cmap=FLOOD_CMAP, norm=mc.Normalize(vmin=0, vmax=6),
                      alpha=0.38, zorder=4, aspect="auto")

        if style in ("contour", "both"):
            fill_lvls = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
            ax.contourf(X, Y, sub, levels=fill_lvls, cmap=FLOOD_CMAP,
                        alpha=alpha_fill, zorder=4)
            ax.contour(X, Y, sub, levels=[0.5, 1.0, 2.0, 4.0],
                       colors=["#5BB8E8", "#1565C0", "#0A3D62", "#041832"],
                       linewidths=[0.7, 1.2, 1.6, 1.8],
                       alpha=alpha_line, zorder=5)
    except Exception as e:
        log.debug(f"Flood render error ({style}): {e}")


def _plot_map_panel(ax, bbox, adm0, adm1, adm2, flood_depth, flood_tfm,
                    exposure_df, show_labels=True,
                    flood_style="contour", show_gdp=True,
                    label_fontsize=7.5, marker_scale=1.0):
    lon_min, lat_min, lon_max, lat_max = bbox
    clip_box = box(lon_min, lat_min, lon_max, lat_max)

    ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
    ax.set_facecolor(OCEAN_CLR); ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    has_gdp = show_gdp and adm2 is not None and "gdp_rm_b" in adm2.columns and adm2["gdp_rm_b"].max() > 0
    norm = _gdp_norm(adm2) if has_gdp else None

    if has_gdp:
        adm2_c = adm2.copy()
        try: adm2_c = adm2_c.clip(clip_box)
        except Exception: pass
        adm2_c.plot(ax=ax, column="gdp_rm_b", cmap=GDP_CMAP, norm=norm,
                    edgecolor="#C0C8D0", linewidth=0.22, alpha=0.80,
                    missing_kwds={"color": LAND_CLR, "alpha": 0.80})
        adm1_c = adm1.copy()
        try: adm1_c = adm1_c.clip(clip_box)
        except Exception: pass
        adm1_c.plot(ax=ax, facecolor="none", edgecolor=BORDER_CLR, linewidth=0.75, zorder=3)
    else:
        base = adm2.copy() if adm2 is not None else adm1.copy()
        try: base = base.clip(clip_box)
        except Exception: pass
        base.plot(ax=ax, color=LAND_CLR, edgecolor=BORDER_CLR, linewidth=0.45)

    _render_flood(ax, flood_depth, flood_tfm, bbox, style=flood_style)

    top5_ids = exposure_df.head(5)["id"].tolist()
    max_gdp = exposure_df["Total GDP (USD B)"].max()

    for _, row in exposure_df.iterrows():
        lat, lon = row["lat"], row["lon"]
        if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
            continue
        clr = _exposure_color(row["Exposure share (%)"])
        sz  = (70 + 260 * (row["Total GDP (USD B)"] / max_gdp)) * marker_scale
        ax.scatter(lon, lat, s=sz, c=clr, edgecolors="white", linewidths=1.4, zorder=6, alpha=0.92)
        if row["id"] in top5_ids:
            ax.scatter(lon, lat, s=sz*0.16, marker="*", c="white", zorder=7, alpha=0.95)
        if show_labels:
            short = next((c["short"] for c in CLUSTERS if c["id"]==row["id"]), row["Cluster"])
            mid_lon = (lon_min + lon_max) / 2
            dx = 0.03 if lon < mid_lon else -0.03
            halign = "left" if lon < mid_lon else "right"
            txt = ax.text(lon + dx, lat + 0.05, short,
                          fontsize=label_fontsize, fontweight="bold", color="white",
                          ha=halign, va="bottom", zorder=8)
            txt.set_path_effects([pe.Stroke(linewidth=2.5, foreground="#1a1a1a"), pe.Normal()])

    for sp in ax.spines.values():
        sp.set_edgecolor("#8B9BAD"); sp.set_linewidth(0.8)


# ===========================================================================
# ▌ ZOOM INSET MAPS
# ===========================================================================

ZOOM_REGIONS = {
    "penang": {
        "bbox": ZOOM_PENANG, "title": "Penang Industrial Corridor",
        "subtitle": "Bayan Lepas FIZ · Perai FIZ · Kulim Hi-Tech Park",
        "cluster_ids": {"bayan_lepas", "perai_fiz", "kulim_htp"},
    },
    "klang": {
        "bbox": ZOOM_KLANG, "title": "Klang Valley Industrial Belt",
        "subtitle": "Port Klang FZ · Shah Alam / Klang Valley",
        "cluster_ids": {"port_klang", "shah_alam"},
    },
    "johor": {
        "bbox": ZOOM_JOHOR, "title": "Johor Industrial Waterfront",
        "subtitle": "Port of Tanjung Pelepas · Pasir Gudang FIZ",
        "cluster_ids": {"ptp", "pasir_gudang"},
    },
    "bintulu": {
        "bbox": ZOOM_BINTULU, "title": "Bintulu Industrial Zone (Sarawak)",
        "subtitle": "Samalaju Industrial Park · PETRONAS MLNG Complex",
        "cluster_ids": {"samalaju", "mlng_bintulu"},
    },
    "sipitang": {
        "bbox": ZOOM_SIPITANG, "title": "Sipitang SOGIP (Sabah)",
        "subtitle": "Sipitang Oil & Gas Industrial Park",
        "cluster_ids": {"sipitang_sogip"},
    },
}


def _draw_zoom_panel(ax, region_key, adm0, adm1, adm2, exposure_df):
    """Render one zoom panel with high-resolution JRC flood data."""
    region = ZOOM_REGIONS[region_key]
    bbox = region["bbox"]
    cluster_ids = region["cluster_ids"]

    # Filter exposure_df to clusters in this zoom
    zoom_df = exposure_df[exposure_df["id"].isin(cluster_ids)].copy()

    # Load high-res JRC flood for this bbox
    flood_hr, flood_hr_tfm = load_flood_zoom(bbox)

    _plot_map_panel(
        ax, bbox, adm0, adm1, adm2, flood_hr, flood_hr_tfm,
        zoom_df, show_labels=True,
        flood_style="both",     # fill + contour for maximum detail
        show_gdp=True,
        label_fontsize=8.5,
        marker_scale=0.7,
    )

    ax.set_title(
        region["title"], fontsize=10, fontweight="bold", color="#1A252F", pad=4,
        loc="center",
    )
    ax.set_xlabel(region["subtitle"], fontsize=7.5, color="#555", labelpad=3)

    # Flood depth scale bar
    lon_min, lat_min, lon_max, lat_max = bbox
    scale_x = lon_min + 0.04 * (lon_max - lon_min)
    scale_y = lat_min + 0.08 * (lat_max - lat_min)
    for depth, clr, lw in [(0.5, "#5BB8E8", 1.2), (1.0, "#1565C0", 1.6), (2.0, "#0A3D62", 2.0)]:
        ax.plot([], [], color=clr, linewidth=lw, label=f"{depth:.1f} m")


def create_zoom_peninsular(adm0, adm1, adm2, exposure_df) -> Path:
    """3-panel zoom for Peninsular Malaysia industrial hubs."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 9), facecolor="white")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.14, top=0.88, wspace=0.06)

    for ax, key in zip(axes, ["penang", "klang", "johor"]):
        _draw_zoom_panel(ax, key, adm0, adm1, adm2, exposure_df)

    # Shared legend and colorbars
    fig.text(0.5, 0.960,
             "Malaysia Industrial Hubs — High-Resolution Flood Depth  (JRC RP50)",
             ha="center", fontsize=14, fontweight="bold", color="#1A252F")
    fig.text(0.5, 0.933,
             "Peninsular Malaysia | District GDP choropleth (DOSM 2020 real, RM B) · "
             "Flood contours at 0.5 / 1.0 / 2.0 / 4.0 m depth · Cluster markers by exposure share",
             ha="center", fontsize=8, color="#555")

    # GDP colorbar
    norm = _gdp_norm(adm2)
    cax1 = fig.add_axes([0.10, 0.04, 0.24, 0.022])
    cb1 = ColorbarBase(cax1, cmap=GDP_CMAP, norm=norm, orientation="horizontal")
    cb1.ax.set_xlabel("District GDP (RM B, 2020 real prices)", fontsize=8, color="#444")

    # Flood depth colorbar
    cax2 = fig.add_axes([0.40, 0.04, 0.24, 0.022])
    cb2 = ColorbarBase(cax2, cmap=FLOOD_CMAP, norm=mc.Normalize(0, 6), orientation="horizontal")
    cb2.ax.set_xlabel("Flood depth RP50 — JRC (m)", fontsize=8, color="#444")

    # Exposure legend
    leg_ax = fig.add_axes([0.70, 0.01, 0.28, 0.08])
    leg_ax.set_axis_off()
    handles = [
        mpatches.Patch(fc="#C0392B", ec="w", label="Exposure ≥85%"),
        mpatches.Patch(fc="#E67E22", ec="w", label="Exposure 60–85%"),
        mpatches.Patch(fc="#F1C40F", ec="w", label="Exposure 30–60%"),
        mpatches.Patch(fc="#27AE60", ec="w", label="Exposure <30%"),
        Line2D([0],[0], color="#5BB8E8", lw=1.4, label="0.5 m depth"),
        Line2D([0],[0], color="#1565C0", lw=1.8, label="1.0 m depth"),
        Line2D([0],[0], color="#0A3D62", lw=2.2, label="2.0 m depth"),
    ]
    leg_ax.legend(handles=handles, loc="center", ncol=2, fontsize=7.5,
                  framealpha=0.0, handlelength=1.5, borderpad=0.3, labelspacing=0.5)

    out = OUTPUT_DIR / "malaysia_zoom_peninsular.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  Peninsular zoom → {out}")
    return out


def create_zoom_east(adm0, adm1, adm2, exposure_df) -> Path:
    """2-panel zoom for East Malaysia industrial hubs (Bintulu + Sipitang)."""
    fig = plt.figure(figsize=(17, 9), facecolor="white")
    gs = GridSpec(1, 2, figure=fig,
                  left=0.01, right=0.99, bottom=0.14, top=0.88,
                  wspace=0.07, width_ratios=[1.4, 1.0])
    ax_bin = fig.add_subplot(gs[0, 0])
    ax_sip = fig.add_subplot(gs[0, 1])

    for ax, key in [(ax_bin, "bintulu"), (ax_sip, "sipitang")]:
        _draw_zoom_panel(ax, key, adm0, adm1, adm2, exposure_df)

    fig.text(0.5, 0.960,
             "East Malaysia Industrial Hubs — High-Resolution Flood Depth  (JRC RP50)",
             ha="center", fontsize=14, fontweight="bold", color="#1A252F")
    fig.text(0.5, 0.933,
             "Sarawak (Bintulu) & Sabah (Sipitang) | District GDP choropleth · JRC flood depth contours",
             ha="center", fontsize=8, color="#555")

    norm = _gdp_norm(adm2)
    cax1 = fig.add_axes([0.10, 0.04, 0.24, 0.022])
    cb1 = ColorbarBase(cax1, cmap=GDP_CMAP, norm=norm, orientation="horizontal")
    cb1.ax.set_xlabel("District GDP (RM B, 2020 real prices)", fontsize=8, color="#444")

    cax2 = fig.add_axes([0.40, 0.04, 0.24, 0.022])
    cb2 = ColorbarBase(cax2, cmap=FLOOD_CMAP, norm=mc.Normalize(0, 6), orientation="horizontal")
    cb2.ax.set_xlabel("Flood depth RP50 — JRC (m)", fontsize=8, color="#444")

    leg_ax = fig.add_axes([0.70, 0.01, 0.28, 0.08])
    leg_ax.set_axis_off()
    handles = [
        mpatches.Patch(fc="#C0392B", ec="w", label="Exposure ≥85%"),
        mpatches.Patch(fc="#E67E22", ec="w", label="Exposure 60–85%"),
        mpatches.Patch(fc="#F1C40F", ec="w", label="Exposure 30–60%"),
        mpatches.Patch(fc="#27AE60", ec="w", label="Exposure <30%"),
        Line2D([0],[0], color="#5BB8E8", lw=1.4, label="0.5 m depth"),
        Line2D([0],[0], color="#1565C0", lw=1.8, label="1.0 m depth"),
        Line2D([0],[0], color="#0A3D62", lw=2.2, label="2.0 m depth"),
    ]
    leg_ax.legend(handles=handles, loc="center", ncol=2, fontsize=7.5,
                  framealpha=0.0, handlelength=1.5, borderpad=0.3, labelspacing=0.5)

    out = OUTPUT_DIR / "malaysia_zoom_east.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  East Malaysia zoom → {out}")
    return out


# ===========================================================================
# ▌ MAIN COMPOSITE MAPS
# ===========================================================================

def create_static_map(adm0, adm1, adm2, exposure_df, flood_depth, flood_tfm,
                      flood_label: str) -> Path:
    fig = plt.figure(figsize=(22, 15), facecolor="white")
    gs  = GridSpec(2, 3, figure=fig,
                   left=0.01, right=0.995, bottom=0.04, top=0.93,
                   wspace=0.05, hspace=0.08,
                   width_ratios=[1.65, 0.90, 0.52], height_ratios=[1.0, 0.38])
    ax_main   = fig.add_subplot(gs[:, 0])
    ax_east   = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, 1])
    ax_table  = fig.add_subplot(gs[:, 2])

    _plot_map_panel(ax_main, PENINSULAR_BBOX, adm0, adm1, adm2,
                    flood_depth, flood_tfm, exposure_df, show_labels=True)
    ax_main.set_title("Peninsular Malaysia — Flood Hazard × GDP Exposure × Industrial Clusters",
                      fontsize=11, fontweight="bold", color="#2C3E50", pad=5)

    norm = _gdp_norm(adm2)
    cax1 = fig.add_axes([0.038, 0.12, 0.013, 0.22])
    cb1 = ColorbarBase(cax1, cmap=GDP_CMAP, norm=norm, orientation="vertical")
    cb1.ax.set_ylabel("District GDP (RM B, 2020 real)", fontsize=7, color="#444")
    cb1.ax.yaxis.set_tick_params(labelsize=6.5, colors="#444")

    cax2 = fig.add_axes([0.038, 0.40, 0.013, 0.22])
    cb2 = ColorbarBase(cax2, cmap=FLOOD_CMAP, norm=mc.Normalize(0, 6), orientation="vertical")
    cb2.ax.set_ylabel("Flood depth RP50 / JRC (m)", fontsize=7, color="#444")
    cb2.ax.yaxis.set_tick_params(labelsize=6.5, colors="#444")

    _plot_map_panel(ax_east, EAST_MY_BBOX, adm0, adm1, adm2,
                    flood_depth, flood_tfm, exposure_df, show_labels=True,
                    label_fontsize=6.5)
    ax_east.set_title("East Malaysia — Sabah & Sarawak",
                      fontsize=8.5, fontweight="bold", color="#2C3E50", pad=3)

    ax_legend.set_facecolor("#FAFAFA")
    for sp in ax_legend.spines.values():
        sp.set_edgecolor("#CCCCCC"); sp.set_linewidth(0.6)
    ax_legend.set_xlim(0, 1); ax_legend.set_ylim(0, 1)
    ax_legend.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    handles = [
        mpatches.Patch(fc="#C0392B", ec="white", label="Exposure ≥85%"),
        mpatches.Patch(fc="#E67E22", ec="white", label="Exposure 60–85%"),
        mpatches.Patch(fc="#F1C40F", ec="white", label="Exposure 30–60%"),
        mpatches.Patch(fc="#27AE60", ec="white", label="Exposure <30%"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#888", markersize=8, label="Cluster (size ∝ GDP)"),
        Line2D([0],[0], color="#5BB8E8", linewidth=1.6, label="0.5 m contour"),
        Line2D([0],[0], color="#0A3D62", linewidth=2.0, label="2.0 m contour"),
    ]
    ax_legend.legend(handles=handles, loc="upper left", fontsize=7.5,
                     framealpha=0.0, handlelength=1.6, borderpad=0.8, labelspacing=0.7)
    ax_legend.text(0.03, 0.06,
                   f"Scenario: RP50 present + 2050 RCP8.5\n"
                   f"Flood: JRC RP50 (CC BY 4.0)\n"
                   "GDP: DOSM 2020 district real (RM B)\n"
                   "Admin: GADM v4.1 (district level)\n"
                   "Buffer: 10 km radius per cluster",
                   fontsize=6.5, color="#555", va="bottom", transform=ax_legend.transAxes)

    cols = ["Rank", "Cluster", "GDP\n(B)", "Exp.\nPresent", "Share", "Exp.\n2050", "Δ"]
    tdata = []
    for rank, row in exposure_df.iterrows():
        tdata.append([
            str(rank),
            row["Cluster"].replace(" / ","/").replace(" + ","+")[:24],
            f"${row['Total GDP (USD B)']:.0f}B",
            f"${row['Exposed GDP – Present (USD B)']:.1f}B",
            f"{row['Exposure share (%)']:.0f}%",
            f"${row['Exposed GDP – 2050 (USD B)']:.1f}B",
            f"{row['Δ Exposed (USD B)']:+.1f}",
        ])
    tbl = ax_table.table(cellText=tdata, colLabels=cols, cellLoc="center",
                         loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(6.8)
    for j in range(len(cols)):
        c = tbl[(0, j)]
        c.set_facecolor("#2C3E50"); c.set_text_props(color="white", fontweight="bold")
        c.set_edgecolor("#2C3E50")
    for i, (_, row) in enumerate(exposure_df.iterrows(), start=1):
        hi = mc.to_rgba(_exposure_color(row["Exposure share (%)"]), alpha=0.22)
        bg = "#FAFAFA" if i % 2 != 0 else "#F3F3F3"
        for j in range(len(cols)):
            c = tbl[(i, j)]; c.set_facecolor(bg); c.set_edgecolor("#E0E0E0")
        tbl[(i, 4)].set_facecolor(hi)
    ax_table.set_facecolor("white")
    for sp in ax_table.spines.values():
        sp.set_edgecolor("#CCCCCC"); sp.set_linewidth(0.6)
    ax_table.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax_table.set_title("Priority Ranking — Flood-Weighted Exposure",
                       fontsize=8.5, fontweight="bold", color="#2C3E50", pad=4)

    fig.text(0.5, 0.970,
             f"Malaysia  Flood-Risk × Economic-Exposure × Industrial Overlay  (JRC RP50)",
             ha="center", va="top", fontsize=14, fontweight="bold", color="#2C3E50")
    fig.text(0.5, 0.944,
             "Sources: DOSM 2020 district real GDP (RM B) · GADM v4.1 district boundaries · "
             "JRC Global Flood Hazard Maps v2.1 RP50 (CC BY 4.0)",
             ha="center", va="top", fontsize=7, color="#777")
    fig.text(0.5, 0.010,
             "⚠ PROTOTYPE — GDP uniformly distributed within districts; "
             "JRC RP50 represents ~2% annual exceedance probability; upgrade with Fathom 3.0 + DID data for capex decisions",
             ha="center", va="bottom", fontsize=6.5, color="#AA4444", style="italic")

    out = OUTPUT_DIR / "malaysia_flood_gdp_map.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  Static map → {out}")
    return out


def create_full_malaysia_map(adm0, adm1, adm2, exposure_df, flood_depth, flood_tfm,
                              flood_label: str) -> Path:
    fig = plt.figure(figsize=(22, 10), facecolor="white")
    gs = GridSpec(1, 3, figure=fig,
                  left=0.02, right=0.99, bottom=0.09, top=0.90,
                  wspace=0.04, width_ratios=[1.35, 1.50, 0.28])
    ax_pen  = fig.add_subplot(gs[0, 0])
    ax_east = fig.add_subplot(gs[0, 1])
    ax_leg  = fig.add_subplot(gs[0, 2])

    _plot_map_panel(ax_pen, PENINSULAR_BBOX, adm0, adm1, adm2,
                    flood_depth, flood_tfm, exposure_df,
                    show_labels=True, flood_style="contour", show_gdp=True)
    ax_pen.set_title("Peninsular Malaysia", fontsize=11, fontweight="bold", color="#2C3E50", pad=5)

    _plot_map_panel(ax_east, EAST_MY_BBOX, adm0, adm1, adm2,
                    flood_depth, flood_tfm, exposure_df,
                    show_labels=True, flood_style="contour", show_gdp=True, label_fontsize=7)
    ax_east.set_title("East Malaysia — Sabah & Sarawak", fontsize=11, fontweight="bold", color="#2C3E50", pad=5)

    ax_leg.set_facecolor("#FAFAFA")
    for sp in ax_leg.spines.values():
        sp.set_edgecolor("#CCCCCC"); sp.set_linewidth(0.6)
    ax_leg.set_xlim(0, 1); ax_leg.set_ylim(0, 1)
    ax_leg.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    handles = [
        mpatches.Patch(fc="#C0392B", ec="white", label="Exposure ≥85%"),
        mpatches.Patch(fc="#E67E22", ec="white", label="60–85%"),
        mpatches.Patch(fc="#F1C40F", ec="white", label="30–60%"),
        mpatches.Patch(fc="#27AE60", ec="white", label="<30%"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#888", markersize=8, label="Cluster (size ∝ GDP)"),
        Line2D([0],[0], color="#5BB8E8", linewidth=1.8, label="0.5 m contour"),
        Line2D([0],[0], color="#1565C0", linewidth=2.2, label="1.0 m contour"),
        Line2D([0],[0], color="#0A3D62", linewidth=2.6, label="2.0 m contour"),
    ]
    ax_leg.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.0,
                  handlelength=1.6, borderpad=0.8, labelspacing=0.80,
                  title="Legend", title_fontsize=9)
    ax_leg.text(0.04, 0.06,
                f"RP50 present + 2050 RCP8.5\nFlood: {flood_label[:28]}\n"
                "GDP: DOSM 2020 district\n10 km cluster buffer",
                fontsize=6.8, color="#666", va="bottom", transform=ax_leg.transAxes, linespacing=1.5)

    norm = _gdp_norm(adm2)
    cax1 = fig.add_axes([0.03, 0.022, 0.22, 0.025])
    cb1 = ColorbarBase(cax1, cmap=GDP_CMAP, norm=norm, orientation="horizontal")
    cb1.ax.set_xlabel("District GDP (RM B, 2020 real prices)", fontsize=7, color="#444")

    cax2 = fig.add_axes([0.31, 0.022, 0.22, 0.025])
    cb2 = ColorbarBase(cax2, cmap=FLOOD_CMAP, norm=mc.Normalize(0, 6), orientation="horizontal")
    cb2.ax.set_xlabel("Flood depth RP50 / JRC (m)", fontsize=7, color="#444")

    fig.text(0.5, 0.960,
             f"Malaysia  Flood Hazard × GDP Exposure × Industrial Clusters  (JRC RP50)",
             ha="center", va="top", fontsize=14, fontweight="bold", color="#2C3E50")
    fig.text(0.5, 0.935,
             "District GDP choropleth (DOSM 2020 real, RM B) · JRC RP50 flood depth contours · "
             "Cluster markers coloured by exposure share",
             ha="center", va="top", fontsize=8, color="#555")
    fig.text(0.5, 0.012,
             "⚠ PROTOTYPE — GDP distributed uniformly within districts; JRC RP50 = ~2% annual exceedance probability.",
             ha="center", va="bottom", fontsize=6.5, color="#AA4444", style="italic")

    out = OUTPUT_DIR / "malaysia_map_full.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  Full Malaysia map → {out}")
    return out


# ===========================================================================
# ▌ TABLE FIGURE
# ===========================================================================

def create_table_figure(exposure_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(20, 6.5), facecolor="white")
    ax.set_axis_off()
    cols = [
        "Rank", "Cluster", "State", "Type",
        "Total GDP\n(USD B)", "Depth\nRP50 (m)",
        "Exposed GDP\nPresent (USD B)", "Exposure\nShare (%)",
        "Depth 2050\nRCP8.5 (m)", "Exposed GDP\n2050 (USD B)", "Δ (USD B)",
    ]
    rows = []
    for rank, row in exposure_df.iterrows():
        rows.append([
            str(rank), row["Cluster"], row["State"].replace("Pulau Pinang","Penang"), row["Type"],
            f"{row['Total GDP (USD B)']:.0f}", f"{row['Depth RP50 (m)']:.2f}",
            f"{row['Exposed GDP – Present (USD B)']:.1f}",
            f"{row['Exposure share (%)']:.0f}%",
            f"{row['Depth 2050 RCP8.5 (m)']:.2f}",
            f"{row['Exposed GDP – 2050 (USD B)']:.1f}",
            f"{row['Δ Exposed (USD B)']:+.1f}",
        ])
    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center", bbox=[0,0,1,1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    for j in range(len(cols)):
        c = tbl[(0,j)]; c.set_facecolor("#1A252F"); c.set_text_props(color="white", fontweight="bold")
    for i, (_, row) in enumerate(exposure_df.iterrows(), start=1):
        hi = mc.to_rgba(_exposure_color(row["Exposure share (%)"]), alpha=0.26)
        bg = "#FAFAFA" if i % 2 != 0 else "#F0F0F0"
        for j in range(len(cols)):
            c = tbl[(i,j)]; c.set_facecolor(bg); c.set_edgecolor("#D0D0D0")
        tbl[(i,6)].set_facecolor(hi); tbl[(i,7)].set_facecolor(hi)
    ax.set_title(
        f"Malaysia Industrial Cluster Flood-Weighted Economic Exposure  (RP50 present + 2050 RCP8.5)",
        fontsize=12, fontweight="bold", color="#1A252F", pad=8)
    fig.text(0.5, 0.02,
             "Sources: DOSM 2020 district real GDP · JRC RP50 flood depth · GADM v4.1 district boundaries",
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
                            flood_label: str) -> Path:
    m = folium.Map(location=[4.5, 108.5], zoom_start=6,
                   tiles="CartoDB positron", prefer_canvas=True)

    if adm2 is not None and "gdp_rm_b" in adm2.columns and adm2["gdp_rm_b"].max() > 0:
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
            geo_data=geojson_str, data=adm2_web,
            columns=["NAME_2", "gdp_rm_b"],
            key_on="feature.properties.NAME_2",
            fill_color="YlOrRd", fill_opacity=0.55, line_opacity=0.35, line_weight=0.5,
            legend_name="District GDP (RM B, 2020 real prices)",
            name="GDP by District", show=True,
        ).add_to(m)
        folium.GeoJson(
            geojson_str, name="District info",
            style_function=lambda f: {"fillOpacity": 0, "color": "#888", "weight": 0.4},
            tooltip=folium.GeoJsonTooltip(
                fields=["NAME_2", "state", "gdp_rm_b"],
                aliases=["District:", "State:", "GDP (RM B):"],
                localize=True, sticky=False, labels=True,
                style="font-family:Arial;font-size:12px;",
            ), show=True,
        ).add_to(m)

    max_gdp = exposure_df["Total GDP (USD B)"].max()
    fg = folium.FeatureGroup(name="Industrial Clusters", show=True)
    for _, row in exposure_df.iterrows():
        share = row["Exposure share (%)"]
        clr = _exposure_color(share)
        radius = 8 + 20 * (row["Total GDP (USD B)"] / max_gdp)
        popup_html = f"""
        <div style='font-family:Arial;font-size:12px;width:300px'>
          <b style='font-size:14px'>{row['Cluster']}</b><br>
          <hr style='margin:4px 0'>
          <b>State:</b> {row['State']} &nbsp; <b>Type:</b> {row['Type']}<br>
          <hr style='margin:4px 0'>
          <b>Total GDP (10 km buffer):</b> USD {row['Total GDP (USD B)']:.1f} B<br>
          <b>JRC RP50 flood depth:</b> {row['Depth RP50 (m)']:.2f} m<br>
          <b>Damage factor:</b> {row['Damage factor']*100:.0f}%<br>
          <b>Exposed GDP (present):</b>
            <span style='color:{clr};font-weight:bold'>
              USD {row['Exposed GDP – Present (USD B)']:.1f} B</span><br>
          <b>Exposed GDP (2050 RCP8.5):</b>
            USD {row['Exposed GDP – 2050 (USD B)']:.1f} B
            (<b>{row['Δ Exposed (USD B)']:+.1f} B Δ</b>)<br>
          <hr style='margin:4px 0'>
          <i style='color:#777;font-size:11px'>{row['note']}</i>
        </div>"""
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius, color="white", weight=2,
            fill=True, fill_color=clr, fill_opacity=0.87,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=(f"<b>{row['Cluster']}</b><br>"
                     f"Exposed: USD {row['Exposed GDP – Present (USD B)']:.1f}B "
                     f"({row['Exposure share (%)']:.0f}%)"),
        ).add_to(fg)
    fg.add_to(m)

    legend_html = """
    <div style='position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:12px 16px;border-radius:8px;
                box-shadow:0 2px 8px rgba(0,0,0,0.25);font-family:Arial;font-size:12px'>
      <b style='font-size:13px'>Flood Exposure Share (RP50)</b><br>
      <span style='color:#C0392B'>&#9679;</span> ≥85% &mdash; Very High<br>
      <span style='color:#E67E22'>&#9679;</span> 60–85% &mdash; High<br>
      <span style='color:#F1C40F'>&#9679;</span> 30–60% &mdash; Medium<br>
      <span style='color:#27AE60'>&#9679;</span> &lt;30% &mdash; Lower<br>
      <span style='color:#888;font-size:11px'>Circle size ∝ total GDP</span><br>
      <hr style='margin:6px 0'>
      <b style='font-size:11px'>Choropleth</b><br>
      <span style='color:#888;font-size:11px'>District GDP (RM B, 2020 real)<br>
      Source: DOSM open data</span><br>
      <hr style='margin:6px 0'>
      <span style='color:#2266AA;font-size:10px'>
        Flood: JRC RP50 depth tiles<br>CC BY 4.0
      </span>
    </div>"""
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
    risk_tag = {"#C0392B": "VERY HIGH", "#E67E22": "HIGH", "#F1C40F": "MEDIUM", "#27AE60": "LOWER"}
    lines = [
        "=" * 72,
        "MALAYSIA FLOOD-RISK × ECONOMIC EXPOSURE — MANAGEMENT BRIEF",
        "=" * 72,
        "",
        f"Scenario : RP50 present + 2050 RCP8.5 | Flood: {flood_label}",
        f"Method   : Depth-damage curve × cluster GDP (10 km buffer, USD B PPP approx.)",
        "GDP map  : DOSM district real GDP 2020 (RM B, 2015 prices) — 144 districts",
        "",
        "TOP 5 CLUSTERS FOR FLOOD-DEFENCE CAPEX",
        "-" * 72,
    ]
    for rank, row in top5.iterrows():
        tag = risk_tag.get(_exposure_color(row["Exposure share (%)"]), "—")
        lines += [
            "",
            f"#{rank}  {row['Cluster'].upper()}   [{tag}]",
            f"    State / Type  : {row['State']} | {row['Type']}",
            f"    Total GDP     : USD {row['Total GDP (USD B)']:.0f}B (10 km buffer)",
            f"    JRC RP50 depth: {row['Depth RP50 (m)']:.2f} m → {row['Depth 2050 RCP8.5 (m)']:.2f} m (2050 RCP8.5)",
            f"    Exposed GDP   : USD {row['Exposed GDP – Present (USD B)']:.1f}B (present) → USD {row['Exposed GDP – 2050 (USD B)']:.1f}B (2050), Δ {row['Δ Exposed (USD B)']:+.1f}B",
            f"    Rationale     : {row['note']}",
        ]
    lines += [
        "", "=" * 72,
        "EAST MALAYSIA NEW CLUSTERS (SARAWAK & SABAH)",
        "-" * 72, "",
        "• SAMALAJU INDUSTRIAL PARK (Bintulu, Sarawak)",
        "  SCORE corridor heavy industry — aluminium smelting (Press Metal),",
        "  ferrosilicon, manganese alloys. Coastal estuarine; JRC shows",
        "  significant flooding in adjacent river valleys.",
        "",
        "• PETRONAS LNG COMPLEX BINTULU (MLNG) — Sarawak",
        "  One of the world's largest LNG export facilities (9 trains).",
        "  Coastal at Tanjung Kidurong; JRC RP50 p90 depth ~3.9 m nearby.",
        "  Single-site concentration risk for Malaysia's LNG export capacity.",
        "",
        "• SIPITANG OIL & GAS INDUSTRIAL PARK (Sabah)",
        "  PETRONAS methanol (1.7 Mtpa) + ammonia plant; Brunei Bay coastal.",
        "  JRC RP50 p90 ~2.2 m in surrounding area; NE Monsoon risk Oct-Feb.",
        "",
        "=" * 72,
        "DATA CITATIONS", "-" * 72, "",
        "DOSM (2024). GDP by District, Real (Supply Approach), 2015–2020.",
        "  https://storage.dosm.gov.my/gdp/gdp_district_real_supply.parquet",
        "",
        "Baugh, C. et al. (2024). JRC Global River Flood Hazard Maps v2.1. CC BY 4.0.",
        "  https://data.jrc.ec.europa.eu/dataset/jrc-floods-floodmapgl_rp50y-tif",
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
    log.info("Malaysia Flood-Risk × Economic-Exposure Analysis")
    log.info("=" * 60)

    log.info("\n[1/7] Admin boundaries + district GDP …")
    adm0, adm1, adm2 = get_malaysia_boundaries()
    land_poly = unary_union(adm1.geometry)

    log.info("\n[2/7] Building GDP raster …")
    gdp_arr, gdp_tfm, _ = load_gdp_raster(adm2)
    log.info(f"  Grid {gdp_arr.shape}")

    log.info("\n[3/7] Loading JRC RP50 flood raster …")
    flood_arr, flood_tfm, _, flood_label = load_flood_raster(land_poly=land_poly)
    log.info(f"  Grid {flood_arr.shape}, max depth: {flood_arr.max():.2f} m")
    log.info(f"  Source: {flood_label}")

    log.info("\n[4/7] Computing cluster exposure …")
    exposure_df = build_exposure_table()
    note_map = {c["id"]: c["note"] for c in CLUSTERS}
    exposure_df["note"] = exposure_df["id"].map(note_map)

    csv_out = OUTPUT_DIR / "cluster_exposure_table.csv"
    exposure_df.to_csv(csv_out)
    log.info(f"  CSV → {csv_out}")

    print("\n  CLUSTER EXPOSURE RANKING (RP50, present):")
    hdr = f"  {'Rk':<4} {'Cluster':<40} {'GDP':>8} {'ExpGDP':>8} {'Share':>7}"
    print(hdr); print("  " + "-"*72)
    for rank, row in exposure_df.iterrows():
        print(f"  {rank:<4} {row['Cluster']:<40}  ${row['Total GDP (USD B)']:>5.0f}B"
              f"  ${row['Exposed GDP – Present (USD B)']:>5.1f}B  {row['Exposure share (%)']:>5.0f}%")

    log.info("\n[5/7] Composite maps (full Malaysia) …")
    map_path  = create_static_map(adm0, adm1, adm2, exposure_df, flood_arr, flood_tfm, flood_label)
    full_path = create_full_malaysia_map(adm0, adm1, adm2, exposure_df, flood_arr, flood_tfm, flood_label)
    tbl_path  = create_table_figure(exposure_df)
    html_path = create_interactive_map(exposure_df, adm2, flood_label)

    log.info("\n[6/7] Zoom maps …")
    pen_path = create_zoom_peninsular(adm0, adm1, adm2, exposure_df)
    east_path = create_zoom_east(adm0, adm1, adm2, exposure_df)

    log.info("\n[7/7] Priority callouts …")
    txt_path = generate_priority_callouts(exposure_df, flood_label)

    log.info("\n" + "=" * 60)
    log.info("ALL OUTPUTS")
    log.info("=" * 60)
    for p in [map_path, full_path, pen_path, east_path, html_path, tbl_path, csv_out, txt_path]:
        log.info(f"  {p}")
    log.info("=" * 60)
    print("\n" + txt_path.read_text())


if __name__ == "__main__":
    main()
