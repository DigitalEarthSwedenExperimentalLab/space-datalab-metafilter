"""Open-Meteo Historical Archive backend for ERA5 retrieval.

This is the second supported source for the ERA5 reanalysis that
`scripts/download_era5.py` fetches from the CDS API. Both backends produce
datasets that `calculate_daily_metrics()` consumes identically, but they do
not take every variable from the same product: see "Sources" below.

Differences from the CDS path:

* **Wire format:** JSON over HTTP (no auth, no quota) versus binary NetCDF.
* **Spatial:** A single point snapped to the ~0.25° ERA5 grid centroid of the
  requested AOI, versus the full bbox raster.
* **Latency:** ~200 ms per call versus minutes/hours in the CDS queue.
* **Sources:** Open-Meteo's ERA5-Land has no precipitation, radiation or cloud
  cover (it returns null), so those come from ERA5 in a second request per
  month and the land-surface variables from ERA5-Land. The CDS backend takes
  precipitation and radiation from ERA5-Land and cloud cover from ERA5 single
  levels, so precipitation and radiation differ in source and resolution
  (ERA5, about 31 km, versus ERA5-Land, about 9 km) between the backends.
* **Cloud cover:** Written into the same NetCDF as the land variables —
  `download_period(backend="open-meteo")` returns `cloud=[]`.

The adapter writes the JSON responses to a NetCDF file with the same variable
names, dimensions, and coordinate conventions that ERA5-Land emits, so the
downstream code path is the same as in the CDS case.
"""
from __future__ import annotations

import calendar
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .config import AREA, OUTPUT_DIR
from .core import ACCUMULATION_ATTR, ACCUMULATION_HOURLY


# Open-Meteo hourly variables that map to the columns in calculate_daily_metrics.
# Requested with one call per source model; see _VAR_MAPPING.
OPEN_METEO_DEFAULT_HOURLY = [
    "temperature_2m",
    "precipitation",
    "shortwave_radiation",
    "cloud_cover",
    "cloud_cover_low",
    "soil_temperature_0_to_7cm",
    "soil_moisture_0_to_7cm",
    "snow_depth",
    "dew_point_2m",
]


# Mapping: Open-Meteo variable name → (CDS-equivalent variable name, source model,
# conversion function).
#
# Source model: without `models` Open-Meteo answers with "Best Match", a blend of
# IFS HRES, ERA5 and ERA5-Land whose composition changes across years - not one
# source to rank analog dates against. Land-surface variables come from ERA5-Land,
# the product the CDS backend retrieves. Open-Meteo's ERA5-Land returns null for
# precipitation, radiation and cloud cover, so those come from ERA5.
# The conversion function takes the raw Open-Meteo values (already a numpy array)
# and returns the array in the same units the CDS-side NetCDF would carry, so
# calculate_daily_metrics() doesn't need to know which backend produced the data.
#
# Notes on each conversion:
#   * temperature_2m: Open-Meteo gives °C, ERA5-Land gives Kelvin
#   * precipitation: Open-Meteo gives mm/h, ERA5-Land gives m/h (we want m/h)
#   * shortwave_radiation: Open-Meteo gives W/m² (instantaneous), ERA5-Land gives
#     J/m² (accumulated over the hour). For a constant-W reading, J/m² = W * 3600 s.
#   * cloud_cover, cloud_cover_low: Open-Meteo gives percentage 0–100,
#     ERA5 single-levels gives fraction 0–1.
#   * soil_moisture/snow_depth: same units (m³/m³, m respectively). Snow depth
#     is emitted as `sde`, the name CDS ERA5-Land uses, so both backends agree;
#     caches written before this carry `sd` and are still read.
#   * soil_temperature: Open-Meteo °C, ERA5-Land Kelvin.
_VAR_MAPPING = {
    "temperature_2m":             ("t2m",   "era5_land", lambda v: v + 273.15),
    "precipitation":              ("tp",    "era5",      lambda v: v / 1000.0),
    "shortwave_radiation":        ("ssrd",  "era5",      lambda v: v * 3600.0),
    "cloud_cover":                ("tcc",   "era5",      lambda v: v / 100.0),
    "cloud_cover_low":            ("lcc",   "era5",      lambda v: v / 100.0),
    "soil_temperature_0_to_7cm":  ("stl1",  "era5_land", lambda v: v + 273.15),
    "soil_moisture_0_to_7cm":     ("swvl1", "era5_land", lambda v: v),
    "snow_depth":                 ("sde",   "era5_land", lambda v: v),
    "dew_point_2m":               ("d2m",   "era5_land", lambda v: v + 273.15),
}

def _routing_identity(mapping=None):
    """Identity of a routing table, written into datasets and cache names.

    Derived from every variable's CDS name and source model, so any routing
    change gives new cache names and old caches are refetched.
    """
    rows = sorted((name, row[0], row[1]) for name, row in (mapping or _VAR_MAPPING).items())
    return "era5_land+era5-" + hashlib.sha256(repr(rows).encode()).hexdigest()[:8]


OPEN_METEO_MODEL = _routing_identity()


def _source_model(name):
    """Open-Meteo model that serves one hourly variable."""
    try:
        return _VAR_MAPPING[name][1]
    except KeyError:
        raise ValueError(f"no source model configured for Open-Meteo variable {name!r}") from None


def _snap_to_era5_grid(area, grid_deg=0.25):
    """Snap area centroid to the nearest ~0.25° cell so neighbouring AOIs share cache."""
    lat = round(((area["south"] + area["north"]) / 2) / grid_deg) * grid_deg
    lon = round(((area["west"] + area["east"]) / 2) / grid_deg) * grid_deg
    return lat, lon


def open_meteo_json_to_dataset(hourly_payload, lat, lon):
    """Convert an Open-Meteo /v1/archive 'hourly' payload to an xarray.Dataset.

    The returned dataset has the same variable names and structure as the
    NetCDF that scripts/download_era5.py produces, with a single 1×1 spatial
    grid (the snapped centroid). Downstream code consumes it identically.
    """
    times = pd.to_datetime(hourly_payload["time"])
    lats = np.array([lat], dtype=np.float32)
    lons = np.array([lon], dtype=np.float32)

    data_vars = {}
    for om_name, (cds_name, _model, convert) in _VAR_MAPPING.items():
        if om_name not in hourly_payload:
            continue
        raw = np.asarray(hourly_payload[om_name], dtype=np.float64)
        # Open-Meteo returns None for missing values; replace with NaN.
        # np.asarray on a list with None gives dtype=object, but with the
        # explicit dtype=float64 above, None becomes NaN automatically.
        converted = convert(raw).astype(np.float32)
        # Reshape to (n_time, n_lat=1, n_lon=1) to match ERA5-Land grid shape.
        reshaped = converted.reshape(-1, 1, 1)
        data_vars[cds_name] = (("time", "latitude", "longitude"), reshaped)

    return xr.Dataset(
        data_vars,
        coords={"time": times, "latitude": lats, "longitude": lons},
        # Open-Meteo hourly values are per-hour sums/means, never running
        # totals; without this tag calculate_daily_metrics would treat them
        # as CDS forecast-start accumulations.
        attrs={
            ACCUMULATION_ATTR: ACCUMULATION_HOURLY,
            "metafilter_source": "open-meteo",
            "metafilter_source_model": OPEN_METEO_MODEL,
        },
    )


def fetch_open_meteo_archive(
    year,
    month,
    *,
    area=None,
    variables=None,
    timeout=60,
):
    """Fetch a month plus the following midnight, completing its last day.

    Requests each variable from its source model (one call per model), joins
    the 'hourly' blocks on their shared time axis and returns
    ``(hourly, lat, lon)``. Pure HTTP — no NetCDF I/O — so this function is
    straightforward to mock in tests. Raises RuntimeError when a response has
    no 'hourly' block, when a requested variable is absent or entirely null,
    or when the two models return different hours; ValueError for a variable
    with no configured source model.
    """
    import requests

    area = area or AREA
    variables = variables or OPEN_METEO_DEFAULT_HOURLY

    lat, lon = _snap_to_era5_grid(area)
    last_day = calendar.monthrange(year, month)[1]
    start_date = f"{year}-{month:02d}-01"
    boundary = pd.Timestamp(year=year, month=month, day=last_day) + pd.Timedelta(days=1)
    # The API accepts dates, so request the following day and retain only
    # its 00:00 sample (the hour ending the requested month's final day).
    end_date = boundary.strftime("%Y-%m-%d")

    by_model = {}
    for name in variables:
        by_model.setdefault(_source_model(name), []).append(name)

    hourly = {}
    for model, names in by_model.items():
        response = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": f"{lat:.4f}",
                "longitude": f"{lon:.4f}",
                "start_date": start_date,
                "end_date": end_date,
                "hourly": ",".join(names),
                "timezone": "UTC",
                "models": model,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if "hourly" not in payload:
            raise RuntimeError(
                f"Open-Meteo response did not contain 'hourly' block: keys={list(payload.keys())}"
            )
        block = payload["hourly"]
        empty = [name for name in names if all(v is None for v in block.get(name, []))]
        if empty:  # an absent or all-null variable would silently fail every rule on it
            raise RuntimeError(f"Open-Meteo {model} returned no data for {empty}")
        if "time" in hourly and block["time"] != hourly["time"]:
            raise RuntimeError(f"Open-Meteo {model} hours differ from the other model's")
        hourly["time"] = block["time"]
        hourly.update({name: block[name] for name in names})
    keep = pd.to_datetime(hourly["time"]) <= boundary
    hourly = {name: [value for value, include in zip(values, keep) if include]
              for name, values in hourly.items()}
    return hourly, lat, lon


def download_open_meteo_land(
    year=2024,
    month=8,
    area=None,
    variables=None,
):
    """Fetch one month from Open-Meteo and persist as NetCDF.

    The output file has the same shape downstream code expects from a CDS
    `reanalysis-era5-land` retrieve, with the same variable names. The
    spatial grid is 1×1 (centroid only) — small AOIs see the same value
    everywhere because Open-Meteo serves one grid cell per source model
    (ERA5-Land and ERA5).

    Returns the absolute path of the written .nc file.
    """
    hourly, lat, lon = fetch_open_meteo_archive(
        year, month, area=area, variables=variables,
    )
    ds = open_meteo_json_to_dataset(hourly, lat, lon)

    out_path = Path(OUTPUT_DIR) / "era5" / f"openmeteo_land_{year}_{month:02d}.nc"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # scipy engine so we don't depend on h5netcdf/netCDF4 — same as tests.
    ds.to_netcdf(
        out_path,
        engine="scipy",
        encoding={"time": {"units": "seconds since 1970-01-01", "dtype": "float64"}},
    )
    return str(out_path)


if __name__ == "__main__":
    download_open_meteo_land()
