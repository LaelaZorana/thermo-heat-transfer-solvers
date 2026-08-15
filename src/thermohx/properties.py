"""Tabulated fluid properties with linear interpolation.

This is the CoolProp free fallback path. The tables in data/fluids/ hold
air at 100 kPa and saturated liquid water, with values consistent with the
appendix tables of a standard heat transfer textbook (Incropera Tables A.4
and A.6 style). Tests compare every tabulated value against CoolProp and
hold the agreement to stated tolerances, so either path gives essentially
the same answers over the tabulated range.

Columns: T_K, rho_kg_m3, cp_J_kgK, mu_Pa_s, k_W_mK, Pr.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "fluids"

TABLES = {
    "air": "air_100kPa.csv",
    "water": "water_sat_liquid.csv",
}

COLUMNS = ("rho_kg_m3", "cp_J_kgK", "mu_Pa_s", "k_W_mK", "Pr")

_cache: dict = {}


def load_table(fluid, data_dir=None):
    """Load a property table as a dict of numpy arrays keyed by column name.
    fluid is "air" or "water", or a path to a CSV with the same columns."""
    key = (str(fluid), str(data_dir))
    if key in _cache:
        return _cache[key]
    base = Path(data_dir) if data_dir is not None else DATA_DIR
    name = TABLES.get(str(fluid).lower())
    path = base / name if name else Path(fluid)
    if not path.exists():
        raise FileNotFoundError(f"no property table at {path}; known fluids: "
                                f"{sorted(TABLES)}")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    table = {c: np.array([float(r[c]) for r in rows]) for c in ("T_K",) + COLUMNS}
    if np.any(np.diff(table["T_K"]) <= 0):
        raise ValueError(f"temperature column in {path} must increase strictly")
    _cache[key] = table
    return table


def prop(fluid, T, name, data_dir=None):
    """Linearly interpolated property at temperature T in kelvin. name is one
    of rho_kg_m3, cp_J_kgK, mu_Pa_s, k_W_mK, Pr. Raises outside the table."""
    table = load_table(fluid, data_dir)
    if name not in COLUMNS:
        raise ValueError(f"unknown property {name}; choose from {COLUMNS}")
    T = np.asarray(T, dtype=float)
    Tmin, Tmax = table["T_K"][0], table["T_K"][-1]
    if np.any(T < Tmin) or np.any(T > Tmax):
        raise ValueError(f"T = {T} K outside the {fluid} table range "
                         f"[{Tmin:g}, {Tmax:g}] K")
    out = np.interp(T, table["T_K"], table[name])
    return float(out) if out.ndim == 0 else out


def properties(fluid, T, data_dir=None):
    """All tabulated properties at T as a dict."""
    return {name: prop(fluid, T, name, data_dir) for name in COLUMNS}
