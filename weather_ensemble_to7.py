"""Joint annual weather permutations for the existing to7 calendar/model.

Weather variables remain aligned across stations and within each source year.
Mapping is by month/day. Missing February 29 is the joint mean of February 28
and March 1; a source February 29 is omitted in a non-leap target year.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib

import numpy as np
import pandas as pd
import xarray as xr


START_YEAR = 2009
YEARS = tuple(range(START_YEAR, 2024))
WEATHER_COMPONENTS = tuple(
    f"{name}_{station}" for station in ("up", "down")
    for name in ("daily_precip", "daily_ave_temp", "daily_min_temp", "daily_max_temp", "solar_radiation")
)
DATES = pd.date_range("2009-01-01", "2023-12-31", freq="D")


def year_permutations(n_sequences: int, seed: int) -> list[dict]:
    if n_sequences < 1:
        raise ValueError("At least one random sequence is required.")
    rng = np.random.default_rng(seed)
    seen = {YEARS}
    rows = [{"sequence_id": "original", "source_years": list(YEARS), "is_original": True}]
    while len(rows) <= n_sequences:
        order = tuple(int(x) for x in rng.permutation(YEARS))
        if order not in seen:
            seen.add(order)
            rows.append({"sequence_id": f"perm_{len(rows):04d}", "source_years": list(order), "is_original": False})
    return rows


def extract_weather(model) -> pd.DataFrame:
    """Read PySD's already initialized/imputed external series, before climate shifts."""
    values = {}
    for component in WEATHER_COMPONENTS:
        external = getattr(model.components, f"_ext_data_{component}")
        data = external.data
        times = data.coords["time"].to_numpy()
        if not np.array_equal(times, np.arange(len(DATES))):
            raise ValueError(f"Unexpected source time coordinates: {component}")
        values[component] = data.to_numpy().reshape(-1).astype(float)
    frame = pd.DataFrame(values, index=DATES)
    validate_weather(frame)
    return frame


def validate_weather(frame: pd.DataFrame) -> None:
    if not frame.index.equals(DATES) or tuple(frame.columns) != WEATHER_COMPONENTS:
        raise ValueError("Expected the full 2009–2023 calendar and all ten joint weather variables.")
    values = frame.to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Weather contains non-finite values.")
    for station in ("up", "down"):
        if (frame[f"daily_precip_{station}"] < 0).any():
            raise ValueError("Weather precipitation must be nonnegative.")
        if (frame[f"daily_max_temp_{station}"] < frame[f"daily_min_temp_{station}"]).any():
            raise ValueError("Maximum temperature is below minimum temperature.")


def permute_weather(frame: pd.DataFrame, order: list[int]) -> tuple[pd.DataFrame, dict]:
    if len(order) != len(YEARS) or set(order) != set(YEARS):
        raise ValueError("A permutation must use each source year exactly once.")
    blocks = {year: frame.loc[frame.index.year == year] for year in YEARS}
    output = np.empty_like(frame.to_numpy(float))
    inserted = removed = 0
    offset = 0
    for target_year, source_year in zip(YEARS, order):
        target_dates = DATES[DATES.year == target_year]
        source = blocks[source_year]
        lookup = {(d.month, d.day): row for d, row in zip(source.index, source.to_numpy(float))}
        for day in target_dates:
            key = (day.month, day.day)
            if key == (2, 29) and key not in lookup:
                output[offset] = (lookup[(2, 28)] + lookup[(3, 1)]) / 2
                inserted += 1
            else:
                output[offset] = lookup[key]
            offset += 1
        if (2, 29) in lookup and (2, 29) not in {(d.month, d.day) for d in target_dates}:
            removed += 1
    reordered = pd.DataFrame(output, index=DATES, columns=frame.columns)
    audit = {"leap_days_inserted": inserted, "leap_days_removed": removed}
    for station in ("up", "down"):
        audit[f"precip_total_difference_{station}_mm"] = float(
            reordered[f"daily_precip_{station}"].sum() - frame[f"daily_precip_{station}"].sum())
    return reordered, audit


@contextmanager
def weather_override(model, frame: pd.DataFrame):
    """Temporarily replace initialized external arrays, then restore them exactly.

Public temperature components still add the scenario shift, and hydrology
still multiplies raw precipitation. Direct array replacement also avoids the
costly per-cell Series setter on every annual model.run call.
"""
    original = []
    try:
        for name in WEATHER_COMPONENTS:
            external = getattr(model.components, f"_ext_data_{name}")
            original.append((external, external.data))
            values = frame[name].to_numpy(float)
            external.data = xr.DataArray(np.append(values, values[-1]),
                                         coords={"time": np.arange(len(frame) + 1, dtype=float)}, dims="time")
        model.cache.clean()
        yield
    finally:
        for external, data in original:
            external.data = data
        model.cache.clean()


def evaluate_weather_policy(scenario: str, decision: dict, reordered: pd.DataFrame):
    core = importlib.import_module("run_vensim_with_pysd_to7_opt")
    model = core.get_model()
    with weather_override(model, reordered):
        return core.evaluate_policy(scenario, decision)
