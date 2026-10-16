"""NumPy implementation of the to7 hydrological submodel.

This module leaves ``River_management_xls_to7.py`` as the reference PySD
implementation. It reproduces the daily Euler updates for the five hydrology
stocks used by that model:

* upstream surface storage
* upstream underground storage
* dam storage
* downstream surface storage
* downstream underground storage

The default driver construction represents the no-additional-policy base case.
Time-varying capacity arrays can be supplied through ``drivers`` so the module
can later be connected to the adaptation optimisation without changing the
hydrology equations again.
"""

from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


CALENDAR_START_YEAR = 2009
CALENDAR_NUM_YEARS = 15
TIME_STEP = 1.0


def _is_leap_year(year):
    return (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0)


YEAR_LENGTHS = [
    366 if _is_leap_year(CALENDAR_START_YEAR + offset) else 365
    for offset in range(CALENDAR_NUM_YEARS)
]
TOTAL_DAYS = int(np.sum(YEAR_LENGTHS))


DEFAULT_PARAMETERS = {
    "daily_precipitation_future_ratio": 1.0,
    "temperature_scenario_shift": 0.0,
    "upstream_area": 157_585.0,
    "downstream_area": 143_951.0,
    "forest_area_ratio": 0.92,
    "unmanaged_plantation_forest_coef": 0.7,
    "waterholding_capacity_of_forest_base": 225.0,
    "paddy_field_ratio": 0.15,
    "initial_paddy_dam_ratio": 0.1,
    "paddy_field_capacity_per_area": 500.0,
    "paddy_dam_capacity_per_area": 1_500.0,
    "downstream_storage_depth": 0.05,
    "innundation_risky_area_ratio": 0.8,
    "initial_drainage_capacity": 50.0,
    "initial_dam_capacity": 74_200_000.0,
    "predischarge_capacity_ratio": 0.10,
    "upstream_outflow_ratio": 0.272198,
    "downstream_outflow_ratio": 0.240634,
    "direct_discharge_ratio": 0.815126,
    "upstream_percolation_ratio": 0.5,
    "downstream_deep_percolation_ratio": 0.0263544,
    "upstream_middle_flow_ratio": 0.443267,
    "downstream_percolation_ratio": 0.5,
    "downstream_middle_flow_ratio": 0.0326612,
    "upstream_deep_percolation_ratio": 0.540584,
}


HYDROLOGY_OUTPUT_COLUMNS = [
    "daily_precip_up",
    "daily_precip_down",
    "scenario_adjusted_daily_precip_up",
    "upstream_inflow",
    "evaporation_ratio_up",
    "evaporation_upstream",
    "upstream_storage",
    "upstream_storage_capacity",
    "upstream_percolation",
    "excessive_surface_flow",
    "upstream_outflow",
    "upstream_underground",
    "middle_flow",
    "upstream_deep_percolation",
    "dam_capacity",
    "dam_storage",
    "dam_inflow",
    "predischarge_capacity",
    "predischarge_control",
    "dam_spill",
    "dam_outflow",
    "river_discharge_upstream",
    "downstream_inflow",
    "evaporation_ratio_down",
    "evaporation_downstream",
    "downstream_storage",
    "downstream_storage_capacity",
    "paddyfield_storage_capacity",
    "downstream_percolation",
    "downstream_outflow",
    "drainage_capacity",
    "drainage",
    "downstream_underground",
    "downstream_middle_flow",
    "downstream_deep_percolation",
    "river_discharge_downstream",
    "inside_water_innundation_level",
    "innundation_level",
]

HYDROLOGY_STATE_NAMES = (
    "upstream_storage",
    "upstream_underground",
    "dam_storage",
    "downstream_storage",
    "downstream_underground",
)


def _filled_values(frame, time_column, value_column, target_time):
    source_time = pd.to_numeric(frame.iloc[:, time_column], errors="coerce")
    source_value = pd.to_numeric(frame.iloc[:, value_column], errors="coerce")
    valid_time = source_time.notna()
    x = source_time.loc[valid_time].to_numpy(dtype=float)
    y = source_value.loc[valid_time].to_numpy(dtype=float)
    valid_value = np.isfinite(y)
    if not valid_value.any():
        raise ValueError(f"No valid weather values in column {value_column}.")
    # PySD ExtData uses np.interp for missing values and for daily lookup.
    return np.interp(target_time, x[valid_value], y[valid_value])


@lru_cache(maxsize=4)
def _load_weather_cached(model_root, number_of_days):
    root = Path(model_root)
    target_time = np.arange(number_of_days, dtype=float)
    locations = {
        "up": root / "data" / "jma_hita_2009_2023.xlsx",
        "down": root / "data" / "jma_asakura_2009_2023.xlsx",
    }
    weather = {}
    for suffix, path in locations.items():
        frame = pd.read_excel(path, sheet_name="input", header=None, usecols="A:F")
        weather[f"daily_precip_{suffix}"] = _filled_values(
            frame, 0, 1, target_time
        )
        weather[f"daily_ave_temp_{suffix}"] = _filled_values(
            frame, 0, 2, target_time
        )
        weather[f"daily_max_temp_{suffix}"] = _filled_values(
            frame, 0, 3, target_time
        )
        weather[f"daily_min_temp_{suffix}"] = _filled_values(
            frame, 0, 4, target_time
        )
        weather[f"solar_radiation_{suffix}"] = _filled_values(
            frame, 0, 5, target_time
        )
    return weather


def load_weather(model_root=None, number_of_days=TOTAL_DAYS):
    """Load and interpolate the same Hita/Asakura inputs used by PySD."""
    root = Path(model_root or Path(__file__).parent).resolve()
    cached = _load_weather_cached(str(root), int(number_of_days))
    return {name: values.copy() for name, values in cached.items()}


def _series_driver(name, drivers, default, number_of_days):
    value = drivers.get(name, default)
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(number_of_days, float(array), dtype=float)
    if array.shape != (number_of_days,):
        raise ValueError(
            f"Driver '{name}' has shape {array.shape}; expected "
            f"({number_of_days},)."
        )
    return array.copy()


def build_base_capacity_drivers(parameters, number_of_days):
    """Build no-policy capacity series from the current to7 assumptions."""
    upstream_area = parameters["upstream_area"]
    downstream_area = parameters["downstream_area"]
    forest_area = upstream_area * parameters["forest_area_ratio"]
    forest_function_coef = (
        0.4 + 0.3 + 0.3 * parameters["unmanaged_plantation_forest_coef"]
    )
    forest_capacity = (
        forest_area
        * parameters["waterholding_capacity_of_forest_base"]
        * forest_function_coef
        * 10.0
    )

    physical_paddy_field = downstream_area * parameters["paddy_field_ratio"]
    initial_paddy_dam_area = (
        physical_paddy_field * parameters["initial_paddy_dam_ratio"]
    )
    paddy_capacity = (
        initial_paddy_dam_area * parameters["paddy_dam_capacity_per_area"]
        + physical_paddy_field * parameters["paddy_field_capacity_per_area"]
    )

    return {
        "upstream_storage_capacity": np.full(number_of_days, forest_capacity),
        "paddyfield_storage_capacity": np.full(number_of_days, paddy_capacity),
        "drainage_capacity": np.full(
            number_of_days, parameters["initial_drainage_capacity"]
        ),
        "dam_capacity": np.full(
            number_of_days, parameters["initial_dam_capacity"]
        ),
    }


def simulate_hydrology_batch(
    parameter_sets,
    weather=None,
    output_columns=None,
    model_root=None,
):
    """Run multiple independent hydrology cases in one NumPy time loop.

    The case dimension is vectorised while the daily state recurrence remains
    explicit. This is intended for LHS and optimisation workloads where
    constructing a pandas ``DataFrame`` for every candidate is unnecessary.

    Parameters
    ----------
    parameter_sets:
        Iterable of scalar parameter mappings. Missing values use
        ``DEFAULT_PARAMETERS``.
    weather:
        Shared weather input. One scenario should be evaluated per call.
    output_columns:
        Hydrology outputs to retain. Omitting it retains all 38 outputs.

    Returns
    -------
    dict[str, numpy.ndarray]
        Arrays have shape ``(number_of_cases, number_of_days)``.
    """
    cases = list(parameter_sets)
    if not cases:
        raise ValueError("parameter_sets must contain at least one case.")

    unknown = set().union(*(set(case) for case in cases)) - set(DEFAULT_PARAMETERS)
    if unknown:
        raise KeyError(f"Unknown NumPy hydrology parameters: {sorted(unknown)}")

    requested = list(output_columns or HYDROLOGY_OUTPUT_COLUMNS)
    unknown_outputs = set(requested) - set(HYDROLOGY_OUTPUT_COLUMNS)
    if unknown_outputs:
        raise KeyError(f"Unknown hydrology outputs: {sorted(unknown_outputs)}")

    # Small arrays pay more Python/NumPy dispatch overhead than the scalar
    # recurrence. Large LHS/DE populations benefit strongly from case-axis
    # vectorisation, while ordinary one-to-three-scenario runs do not.
    if len(cases) < 8:
        if weather is None:
            weather = load_weather(model_root=model_root)
        frames = [
            simulate_hydrology(
                weather=weather,
                params=case,
                model_root=model_root,
            )
            for case in cases
        ]
        return {
            name: np.vstack([frame[name].to_numpy(dtype=float) for frame in frames])
            for name in requested
        }

    if weather is None:
        weather = load_weather(model_root=model_root)
    required_weather = {
        f"{variable}_{suffix}"
        for suffix in ("up", "down")
        for variable in (
            "daily_precip",
            "daily_ave_temp",
            "daily_max_temp",
            "daily_min_temp",
            "solar_radiation",
        )
    }
    missing_weather = required_weather - set(weather)
    if missing_weather:
        raise KeyError(f"Missing weather inputs: {sorted(missing_weather)}")

    number_of_days = len(np.asarray(weather["daily_precip_up"]))
    weather_arrays = {}
    for name in required_weather:
        values = np.asarray(weather[name], dtype=float)
        if values.shape != (number_of_days,):
            raise ValueError(
                f"Weather input '{name}' has shape {values.shape}; expected "
                f"({number_of_days},)."
            )
        weather_arrays[name] = values

    parameters = {
        name: np.asarray(
            [case.get(name, default) for case in cases], dtype=float
        )
        for name, default in DEFAULT_PARAMETERS.items()
    }
    case_count = len(cases)
    output = {
        name: np.empty((case_count, number_of_days), dtype=float)
        for name in requested
    }
    retained = set(requested)

    forest_area = parameters["upstream_area"] * parameters["forest_area_ratio"]
    forest_function_coef = (
        0.4 + 0.3 + 0.3 * parameters["unmanaged_plantation_forest_coef"]
    )
    upstream_capacity = (
        forest_area
        * parameters["waterholding_capacity_of_forest_base"]
        * forest_function_coef
        * 10.0
    )
    physical_paddy_field = (
        parameters["downstream_area"] * parameters["paddy_field_ratio"]
    )
    paddy_capacity = (
        physical_paddy_field
        * parameters["initial_paddy_dam_ratio"]
        * parameters["paddy_dam_capacity_per_area"]
        + physical_paddy_field * parameters["paddy_field_capacity_per_area"]
    )
    drainage_capacity = parameters["initial_drainage_capacity"]
    dam_capacity = parameters["initial_dam_capacity"]
    downstream_capacity = (
        parameters["downstream_area"]
        * 10_000.0
        * parameters["downstream_storage_depth"]
        + paddy_capacity
    )

    upstream_storage = np.full(case_count, 15_000_000.0)
    upstream_underground = np.full(case_count, 135_000_000.0)
    dam_storage = dam_capacity * 0.6
    downstream_storage = np.full(case_count, 5_000_000.0)
    downstream_underground = np.full(case_count, 45_000_000.0)

    precip_up = weather_arrays["daily_precip_up"]
    precip_down = weather_arrays["daily_precip_down"]
    solar_up = weather_arrays["solar_radiation_up"]
    solar_down = weather_arrays["solar_radiation_down"]

    for day in range(number_of_days):
        precip_ratio = parameters["daily_precipitation_future_ratio"]
        temperature_shift = parameters["temperature_scenario_shift"]
        adjusted_precip_up = precip_up[day] * precip_ratio
        ave_up = weather_arrays["daily_ave_temp_up"][day] + temperature_shift
        max_up = weather_arrays["daily_max_temp_up"][day] + temperature_shift
        min_up = weather_arrays["daily_min_temp_up"][day] + temperature_shift
        ave_down = weather_arrays["daily_ave_temp_down"][day] + temperature_shift
        max_down = weather_arrays["daily_max_temp_down"][day] + temperature_shift
        min_down = weather_arrays["daily_min_temp_down"][day] + temperature_shift

        evaporation_ratio_up = (
            0.0023 * (ave_up + 17.8) * np.sqrt(max_up - min_up)
            * solar_up[day] / 2.45
        )
        evaporation_ratio_down = (
            0.0023 * (ave_down + 17.8) * np.sqrt(max_down - min_down)
            * solar_down[day] / 2.45
        )
        upstream_inflow = (
            adjusted_precip_up * parameters["upstream_area"] * 10.0
        )
        downstream_inflow = (
            precip_down[day]
            * precip_ratio
            * parameters["downstream_area"]
            * 10.0
        )
        evaporation_upstream = (
            np.minimum(
                evaporation_ratio_up * parameters["upstream_area"] * 10.0,
                upstream_storage,
            )
            * 0.01
        )
        upstream_percolation = (
            parameters["upstream_percolation_ratio"] * upstream_inflow
        )
        excessive_surface_flow = np.maximum(
            upstream_storage - upstream_capacity, 0.0
        )
        upstream_outflow = np.maximum(
            upstream_storage * parameters["upstream_outflow_ratio"]
            + excessive_surface_flow,
            1_900_000.0,
        )
        middle_flow = (
            upstream_underground * parameters["upstream_middle_flow_ratio"]
        )
        upstream_deep_percolation = (
            upstream_underground * parameters["upstream_deep_percolation_ratio"]
        )

        dam_inflow = upstream_outflow * (1.0 - parameters["direct_discharge_ratio"])
        predischarge_capacity = (
            dam_capacity * parameters["predischarge_capacity_ratio"]
        )
        storage_ratio = dam_storage / np.maximum(dam_capacity, 1e-12)
        release_factor = np.clip((storage_ratio - 0.6) / 0.3, 0.0, 1.0)
        target_release = release_factor * predischarge_capacity
        releasable_storage = np.maximum(
            dam_storage - 0.6 * dam_capacity, 0.0
        ) / TIME_STEP
        predischarge_control = np.minimum(target_release, releasable_storage)
        projected_storage = dam_storage + (
            dam_inflow - predischarge_control
        ) * TIME_STEP
        dam_spill = np.maximum(projected_storage - dam_capacity, 0.0) / TIME_STEP
        available_water = dam_storage / TIME_STEP + dam_inflow
        dam_outflow = np.minimum(
            predischarge_control + dam_spill, available_water
        )
        river_discharge_upstream = np.maximum(
            upstream_outflow * parameters["direct_discharge_ratio"]
            + dam_outflow
            + middle_flow,
            2_500_000.0,
        )

        evaporation_downstream = (
            np.minimum(
                evaporation_ratio_down * parameters["downstream_area"] * 10.0,
                downstream_storage,
            )
            * 0.01
        )
        downstream_percolation = np.maximum(
            np.minimum(
                downstream_storage * parameters["downstream_percolation_ratio"],
                downstream_storage - evaporation_downstream,
            ),
            0.0,
        )
        downstream_outflow = np.maximum(
            np.minimum(
                parameters["downstream_outflow_ratio"] * downstream_storage,
                downstream_storage
                - evaporation_downstream
                - downstream_percolation,
            ),
            600_000.0,
        )
        drainage = np.maximum(
            np.minimum(
                drainage_capacity * 3_600.0 * 24.0,
                downstream_storage
                - evaporation_downstream
                - downstream_percolation
                - downstream_outflow,
            ),
            0.0,
        )
        downstream_middle_flow = (
            downstream_underground * parameters["downstream_middle_flow_ratio"]
        )
        downstream_deep_percolation = (
            downstream_underground
            * parameters["downstream_deep_percolation_ratio"]
        )
        river_discharge_downstream = np.maximum(
            river_discharge_upstream
            + downstream_outflow
            + downstream_middle_flow,
            2_500_000.0,
        )
        inside_water_innundation_level = np.maximum(
            downstream_storage - downstream_capacity, 0.0
        )
        innundation_level = inside_water_innundation_level / (
            parameters["downstream_area"]
            * 10_000.0
            * parameters["innundation_risky_area_ratio"]
        )

        if "daily_precip_up" in retained:
            output["daily_precip_up"][:, day] = precip_up[day]
        if "daily_precip_down" in retained:
            output["daily_precip_down"][:, day] = precip_down[day]
        if "scenario_adjusted_daily_precip_up" in retained:
            output["scenario_adjusted_daily_precip_up"][:, day] = adjusted_precip_up
        if "upstream_inflow" in retained:
            output["upstream_inflow"][:, day] = upstream_inflow
        if "evaporation_ratio_up" in retained:
            output["evaporation_ratio_up"][:, day] = evaporation_ratio_up
        if "evaporation_upstream" in retained:
            output["evaporation_upstream"][:, day] = evaporation_upstream
        if "upstream_storage" in retained:
            output["upstream_storage"][:, day] = upstream_storage
        if "upstream_storage_capacity" in retained:
            output["upstream_storage_capacity"][:, day] = upstream_capacity
        if "upstream_percolation" in retained:
            output["upstream_percolation"][:, day] = upstream_percolation
        if "excessive_surface_flow" in retained:
            output["excessive_surface_flow"][:, day] = excessive_surface_flow
        if "upstream_outflow" in retained:
            output["upstream_outflow"][:, day] = upstream_outflow
        if "upstream_underground" in retained:
            output["upstream_underground"][:, day] = upstream_underground
        if "middle_flow" in retained:
            output["middle_flow"][:, day] = middle_flow
        if "upstream_deep_percolation" in retained:
            output["upstream_deep_percolation"][:, day] = upstream_deep_percolation
        if "dam_capacity" in retained:
            output["dam_capacity"][:, day] = dam_capacity
        if "dam_storage" in retained:
            output["dam_storage"][:, day] = dam_storage
        if "dam_inflow" in retained:
            output["dam_inflow"][:, day] = dam_inflow
        if "predischarge_capacity" in retained:
            output["predischarge_capacity"][:, day] = predischarge_capacity
        if "predischarge_control" in retained:
            output["predischarge_control"][:, day] = predischarge_control
        if "dam_spill" in retained:
            output["dam_spill"][:, day] = dam_spill
        if "dam_outflow" in retained:
            output["dam_outflow"][:, day] = dam_outflow
        if "river_discharge_upstream" in retained:
            output["river_discharge_upstream"][:, day] = river_discharge_upstream
        if "downstream_inflow" in retained:
            output["downstream_inflow"][:, day] = downstream_inflow
        if "evaporation_ratio_down" in retained:
            output["evaporation_ratio_down"][:, day] = evaporation_ratio_down
        if "evaporation_downstream" in retained:
            output["evaporation_downstream"][:, day] = evaporation_downstream
        if "downstream_storage" in retained:
            output["downstream_storage"][:, day] = downstream_storage
        if "downstream_storage_capacity" in retained:
            output["downstream_storage_capacity"][:, day] = downstream_capacity
        if "paddyfield_storage_capacity" in retained:
            output["paddyfield_storage_capacity"][:, day] = paddy_capacity
        if "downstream_percolation" in retained:
            output["downstream_percolation"][:, day] = downstream_percolation
        if "downstream_outflow" in retained:
            output["downstream_outflow"][:, day] = downstream_outflow
        if "drainage_capacity" in retained:
            output["drainage_capacity"][:, day] = drainage_capacity
        if "drainage" in retained:
            output["drainage"][:, day] = drainage
        if "downstream_underground" in retained:
            output["downstream_underground"][:, day] = downstream_underground
        if "downstream_middle_flow" in retained:
            output["downstream_middle_flow"][:, day] = downstream_middle_flow
        if "downstream_deep_percolation" in retained:
            output["downstream_deep_percolation"][:, day] = downstream_deep_percolation
        if "river_discharge_downstream" in retained:
            output["river_discharge_downstream"][:, day] = river_discharge_downstream
        if "inside_water_innundation_level" in retained:
            output["inside_water_innundation_level"][:, day] = (
                inside_water_innundation_level
            )
        if "innundation_level" in retained:
            output["innundation_level"][:, day] = innundation_level

        if day < number_of_days - 1:
            upstream_storage = upstream_storage + (
                upstream_inflow
                - evaporation_upstream
                - upstream_percolation
                - upstream_outflow
            ) * TIME_STEP
            upstream_underground = upstream_underground + (
                upstream_percolation - middle_flow - upstream_deep_percolation
            ) * TIME_STEP
            dam_storage = dam_storage + (dam_inflow - dam_outflow) * TIME_STEP
            downstream_storage = downstream_storage + (
                downstream_inflow
                - downstream_outflow
                - downstream_percolation
                - drainage
                - evaporation_downstream
            ) * TIME_STEP
            downstream_underground = downstream_underground + (
                downstream_percolation
                - downstream_deep_percolation
                - downstream_middle_flow
            ) * TIME_STEP

    return output


def simulate_hydrology(
    weather=None,
    params=None,
    drivers=None,
    model_root=None,
    initial_state=None,
    return_final_state=False,
):
    """Run the to7 hydrology equations with one-day explicit Euler steps.

    Parameters
    ----------
    weather:
        Mapping of the ten required weather arrays. If omitted, the existing
        Hita and Asakura workbooks are loaded.
    params:
        Scalar overrides using the PySD component names.
    drivers:
        Optional scalar or daily arrays for ``upstream_storage_capacity``,
        ``paddyfield_storage_capacity``, ``drainage_capacity`` and
        ``dam_capacity``. These are the policy-coupling boundary.
    model_root:
        Repository root used when loading weather.
    initial_state:
        Optional values of the five hydrology stocks at the first requested
        timestamp. If omitted, the original to7 initial values are used.
    return_final_state:
        If true, also return the stock values at the final output timestamp.
    """
    parameters = DEFAULT_PARAMETERS.copy()
    if params:
        unknown = set(params) - set(parameters)
        if unknown:
            raise KeyError(f"Unknown NumPy hydrology parameters: {sorted(unknown)}")
        parameters.update(params)

    if weather is None:
        weather = load_weather(model_root=model_root)
    required_weather = {
        f"{variable}_{suffix}"
        for suffix in ("up", "down")
        for variable in (
            "daily_precip",
            "daily_ave_temp",
            "daily_max_temp",
            "daily_min_temp",
            "solar_radiation",
        )
    }
    missing_weather = required_weather - set(weather)
    if missing_weather:
        raise KeyError(f"Missing weather inputs: {sorted(missing_weather)}")

    number_of_days = len(np.asarray(weather["daily_precip_up"]))
    if number_of_days == 0:
        raise ValueError("Weather arrays must not be empty.")
    weather_arrays = {}
    for name in required_weather:
        values = np.asarray(weather[name], dtype=float)
        if values.shape != (number_of_days,):
            raise ValueError(
                f"Weather input '{name}' has shape {values.shape}; expected "
                f"({number_of_days},)."
            )
        weather_arrays[name] = values

    capacity_drivers = build_base_capacity_drivers(parameters, number_of_days)
    if drivers:
        capacity_drivers.update(drivers)
    upstream_capacity = _series_driver(
        "upstream_storage_capacity",
        capacity_drivers,
        capacity_drivers["upstream_storage_capacity"],
        number_of_days,
    )
    paddy_capacity = _series_driver(
        "paddyfield_storage_capacity",
        capacity_drivers,
        capacity_drivers["paddyfield_storage_capacity"],
        number_of_days,
    )
    drainage_capacity = _series_driver(
        "drainage_capacity",
        capacity_drivers,
        capacity_drivers["drainage_capacity"],
        number_of_days,
    )
    dam_capacity = _series_driver(
        "dam_capacity",
        capacity_drivers,
        capacity_drivers["dam_capacity"],
        number_of_days,
    )

    output = {
        name: np.empty(number_of_days, dtype=float)
        for name in HYDROLOGY_OUTPUT_COLUMNS
    }
    output["upstream_storage_capacity"][:] = upstream_capacity
    output["paddyfield_storage_capacity"][:] = paddy_capacity
    output["downstream_storage_capacity"][:] = (
        parameters["downstream_area"]
        * 10_000.0
        * parameters["downstream_storage_depth"]
        + paddy_capacity
    )
    output["drainage_capacity"][:] = drainage_capacity
    output["dam_capacity"][:] = dam_capacity

    precip_ratio = parameters["daily_precipitation_future_ratio"]
    temperature_shift = parameters["temperature_scenario_shift"]
    precip_up = weather_arrays["daily_precip_up"]
    precip_down = weather_arrays["daily_precip_down"]
    adjusted_precip_up = precip_up * precip_ratio
    output["daily_precip_up"][:] = precip_up
    output["daily_precip_down"][:] = precip_down
    output["scenario_adjusted_daily_precip_up"][:] = adjusted_precip_up

    ave_up = weather_arrays["daily_ave_temp_up"] + temperature_shift
    max_up = weather_arrays["daily_max_temp_up"] + temperature_shift
    min_up = weather_arrays["daily_min_temp_up"] + temperature_shift
    ave_down = weather_arrays["daily_ave_temp_down"] + temperature_shift
    max_down = weather_arrays["daily_max_temp_down"] + temperature_shift
    min_down = weather_arrays["daily_min_temp_down"] + temperature_shift
    evaporation_ratio_up = (
        0.0023
        * (ave_up + 17.8)
        * np.sqrt(max_up - min_up)
        * weather_arrays["solar_radiation_up"]
        / 2.45
    )
    evaporation_ratio_down = (
        0.0023
        * (ave_down + 17.8)
        * np.sqrt(max_down - min_down)
        * weather_arrays["solar_radiation_down"]
        / 2.45
    )
    output["evaporation_ratio_up"][:] = evaporation_ratio_up
    output["evaporation_ratio_down"][:] = evaporation_ratio_down

    upstream_inflow = adjusted_precip_up * parameters["upstream_area"] * 10.0
    downstream_inflow = (
        precip_down
        * precip_ratio
        * parameters["downstream_area"]
        * 10.0
    )
    output["upstream_inflow"][:] = upstream_inflow
    output["downstream_inflow"][:] = downstream_inflow

    state = {
        "upstream_storage": 15_000_000.0,
        "upstream_underground": 135_000_000.0,
        "dam_storage": parameters["initial_dam_capacity"] * 0.6,
        "downstream_storage": 5_000_000.0,
        "downstream_underground": 45_000_000.0,
    }
    if initial_state is not None:
        missing_state = set(HYDROLOGY_STATE_NAMES) - set(initial_state)
        if missing_state:
            raise KeyError(f"Missing hydrology state values: {sorted(missing_state)}")
        state.update(
            {name: float(initial_state[name]) for name in HYDROLOGY_STATE_NAMES}
        )

    upstream_storage = state["upstream_storage"]
    upstream_underground = state["upstream_underground"]
    dam_storage = state["dam_storage"]
    downstream_storage = state["downstream_storage"]
    downstream_underground = state["downstream_underground"]

    for day in range(number_of_days):
        evaporation_upstream = (
            min(
                evaporation_ratio_up[day]
                * parameters["upstream_area"]
                * 10.0,
                upstream_storage,
            )
            * 0.01
        )
        upstream_percolation = (
            parameters["upstream_percolation_ratio"] * upstream_inflow[day]
        )
        excessive_surface_flow = max(
            upstream_storage - upstream_capacity[day], 0.0
        )
        upstream_outflow = max(
            upstream_storage * parameters["upstream_outflow_ratio"]
            + excessive_surface_flow,
            1_900_000.0,
        )
        middle_flow = (
            upstream_underground * parameters["upstream_middle_flow_ratio"]
        )
        upstream_deep_percolation = (
            upstream_underground
            * parameters["upstream_deep_percolation_ratio"]
        )

        current_dam_capacity = dam_capacity[day]
        dam_inflow = upstream_outflow * (1.0 - parameters["direct_discharge_ratio"])
        predischarge_capacity = (
            current_dam_capacity * parameters["predischarge_capacity_ratio"]
        )
        storage_ratio = dam_storage / max(current_dam_capacity, 1e-12)
        release_factor = np.clip((storage_ratio - 0.60) / 0.30, 0.0, 1.0)
        target_release = release_factor * predischarge_capacity
        releasable_storage = max(
            dam_storage - 0.60 * current_dam_capacity, 0.0
        ) / TIME_STEP
        predischarge_control = min(target_release, releasable_storage)
        projected_storage = dam_storage + (
            dam_inflow - predischarge_control
        ) * TIME_STEP
        dam_spill = max(projected_storage - current_dam_capacity, 0.0) / TIME_STEP
        available_water = dam_storage / TIME_STEP + dam_inflow
        dam_outflow = min(predischarge_control + dam_spill, available_water)
        river_discharge_upstream = max(
            upstream_outflow * parameters["direct_discharge_ratio"]
            + dam_outflow
            + middle_flow,
            2_500_000.0,
        )

        evaporation_downstream = (
            min(
                evaporation_ratio_down[day]
                * parameters["downstream_area"]
                * 10.0,
                downstream_storage,
            )
            * 0.01
        )
        downstream_percolation = max(
            min(
                downstream_storage * parameters["downstream_percolation_ratio"],
                downstream_storage - evaporation_downstream,
            ),
            0.0,
        )
        downstream_outflow = max(
            min(
                parameters["downstream_outflow_ratio"] * downstream_storage,
                downstream_storage
                - evaporation_downstream
                - downstream_percolation,
            ),
            600_000.0,
        )
        drainage = max(
            min(
                drainage_capacity[day] * 3_600.0 * 24.0,
                downstream_storage
                - evaporation_downstream
                - downstream_percolation
                - downstream_outflow,
            ),
            0.0,
        )
        downstream_middle_flow = (
            downstream_underground * parameters["downstream_middle_flow_ratio"]
        )
        downstream_deep_percolation = (
            downstream_underground
            * parameters["downstream_deep_percolation_ratio"]
        )
        river_discharge_downstream = max(
            river_discharge_upstream
            + downstream_outflow
            + downstream_middle_flow,
            2_500_000.0,
        )
        inside_water_innundation_level = max(
            downstream_storage
            - output["downstream_storage_capacity"][day],
            0.0,
        )

        daily_values = {
            "evaporation_upstream": evaporation_upstream,
            "upstream_storage": upstream_storage,
            "upstream_percolation": upstream_percolation,
            "excessive_surface_flow": excessive_surface_flow,
            "upstream_outflow": upstream_outflow,
            "upstream_underground": upstream_underground,
            "middle_flow": middle_flow,
            "upstream_deep_percolation": upstream_deep_percolation,
            "dam_storage": dam_storage,
            "dam_inflow": dam_inflow,
            "predischarge_capacity": predischarge_capacity,
            "predischarge_control": predischarge_control,
            "dam_spill": dam_spill,
            "dam_outflow": dam_outflow,
            "river_discharge_upstream": river_discharge_upstream,
            "evaporation_downstream": evaporation_downstream,
            "downstream_storage": downstream_storage,
            "downstream_percolation": downstream_percolation,
            "downstream_outflow": downstream_outflow,
            "drainage": drainage,
            "downstream_underground": downstream_underground,
            "downstream_middle_flow": downstream_middle_flow,
            "downstream_deep_percolation": downstream_deep_percolation,
            "river_discharge_downstream": river_discharge_downstream,
            "inside_water_innundation_level": inside_water_innundation_level,
            "innundation_level": inside_water_innundation_level
            / (
                parameters["downstream_area"]
                * 10_000.0
                * parameters["innundation_risky_area_ratio"]
            ),
        }
        for name, value in daily_values.items():
            output[name][day] = value

        if day < number_of_days - 1:
            upstream_storage += (
                upstream_inflow[day]
                - evaporation_upstream
                - upstream_percolation
                - upstream_outflow
            ) * TIME_STEP
            upstream_underground += (
                upstream_percolation
                - middle_flow
                - upstream_deep_percolation
            ) * TIME_STEP
            dam_storage += (dam_inflow - dam_outflow) * TIME_STEP
            downstream_storage += (
                downstream_inflow[day]
                - downstream_outflow
                - downstream_percolation
                - drainage
                - evaporation_downstream
            ) * TIME_STEP
            downstream_underground += (
                downstream_percolation
                - downstream_deep_percolation
                - downstream_middle_flow
            ) * TIME_STEP

    result = pd.DataFrame(output, index=np.arange(number_of_days, dtype=float))
    if not return_final_state:
        return result
    final_state = {
        "upstream_storage": upstream_storage,
        "upstream_underground": upstream_underground,
        "dam_storage": dam_storage,
        "downstream_storage": downstream_storage,
        "downstream_underground": downstream_underground,
    }
    return result, final_state
