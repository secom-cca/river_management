"""Compatibility layer between NumPy hydrology and the remaining PySD model."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from pysd import load

from River_management_xls_to7_numpy import (
    DEFAULT_PARAMETERS,
    HYDROLOGY_OUTPUT_COLUMNS,
    TOTAL_DAYS,
    build_base_capacity_drivers,
    load_weather,
    simulate_hydrology,
)


CAPACITY_DRIVER_COLUMNS = [
    "upstream_storage_capacity",
    "paddyfield_storage_capacity",
    "drainage_capacity",
    "dam_capacity",
]

DYNAMIC_CAPACITY_PARAMETERS = {
    "annual_forest_management_conversion_area",
    "annual_paddy_dam_investment",
    "drainage_investment_amount",
    "dam_investment_amount",
}

REPLACED_HYDROLOGY_STATEFULS = {
    "_integ_upstream_storage",
    "_integ_upstream_underground",
    "_integ_dam_storage",
    "_integ_downstream_storage",
    "_integ_downstream_underground",
}

# Keep PySD's already-loaded external precipitation objects. Passing a Series
# to a Data component invokes its interpolation machinery and removes most of
# the performance benefit of precomputing hydrology.
NUMPY_INJECTED_COLUMNS = [
    name
    for name in HYDROLOGY_OUTPUT_COLUMNS
    if name not in {"daily_precip_up", "daily_precip_down"}
]


def _remove_replaced_hydrology_statefuls(model):
    """Prevent PySD from integrating stocks supplied by NumPy time series."""
    removed_objects = {
        element
        for name, element in model._stateful_elements.items()
        if name in REPLACED_HYDROLOGY_STATEFULS
    }
    model._stateful_elements = {
        name: element
        for name, element in model._stateful_elements.items()
        if name not in REPLACED_HYDROLOGY_STATEFULS
    }
    model._dynamicstateful_elements = [
        element
        for element in model._dynamicstateful_elements
        if element not in removed_objects
    ]
    model.initialize_order = [
        name
        for name in model.initialize_order
        if name not in REPLACED_HYDROLOGY_STATEFULS
    ]
    if model.stateful_initial_dependencies is not None:
        model.stateful_initial_dependencies = {
            name: dependencies
            for name, dependencies in model.stateful_initial_dependencies.items()
            if name not in REPLACED_HYDROLOGY_STATEFULS
        }


class NumpyHydrologyAdapter:
    """Expose a PySD-compatible ``run`` method backed by NumPy hydrology.

    The socioeconomic, agriculture and forest equations remain in the PySD
    model. Only the hydrological equations are replaced by daily NumPy series.
    ``capacity_mode='static'`` is intended for no-policy base and parameter
    studies. ``capacity_mode='pysd'`` evaluates the four policy-dependent
    capacity drivers with a small PySD dependency graph before running NumPy
    hydrology, which preserves the existing delay and forest-recovery logic.
    """

    def __init__(self, model_path="River_management_xls_to7.py", capacity_mode="static"):
        if capacity_mode not in {"static", "pysd"}:
            raise ValueError("capacity_mode must be 'static' or 'pysd'.")
        self.model_path = Path(model_path)
        self.capacity_mode = capacity_mode
        self.model = load(self.model_path.as_posix())
        _remove_replaced_hydrology_statefuls(self.model)
        self.driver_model = (
            load(self.model_path.as_posix()) if capacity_mode == "pysd" else None
        )
        self.weather = load_weather(
            model_root=self.model_path.resolve().parent,
            number_of_days=TOTAL_DAYS,
        )
        self.hydrology_state = None
        self.last_time = None

    @staticmethod
    def _timestamps(values):
        timestamps = np.asarray(values, dtype=float)
        if timestamps.ndim != 1 or len(timestamps) == 0:
            raise ValueError("return_timestamps must be a non-empty one-dimensional list.")
        integer_times = timestamps.astype(int)
        if not np.array_equal(timestamps, integer_times.astype(float)):
            raise ValueError("NumPy hydrology currently requires integer daily timestamps.")
        if np.any(np.diff(integer_times) != 1):
            raise ValueError("NumPy hydrology requires contiguous daily timestamps.")
        if integer_times[0] < 0 or integer_times[-1] >= TOTAL_DAYS:
            raise ValueError(
                f"Timestamps must remain within 0 through {TOTAL_DAYS - 1}."
            )
        return integer_times

    @staticmethod
    def _scalar_hydrology_parameters(params):
        hydrology_params = DEFAULT_PARAMETERS.copy()
        for name in hydrology_params:
            if name not in params:
                continue
            value = params[name]
            if isinstance(value, pd.Series) or callable(value):
                raise TypeError(
                    f"Time-varying hydrology parameter '{name}' is not supported."
                )
            hydrology_params[name] = float(value)
        return hydrology_params

    @staticmethod
    def _has_dynamic_capacity_policy(params):
        for name in DYNAMIC_CAPACITY_PARAMETERS:
            value = params.get(name, 0.0)
            if isinstance(value, pd.Series):
                if np.any(np.abs(value.to_numpy(dtype=float)) > 0):
                    return True
            elif callable(value):
                return True
            elif abs(float(value)) > 0:
                return True
        return False

    def _array_component(self, values, first_time):
        values = np.asarray(values, dtype=float)
        first_time = int(first_time)

        def value_at_model_time():
            position = int(round(float(self.model.time()))) - first_time
            return float(values[position])

        return value_at_model_time

    def _capacity_drivers(
        self, params, hydrology_params, hydrology_times, initial_condition
    ):
        if self.capacity_mode == "static":
            if self._has_dynamic_capacity_policy(params):
                raise ValueError(
                    "Dynamic dam, drainage, paddy-dam or forest-management policy "
                    "requires capacity_mode='pysd'."
                )
            return build_base_capacity_drivers(
                hydrology_params, len(hydrology_times)
            )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Replacing a variable by a constant value\.",
                category=UserWarning,
            )
            driver_result = self.driver_model.run(
                params=params,
                return_timestamps=hydrology_times.tolist(),
                return_columns=CAPACITY_DRIVER_COLUMNS,
                initial_condition=initial_condition,
            )
        return {
            name: driver_result[name].to_numpy(dtype=float)
            for name in CAPACITY_DRIVER_COLUMNS
        }

    def run(
        self,
        params=None,
        return_columns=None,
        return_timestamps=None,
        initial_condition="original",
        **kwargs,
    ):
        """Run the hybrid model using the subset of the PySD ``run`` API used here."""
        if return_timestamps is None:
            raise ValueError("return_timestamps must be supplied explicitly.")
        params = dict(params or {})
        requested_times = self._timestamps(return_timestamps)

        if initial_condition == "original":
            if requested_times[0] != 0:
                raise ValueError("An original NumPy run must start at day 0.")
            self.hydrology_state = None
            self.last_time = None
            hydrology_times = requested_times
        elif initial_condition == "current":
            if self.hydrology_state is None or self.last_time is None:
                raise RuntimeError("No previous NumPy hydrology state is available.")
            if requested_times[0] != self.last_time + 1:
                raise ValueError(
                    "A current run must start one day after the previous output."
                )
            # PySD changes annual parameters at the previous final timestamp and
            # then integrates to the first timestamp of the new annual segment.
            hydrology_times = np.concatenate(([self.last_time], requested_times))
        else:
            raise ValueError("Only 'original' and 'current' initial conditions are supported.")

        hydrology_params = self._scalar_hydrology_parameters(params)
        drivers = self._capacity_drivers(
            params, hydrology_params, hydrology_times, initial_condition
        )
        weather = {
            name: values[hydrology_times]
            for name, values in self.weather.items()
        }
        hydrology, final_state = simulate_hydrology(
            weather=weather,
            params=hydrology_params,
            drivers=drivers,
            initial_state=self.hydrology_state,
            return_final_state=True,
        )
        hydrology.index = hydrology_times.astype(float)

        hybrid_params = params.copy()
        hybrid_params.update(
            {
                name: self._array_component(
                    hydrology[name].to_numpy(dtype=float), hydrology_times[0]
                )
                for name in NUMPY_INJECTED_COLUMNS
            }
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Replacing the value of Stateful variable with an expression\..*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Replacing a variable by a constant value\.",
                category=UserWarning,
            )
            result = self.model.run(
                params=hybrid_params,
                return_columns=return_columns,
                return_timestamps=requested_times.tolist(),
                initial_condition=initial_condition,
                **kwargs,
            )

        self.hydrology_state = final_state
        self.last_time = int(requested_times[-1])
        return result
