"""Load and validate basin-specific settings shared by the current workflows."""

from __future__ import annotations

import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


BASE_DIR = Path(__file__).resolve().parent
BASIN_CONFIG_DIR = BASE_DIR / "config" / "basins"
ACTIVE_BASIN_ENV = "RIVER_BASIN"
DEFAULT_BASIN = "chikugo"

REQUIRED_INPUTS = {"weather_up", "weather_down", "flow_up", "flow_down"}
REQUIRED_CALENDAR = {"start_year", "num_years", "use_leap_years"}


def _config_path(basin: str) -> Path:
    candidate = Path(basin)
    if candidate.suffix in {".yaml", ".yml"} or candidate.parent != Path("."):
        return candidate.expanduser().resolve()
    return BASIN_CONFIG_DIR / f"{basin}.yaml"


@lru_cache(maxsize=16)
def load_basin_config(basin: str) -> dict[str, Any]:
    """Return a validated basin config by key or YAML path."""
    path = _config_path(basin)
    if not path.exists():
        raise FileNotFoundError(f"流域設定が見つかりません: {path}")

    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"流域設定のルートはmappingである必要があります: {path}")

    missing_sections = {
        "calendar",
        "inputs",
        "climate_scenarios",
        "model_parameters",
        "sensitivity_bounds",
        "hydrology_calibration",
    } - set(config)
    if missing_sections:
        raise ValueError(f"{path} に不足している設定: {sorted(missing_sections)}")

    missing_calendar = REQUIRED_CALENDAR - set(config["calendar"])
    missing_inputs = REQUIRED_INPUTS - set(config["inputs"])
    if missing_calendar or missing_inputs:
        raise ValueError(
            f"{path} の必須項目不足: calendar={sorted(missing_calendar)}, "
            f"inputs={sorted(missing_inputs)}"
        )

    model_parameters = config["model_parameters"]
    unknown_sensitivity = set(config["sensitivity_bounds"]) - set(model_parameters)
    calibration_bounds = config["hydrology_calibration"].get("bounds", {})
    unknown_calibration = set(calibration_bounds) - set(model_parameters)
    if unknown_sensitivity or unknown_calibration:
        raise ValueError(
            f"{path} にモデル変数として未定義の設定があります: "
            f"sensitivity={sorted(unknown_sensitivity)}, "
            f"calibration={sorted(unknown_calibration)}"
        )

    config = deepcopy(config)
    config["_path"] = path
    config.setdefault("key", path.stem)
    config.setdefault("name", config["key"])
    return config


def load_active_basin() -> dict[str, Any]:
    """Load the basin selected by RIVER_BASIN, defaulting to Chikugo."""
    return load_basin_config(os.environ.get(ACTIVE_BASIN_ENV, DEFAULT_BASIN))


def resolve_input_path(config: dict[str, Any], input_name: str) -> str:
    """Resolve an input path relative to the repository for PySD ExtData."""
    value = config["inputs"][input_name]
    path = Path(value)
    if path.is_absolute():
        return path.as_posix()
    return (BASE_DIR / path).as_posix()


def sensitivity_bounds(config: dict[str, Any]) -> dict[str, tuple[float, float, float]]:
    """Convert YAML [low, high, base] entries to validated tuples."""
    result = {}
    for name, values in config["sensitivity_bounds"].items():
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(f"sensitivity_bounds.{name} は [low, high, base] 形式が必要です")
        low, high, base = map(float, values)
        if not low <= base <= high:
            raise ValueError(f"sensitivity_bounds.{name}: low <= base <= high ではありません")
        configured_base = float(config["model_parameters"][name])
        if not abs(configured_base - base) <= 1e-9 * max(1.0, abs(base)):
            raise ValueError(
                f"sensitivity_bounds.{name} のbase ({base}) と "
                f"model_parameters ({configured_base}) が一致しません"
            )
        result[name] = (low, high, base)
    return result


def scenario_maps(config: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    precip = {}
    temperature = {}
    for name, values in config["climate_scenarios"].items():
        precip[name] = float(values["precip_ratio"])
        temperature[name] = float(values["temperature_shift"])
    return precip, temperature
