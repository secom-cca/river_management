"""
Python model 'River_management_xls_to3.py'
Translated using PySD
"""

from pathlib import Path
import numpy as np

from pysd.py_backend.functions import if_then_else, pulse
from pysd.py_backend.statefuls import Delay, Integ
from pysd.py_backend.external import ExtData
from pysd import Component

__pysd_version__ = "3.14.3"

__data = {"scope": None, "time": lambda: 0}

_root = Path(__file__).parent


component = Component()

#######################################################################
#                          CONTROL VARIABLES                          #
#######################################################################

_control_vars = {
    "initial_time": lambda: 0,
    "final_time": lambda: 364,
    "time_step": lambda: 1,
    "saveper": lambda: time_step(),
}


@component.add(
    name="Day of year",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"time": 1},
)
def day_of_year():
    """
    0始まりで年内の日付を返す。複数年シミュレーション対応。
    """
    return float(np.mod(np.floor(time()), 365))


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


@component.add(name="Time")
def time():
    """
    Current time of the model.
    """
    return __data["time"]()


@component.add(
    name="FINAL TIME", units="day", comp_type="Constant", comp_subtype="Normal"
)
def final_time():
    """
    The final time for the simulation.
    """
    return __data["time"].final_time()


@component.add(
    name="INITIAL TIME", units="day", comp_type="Constant", comp_subtype="Normal"
)
def initial_time():
    """
    The initial time for the simulation.
    """
    return __data["time"].initial_time()


@component.add(
    name="SAVEPER",
    units="day",
    limits=(0.0, np.nan),
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"time_step": 1},
)
def saveper():
    """
    The frequency with which output is stored.
    """
    return __data["time"].saveper()


@component.add(
    name="TIME STEP",
    units="day",
    limits=(0.0, np.nan),
    comp_type="Constant",
    comp_subtype="Normal",
)
def time_step():
    """
    The time step for the simulation.
    """
    return __data["time"].time_step()


#######################################################################
#                           MODEL VARIABLES                           #
#######################################################################


@component.add(
    name="Heat excess",
    units="day・℃",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"daily_ave_temp": 2},
)
def heat_excess():
    return if_then_else(daily_ave_temp() > 26, lambda: daily_ave_temp() - 26, lambda: 0)


@component.add(
    name="Accumulated heat stress",
    units="degree day",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_accumulated_heat_stress": 1},
    other_deps={
        "_integ_accumulated_heat_stress": {
            "initial": {},
            "step": {
                "heat_excess": 1,
                "post_heading_accumulation_window": 1,
                "reset_heat_stress_each_year": 1,
            },
        }
    },
)
def accumulated_heat_stress():
    return _integ_accumulated_heat_stress()


_integ_accumulated_heat_stress = Integ(
    lambda: heat_excess() * post_heading_accumulation_window()
    - reset_heat_stress_each_year() * accumulated_heat_stress() / time_step(),
    lambda: 0,
    "_integ_accumulated_heat_stress",
)


@component.add(
    name="Heading day", units="day", comp_type="Constant", comp_subtype="Normal"
)
def heading_day():
    """
    出穂日を8/1と仮定
    """
    return 213


@component.add(
    name="Post heading accumulation window",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"day_of_year": 2, "heading_day": 2},
)
def post_heading_accumulation_window():
    return if_then_else(
        day_of_year() >= heading_day(), lambda: 1, lambda: 0
    ) * if_then_else(day_of_year() < heading_day() + 30, lambda: 1, lambda: 0)


@component.add(
    name="Post heading period",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"day_of_year": 2, "heading_day": 2},
)
def post_heading_period():
    return if_then_else(
        day_of_year() >= heading_day(), lambda: 1, lambda: 0
    ) * if_then_else(day_of_year() < heading_day() + 30, lambda: 1, lambda: 0)


@component.add(
    name="Reset heat stress each year",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"day_of_year": 1},
)
def reset_heat_stress_each_year():
    """
    年頭で蓄積熱ストレスをリセットするトリガー（ステップ微分でゼロ化）。
    """
    return if_then_else(day_of_year() == 0, lambda: 1, lambda: 0)


@component.add(
    name="Heat stress sample trigger",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"day_of_year": 1, "heading_day": 1, "time_step": 1},
)
def heat_stress_sample_trigger():
    """
    出穂日＋30日目に蓄積熱ストレスをサンプルして固定する。
    """
    return if_then_else(
        (day_of_year() >= heading_day() + 30 - time_step() * 0.5)
        & (day_of_year() < heading_day() + 30 + time_step() * 0.5),
        lambda: 1,
        lambda: 0,
    )


@component.add(
    name="Heat stress at heading plus 30 state",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_heat_stress_at_heading_plus_30_state": 1},
    other_deps={
        "_integ_heat_stress_at_heading_plus_30_state": {
            "initial": {},
            "step": {
                "heat_stress_sample_trigger": 1,
                "accumulated_heat_stress": 1,
                "reset_heat_stress_each_year": 1,
            },
        }
    },
)
def heat_stress_at_heading_plus_30_state():
    return _integ_heat_stress_at_heading_plus_30_state()


_integ_heat_stress_at_heading_plus_30_state = Integ(
    lambda: heat_stress_sample_trigger()
    * (accumulated_heat_stress() - heat_stress_at_heading_plus_30_state())
    / time_step()
    - reset_heat_stress_each_year()
    * heat_stress_at_heading_plus_30_state()
    / time_step(),
    lambda: 0,
    "_integ_heat_stress_at_heading_plus_30_state",
)


@component.add(
    name="Heat stress at heading plus 30",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "heat_stress_sample_trigger": 1,
        "accumulated_heat_stress": 1,
        "heat_stress_at_heading_plus_30_state": 1,
    },
)
def heat_stress_at_heading_plus_30():
    return if_then_else(
        heat_stress_sample_trigger(),
        lambda: accumulated_heat_stress(),
        lambda: heat_stress_at_heading_plus_30_state(),
    )


@component.add(name="Zero", comp_type="Constant", comp_subtype="Unchangeable")
def zero():
    return 0


@component.add(
    name="top flow error",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"flow": 2, "river_discharge_downstream": 1},
)
def top_flow_error():
    """
    修正前： IF THEN ELSE(River discharge downstream > 1.23e+08, ((River discharge downstream - flow)/1.23e+08)^2, 0) ※これだとモデルの値が小さいとエラーがカウントされないの 、一旦観測値ベースに戻してみる。 修正後： IF THEN ELSE(flow > 1.23e+08, ((flow - River discharge downstream)/1.23e+08)^2, 0)
    """
    return if_then_else(
        flow() > 123000000.0,
        lambda: ((flow() - river_discharge_downstream()) / 123000000.0) ** 2,
        lambda: 0,
    )


@component.add(
    name="high flow error",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"flow": 2, "river_discharge_downstream": 1},
)
def high_flow_error():
    """
    修正前： IF THEN ELSE(River discharge downstream > 3.6e+07, ((River discharge downstream - flow)/3.6e+07)^2, 0) ※これだとモデルの値が小さいとエラーがカウントされないの 、一旦観測値ベースに戻してみる。 修正後： IF THEN ELSE(flow > 3.6e+07, ((flow - River discharge downstream)/3.6e+07)^2, 0)
    """
    return if_then_else(
        flow() > 36000000.0,
        lambda: ((flow() - river_discharge_downstream()) / 36000000.0) ** 2,
        lambda: 0,
    )


@component.add(
    name="flow flag",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"flow": 1},
)
def flow_flag():
    return if_then_else(flow() > 123000000.0, lambda: 1, lambda: 0)


@component.add(
    name="flow",
    comp_type="Data",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_data_flow",
        "__data__": "_ext_data_flow",
        "time": 1,
    },
)
def flow():
    return _ext_data_flow(time())


_ext_data_flow = ExtData(
    r"data\flow_arase_2023_x100000_0to364.csv",
    ",",
    "A",
    "D2",
    "interpolate",
    {},
    _root,
    {},
    "_ext_data_flow",
)


@component.add(
    name="flow log error sq",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"time": 1, "flow": 1, "river_discharge_downstream": 1},
)
def flow_log_error_sq():
    return if_then_else(
        time() < 10,
        lambda: 0,
        lambda: (float(np.log(flow())) - float(np.log(river_discharge_downstream())))
        ** 2,
    )


@component.add(
    name="Accumulated flood damage",
    units="Yen",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_accumulated_flood_damage": 1},
    other_deps={
        "_integ_accumulated_flood_damage": {
            "initial": {},
            "step": {"financial_damage_by_flood": 1},
        }
    },
)
def accumulated_flood_damage():
    return _integ_accumulated_flood_damage()


_integ_accumulated_flood_damage = Integ(
    lambda: financial_damage_by_flood(), lambda: 0, "_integ_accumulated_flood_damage"
)


@component.add(
    name="Accumulated innundation damage",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_accumulated_innundation_damage": 1},
    other_deps={
        "_integ_accumulated_innundation_damage": {
            "initial": {},
            "step": {"financial_damage_by_innundation": 1},
        }
    },
)
def accumulated_innundation_damage():
    return _integ_accumulated_innundation_damage()


_integ_accumulated_innundation_damage = Integ(
    lambda: financial_damage_by_innundation(),
    lambda: 0,
    "_integ_accumulated_innundation_damage",
)


@component.add(
    name="Dam storage",
    units="m3",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_dam_storage": 1},
    other_deps={
        "_integ_dam_storage": {
            "initial": {"initial_dam_capacity": 1},
            "step": {"dam_inflow": 1, "dam_outflow": 1},
        }
    },
)
def dam_storage():
    return _integ_dam_storage()


_integ_dam_storage = Integ(
    lambda: dam_inflow() - dam_outflow(),
    lambda: initial_dam_capacity() * 0.6,
    "_integ_dam_storage",
)


@component.add(
    name="Wild animal damage",
    units="ha/day",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"daily_ave_temp": 1},
)
def wild_animal_damage():
    return if_then_else(daily_ave_temp() > 25, lambda: 150 * 2 / 365, lambda: 150 / 365)


@component.add(
    name="Landslide disaster risk",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "daily_precip": 1,
        "daily_precipitation_future_ratio": 1,
        "forest_area": 1,
        "upstream_area": 1,
        "erosion_control_dam_capacity": 2,
        "erosion_control_of_forest": 1,
    },
)
def landslide_disaster_risk():
    return float(
        np.maximum(
            (
                daily_precip() * daily_precipitation_future_ratio()
                - (
                    erosion_control_dam_capacity()
                    + forest_area() / upstream_area() * erosion_control_of_forest()
                )
            )
            / erosion_control_dam_capacity(),
            0,
        )
    )


@component.add(
    name="Deterioration ratio by heat",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"daily_ave_temp": 2},
)
def deterioration_ratio_by_heat():
    return if_then_else(
        daily_ave_temp() < 25, lambda: 0, lambda: (daily_ave_temp() - 25) / 25
    )


@component.add(
    name="Upstream inflow",
    units="m3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "daily_precip": 1,
        "upstream_area": 1,
        "daily_precipitation_future_ratio": 1,
    },
)
def upstream_inflow():
    return (
        daily_precip()
        * upstream_area()
        * 10000
        / 1000
        * daily_precipitation_future_ratio()
    )


@component.add(
    name="Evaporation ratio",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "daily_ave_temp": 1,
        "daily_max_temp": 1,
        "daily_min_temp": 1,
        "solar_radiation_time": 1,
    },
)
def evaporation_ratio():
    return (
        0.0023
        * (daily_ave_temp() + 17.8)
        * (daily_max_temp() - daily_min_temp()) ** 0.5
        * solar_radiation_time()
    )


@component.add(
    name="Downstream inflow",
    units="m3/day",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "daily_precip": 1,
        "downstream_area": 1,
        "daily_precipitation_future_ratio": 1,
    },
)
def downstream_inflow():
    return (
        daily_precip()
        * downstream_area()
        * 10000
        / 1000
        * daily_precipitation_future_ratio()
    )


@component.add(
    name="Daily min temp",
    comp_type="Data",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_data_daily_min_temp",
        "__data__": "_ext_data_daily_min_temp",
        "time": 1,
    },
)
def daily_min_temp():
    return _ext_data_daily_min_temp(time())


_ext_data_daily_min_temp = ExtData(
    r"jma_kurume_2023.xlsx",
    "input",
    "A",
    "E2",
    "interpolate",
    {},
    _root,
    {},
    "_ext_data_daily_min_temp",
)


@component.add(
    name="Solar radiation time",
    comp_type="Data",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_data_solar_radiation_time",
        "__data__": "_ext_data_solar_radiation_time",
        "time": 1,
    },
)
def solar_radiation_time():
    return _ext_data_solar_radiation_time(time())


_ext_data_solar_radiation_time = ExtData(
    r"jma_kurume_2023.xlsx",
    "input",
    "A",
    "F2",
    "interpolate",
    {},
    _root,
    {},
    "_ext_data_solar_radiation_time",
)


@component.add(
    name="Daily ave temp",
    comp_type="Data",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_data_daily_ave_temp",
        "__data__": "_ext_data_daily_ave_temp",
        "time": 1,
    },
)
def daily_ave_temp():
    return _ext_data_daily_ave_temp(time())


_ext_data_daily_ave_temp = ExtData(
    r"jma_kurume_2023.xlsx",
    "input",
    "A",
    "C2",
    "interpolate",
    {},
    _root,
    {},
    "_ext_data_daily_ave_temp",
)


@component.add(
    name="Daily max temp",
    comp_type="Data",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_data_daily_max_temp",
        "__data__": "_ext_data_daily_max_temp",
        "time": 1,
    },
)
def daily_max_temp():
    return _ext_data_daily_max_temp(time())


_ext_data_daily_max_temp = ExtData(
    r"jma_kurume_2023.xlsx",
    "input",
    "A",
    "D2",
    "interpolate",
    {},
    _root,
    {},
    "_ext_data_daily_max_temp",
)


@component.add(
    name="Daily precip",
    units="mm",
    comp_type="Data",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_data_daily_precip",
        "__data__": "_ext_data_daily_precip",
        "time": 1,
    },
)
def daily_precip():
    return _ext_data_daily_precip(time())


_ext_data_daily_precip = ExtData(
    r"jma_kurume_2023.xlsx",
    "input",
    "A",
    "B2",
    "interpolate",
    {},
    _root,
    {},
    "_ext_data_daily_precip",
)


@component.add(
    name="Paddy dam area",
    units="ha",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_paddy_dam_area": 1},
    other_deps={
        "_integ_paddy_dam_area": {
            "initial": {
                "downstream_area": 1,
                "paddy_field_ratio": 1,
                "initial_paddy_dam_ratio": 1,
            },
            "step": {"paddy_dam_investment": 1, "paddy_dam_cost_per_area": 1},
        }
    },
)
def paddy_dam_area():
    return _integ_paddy_dam_area()


_integ_paddy_dam_area = Integ(
    lambda: paddy_dam_investment() / paddy_dam_cost_per_area(),
    lambda: downstream_area() * paddy_field_ratio() * initial_paddy_dam_ratio(),
    "_integ_paddy_dam_area",
)


@component.add(
    name="Initial paddy dam ratio",
    limits=(0.0, 1.0, 0.1),
    comp_type="Constant",
    comp_subtype="Normal",
)
def initial_paddy_dam_ratio():
    return 0.1


@component.add(
    name="Biodiversity",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "coefficient_for_ecosystem": 1,
        "dam_capacity": 1,
        "forest_area_storage_capacity": 1,
    },
)
def biodiversity():
    return (
        1
        - coefficient_for_ecosystem() * dam_capacity() / forest_area_storage_capacity()
    )


@component.add(
    name="Degradation of paddy dam",
    limits=(0.0, 0.1, 0.01),
    comp_type="Constant",
    comp_subtype="Normal",
)
def degradation_of_paddy_dam():
    return 0.05


@component.add(
    name="Daily crop production",
    units="Yen",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "paddy_field": 1,
        "paddy_field_productivity": 1,
        "deterioration_ratio_by_heat": 1,
        "quality_adjusted_crop_price": 1,
        "paddy_dam_ratio": 2,
        "degradation_of_paddy_dam": 1,
    },
)
def daily_crop_production():
    return (
        paddy_field()
        * paddy_field_productivity()
        * (1 - deterioration_ratio_by_heat())
        * quality_adjusted_crop_price()
        * (
            (1 - paddy_dam_ratio())
            + paddy_dam_ratio() * (1 - degradation_of_paddy_dam())
        )
    )


@component.add(
    name="Accumulated crop production within year",
    units="Yen",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_accumulated_crop_production_within_year": 1},
    other_deps={
        "_integ_accumulated_crop_production_within_year": {
            "initial": {},
            "step": {
                "daily_crop_production": 1,
                "reset_heat_stress_each_year": 1,
            },
        }
    },
)
def accumulated_crop_production_within_year():
    return _integ_accumulated_crop_production_within_year()


_integ_accumulated_crop_production_within_year = Integ(
    lambda: daily_crop_production()
    - reset_heat_stress_each_year()
    * accumulated_crop_production_within_year()
    / time_step(),
    lambda: 0,
    "_integ_accumulated_crop_production_within_year",
)


@component.add(
    name="Year end trigger",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"day_of_year": 1, "time_step": 1},
)
def year_end_trigger():
    return if_then_else(day_of_year() >= 364 - time_step() * 0.5, lambda: 1, lambda: 0)


@component.add(
    name="Yearly crop production state",
    units="Yen/Year",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_yearly_crop_production_state": 1},
    other_deps={
        "_integ_yearly_crop_production_state": {
            "initial": {},
            "step": {
                "year_end_trigger": 1,
                "accumulated_crop_production_within_year": 1,
                "reset_heat_stress_each_year": 1,
            },
        }
    },
)
def yearly_crop_production_state():
    return _integ_yearly_crop_production_state()


_integ_yearly_crop_production_state = Integ(
    lambda: year_end_trigger()
    * (accumulated_crop_production_within_year() - yearly_crop_production_state())
    / time_step()
    - reset_heat_stress_each_year() * yearly_crop_production_state() / time_step(),
    lambda: 0,
    "_integ_yearly_crop_production_state",
)


@component.add(
    name="Yearly crop production",
    units="Yen/Year",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "year_end_trigger": 1,
        "accumulated_crop_production_within_year": 1,
        "yearly_crop_production_state": 1,
    },
)
def yearly_crop_production():
    return if_then_else(
        year_end_trigger(),
        lambda: accumulated_crop_production_within_year(),
        lambda: yearly_crop_production_state(),
    )


@component.add(
    name="Yearly crop production per day",
    units="Yen/day",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"yearly_crop_production": 1},
)
def yearly_crop_production_per_day():
    return yearly_crop_production() / 365


@component.add(
    name="Crop production cashflow",
    units="Yen/day",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "year_end_trigger": 1,
        "accumulated_crop_production_within_year": 1,
        "time_step": 1,
    },
)
def crop_production_cashflow():
    """
    年末に一括で作況収入を計上するためのキャッシュフロー。
    """
    return (
        year_end_trigger()
        * accumulated_crop_production_within_year()
        / time_step()
    )


@component.add(
    name="Coefficient for ecosystem",
    limits=(0.0, 10.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def coefficient_for_ecosystem():
    return 1


@component.add(
    name="Delay of recovery",
    units="day",
    limits=(0.0, 770.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def delay_of_recovery():
    return 365


@component.add(
    name="Outflow of damaged houses",
    units="house",
    comp_type="Stateful",
    comp_subtype="Delay",
    depends_on={"_delay_outflow_of_damaged_houses": 1, "damaged_houses": 1},
    other_deps={
        "_delay_outflow_of_damaged_houses": {
            "initial": {
                "damaged_houses": 1,
                "recovery_ratio": 1,
                "delay_of_recovery": 1,
            },
            "step": {"damaged_houses": 1, "recovery_ratio": 1, "delay_of_recovery": 1},
        }
    },
)
def outflow_of_damaged_houses():
    return float(
        np.maximum(
            0, float(np.minimum(_delay_outflow_of_damaged_houses(), damaged_houses()))
        )
    )


_delay_outflow_of_damaged_houses = Delay(
    lambda: damaged_houses() * recovery_ratio(),
    lambda: delay_of_recovery(),
    lambda: damaged_houses() * recovery_ratio(),
    lambda: 3,
    time_step,
    "_delay_outflow_of_damaged_houses",
)


@component.add(
    name="Recovery ratio",
    limits=(0.0, 1.0, 0.01),
    comp_type="Constant",
    comp_subtype="Normal",
)
def recovery_ratio():
    return 0.9


@component.add(
    name="Paddy dam investment",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"paddy_dam_ratio": 1, "annual_paddy_dam_investment": 1},
)
def paddy_dam_investment():
    return if_then_else(
        paddy_dam_ratio() < 1, lambda: annual_paddy_dam_investment(), lambda: 0
    )


@component.add(
    name="Inflow of houses in risky area",
    units="house",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "houses_in_inundation_risk": 1,
        "inflow_rate_of_residents": 1,
        "outflow_of_damaged_houses": 1,
    },
)
def inflow_of_houses_in_risky_area():
    return (
        houses_in_inundation_risk() * inflow_rate_of_residents()
        + outflow_of_damaged_houses()
    )


@component.add(
    name="Damaged houses",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_damaged_houses": 1},
    other_deps={
        "_integ_damaged_houses": {
            "initial": {},
            "step": {"inflow_of_damaged_houses": 1, "outflow_of_damaged_houses": 1},
        }
    },
)
def damaged_houses():
    return _integ_damaged_houses()


_integ_damaged_houses = Integ(
    lambda: inflow_of_damaged_houses() - outflow_of_damaged_houses(),
    lambda: 0,
    "_integ_damaged_houses",
)


@component.add(
    name="Outflow of houses in risky area",
    units="house",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "houses_in_inundation_risk": 1,
        "outflow_rate_of_residents": 1,
        "number_of_house_elevation": 1,
        "number_of_migration": 1,
        "houses_damaged_by_inundation": 1,
    },
)
def outflow_of_houses_in_risky_area():
    return (
        houses_in_inundation_risk() * outflow_rate_of_residents()
        + number_of_house_elevation() / 365
        + number_of_migration() / 365
        + houses_damaged_by_inundation()
    )


@component.add(
    name="Inflow of damaged houses",
    units="house",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"houses_damaged_by_inundation": 1},
)
def inflow_of_damaged_houses():
    return houses_damaged_by_inundation()


@component.add(
    name="Municipality cost",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "green_infrastructure_cost": 1,
        "infrastructure_cost": 1,
        "house_elevation_cost": 1,
        "migration_cost": 1,
        "drainage_investment": 1,
        "paddy_dam_investment": 1,
    },
)
def municipality_cost():
    return (
        green_infrastructure_cost()
        + infrastructure_cost()
        + house_elevation_cost()
        + migration_cost()
        + drainage_investment()
        + paddy_dam_investment()
    )


@component.add(
    name="Annual Paddy dam investment",
    units="Yen/Year",
    limits=(0.0, 50000000.0, 1000000.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def annual_paddy_dam_investment():
    return 1000000.0


@component.add(
    name="Upstream percolation",
    units="m3/day",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"upstream_percolation_ratio": 1, "upstream_inflow": 1},
)
def upstream_percolation():
    return upstream_percolation_ratio() * upstream_inflow()


@component.add(
    name="Paddy dam ratio",
    limits=(0.0, 1.0, 0.05),
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"paddy_dam_area": 1, "paddy_field_ratio": 1, "downstream_area": 1},
)
def paddy_dam_ratio():
    return paddy_dam_area() / (paddy_field_ratio() * downstream_area())


@component.add(
    name="Downstream percolation",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "downstream_storage": 2,
        "downstream_percolation_ratio": 1,
        "evaporation_downstream": 1,
    },
)
def downstream_percolation():
    return float(
        np.maximum(
            float(
                np.minimum(
                    downstream_storage() * downstream_percolation_ratio(),
                    downstream_storage() - evaporation_downstream(),
                )
            ),
            0,
        )
    )


@component.add(
    name="Drainage",
    units="m3/day",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "drainage_capacity": 1,
        "downstream_storage": 1,
        "downstream_outflow": 1,
        "downstream_percolation": 1,
        "evaporation_downstream": 1,
    },
)
def drainage():
    return float(
        np.maximum(
            float(
                np.minimum(
                    drainage_capacity() * 3600 * 24,
                    downstream_storage()
                    - evaporation_downstream()
                    - downstream_percolation()
                    - downstream_outflow(),
                )
            ),
            0,
        )
    )


@component.add(
    name="Downstream outflow",
    units="m3/day",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "downstream_outflow_ratio": 1,
        "downstream_storage": 2,
        "downstream_percolation": 1,
        "evaporation_downstream": 1,
    },
)
def downstream_outflow():
    return float(
        np.maximum(
            float(
                np.minimum(
                    downstream_outflow_ratio() * downstream_storage(),
                    downstream_storage()
                    - evaporation_downstream()
                    - downstream_percolation(),
                )
            ),
            600000,
        )
    )


@component.add(
    name="Drainage investment amount",
    units="Yen",
    limits=(0.0, 10000000000.0, 100000000.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def drainage_investment_amount():
    return 0


@component.add(
    name="Drainage investment",
    units="Yen",
    limits=(0.0, 10000000000.0, 100000000.0),
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"drainage_investment_amount": 1, "time": 1},
)
def drainage_investment():
    return drainage_investment_amount() * pulse(__data["time"], 0, width=365) / 365


@component.add(
    name="Discharge allowance",
    units="m3/day",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_discharge_allowance": 1},
    other_deps={
        "_integ_discharge_allowance": {
            "initial": {"current_highwater_discharge": 1},
            "step": {"levee_level_increase": 1},
        }
    },
)
def discharge_allowance():
    return _integ_discharge_allowance()


_integ_discharge_allowance = Integ(
    lambda: levee_level_increase(),
    lambda: current_highwater_discharge() * 60 * 60 * 24,
    "_integ_discharge_allowance",
)


@component.add(
    name="Green infrastructure cost",
    units="Yen",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "number_of_planting_trees": 1,
        "trees_per_area": 1,
        "unit_cost_of_planting_trees": 1,
    },
)
def green_infrastructure_cost():
    return (
        number_of_planting_trees()
        / 365
        / trees_per_area()
        * unit_cost_of_planting_trees()
    )


@component.add(
    name="Level per levee investment",
    units="m3/day/Yen",
    limits=(0.0, 10000.0, 100.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def level_per_levee_investment():
    return 1000


@component.add(
    name="Levee level increase",
    units="m3/day",
    comp_type="Stateful",
    comp_subtype="Delay",
    depends_on={"_delay_levee_level_increase": 1},
    other_deps={
        "_delay_levee_level_increase": {
            "initial": {"levee_construction_time": 1},
            "step": {
                "levee_investment": 1,
                "level_per_levee_investment": 1,
                "levee_construction_time": 1,
            },
        }
    },
)
def levee_level_increase():
    return _delay_levee_level_increase()


_delay_levee_level_increase = Delay(
    lambda: levee_investment() * level_per_levee_investment(),
    lambda: levee_construction_time() * 365,
    lambda: 0,
    lambda: 3,
    time_step,
    "_delay_levee_level_increase",
)


@component.add(
    name="Levee investment amount",
    units="Yen",
    limits=(0.0, 100000000.0, 10000000.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def levee_investment_amount():
    return 0


@component.add(
    name="Levee investment",
    units="Yen",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "levee_investment_amount": 1,
        "time": 1,
        "levee_investment_start_time": 1,
    },
)
def levee_investment():
    return (
        levee_investment_amount()
        * pulse(__data["time"], levee_investment_start_time() * 365, width=365)
        / 365
    )


@component.add(
    name="Levee investment start time",
    units="Year",
    limits=(0.0, 10.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def levee_investment_start_time():
    return 0


@component.add(
    name="Dam investment amount",
    limits=(0.0, 10000000000.0, 1000000000.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def dam_investment_amount():
    return 0


@component.add(
    name="Dam investment start time",
    units="Year",
    limits=(0.0, 11.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def dam_investment_start_time():
    return 0


@component.add(
    name="Dam capacity",
    units="m3",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_dam_capacity": 1},
    other_deps={
        "_integ_dam_capacity": {
            "initial": {"initial_dam_capacity": 1},
            "step": {"dam_capacity_increase": 1},
        }
    },
)
def dam_capacity():
    return _integ_dam_capacity()


_integ_dam_capacity = Integ(
    lambda: dam_capacity_increase(),
    lambda: initial_dam_capacity(),
    "_integ_dam_capacity",
)


@component.add(
    name="Initial dam capacity", units="m3", comp_type="Constant", comp_subtype="Normal"
)
def initial_dam_capacity():
    return 74200000.0


@component.add(
    name="Dam investment",
    units="Yen/Year",
    limits=(0.0, 100000000000.0, 1000000000.0),
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"dam_investment_amount": 1, "time": 1, "dam_investment_start_time": 1},
)
def dam_investment():
    return (
        dam_investment_amount()
        * pulse(__data["time"], dam_investment_start_time() * 365, width=365)
        / 365
    )


@component.add(
    name="Downstream deep percolation",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"downstream_underground": 1, "downstream_deep_percolation_ratio": 1},
)
def downstream_deep_percolation():
    return downstream_underground() * downstream_deep_percolation_ratio()


@component.add(
    name="Downstream deep percolation ratio",
    limits=(0.0, 1.0, 0.1),
    comp_type="Constant",
    comp_subtype="Normal",
)
def downstream_deep_percolation_ratio():
    return 0.2


@component.add(
    name="Downstream middle flow ratio",
    limits=(0.0, 1.0, 0.1),
    comp_type="Constant",
    comp_subtype="Normal",
)
def downstream_middle_flow_ratio():
    return 0.5


@component.add(
    name="Downstream middle flow",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"downstream_underground": 1, "downstream_middle_flow_ratio": 1},
)
def downstream_middle_flow():
    return downstream_underground() * downstream_middle_flow_ratio()


@component.add(
    name="Downstream percolation ratio",
    limits=(0.0, 1.0, 0.1),
    comp_type="Constant",
    comp_subtype="Normal",
)
def downstream_percolation_ratio():
    return 0.2


@component.add(
    name="Upstream deep Percolation",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"upstream_underground": 1, "upstream_deep_percolation_ratio": 1},
)
def upstream_deep_percolation():
    return upstream_underground() * upstream_deep_percolation_ratio()


@component.add(
    name="Upstream deep percolation ratio",
    limits=(0.0, 1.0, 0.1),
    comp_type="Constant",
    comp_subtype="Normal",
)
def upstream_deep_percolation_ratio():
    return 0.5


@component.add(
    name="Paddyfield storage capacity",
    units="m2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "paddy_field": 1,
        "paddy_dam_ratio": 1,
        "paddy_field_capacity_per_area": 1,
    },
)
def paddyfield_storage_capacity():
    return paddy_field() * paddy_dam_ratio() * paddy_field_capacity_per_area()


@component.add(
    name="Drainage capacity",
    units="m3/s",
    limits=(0.0, 100.0, 1.0),
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_drainage_capacity": 1},
    other_deps={
        "_integ_drainage_capacity": {
            "initial": {"initial_drainage_capacity": 1},
            "step": {"drainage_investment": 1, "cost_for_unit_drainage_increase": 1},
        }
    },
)
def drainage_capacity():
    return _integ_drainage_capacity()


_integ_drainage_capacity = Integ(
    lambda: drainage_investment() / cost_for_unit_drainage_increase(),
    lambda: initial_drainage_capacity(),
    "_integ_drainage_capacity",
)


@component.add(
    name="Paddy field",
    units="ha",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_paddy_field": 1},
    other_deps={
        "_integ_paddy_field": {
            "initial": {"downstream_area": 1, "paddy_field_ratio": 1},
            "step": {"inflow_of_paddy_field": 1, "outflow_of_paddy_field": 1},
        }
    },
)
def paddy_field():
    return _integ_paddy_field()


_integ_paddy_field = Integ(
    lambda: inflow_of_paddy_field() - outflow_of_paddy_field(),
    lambda: downstream_area() * paddy_field_ratio(),
    "_integ_paddy_field",
)


@component.add(
    name="Initial drainage capacity",
    units="m3/s",
    limits=(0.0, 50.0, 10.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def initial_drainage_capacity():
    return 50


@component.add(
    name="Cost for unit drainage increase",
    units="Yen",
    limits=(0.0, 200000000.0, 10000000.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def cost_for_unit_drainage_increase():
    return 100000000.0


@component.add(
    name="Chalky kernel ratio",
    units="Percent",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"heat_stress_at_heading_plus_30": 1},
)
def chalky_kernel_ratio():
    """
    白未熟粒率(%): y = 0.0001x^3 - 0.0045x^2 + 0.2103x + 1.7942
    出典: 西森ら(2020) 作況基準筆データを用いたコメ品質に対する気候影響の統計解析
    """
    return float(
        np.maximum(
            0,
            0.0001 * heat_stress_at_heading_plus_30() ** 3
            - 0.0045 * heat_stress_at_heading_plus_30() ** 2
            + 0.2103 * heat_stress_at_heading_plus_30()
            + 1.7942,
        )
    )


@component.add(
    name="Crop price quality factor",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"chalky_kernel_ratio": 1},
)
def crop_price_quality_factor():
    """
    等級シェアを白未熟粒率のビンごとに固定し、価格係数を加重平均。
    1等=100%, 2等=90%, 3等=80%, 規格外=70%。
    """

    def grade_factor(s1, s2, s3, sreject):
        return s1 * 1.0 + s2 * 0.9 + s3 * 0.8 + sreject * 0.7

    x = chalky_kernel_ratio()
    if x <= 0:
        return grade_factor(0.5, 0.3, 0.1, 0.1)
    elif x <= 5:
        return grade_factor(0.5, 0.3, 0.1, 0.1)
    elif x <= 10:
        return grade_factor(0.3, 0.4, 0.2, 0.1)
    elif x <= 15:
        return grade_factor(0.1, 0.5, 0.3, 0.1)
    elif x <= 20:
        return grade_factor(0.1, 0.3, 0.4, 0.2)
    elif x <= 25:
        return grade_factor(0.1, 0.2, 0.5, 0.2)
    elif x <= 30:
        return grade_factor(0.0, 0.3, 0.4, 0.3)
    elif x <= 35:
        return grade_factor(0.0, 0.1, 0.2, 0.7)
    elif x <= 40:
        return grade_factor(0.0, 0.1, 0.2, 0.7)
    elif x <= 45:
        return grade_factor(0.0, 0.0, 0.1, 0.9)
    else:
        return grade_factor(0.0, 0.0, 0.1, 0.9)


@component.add(
    name="Crop price",
    units="Yen/kg",
    limits=(0.0, 10000.0, 100.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def crop_price():
    return 4000


@component.add(
    name="Quality adjusted crop price",
    units="Yen/kg",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"crop_price": 1, "crop_price_quality_factor": 1},
)
def quality_adjusted_crop_price():
    return crop_price() * crop_price_quality_factor()


@component.add(
    name="Forest area",
    units="ha",
    limits=(0.0, np.nan),
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_forest_area": 1},
    other_deps={
        "_integ_forest_area": {
            "initial": {"upstream_area": 1, "forest_area_ratio": 1},
            "step": {"inflow_of_forest_area": 1, "outflow_of_forest_area": 1},
        }
    },
)
def forest_area():
    return _integ_forest_area()


_integ_forest_area = Integ(
    lambda: inflow_of_forest_area() - outflow_of_forest_area(),
    lambda: upstream_area() * forest_area_ratio(),
    "_integ_forest_area",
)


@component.add(
    name="Forest area ratio",
    limits=(0.0, 1.0, 0.01),
    comp_type="Constant",
    comp_subtype="Normal",
)
def forest_area_ratio():
    return 0.92


@component.add(
    name="Daily Total GDP",
    units="Yen/day",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "houses_in_inundation_risk": 1,
        "houses_in_nonrisky_area": 1,
        "gdp_per_resident": 1,
        "crop_production_cashflow": 1,
        "financial_damage_by_innundation": 1,
        "financial_damage_by_flood": 1,
    },
)
def daily_total_gdp():
    return (
        (houses_in_inundation_risk() + houses_in_nonrisky_area()) * gdp_per_resident()
        + crop_production_cashflow()
        - financial_damage_by_innundation()
        - financial_damage_by_flood()
    )


@component.add(
    name="Paddy field ratio",
    limits=(0.0, 1.0, 0.1),
    comp_type="Constant",
    comp_subtype="Normal",
)
def paddy_field_ratio():
    return 0.12


@component.add(
    name="Paddy dam cost per area",
    units="Yen/ha",
    comp_type="Constant",
    comp_subtype="Normal",
)
def paddy_dam_cost_per_area():
    return 2500000.0 / 30 + 140000 * 260 / 100 / 0.3


@component.add(
    name="River water level downstream",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"river_discharge_downstream": 1},
)
def river_water_level_downstream():
    return 10 + float(np.sqrt(river_discharge_downstream())) / 1000


@component.add(
    name="Upstream middle flow ratio",
    limits=(0.0, 1.0, 0.1),
    comp_type="Constant",
    comp_subtype="Normal",
)
def upstream_middle_flow_ratio():
    return 0.5


@component.add(
    name="Middle flow",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"upstream_underground": 1, "upstream_middle_flow_ratio": 1},
)
def middle_flow():
    return upstream_underground() * upstream_middle_flow_ratio()


@component.add(
    name="River discharge downstream",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "river_discharge_upstream": 1,
        "downstream_outflow": 1,
        "downstream_middle_flow": 1,
    },
)
def river_discharge_downstream():
    """
    はじめは River discharge upstream + Downstream outflow + Downstream middle flow だったが、 ０の時にlogを適用できなくなるため、以下のように変更した。 こを250とかにしてもよいのかもしれないが。 MAX( River discharge upstream + Downstream outflow + Downstream middle flow , 1 ) 2025/11/21 一度基底流量を2.5Mに設定してみよう。 MAX( River discharge upstream + Downstream outflow + Downstream middle flow , 2500000 )
    """
    return float(
        np.maximum(
            river_discharge_upstream()
            + downstream_outflow()
            + downstream_middle_flow(),
            2500000.0,
        )
    )


@component.add(
    name="River discharge upstream",
    units="m3/day",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "upstream_outflow": 1,
        "direct_discharge_ratio": 1,
        "dam_outflow": 1,
        "middle_flow": 1,
    },
)
def river_discharge_upstream():
    return upstream_outflow() * direct_discharge_ratio() + dam_outflow() + middle_flow()


@component.add(
    name="Flood water amount",
    units="m3/day",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"river_discharge_downstream": 1, "discharge_allowance": 1},
)
def flood_water_amount():
    return float(np.maximum(river_discharge_downstream() - discharge_allowance(), 0))


@component.add(
    name="Discharge amount ratio",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"river_discharge_downstream": 1, "discharge_allowance": 1},
)
def discharge_amount_ratio():
    return river_discharge_downstream() / discharge_allowance()


@component.add(
    name="Upstream storage capacity",
    units="m3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"forest_area_storage_capacity": 1},
)
def upstream_storage_capacity():
    return forest_area_storage_capacity()


@component.add(
    name="Downstream storage capacity",
    units="m3",
    limits=(0.0, 1000000000.0, 10000000.0),
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "downstream_area": 1,
        "downstream_storage_depth": 1,
        "paddyfield_storage_capacity": 1,
    },
)
def downstream_storage_capacity():
    return (
        downstream_area() * 10000 * downstream_storage_depth()
        + paddyfield_storage_capacity()
    )


@component.add(
    name="Daily precipitation future ratio",
    limits=(1.0, 2.0, 0.01),
    comp_type="Constant",
    comp_subtype="Normal",
)
def daily_precipitation_future_ratio():
    return 1


@component.add(
    name="Evaporation downstream",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"evaporation_ratio": 2, "downstream_area": 2, "downstream_storage": 2},
)
def evaporation_downstream():
    return (
        if_then_else(
            evaporation_ratio() * downstream_area() * 0.001 * 100 * 100
            < downstream_storage(),
            lambda: evaporation_ratio() * downstream_area() * 0.001 * 100 * 100,
            lambda: downstream_storage(),
        )
        * 0.01
    )


@component.add(
    name="Evaporation upstream",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"evaporation_ratio": 2, "upstream_area": 2, "upstream_storage": 2},
)
def evaporation_upstream():
    return (
        if_then_else(
            evaporation_ratio() * upstream_area() * 0.001 * 100 * 100
            < upstream_storage(),
            lambda: evaporation_ratio() * upstream_area() * 0.001 * 100 * 100,
            lambda: upstream_storage(),
        )
        * 0.01
    )


@component.add(
    name="Downstream outflow ratio",
    limits=(0.0, 1.0, 0.1),
    comp_type="Constant",
    comp_subtype="Normal",
)
def downstream_outflow_ratio():
    return 0.5


@component.add(
    name="Downstream storage",
    units="m3",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_downstream_storage": 1},
    other_deps={
        "_integ_downstream_storage": {
            "initial": {},
            "step": {
                "downstream_inflow": 1,
                "downstream_outflow": 1,
                "downstream_percolation": 1,
                "drainage": 1,
                "evaporation_downstream": 1,
            },
        }
    },
)
def downstream_storage():
    """
    2025/11/20 initial value: 5000000
    """
    return _integ_downstream_storage()


_integ_downstream_storage = Integ(
    lambda: downstream_inflow()
    - downstream_outflow()
    - downstream_percolation()
    - drainage()
    - evaporation_downstream(),
    lambda: 5000000.0,
    "_integ_downstream_storage",
)


@component.add(
    name="Upstream outflow ratio",
    limits=(0.0, 1.0, 0.1),
    comp_type="Constant",
    comp_subtype="Normal",
)
def upstream_outflow_ratio():
    return 0.5


@component.add(
    name="Upstream outflow",
    units="m3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "upstream_storage": 1,
        "upstream_outflow_ratio": 1,
        "excessive_surface_flow": 1,
    },
)
def upstream_outflow():
    """
    元の式：=Upstream storage * Upstream outflow ratio + Excessive surface flow 2025/11/21：常に一定の基底流量が流れるようにして実験 てみる。200。
    """
    return float(
        np.maximum(
            upstream_storage() * upstream_outflow_ratio() + excessive_surface_flow(),
            1900000.0,
        )
    )


@component.add(
    name="Downstream underground",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_downstream_underground": 1},
    other_deps={
        "_integ_downstream_underground": {
            "initial": {},
            "step": {
                "downstream_percolation": 1,
                "downstream_deep_percolation": 1,
                "downstream_middle_flow": 1,
            },
        }
    },
)
def downstream_underground():
    """
    2025/11/20 initial value 45000000
    """
    return _integ_downstream_underground()


_integ_downstream_underground = Integ(
    lambda: downstream_percolation()
    - downstream_deep_percolation()
    - downstream_middle_flow(),
    lambda: 45000000.0,
    "_integ_downstream_underground",
)


@component.add(
    name="River water level upstream",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"river_discharge_upstream": 1},
)
def river_water_level_upstream():
    return 10 + float(np.sqrt(river_discharge_upstream())) / 1000


@component.add(
    name="Upstream storage",
    units="m3",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_upstream_storage": 1},
    other_deps={
        "_integ_upstream_storage": {
            "initial": {},
            "step": {
                "upstream_inflow": 1,
                "evaporation_upstream": 1,
                "upstream_percolation": 1,
                "upstream_outflow": 1,
            },
        }
    },
)
def upstream_storage():
    """
    流域全体ではじめに持っておくべき水＝initial valueは226,778,400 (2519760 m3/日×90日)。 上流域面積は157,585ha, 下流域面積は50,000haで定義されているので、ざっくり3:1 初期の水量を割り振ると、 上流域：150,000,000m3、下流域50,000,000m3 Up(down)stream storageが河川水、undergroundが土壌水として、滞留時間は１ ：９ぐらいなので、その割合で割って、 ・上流域： storage=15,000,000m3, underground=135,000,000m3 ・下流域： storage=5,000,000m3, underground=45,000,000m3 前：initial value=0 後：initial value=15000000 m3
    """
    return _integ_upstream_storage()


_integ_upstream_storage = Integ(
    lambda: upstream_inflow()
    - evaporation_upstream()
    - upstream_percolation()
    - upstream_outflow(),
    lambda: 15000000.0,
    "_integ_upstream_storage",
)


@component.add(
    name="Upstream underground",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_upstream_underground": 1},
    other_deps={
        "_integ_upstream_underground": {
            "initial": {},
            "step": {
                "upstream_percolation": 1,
                "middle_flow": 1,
                "upstream_deep_percolation": 1,
            },
        }
    },
)
def upstream_underground():
    """
    2025/11/20：initial valueを0→135,000,000に
    """
    return _integ_upstream_underground()


_integ_upstream_underground = Integ(
    lambda: upstream_percolation() - middle_flow() - upstream_deep_percolation(),
    lambda: 135000000.0,
    "_integ_upstream_underground",
)


@component.add(
    name="Flood risky area ratio", comp_type="Constant", comp_subtype="Normal"
)
def flood_risky_area_ratio():
    return 0.05


@component.add(
    name="Inflow of houses in flood risk", comp_type="Constant", comp_subtype="Normal"
)
def inflow_of_houses_in_flood_risk():
    return 0


@component.add(
    name="Flood water level",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "flood_water_amount": 1,
        "downstream_area": 1,
        "flood_risky_area_ratio": 1,
    },
)
def flood_water_level():
    return (
        flood_water_amount()
        / (downstream_area() * 10000 * flood_risky_area_ratio())
        * 1000
    )


@component.add(
    name="Houses in flood risk",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_houses_in_flood_risk": 1},
    other_deps={
        "_integ_houses_in_flood_risk": {
            "initial": {},
            "step": {
                "inflow_of_houses_in_flood_risk": 1,
                "outflow_of_houses_in_flood_risk": 1,
            },
        }
    },
)
def houses_in_flood_risk():
    return _integ_houses_in_flood_risk()


_integ_houses_in_flood_risk = Integ(
    lambda: inflow_of_houses_in_flood_risk() - outflow_of_houses_in_flood_risk(),
    lambda: 282000,
    "_integ_houses_in_flood_risk",
)


@component.add(
    name="Financial damage by flood",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"flood_damage_per_resident": 1, "houses_damaged_by_flood": 1},
)
def financial_damage_by_flood():
    return flood_damage_per_resident() * houses_damaged_by_flood()


@component.add(
    name="Flood damage per resident", comp_type="Constant", comp_subtype="Normal"
)
def flood_damage_per_resident():
    return 10000000.0 / 365


@component.add(
    name="Flood damage ratio",
    comp_type="Auxiliary",
    comp_subtype="with Lookup",
    depends_on={"flood_water_level": 1},
)
def flood_damage_ratio():
    return np.interp(
        flood_water_level(), [0.0, 0.5, 1.0, 1.8, 10.0], [0.0, 0.2, 0.4, 0.5, 1.0]
    )


@component.add(
    name="Houses damaged by flood",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "houses_in_flood_risk": 1,
        "flood_damage_ratio": 1,
        "evacuation_ratio": 1,
    },
)
def houses_damaged_by_flood():
    return houses_in_flood_risk() * flood_damage_ratio() * (1 - evacuation_ratio())


@component.add(
    name="Outflow of houses in flood risk", comp_type="Constant", comp_subtype="Normal"
)
def outflow_of_houses_in_flood_risk():
    return 0


@component.add(
    name="Downstream storage depth",
    units="m",
    limits=(0.0, 0.1, 0.01),
    comp_type="Constant",
    comp_subtype="Normal",
)
def downstream_storage_depth():
    return 0.03


@component.add(
    name="Outflow of forest area",
    units="ha/day",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "forest_area": 1,
        "tree_die_ratio": 1,
        "number_of_lumbering_trees": 1,
        "trees_per_area": 1,
        "wild_animal_damage": 1,
    },
)
def outflow_of_forest_area():
    return (
        forest_area() * tree_die_ratio() / 365
        + number_of_lumbering_trees() / trees_per_area() / 365
        + wild_animal_damage() * 0
    )


@component.add(
    name="Infrastructure Cost",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"dam_investment": 1, "levee_investment": 1},
)
def infrastructure_cost():
    return dam_investment() + levee_investment()


@component.add(
    name="Inundation damage ratio",
    limits=(0.0, 1.0),
    comp_type="Auxiliary",
    comp_subtype="with Lookup",
    depends_on={"innundation_level": 1},
)
def inundation_damage_ratio():
    return np.interp(
        innundation_level(), [0.0, 0.5, 1.0, 1.8, 10.0], [0.0, 0.2, 0.4, 0.5, 1.0]
    )


@component.add(
    name="Houses damaged by inundation",
    units="house",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "houses_in_inundation_risk": 1,
        "inundation_damage_ratio": 1,
        "evacuation_ratio": 1,
    },
)
def houses_damaged_by_inundation():
    return (
        houses_in_inundation_risk()
        * inundation_damage_ratio()
        * (1 - evacuation_ratio())
    )


@component.add(
    name="Innundation level",
    units="m",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "inside_water_innundation_level": 1,
        "innundation_risky_area_ratio": 1,
        "downstream_area": 1,
    },
)
def innundation_level():
    return inside_water_innundation_level() / (
        downstream_area() * 10000 * innundation_risky_area_ratio()
    )


@component.add(
    name="Innundation risky area ratio", comp_type="Constant", comp_subtype="Normal"
)
def innundation_risky_area_ratio():
    return 0.2


@component.add(
    name="Upstream percolation ratio",
    limits=(0.0, 1.0, 0.1),
    comp_type="Constant",
    comp_subtype="Normal",
)
def upstream_percolation_ratio():
    return 0.2


@component.add(
    name="current planting trees",
    units="tree/Year",
    limits=(0.0, 20000.0, 1000.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def current_planting_trees():
    return 10000


@component.add(
    name="Excessive surface flow",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"upstream_storage": 1, "upstream_storage_capacity": 1},
)
def excessive_surface_flow():
    return float(np.maximum(upstream_storage() - upstream_storage_capacity(), 0))


@component.add(
    name="Inflow of forest area",
    units="ha",
    comp_type="Stateful",
    comp_subtype="Delay",
    depends_on={"_delay_inflow_of_forest_area": 1},
    other_deps={
        "_delay_inflow_of_forest_area": {
            "initial": {
                "current_planting_trees": 1,
                "trees_per_area": 1,
                "planting_start_time": 1,
                "tree_growth_time": 1,
            },
            "step": {
                "number_of_planting_trees": 1,
                "trees_per_area": 2,
                "current_planting_trees": 1,
                "planting_start_time": 1,
                "tree_growth_time": 1,
            },
        }
    },
)
def inflow_of_forest_area():
    return _delay_inflow_of_forest_area()


_delay_inflow_of_forest_area = Delay(
    lambda: number_of_planting_trees() / 365 / trees_per_area()
    + current_planting_trees() / 365 / trees_per_area(),
    lambda: (tree_growth_time() + planting_start_time()) * 365,
    lambda: current_planting_trees() / trees_per_area() / 365,
    lambda: 3,
    time_step,
    "_delay_inflow_of_forest_area",
)


@component.add(
    name="Dam outflow",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"predischarge_control": 1, "dam_spill": 1},
)
def dam_outflow():
    return predischarge_control() + dam_spill()


@component.add(
    name="Dam spill",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"dam_storage": 1, "dam_capacity": 1},
)
def dam_spill():
    return float(np.maximum(dam_storage() - dam_capacity(), 0))


@component.add(
    name="Price per tree",
    units="Yen/tree",
    limits=(0.0, 100000.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def price_per_tree():
    return 10000


@component.add(
    name="Sales of forestry",
    units="Yen/day",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"number_of_lumbering_trees": 1, "price_per_tree": 1},
)
def sales_of_forestry():
    return number_of_lumbering_trees() * price_per_tree() / 365


@component.add(
    name="Dam inflow",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"upstream_outflow": 1, "direct_discharge_ratio": 1},
)
def dam_inflow():
    return upstream_outflow() * (1 - direct_discharge_ratio())


@component.add(
    name="Predischarge control",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"dam_storage": 3, "dam_capacity": 2, "predischarge_capacity": 1},
)
def predischarge_control():
    return if_then_else(
        dam_storage() / dam_capacity() > 0.6,
        lambda: if_then_else(
            dam_storage() / dam_capacity() < 0.9,
            lambda: float(np.maximum(predischarge_capacity(), dam_storage() * 0.1)),
            lambda: 0,
        ),
        lambda: 0,
    )


@component.add(
    name="Direct discharge ratio",
    limits=(0.0, 1.0, 0.1),
    comp_type="Constant",
    comp_subtype="Normal",
)
def direct_discharge_ratio():
    return 0.97


@component.add(
    name="Predischarge capacity",
    units="m3/day",
    limits=(0.0, 10000.0, 100.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def predischarge_capacity():
    return 10000


@component.add(
    name="Evacuation ratio",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"eldery_people_ratio": 1, "capacity_building": 1},
)
def evacuation_ratio():
    return (1 - eldery_people_ratio()) * capacity_building()


@component.add(
    name="Eldery people ratio",
    limits=(0.0, 1.0, 0.01),
    comp_type="Constant",
    comp_subtype="Normal",
)
def eldery_people_ratio():
    return 0.6


@component.add(
    name="Capacity Building",
    limits=(0.0, 1.0, 0.1),
    comp_type="Constant",
    comp_subtype="Normal",
)
def capacity_building():
    return 0.5


@component.add(
    name="Inside water innundation level",
    units="m3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"downstream_storage": 1, "downstream_storage_capacity": 1},
)
def inside_water_innundation_level():
    return float(np.maximum(downstream_storage() - downstream_storage_capacity(), 0))


@component.add(
    name="Planting start time",
    units="Year",
    limits=(0.0, 11.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def planting_start_time():
    return 5


@component.add(
    name="Number of planting trees",
    units="tree/Year",
    limits=(0.0, 1000000.0, 100000.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def number_of_planting_trees():
    return 1000000.0


@component.add(
    name="Inflow of damaged paddy field",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "innundation_level": 1,
        "ratio_of_paddy_field_in_risky_area": 1,
        "paddy_field": 1,
    },
)
def inflow_of_damaged_paddy_field():
    return if_then_else(
        innundation_level(),
        lambda: ratio_of_paddy_field_in_risky_area() * paddy_field(),
        lambda: 0,
    )


@component.add(
    name='"Current high-water discharge"',
    units="m3/s",
    limits=(0.0, 30000.0, 1000.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def current_highwater_discharge():
    return 11500


@component.add(
    name="Paddy field capacity per area",
    units="m3/ha",
    limits=(0.0, 2000.0, 100.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def paddy_field_capacity_per_area():
    return 1000


@component.add(
    name="Paddy field productivity",
    units="kg/ha/day",
    comp_type="Constant",
    comp_subtype="Normal",
)
def paddy_field_productivity():
    return 5400 / 365


@component.add(
    name="Outflow of damaged paddy field",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"damaged_paddy_field": 1},
)
def outflow_of_damaged_paddy_field():
    return damaged_paddy_field() * 0.1


@component.add(
    name="Downstream area",
    units="ha",
    limits=(0.0, 10000000000.0, 100000000.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def downstream_area():
    return 143951


@component.add(
    name="Upstream area", units="ha", comp_type="Constant", comp_subtype="Normal"
)
def upstream_area():
    return 157585


@component.add(
    name="Erosion control dam capacity",
    units="mm/day",
    limits=(0.0, 1000.0, 100.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def erosion_control_dam_capacity():
    return 200


@component.add(
    name="Dam capacity increase",
    units="m3",
    comp_type="Stateful",
    comp_subtype="Delay",
    depends_on={"_delay_dam_capacity_increase": 1},
    other_deps={
        "_delay_dam_capacity_increase": {
            "initial": {"dam_construction_time": 1},
            "step": {
                "dam_investment": 1,
                "capacity_per_dam_investment": 1,
                "dam_construction_time": 1,
            },
        }
    },
)
def dam_capacity_increase():
    return _delay_dam_capacity_increase()


_delay_dam_capacity_increase = Delay(
    lambda: dam_investment() * capacity_per_dam_investment(),
    lambda: dam_construction_time() * 365,
    lambda: 0,
    lambda: 3,
    time_step,
    "_delay_dam_capacity_increase",
)


@component.add(
    name="Capacity per dam investment",
    units="m3/Yen",
    limits=(0.0, 0.005, 0.0005),
    comp_type="Constant",
    comp_subtype="Normal",
)
def capacity_per_dam_investment():
    return 0.0002


@component.add(
    name="Erosion control of forest",
    units="mm/ha",
    limits=(0.0, 100.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def erosion_control_of_forest():
    return 50


@component.add(
    name="Number of migration",
    units="house",
    limits=(0.0, 1000.0, 10.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def number_of_migration():
    return 10


@component.add(
    name='"Houses in non-risky area"',
    units="house",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_houses_in_nonrisky_area": 1},
    other_deps={
        "_integ_houses_in_nonrisky_area": {
            "initial": {},
            "step": {
                "inflow_of_houses_in_nonrisky_area": 1,
                "outflow_of_houses_in_nonrisky_area": 1,
            },
        }
    },
)
def houses_in_nonrisky_area():
    return _integ_houses_in_nonrisky_area()


_integ_houses_in_nonrisky_area = Integ(
    lambda: inflow_of_houses_in_nonrisky_area() - outflow_of_houses_in_nonrisky_area(),
    lambda: 10000,
    "_integ_houses_in_nonrisky_area",
)


@component.add(
    name="Elevated houses",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_elevated_houses": 1},
    other_deps={
        "_integ_elevated_houses": {
            "initial": {},
            "step": {"inflow_of_elevated_houses": 1, "outflow_of_elevated_houses": 1},
        }
    },
)
def elevated_houses():
    return _integ_elevated_houses()


_integ_elevated_houses = Integ(
    lambda: inflow_of_elevated_houses() - outflow_of_elevated_houses(),
    lambda: 0,
    "_integ_elevated_houses",
)


@component.add(
    name="Migration cost",
    units="Yen",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"number_of_migration": 1, "unit_cost_of_migration": 1},
)
def migration_cost():
    return number_of_migration() * unit_cost_of_migration()


@component.add(
    name="Number of house elevation",
    units="house/Year",
    limits=(0.0, 1000.0, 10.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def number_of_house_elevation():
    return 10


@component.add(
    name="House elevation cost",
    units="Yen",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"number_of_house_elevation": 1, "unit_cost_of_elevation": 1},
)
def house_elevation_cost():
    return number_of_house_elevation() * unit_cost_of_elevation()


@component.add(
    name="Unit cost of elevation",
    units="Yen/house",
    limits=(1000000.0, 5000000.0, 100000.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def unit_cost_of_elevation():
    return 3000000.0


@component.add(
    name="Outflow of elevated houses",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"elevated_houses": 1, "outflow_rate_of_residents": 1},
)
def outflow_of_elevated_houses():
    return elevated_houses() * outflow_rate_of_residents()


@component.add(
    name="Damaged paddy field",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_damaged_paddy_field": 1},
    other_deps={
        "_integ_damaged_paddy_field": {
            "initial": {},
            "step": {
                "inflow_of_damaged_paddy_field": 1,
                "outflow_of_damaged_paddy_field": 1,
            },
        }
    },
)
def damaged_paddy_field():
    return _integ_damaged_paddy_field()


_integ_damaged_paddy_field = Integ(
    lambda: inflow_of_damaged_paddy_field() - outflow_of_damaged_paddy_field(),
    lambda: 0,
    "_integ_damaged_paddy_field",
)


@component.add(
    name="Outflow of paddy field",
    units="ha",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"inflow_of_damaged_paddy_field": 1},
)
def outflow_of_paddy_field():
    return inflow_of_damaged_paddy_field()


@component.add(
    name="Inflow of paddy field",
    units="ha",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"outflow_of_damaged_paddy_field": 1},
)
def inflow_of_paddy_field():
    return outflow_of_damaged_paddy_field()


@component.add(
    name="Trees per area",
    units="tree/ha",
    limits=(0.0, 2000.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def trees_per_area():
    return 900


@component.add(
    name='"Inflow of houses in non-risky area"',
    units="house",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "houses_in_nonrisky_area": 1,
        "inflow_rate_of_residents": 1,
        "number_of_migration": 1,
    },
)
def inflow_of_houses_in_nonrisky_area():
    return (
        houses_in_nonrisky_area() * inflow_rate_of_residents()
        + number_of_migration() / 365
    )


@component.add(
    name="Inflow of elevated houses",
    units="house",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"number_of_house_elevation": 1},
)
def inflow_of_elevated_houses():
    return number_of_house_elevation() / 365


@component.add(
    name="Unit cost of migration",
    units="Yen",
    limits=(0.0, 10000000.0, 1000000.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def unit_cost_of_migration():
    return 3000000.0


@component.add(
    name='"Outflow of houses in non-risky area"',
    units="house/Year",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"houses_in_nonrisky_area": 1, "outflow_rate_of_residents": 1},
)
def outflow_of_houses_in_nonrisky_area():
    return houses_in_nonrisky_area() * outflow_rate_of_residents()


@component.add(
    name='"Water-holding Capacity of Forest"',
    units="mm",
    limits=(200.0, 500.0, 10.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def waterholding_capacity_of_forest():
    return 200


@component.add(
    name="Forest area storage capacity",
    units="m3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"forest_area": 1, "waterholding_capacity_of_forest": 1},
)
def forest_area_storage_capacity():
    return forest_area() * waterholding_capacity_of_forest() * 10000 / 1000


@component.add(
    name="Levee construction time",
    units="Year",
    limits=(1.0, 10.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def levee_construction_time():
    return 5


@component.add(
    name="CO2 absorption",
    units="CO2ton/day",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"forest_area": 1, "co2_absorption_per_area": 1},
)
def co2_absorption():
    return forest_area() * co2_absorption_per_area() / 365


@component.add(
    name="CO2 absorption per area",
    units="CO2ton/ha/Year",
    limits=(1.0, 10.0, 0.1),
    comp_type="Constant",
    comp_subtype="Normal",
)
def co2_absorption_per_area():
    return 8.8


@component.add(
    name="Dam construction time",
    limits=(0.0, 20.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def dam_construction_time():
    return 10


@component.add(
    name="Unit cost of planting trees",
    units="Yen/ha",
    limits=(1000000.0, 3000000.0, 100000.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def unit_cost_of_planting_trees():
    return 2310000.0


@component.add(
    name="Tree die ratio",
    units="{(ha/ha)/day}",
    limits=(0.0, 0.01, 0.001),
    comp_type="Constant",
    comp_subtype="Normal",
)
def tree_die_ratio():
    return 0.01


@component.add(
    name="Tree growth time",
    units="Year",
    limits=(0.0, 40.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def tree_growth_time():
    return 40


@component.add(
    name="Number of lumbering trees",
    units="tree/Year",
    limits=(0.0, 1000000.0, 100000.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def number_of_lumbering_trees():
    return 100000


@component.add(
    name="Outflow rate of residents",
    limits=(0.0, 0.1, 0.01),
    comp_type="Constant",
    comp_subtype="Normal",
)
def outflow_rate_of_residents():
    return 0.01 / 365


@component.add(
    name="Financial damage by innundation",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "innundation_damage_per_resident": 1,
        "houses_damaged_by_inundation": 1,
    },
)
def financial_damage_by_innundation():
    return innundation_damage_per_resident() * houses_damaged_by_inundation()


@component.add(
    name="GDP per resident",
    units="Yen",
    limits=(0.0, 5000000.0, 10000.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def gdp_per_resident():
    return 3000000.0 / 365


@component.add(
    name="Inflow rate of residents",
    limits=(0.0, 0.1, 0.01),
    comp_type="Constant",
    comp_subtype="Normal",
)
def inflow_rate_of_residents():
    return 0.01 / 365


@component.add(
    name="Ratio of paddy field in risky area",
    limits=(0.0, 1.0, 0.01),
    comp_type="Constant",
    comp_subtype="Normal",
)
def ratio_of_paddy_field_in_risky_area():
    return 0.01


@component.add(
    name="Innundation damage per resident",
    units="Yen/person",
    comp_type="Constant",
    comp_subtype="Normal",
)
def innundation_damage_per_resident():
    return 10000000.0 / 365


@component.add(
    name="Houses in inundation risk",
    units="house",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_houses_in_inundation_risk": 1},
    other_deps={
        "_integ_houses_in_inundation_risk": {
            "initial": {},
            "step": {
                "inflow_of_houses_in_risky_area": 1,
                "outflow_of_houses_in_risky_area": 1,
            },
        }
    },
)
def houses_in_inundation_risk():
    return _integ_houses_in_inundation_risk()


_integ_houses_in_inundation_risk = Integ(
    lambda: inflow_of_houses_in_risky_area() - outflow_of_houses_in_risky_area(),
    lambda: 500000,
    "_integ_houses_in_inundation_risk",
)
