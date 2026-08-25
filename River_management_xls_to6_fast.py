"""River_management_xls_to6.py の水文サブモデルだけを numpy 逐次ループで再実装した高速版。

較正（calibrate_hydrology_fast.py）で数千回モデルを回すためのもの。
PySD 版との一致は verify_River_management_xls_to6_fast.py で確認する。

目的は較正（scipy.optimize.differential_evolution）で数千回モデルを回すこと。
PySD の Euler 積分（dt=1、initial_condition="original"）を厳密に再現する。

このファイルを直接実行すると、PySD 版と一致するかを検証する:

    RIVER_BASIN=abukuma python River_management_xls_to6_fast.py

River_management_xls_to6.py を書き換えたら必ず流すこと。式を1つ変えただけで
高速版が静かにズレて、較正結果だけが間違うという事故を防ぐための回帰テスト。

外生入力は較正時と同じゼロ条件に固定している:
    levee_investment_amount        = 0
    dam_investment_amount          = 0
    drainage_investment_amount     = 0
    number_of_house_elevation      = 0
    number_of_migration            = 0
    annual_paddy_dam_investment    = 0
    annual_breeding_investment     = 0
    daily_precipitation_future_ratio = 1.0
    temperature_scenario_shift       = 0.0

この条件下で流量に効くストックは
upstream_storage / upstream_underground / downstream_storage /
downstream_underground / dam_storage の 5 本だけである。
時間変化しうる他のストックについては次を確認済み:

- dam_capacity           : dam_investment_amount=0 なので Delay 出力が常に 0。
                           29,800,000 m3 で一定。
- drainage_capacity      : drainage_investment_amount=0 なので 62 m3/s で一定。
- discharge_allowance    : levee_investment_amount=0 なので 125,280,000 m3/day で
                           一定。そもそも flood_water_amount と
                           discharge_amount_ratio にしか使われず、
                           水収支には戻らない。
- upstream_storage_capacity（= forest_area_storage_capacity）:
                           annual_forest_management_conversion_area=0 のため
                           森林の管理状態が動かず 519,573,600 m3 で一定。
- paddy_field / paddy_dam_area / paddyfield_storage_capacity:
                           paddy_field は浸水被害で時間変化する（9,337〜10,374 ha）
                           が、これが効くのは downstream_storage_capacity を経由した
                           inside_water_innundation_level だけで、
                           downstream_storage の収支には戻らない。
                           したがって河川流量には効かない。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# 元モデルの River_management_xls_to6.py に埋め込まれたリテラル定数。
# YAML では設定できないため、ここでも同じ値をハードコードする。
_INITIAL_UPSTREAM_STORAGE = 15_000_000.0        # line 3521
_INITIAL_UPSTREAM_UNDERGROUND = 135_000_000.0   # line 3551
_INITIAL_DOWNSTREAM_STORAGE = 5_000_000.0       # line 3405
_INITIAL_DOWNSTREAM_UNDERGROUND = 45_000_000.0  # line 3470
_DAM_STORAGE_INITIAL_FILL = 0.6                 # line 968 (_integ_dam_storage 初期値)
_UPSTREAM_OUTFLOW_FLOOR = 1_900_000.0           # line 3438
_DOWNSTREAM_OUTFLOW_FLOOR = 600_000.0           # line 2145
_PREDISCHARGE_CAPACITY = 10_000.0               # line 4189 (predischarge_capacity)
_PREDISCHARGE_LOWER_TRIGGER = 0.6               # line 4161 (predischarge_control)
_PREDISCHARGE_UPPER_TRIGGER = 0.9               # line 4163 (predischarge_control)
_PREDISCHARGE_RELEASE_FRACTION = 0.1            # line 4164 (predischarge_control)
_EVAPORATION_SCALE = 0.01                       # line 3347 / 3367
_NATURAL_FOREST_SHARE = 0.4                     # line 2721 natural_forest_area
_MANAGED_PLANTATION_SHARE = 0.3                 # line 2732 initial_managed_plantation_forest_area
_UNMANAGED_PLANTATION_SHARE = 0.3               # line 2786 _integ_unmanaged_plantation_forest_area


def _repo_dir() -> Path:
    """River_management_xls_to6.py があるディレクトリを推定する。"""
    env = os.environ.get("RIVER_MODEL_DIR")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve().parent
    for candidate in (here, here / "repo", here.parent / "repo", here.parent):
        if (candidate / "River_management_xls_to6.py").exists():
            return candidate
    return here


class FastHydrology:
    """PySD 版 SD モデルの水文部分だけを再現する高速エミュレータ。"""

    #: run() が受け取る較正パラメータ名（YAML の hydrology_calibration.bounds と同じ）
    PARAMETER_NAMES = (
        "upstream_outflow_ratio",
        "downstream_outflow_ratio",
        "direct_discharge_ratio",
        "upstream_percolation_ratio",
        "downstream_deep_percolation_ratio",
        "upstream_middle_flow_ratio",
        "downstream_percolation_ratio",
        "downstream_middle_flow_ratio",
        "upstream_deep_percolation_ratio",
    )

    def __init__(
        self,
        basin: str | None = None,
        repo_dir: str | Path | None = None,
        daily_precipitation_future_ratio: float = 1.0,
        temperature_scenario_shift: float = 0.0,
    ) -> None:
        self.repo_dir = Path(repo_dir).resolve() if repo_dir else _repo_dir()
        if basin is None:
            basin = os.environ.get("RIVER_BASIN", "chikugo")
        self.basin = basin
        self.config = self._load_config(basin)
        self.params_yaml = dict(self.config["model_parameters"])

        self.n_steps = self._total_days(self.config)
        self.precip_ratio = float(daily_precipitation_future_ratio)
        self.temperature_shift = float(temperature_scenario_shift)

        self._load_inputs()
        self._precompute_constants()
        self._precompute_forcing()

    # ------------------------------------------------------------------
    # 設定と入力
    # ------------------------------------------------------------------
    def _load_config(self, basin: str) -> dict:
        candidate = Path(basin)
        if candidate.suffix in {".yaml", ".yml"} or candidate.parent != Path("."):
            path = candidate.expanduser().resolve()
        else:
            path = self.repo_dir / "config" / "basins" / f"{basin}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"流域設定が見つかりません: {path}")
        with path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        config["_path"] = path
        return config

    @staticmethod
    def _is_leap_year(year: int) -> bool:
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    @classmethod
    def _total_days(cls, config: dict) -> int:
        calendar = config["calendar"]
        num_years = int(calendar["num_years"])
        start = int(calendar["start_year"])
        if not calendar["use_leap_years"]:
            return 365 * num_years
        return sum(
            366 if cls._is_leap_year(start + offset) else 365
            for offset in range(num_years)
        )

    def _resolve(self, key: str) -> Path:
        value = Path(self.config["inputs"][key])
        return value if value.is_absolute() else (self.repo_dir / value)

    def _read_weather(self, key: str) -> dict[str, np.ndarray]:
        """気象 xlsx を読む。sheet "input"、列は A:time, B:precip, C:Tave, D:Tmax, E:Tmin, F:Rs。"""
        frame = pd.read_excel(self._resolve(key), sheet_name="input", header=None)
        return {
            "precip": self._as_extdata(frame.iloc[:, 1].to_numpy(dtype=float)),
            "tave": self._as_extdata(frame.iloc[:, 2].to_numpy(dtype=float)),
            "tmax": self._as_extdata(frame.iloc[:, 3].to_numpy(dtype=float)),
            "tmin": self._as_extdata(frame.iloc[:, 4].to_numpy(dtype=float)),
            "solar": self._as_extdata(frame.iloc[:, 5].to_numpy(dtype=float)),
        }

    def _read_flow(self, key: str) -> np.ndarray:
        frame = pd.read_csv(self._resolve(key), header=None)
        return self._as_extdata(frame.iloc[:, 3].to_numpy(dtype=float))

    def _as_extdata(self, raw: np.ndarray) -> np.ndarray:
        """PySD の ExtData と同じ時刻対応に直す。

        元モデルの ExtData はセル "B2"（Excel 2 行目、CSV 2 行目）から読むので、
        ファイル 0 行目は使われず、時刻軸は t = 1 .. N-1 になる。
        t = 0 は "extrapolating data below the minimum value of the time" として
        時刻軸先頭（= ファイル 1 行目）の値が返る。したがって

            ext[t] = raw[max(t, 1)]

        となる。
        """
        series = raw[: self.n_steps].astype(float).copy()
        if series.size < self.n_steps:
            raise ValueError(
                f"入力データが短い: {series.size} 行 < {self.n_steps} 日"
            )
        series[0] = raw[1]
        return series

    def _load_inputs(self) -> None:
        self.weather_up = self._read_weather("weather_up")
        self.weather_down = self._read_weather("weather_down")
        self.observed_flow_upstream = self._read_flow("flow_up")
        self.observed_flow_downstream = self._read_flow("flow_down")

    # ------------------------------------------------------------------
    # 定数の事前計算
    # ------------------------------------------------------------------
    def _precompute_constants(self) -> None:
        p = self.params_yaml
        self.upstream_area = float(p["upstream_area"])
        self.downstream_area = float(p["downstream_area"])
        self.minimum_river_discharge = float(p["minimum_river_discharge"])
        self.initial_dam_capacity = float(p["initial_dam_capacity"])
        self.drainage_capacity = float(p["initial_drainage_capacity"])

        # 森林関連。annual_forest_management_conversion_area = 0 なので
        # forest_management_conversion = 0 となり、人工林の管理状態は
        # 初期値のまま動かない（回復ステージ 0-5 と成熟面積は常に 0）。
        # そのため managed_plantation_forest_coef = 1.0、
        # forest_function_coef は定数になる。
        # 丸め誤差まで元コードに合わせるため、演算順序もそのまま写す。
        forest_area = self.upstream_area * float(p["forest_area_ratio"])
        unmanaged_coef = float(p["unmanaged_plantation_forest_coef"])
        natural_forest_area = forest_area * _NATURAL_FOREST_SHARE
        managed_area = forest_area * _MANAGED_PLANTATION_SHARE
        unmanaged_area = forest_area * _UNMANAGED_PLANTATION_SHARE
        managed_plantation_forest_coef = managed_area / managed_area  # = 1.0
        forest_function_coef = (
            natural_forest_area * 1.0
            + managed_area * managed_plantation_forest_coef
            + unmanaged_area * unmanaged_coef
        ) / forest_area
        waterholding = (
            float(p["waterholding_capacity_of_forest_base"]) * forest_function_coef
        )
        self.forest_area = forest_area
        self.forest_function_coef = forest_function_coef
        self.waterholding_capacity_of_forest = waterholding
        # forest_area_storage_capacity = forest_area * whcf * 10000 / 1000
        self.upstream_storage_capacity = forest_area * waterholding * 10000 / 1000

        # 投資ゼロなので dam_capacity_increase の Delay 出力は常に 0。
        self.dam_capacity = self.initial_dam_capacity
        self.initial_dam_storage = self.initial_dam_capacity * _DAM_STORAGE_INITIAL_FILL

        # drainage は m3/s -> m3/day
        self.drainage_capacity_per_day = self.drainage_capacity * 3600.0 * 24.0

    def _hargreaves(self, weather: dict[str, np.ndarray]) -> np.ndarray:
        """evaporation_ratio_up / down（Hargreaves 式、line 1094-1102 / 335-347）。

        丸めまで一致させるため、numpy のベクトル演算ではなく
        元コードと同じ順序のスカラー演算で計算する。
        """
        shift = self.temperature_shift
        tave = weather["tave"].tolist()
        tmax = weather["tmax"].tolist()
        tmin = weather["tmin"].tolist()
        solar = weather["solar"].tolist()
        out = np.empty(self.n_steps, dtype=float)
        for i in range(self.n_steps):
            out[i] = (
                0.0023
                * ((tave[i] + shift) + 17.8)
                * ((tmax[i] + shift) - (tmin[i] + shift)) ** 0.5
                * solar[i]
                / 2.45
            )
        return out

    def _precompute_forcing(self) -> None:
        """状態に依存しない強制項を先に配列化しておく。

        乗算順序は元コードのまま（`* 10000 / 1000`, `* 0.001 * 100 * 100`）に
        しておく。まとめて `* 10` にすると最下位ビットがずれる。
        """
        ratio = self.precip_ratio
        self.upstream_inflow = (
            self.weather_up["precip"] * self.upstream_area * 10000 / 1000 * ratio
        )
        self.downstream_inflow = (
            self.weather_down["precip"] * self.downstream_area * 10000 / 1000 * ratio
        )
        self.evaporation_ratio_up = self._hargreaves(self.weather_up)
        self.evaporation_ratio_down = self._hargreaves(self.weather_down)
        # evaporation_*stream の if_then_else 内の候補値
        # = evaporation_ratio * area * 0.001 * 100 * 100
        self.evap_candidate_up = (
            self.evaporation_ratio_up * self.upstream_area * 0.001 * 100 * 100
        )
        self.evap_candidate_down = (
            self.evaporation_ratio_down * self.downstream_area * 0.001 * 100 * 100
        )

        # ループ内は Python float の方が速いのでリスト化しておく
        self._up_inflow_list = self.upstream_inflow.tolist()
        self._down_inflow_list = self.downstream_inflow.tolist()
        self._evap_cand_up_list = self.evap_candidate_up.tolist()
        self._evap_cand_down_list = self.evap_candidate_down.tolist()

    # ------------------------------------------------------------------
    # 実行
    # ------------------------------------------------------------------
    #: __init__ で定数として畳み込んでしまうため run() では変更できないキー。
    #: 較正時に動かすのは PARAMETER_NAMES の 9 個だけなので実害はないが、
    #: 黙って無視すると誤差の原因になるので明示的に弾く。
    FROZEN_KEYS = (
        "upstream_area",
        "downstream_area",
        "minimum_river_discharge",
        "initial_dam_capacity",
        "initial_drainage_capacity",
        "forest_area_ratio",
        "waterholding_capacity_of_forest_base",
        "unmanaged_plantation_forest_coef",
    )

    def _check_frozen(self, params: dict) -> None:
        for key in self.FROZEN_KEYS:
            if key not in params:
                continue
            configured = float(self.params_yaml[key])
            given = float(params[key])
            if given != configured:
                raise ValueError(
                    f"{key} は FastHydrology の初期化時に定数化されるため "
                    f"run() では変更できない（設定値 {configured}, 指定値 {given}）。"
                    " 変更する場合は FastHydrology を作り直すこと。"
                )

    def run(self, params: dict | None = None) -> dict[str, np.ndarray]:
        """9 つの較正パラメータで水文サブモデルを回す。

        戻り値は PySD の return_timestamps=list(range(n_steps)) と同じ index。
        すなわち index t の値は「時刻 t におけるストック状態から計算した値」。
        """
        values = dict(self.params_yaml)
        if params:
            self._check_frozen(params)
            values.update(params)

        upstream_outflow_ratio = float(values["upstream_outflow_ratio"])
        downstream_outflow_ratio = float(values["downstream_outflow_ratio"])
        direct_discharge_ratio = float(values["direct_discharge_ratio"])
        upstream_percolation_ratio = float(values["upstream_percolation_ratio"])
        downstream_deep_percolation_ratio = float(
            values["downstream_deep_percolation_ratio"]
        )
        upstream_middle_flow_ratio = float(values["upstream_middle_flow_ratio"])
        downstream_percolation_ratio = float(values["downstream_percolation_ratio"])
        downstream_middle_flow_ratio = float(values["downstream_middle_flow_ratio"])
        upstream_deep_percolation_ratio = float(
            values["upstream_deep_percolation_ratio"]
        )

        n = self.n_steps
        up_inflow = self._up_inflow_list
        down_inflow = self._down_inflow_list
        evap_cand_up = self._evap_cand_up_list
        evap_cand_down = self._evap_cand_down_list

        upstream_storage_capacity = self.upstream_storage_capacity
        dam_capacity = self.dam_capacity
        drainage_capacity_day = self.drainage_capacity_per_day
        minimum_discharge = self.minimum_river_discharge
        indirect_ratio = 1.0 - direct_discharge_ratio

        # ストック（PySD の initial_condition="original" と同じ初期値）
        upstream_storage = _INITIAL_UPSTREAM_STORAGE
        upstream_underground = _INITIAL_UPSTREAM_UNDERGROUND
        downstream_storage = _INITIAL_DOWNSTREAM_STORAGE
        downstream_underground = _INITIAL_DOWNSTREAM_UNDERGROUND
        dam_storage = self.initial_dam_storage

        discharge_up = np.empty(n, dtype=float)
        discharge_down = np.empty(n, dtype=float)
        out_up = discharge_up
        out_down = discharge_down

        for t in range(n):
            # ---------------- upstream ----------------
            inflow_up = up_inflow[t]

            candidate = evap_cand_up[t]
            evaporation_upstream = (
                candidate if candidate < upstream_storage else upstream_storage
            ) * _EVAPORATION_SCALE

            upstream_percolation = upstream_percolation_ratio * inflow_up

            excessive = upstream_storage - upstream_storage_capacity
            if excessive < 0.0:
                excessive = 0.0
            upstream_outflow = upstream_storage * upstream_outflow_ratio + excessive
            if upstream_outflow < _UPSTREAM_OUTFLOW_FLOOR:
                upstream_outflow = _UPSTREAM_OUTFLOW_FLOOR

            middle_flow = upstream_underground * upstream_middle_flow_ratio
            upstream_deep_percolation = (
                upstream_underground * upstream_deep_percolation_ratio
            )

            # ---------------- dam ----------------
            dam_inflow = upstream_outflow * indirect_ratio
            fill = dam_storage / dam_capacity
            if fill > _PREDISCHARGE_LOWER_TRIGGER and fill < _PREDISCHARGE_UPPER_TRIGGER:
                release = dam_storage * _PREDISCHARGE_RELEASE_FRACTION
                predischarge_control = (
                    _PREDISCHARGE_CAPACITY
                    if _PREDISCHARGE_CAPACITY > release
                    else release
                )
            else:
                predischarge_control = 0.0
            dam_spill = dam_storage - dam_capacity
            if dam_spill < 0.0:
                dam_spill = 0.0
            dam_outflow = predischarge_control + dam_spill

            river_discharge_upstream = (
                upstream_outflow * direct_discharge_ratio + dam_outflow + middle_flow
            )
            if river_discharge_upstream < minimum_discharge:
                river_discharge_upstream = minimum_discharge

            # ---------------- downstream ----------------
            inflow_down = down_inflow[t]

            candidate_d = evap_cand_down[t]
            evaporation_downstream = (
                candidate_d if candidate_d < downstream_storage else downstream_storage
            ) * _EVAPORATION_SCALE

            perc_a = downstream_storage * downstream_percolation_ratio
            perc_b = downstream_storage - evaporation_downstream
            downstream_percolation = perc_a if perc_a < perc_b else perc_b
            if downstream_percolation < 0.0:
                downstream_percolation = 0.0

            out_a = downstream_outflow_ratio * downstream_storage
            out_b = downstream_storage - evaporation_downstream - downstream_percolation
            downstream_outflow = out_a if out_a < out_b else out_b
            if downstream_outflow < _DOWNSTREAM_OUTFLOW_FLOOR:
                downstream_outflow = _DOWNSTREAM_OUTFLOW_FLOOR

            drain_b = (
                downstream_storage
                - evaporation_downstream
                - downstream_percolation
                - downstream_outflow
            )
            drainage = drainage_capacity_day if drainage_capacity_day < drain_b else drain_b
            if drainage < 0.0:
                drainage = 0.0

            downstream_middle_flow = (
                downstream_underground * downstream_middle_flow_ratio
            )
            downstream_deep_percolation = (
                downstream_underground * downstream_deep_percolation_ratio
            )

            river_discharge_downstream = (
                river_discharge_upstream
                + downstream_outflow
                + downstream_middle_flow
            )
            if river_discharge_downstream < minimum_discharge:
                river_discharge_downstream = minimum_discharge

            out_up[t] = river_discharge_upstream
            out_down[t] = river_discharge_downstream

            # ---------------- Euler 更新 (dt = 1) ----------------
            upstream_storage += (
                inflow_up
                - evaporation_upstream
                - upstream_percolation
                - upstream_outflow
            )
            upstream_underground += (
                upstream_percolation - middle_flow - upstream_deep_percolation
            )
            downstream_storage += (
                inflow_down
                - downstream_outflow
                - downstream_percolation
                - drainage
                - evaporation_downstream
            )
            downstream_underground += (
                downstream_percolation
                - downstream_deep_percolation
                - downstream_middle_flow
            )
            dam_storage += dam_inflow - dam_outflow

        return {
            "river_discharge_upstream": discharge_up,
            "river_discharge_downstream": discharge_down,
        }

    # ------------------------------------------------------------------
    # 便利メソッド
    # ------------------------------------------------------------------
    def run_with_states(self, params: dict | None = None) -> dict[str, np.ndarray]:
        """中間変数も含めて返す（検証・デバッグ用。run() より遅い）。"""
        values = dict(self.params_yaml)
        if params:
            values.update(params)
        keys = [
            "river_discharge_upstream",
            "river_discharge_downstream",
            "upstream_storage",
            "upstream_underground",
            "downstream_storage",
            "downstream_underground",
            "dam_storage",
            "upstream_outflow",
            "excessive_surface_flow",
            "middle_flow",
            "dam_inflow",
            "dam_outflow",
            "predischarge_control",
            "dam_spill",
            "upstream_inflow",
            "upstream_percolation",
            "upstream_deep_percolation",
            "evaporation_upstream",
            "downstream_inflow",
            "downstream_outflow",
            "downstream_percolation",
            "drainage",
            "evaporation_downstream",
            "downstream_middle_flow",
            "downstream_deep_percolation",
        ]
        n = self.n_steps
        rec = {key: np.empty(n, dtype=float) for key in keys}

        upstream_outflow_ratio = float(values["upstream_outflow_ratio"])
        downstream_outflow_ratio = float(values["downstream_outflow_ratio"])
        direct_discharge_ratio = float(values["direct_discharge_ratio"])
        upstream_percolation_ratio = float(values["upstream_percolation_ratio"])
        downstream_deep_percolation_ratio = float(
            values["downstream_deep_percolation_ratio"]
        )
        upstream_middle_flow_ratio = float(values["upstream_middle_flow_ratio"])
        downstream_percolation_ratio = float(values["downstream_percolation_ratio"])
        downstream_middle_flow_ratio = float(values["downstream_middle_flow_ratio"])
        upstream_deep_percolation_ratio = float(
            values["upstream_deep_percolation_ratio"]
        )

        upstream_storage = _INITIAL_UPSTREAM_STORAGE
        upstream_underground = _INITIAL_UPSTREAM_UNDERGROUND
        downstream_storage = _INITIAL_DOWNSTREAM_STORAGE
        downstream_underground = _INITIAL_DOWNSTREAM_UNDERGROUND
        dam_storage = self.initial_dam_storage
        dam_capacity = self.dam_capacity

        for t in range(n):
            upstream_inflow = float(self.upstream_inflow[t])
            candidate = float(self.evap_candidate_up[t])
            evaporation_upstream = (
                candidate if candidate < upstream_storage else upstream_storage
            ) * _EVAPORATION_SCALE
            upstream_percolation = upstream_percolation_ratio * upstream_inflow
            excessive_surface_flow = max(
                upstream_storage - self.upstream_storage_capacity, 0.0
            )
            upstream_outflow = max(
                upstream_storage * upstream_outflow_ratio + excessive_surface_flow,
                _UPSTREAM_OUTFLOW_FLOOR,
            )
            middle_flow = upstream_underground * upstream_middle_flow_ratio
            upstream_deep_percolation = (
                upstream_underground * upstream_deep_percolation_ratio
            )
            dam_inflow = upstream_outflow * (1.0 - direct_discharge_ratio)
            fill = dam_storage / dam_capacity
            if fill > _PREDISCHARGE_LOWER_TRIGGER:
                if fill < _PREDISCHARGE_UPPER_TRIGGER:
                    predischarge_control = max(
                        _PREDISCHARGE_CAPACITY,
                        dam_storage * _PREDISCHARGE_RELEASE_FRACTION,
                    )
                else:
                    predischarge_control = 0.0
            else:
                predischarge_control = 0.0
            dam_spill = max(dam_storage - dam_capacity, 0.0)
            dam_outflow = predischarge_control + dam_spill
            river_discharge_upstream = max(
                upstream_outflow * direct_discharge_ratio + dam_outflow + middle_flow,
                self.minimum_river_discharge,
            )

            downstream_inflow = float(self.downstream_inflow[t])
            candidate_d = float(self.evap_candidate_down[t])
            evaporation_downstream = (
                candidate_d if candidate_d < downstream_storage else downstream_storage
            ) * _EVAPORATION_SCALE
            downstream_percolation = max(
                min(
                    downstream_storage * downstream_percolation_ratio,
                    downstream_storage - evaporation_downstream,
                ),
                0.0,
            )
            downstream_outflow = max(
                min(
                    downstream_outflow_ratio * downstream_storage,
                    downstream_storage
                    - evaporation_downstream
                    - downstream_percolation,
                ),
                _DOWNSTREAM_OUTFLOW_FLOOR,
            )
            drainage = max(
                min(
                    self.drainage_capacity_per_day,
                    downstream_storage
                    - evaporation_downstream
                    - downstream_percolation
                    - downstream_outflow,
                ),
                0.0,
            )
            downstream_middle_flow = (
                downstream_underground * downstream_middle_flow_ratio
            )
            downstream_deep_percolation = (
                downstream_underground * downstream_deep_percolation_ratio
            )
            river_discharge_downstream = max(
                river_discharge_upstream
                + downstream_outflow
                + downstream_middle_flow,
                self.minimum_river_discharge,
            )

            local = locals()
            for key in keys:
                rec[key][t] = local[key]

            upstream_storage += (
                upstream_inflow
                - evaporation_upstream
                - upstream_percolation
                - upstream_outflow
            )
            upstream_underground += (
                upstream_percolation - middle_flow - upstream_deep_percolation
            )
            downstream_storage += (
                downstream_inflow
                - downstream_outflow
                - downstream_percolation
                - drainage
                - evaporation_downstream
            )
            downstream_underground += (
                downstream_percolation
                - downstream_deep_percolation
                - downstream_middle_flow
            )
            dam_storage += dam_inflow - dam_outflow

        return rec


# ----------------------------------------------------------------------
# PySD 版との一致検証（回帰テスト）
# ----------------------------------------------------------------------

_VERIFY_ZERO_EXOGENOUS = {
    "levee_investment_amount": 0,
    "dam_investment_amount": 0,
    "drainage_investment_amount": 0,
    "number_of_house_elevation": 0,
    "number_of_migration": 0,
    "annual_paddy_dam_investment": 0,
    "annual_breeding_investment": 0,
    "daily_precipitation_future_ratio": 1.0,
    "temperature_scenario_shift": 0.0,
}

_VERIFY_TARGETS = ["river_discharge_upstream", "river_discharge_downstream"]

# 一致しないときに二分探索的に追う中間変数（上流 -> 下流の因果順）
_VERIFY_DIAGNOSTIC_COLUMNS = [
    "upstream_inflow",
    "evaporation_upstream",
    "upstream_percolation",
    "excessive_surface_flow",
    "upstream_outflow",
    "upstream_storage",
    "middle_flow",
    "upstream_deep_percolation",
    "upstream_underground",
    "dam_inflow",
    "predischarge_control",
    "dam_spill",
    "dam_outflow",
    "dam_storage",
    "river_discharge_upstream",
    "downstream_inflow",
    "evaporation_downstream",
    "downstream_percolation",
    "downstream_outflow",
    "drainage",
    "downstream_storage",
    "downstream_middle_flow",
    "downstream_deep_percolation",
    "downstream_underground",
    "river_discharge_downstream",
]


def _verify_load_config(basin: str, repo: str | None = None) -> dict:
    root = Path(repo) if repo else _repo_dir()
    with (root / "config" / "basins" / f"{basin}.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _verify_make_parameter_sets(config: dict, seed: int = 20260818) -> list[tuple[str, dict]]:
    bounds = config["hydrology_calibration"]["bounds"]
    names = list(bounds)
    sets = [("YAML既定値", {name: float(config["model_parameters"][name]) for name in names})]
    rng = np.random.default_rng(seed)
    for index in range(5):
        candidate = {}
        for name in names:
            low, high = map(float, bounds[name])
            candidate[name] = float(rng.uniform(low, high))
        sets.append((f"乱数{index + 1}", candidate))
    return sets


def _verify_errors(reference: np.ndarray, emulated: np.ndarray) -> tuple[float, float, int]:
    absolute = np.abs(emulated - reference)
    denominator = np.abs(reference)
    relative = np.where(denominator > 0, absolute / np.maximum(denominator, 1e-300), absolute)
    return float(absolute.max()), float(relative.max()), int(np.argmax(relative))


def _verify__verify_diagnose(model, fast, parameters, base_params, timestamps) -> None:
    """どの中間変数から食い違うかを因果順に調べる。"""
    print("  --- 中間変数の突き合わせ ---")
    params = dict(base_params)
    params.update(parameters)
    frame = model.run(
        params=params,
        return_timestamps=timestamps,
        return_columns=_VERIFY_DIAGNOSTIC_COLUMNS,
        initial_condition="original",
    )
    states = fast.run_with_states(parameters)
    for column in _VERIFY_DIAGNOSTIC_COLUMNS:
        if column not in states:
            continue
        reference = frame[column].to_numpy()
        emulated = states[column]
        max_abs, max_rel, where = _verify_errors(reference, emulated)
        flag = "NG" if max_rel >= 1e-9 else "ok"
        print(
            f"  {flag} {column:<32} maxabs={max_abs:.6e} maxrel={max_rel:.6e} "
            f"first-bad-index={int(np.argmax(np.abs(emulated - reference) > 0)) if max_abs > 0 else -1} "
            f"argmaxrel={where}"
        )


def verify_against_pysd(basin: str | None = None) -> None:
    import time
    import warnings

    warnings.filterwarnings("ignore")
    from pysd import load

    if basin is None:
        basin = os.environ.get("RIVER_BASIN", "abukuma")
    os.environ["RIVER_BASIN"] = basin
    REPO = str(_repo_dir())

    config = _verify_load_config(basin, REPO)
    calendar = config["calendar"]
    days = FastHydrology._total_days(config)
    timestamps = list(range(days))
    print(f"流域: {basin} / 期間: {days} 日 "
          f"({calendar['start_year']}年から{calendar['num_years']}年)")

    base_params = dict(config["model_parameters"])
    base_params.update(_VERIFY_ZERO_EXOGENOUS)

    t0 = time.perf_counter()
    model = load(str(Path(REPO) / "River_management_xls_to6.py"))
    print(f"PySD モデル読み込み: {time.perf_counter() - t0:.2f} 秒")

    t0 = time.perf_counter()
    fast = FastHydrology(basin, repo_dir=REPO)
    print(f"FastHydrology 初期化: {time.perf_counter() - t0:.2f} 秒")

    parameter_sets = _verify_make_parameter_sets(config)
    rows = []
    pysd_times = []
    fast_times = []
    failures = []

    for label, parameters in parameter_sets:
        params = dict(base_params)
        params.update(parameters)

        t0 = time.perf_counter()
        frame = model.run(
            params=params,
            return_timestamps=timestamps,
            return_columns=_VERIFY_TARGETS + ["flow", "flow_d"],
            initial_condition="original",
        )
        pysd_elapsed = time.perf_counter() - t0
        pysd_times.append(pysd_elapsed)

        t0 = time.perf_counter()
        emulated = fast.run(parameters)
        fast_elapsed = time.perf_counter() - t0
        fast_times.append(fast_elapsed)

        row = {"パラメータ": label, "PySD秒": pysd_elapsed, "Fast秒": fast_elapsed}
        bad = False
        for target in _VERIFY_TARGETS:
            max_abs, max_rel, where = _verify_errors(frame[target].to_numpy(), emulated[target])
            key = "上流" if target.endswith("upstream") else "下流"
            row[f"{key}_最大絶対誤差"] = max_abs
            row[f"{key}_最大相対誤差"] = max_rel
            row[f"{key}_相対誤差最大index"] = where
            if not (max_rel < 1e-9):
                bad = True
        rows.append(row)
        status = "NG" if bad else "OK"
        print(
            f"[{status}] {label}: "
            f"上流 maxabs={row['上流_最大絶対誤差']:.6e} maxrel={row['上流_最大相対誤差']:.6e} / "
            f"下流 maxabs={row['下流_最大絶対誤差']:.6e} maxrel={row['下流_最大相対誤差']:.6e} "
            f"(PySD {pysd_elapsed:.2f}s, Fast {fast_elapsed * 1000:.2f}ms)"
        )
        if bad:
            failures.append((label, parameters))
            _verify_diagnose(model, fast, parameters, base_params, timestamps)

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.6e}")
    print("\n=== 一致検証まとめ ===")
    print(table.to_string(index=False))

    # 速度（キャッシュの影響を避けるため FastHydrology は複数回計測する）
    repeats = 50
    default_parameters = parameter_sets[0][1]
    t0 = time.perf_counter()
    for _ in range(repeats):
        fast.run(default_parameters)
    fast_mean = (time.perf_counter() - t0) / repeats
    pysd_mean = float(np.mean(pysd_times))
    print("\n=== 速度 ===")
    print(f"PySD        : {pysd_mean:.3f} 秒/回 (n={len(pysd_times)})")
    print(f"FastHydrology: {fast_mean * 1000:.3f} ミリ秒/回 (n={repeats})")
    print(f"倍率        : {pysd_mean / fast_mean:.0f} 倍")

    if failures:
        print(f"\n不一致 {len(failures)} 件: " + ", ".join(label for label, _ in failures))
        raise SystemExit(1)
    print("\n全 6 通りで相対誤差 < 1e-9 を満たした。")


__all__ = ["FastHydrology", "verify_against_pysd"]


if __name__ == "__main__":
    verify_against_pysd()
