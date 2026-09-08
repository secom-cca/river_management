"""PySD の外部データ参照（ExtData）を numpy 直参照に置き換える高速化パッチ。

背景
    River_management_xls_to6.py を 4,383 日ぶん回すと、実行時間の約 77% が
    pysd/py_backend/data.py の Data.__call__ に費やされている（cProfile 実測）。
    このメソッドは 1 日 1 変数ごとに

        if time in self.data["time"].values   # 4,383 要素の線形走査
        outdata = self.data.sel(time=time)    # xarray の座標検索
        float(outdata)                        # 0 次元 DataArray からの変換

    を行う。気象・流量あわせて 12 系列 × 4,383 日 = 約 52,600 回呼ばれるため、
    1 回あたり 370 マイクロ秒でも合計 19 秒になる。

やること
    Data.__call__ を差し替え、対象が「1 次元・時間軸が単調増加・スカラー値・
    interp='interpolate'」の場合に限り np.interp による直参照に切り替える。
    条件を満たさないものは元の実装をそのまま呼ぶ。

    np.interp は端点で自動的にクランプするため、時間軸の範囲外では元実装の
    「最初/最後の値を返す」挙動と同じ値になる。ただし元実装が出す
    extrapolating 警告は出ない（呼び出し側スクリプトは元々この警告を
    filterwarnings で抑制している）。

    モデルファイル（River_management_xls_to6.py）は一切変更しない。

使い方
    import pysd_fast_extdata
    pysd_fast_extdata.enable()
    model = pysd.load("River_management_xls_to6.py")

    値が元実装と一致するかは verify() で確認できる:
        python pysd_fast_extdata.py
"""

from __future__ import annotations

import numpy as np
from pysd.py_backend.data import Data

_ORIGINAL_CALL = Data.__call__
_CACHE_KEY = "_fast_extdata_lookup"


def _build_lookup(obj: Data):
    """高速参照に使える場合は (time配列, 値配列) を、使えない場合は False を返す。"""
    try:
        array = obj.data
        if array is None:
            return False
        if getattr(obj, "interp", None) != "interpolate":
            return False
        if tuple(getattr(array, "dims", ())) != ("time",):
            return False
        times = np.asarray(array["time"].values, dtype=float)
        values = np.asarray(array.values, dtype=float)
        if times.ndim != 1 or values.shape != times.shape or times.size < 2:
            return False
        steps = np.diff(times)
        if not np.all(steps > 0):
            return False
        if not np.all(np.isfinite(times)):
            return False
    except Exception:
        return False

    # 元実装と値が一致することをその場で確認する。
    # 時間軸内の代表点と、範囲外（クランプされる側）を見る。
    probes = [times[0], times[times.size // 2], times[-1], times[0] - 1.0, times[-1] + 1.0]
    for probe in probes:
        try:
            expected = _ORIGINAL_CALL(obj, probe)
        except Exception:
            return False
        if not isinstance(expected, float):
            return False
        if not np.isclose(expected, float(np.interp(probe, times, values)),
                          rtol=0.0, atol=0.0, equal_nan=True):
            return False
    # 時間軸が刻み 1 の連続整数なら、np.interp を介さず添字で直接引ける。
    if np.all(steps == 1.0) and float(times[0]) == int(times[0]):
        return times, values, int(times[0])
    return times, values, None


def _fast_call(self, time):
    lookup = self.__dict__.get(_CACHE_KEY)
    if lookup is None:
        lookup = _build_lookup(self)
        self.__dict__[_CACHE_KEY] = lookup
    if lookup is False:
        return _ORIGINAL_CALL(self, time)
    times, values, origin = lookup
    if origin is not None:
        index = int(time) - origin
        if index == time - origin:            # 整数時刻ならクランプして直接参照
            if index < 0:
                return values[0]
            if index >= values.size:
                return values[-1]
            return values[index]
    return float(np.interp(time, times, values))


_ORIGINAL_SET_VALUES = Data.set_values


def _set_values_invalidating(self, values):
    """ユーザーが値を差し替えたらキャッシュを捨てる。"""
    self.__dict__.pop(_CACHE_KEY, None)
    return _ORIGINAL_SET_VALUES(self, values)


def enable() -> None:
    Data.__call__ = _fast_call
    Data.set_values = _set_values_invalidating


def disable() -> None:
    Data.__call__ = _ORIGINAL_CALL
    Data.set_values = _ORIGINAL_SET_VALUES


def verify(basin: str | None = None, columns: list[str] | None = None) -> None:
    """パッチの有無で全出力が一致するかと、速度差を確認する。"""
    import os
    import time as _time
    import warnings
    from pathlib import Path

    warnings.filterwarnings("ignore")
    import pandas as pd
    from pysd import load

    from basin_config import load_active_basin

    if basin:
        os.environ["RIVER_BASIN"] = basin
    config = load_active_basin()
    base_dir = Path(__file__).resolve().parent
    model_py = str(base_dir / "River_management_xls_to6.py")

    if columns is None:
        columns = [
            "river_discharge_upstream", "river_discharge_downstream",
            "financial_damage_by_flood", "financial_damage_by_innundation",
            "landslide_disaster_risk", "crop_production_cashflow",
            "biodiversity", "daily_total_gdp", "yearly_crop_production",
            "innundation_level", "flood_water_level", "yield_per_10a",
        ]

    def total_days(cfg):
        calendar = cfg["calendar"]
        start, years = int(calendar["start_year"]), int(calendar["num_years"])
        if not calendar["use_leap_years"]:
            return 365 * years
        leap = lambda y: y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
        return sum(366 if leap(start + i) else 365 for i in range(years))

    timestamps = list(range(total_days(config)))
    params = dict(config["model_parameters"])
    params.update({
        "daily_precipitation_future_ratio": 1.0, "temperature_scenario_shift": 0.0,
        "levee_investment_amount": 0, "dam_investment_amount": 0,
        "drainage_investment_amount": 0, "number_of_house_elevation": 0,
        "number_of_migration": 0, "annual_paddy_dam_investment": 0,
        "annual_breeding_investment": 0,
    })

    def timed_run(repeats: int):
        model = load(model_py)
        model.run(params=params, return_timestamps=timestamps,
                  return_columns=columns, initial_condition="original")  # ウォームアップ
        elapsed = []
        for _ in range(repeats):
            t0 = _time.perf_counter()
            frame = model.run(params=params, return_timestamps=timestamps,
                              return_columns=columns, initial_condition="original")
            elapsed.append(_time.perf_counter() - t0)
        return frame, float(np.mean(elapsed))

    print(f"流域: {config['key']} / {len(timestamps)} 日 / {len(columns)} 列")
    disable()
    reference, slow = timed_run(2)
    enable()
    patched, fast = timed_run(3)

    print("\n=== 一致検証 ===")
    worst = 0.0
    for column in columns:
        a = reference[column].to_numpy(dtype=float)
        b = patched[column].to_numpy(dtype=float)
        max_abs = float(np.nanmax(np.abs(b - a))) if a.size else 0.0
        denominator = np.where(np.abs(a) > 0, np.abs(a), 1.0)
        max_rel = float(np.nanmax(np.abs(b - a) / denominator)) if a.size else 0.0
        worst = max(worst, max_rel)
        flag = "ok" if max_rel == 0.0 else ("近似" if max_rel < 1e-12 else "NG")
        print(f"  {flag:4s} {column:38s} maxabs={max_abs:.3e} maxrel={max_rel:.3e}")

    print("\n=== 速度 ===")
    print(f"  パッチなし: {slow:6.2f} 秒/回")
    print(f"  パッチあり: {fast:6.2f} 秒/回")
    print(f"  倍率      : {slow / fast:5.2f} 倍")

    if worst > 0.0:
        raise SystemExit(f"出力が一致しない（最大相対誤差 {worst:.3e}）")
    print("\n全列で完全一致。")


if __name__ == "__main__":
    verify()
