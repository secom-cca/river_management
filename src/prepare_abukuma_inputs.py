"""阿武隈川の観測データを River_management モデルの入力形式へ変換する。

入力（DMDU研究/data、いずれも 2009-01-01 から 2020-12-31 の 4383 日）
  郡山2009-2020.csv   JMA 郡山       -> weather_up   （福島地点より上流の代表点）
  丸森2009-2020.csv   JMA 丸森       -> weather_down （福島から舘矢間の残流域の代表点）
  福島_日流量.csv      MLIT 福島      -> flow_up      （m3/s）
  舘矢間_日流量.csv    MLIT 舘矢間    -> flow_down    （m3/s）

出力（リポジトリの data/）
  jma_koriyama_2009_2020.xlsx   シート input、A列=モデル日、B降水量、C平均気温、D最高気温、E最低気温、F全天日射量
  jma_marumori_2009_2020.xlsx   同上
  flow_fukushima_2009_2020_m3day_utf8.csv   A列=モデル日、D列=観測流量（m3/day）
  flow_tateyama_2009_2020_m3day_utf8.csv    同上
  obs_mask_abukuma.csv          観測の欠測日フラグ（較正スクリプトで NSE から除外するため）

日射量について
  丸森・郡山とも合計全天日射量は全期間欠測（日射計がない）。モデルの Hargreaves 式は
  全天日射量 Rs を要求するため、観測されている日照時間 n から Angstrom-Prescott 式
      Rs = (a_s + b_s * n / N) * Ra
  で推定する。a_s = 0.25、b_s = 0.50 は FAO-56（Allen et al., 1998, Chapter 3, Eq.35）
  の推奨既定値。Ra（大気外日射量）と N（可照時間）は同 Eq.21, Eq.34 により緯度と
  年内通日から計算する。
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd

# このファイルは src/ に置く想定。リポジトリのルートは1つ上。
REPO = Path(os.environ.get("RIVER_MODEL_DIR",
    Path(__file__).resolve().parent.parent)).expanduser()

SRC = Path(os.environ.get("ABUKUMA_DATA_DIR",
    Path.home() / "Claude/Projects/DMDU研究/data")).expanduser()
OUT = REPO / "data"
OUT.mkdir(parents=True, exist_ok=True)

# JMA 観測所の緯度（度）。気象庁「地点の情報」より。
LATITUDE = {"郡山": 37.3983, "丸森": 37.9050}

GSC = 0.0820  # 太陽定数 MJ m-2 min-1 (FAO-56)
A_S, B_S = 0.25, 0.50  # Angstrom-Prescott 係数 (FAO-56 既定値)


def extraterrestrial_radiation(doy: np.ndarray, lat_deg: float):
    """FAO-56 Eq.21 の Ra [MJ m-2 day-1] と Eq.34 の N [hour] を返す。"""
    phi = np.deg2rad(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)          # Eq.23
    delta = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)     # Eq.24
    x = np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0)
    omega_s = np.arccos(x)                                     # Eq.25
    ra = (24 * 60 / np.pi) * GSC * dr * (
        omega_s * np.sin(phi) * np.sin(delta)
        + np.cos(phi) * np.cos(delta) * np.sin(omega_s)
    )
    n_max = 24.0 / np.pi * omega_s                             # Eq.34
    return ra, n_max


def read_jma(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    names, sub = rows[3], rows[5]
    frame = pd.DataFrame(rows[6:], columns=[f"{i}:{a}" for i, a in enumerate(names)])
    value_cols = [c for i, c in enumerate(frame.columns) if sub[i] == ""]
    out = frame[value_cols].copy()
    out.columns = [c.split(":", 1)[1] for c in value_cols]
    out = out.rename(columns={"年月日": "date"})
    out["date"] = pd.to_datetime(out["date"])
    for col in out.columns[1:]:
        out[col] = pd.to_numeric(
            out[col].replace({"": np.nan, "--": np.nan, "×": np.nan, "///": np.nan}),
            errors="coerce",
        )
    return out


def build_weather(station: str, src_name: str, out_name: str) -> pd.DataFrame:
    raw = read_jma(SRC / src_name)
    doy = raw["date"].dt.dayofyear.to_numpy(dtype=float)
    ra, n_max = extraterrestrial_radiation(doy, LATITUDE[station])

    sunshine = raw["日照時間(時間)"].to_numpy(dtype=float)
    ratio = np.clip(sunshine / n_max, 0.0, 1.0)
    rs = (A_S + B_S * ratio) * ra

    frame = pd.DataFrame(
        {
            "day": np.arange(len(raw), dtype=float),
            "precip_mm": raw["降水量の合計(mm)"].to_numpy(dtype=float),
            "temp_ave_c": raw["平均気温(℃)"].to_numpy(dtype=float),
            "temp_max_c": raw["最高気温(℃)"].to_numpy(dtype=float),
            "temp_min_c": raw["最低気温(℃)"].to_numpy(dtype=float),
            "solar_mj_m2": rs,
        }
    )

    filled = frame.copy()
    gap = int(filled.iloc[:, 1:].isna().sum().sum())
    filled.iloc[:, 1:] = (
        filled.iloc[:, 1:].interpolate(limit_direction="both")
    )
    with pd.ExcelWriter(OUT / out_name, engine="openpyxl") as writer:
        filled.to_excel(writer, sheet_name="input", index=False, header=False)
    print(
        f"{station}: {out_name}  {len(filled)}日  気象欠測を線形補間したセル数={gap}  "
        f"降水平均={filled.precip_mm.mean():.2f} mm/day  "
        f"Rs平均={filled.solar_mj_m2.mean():.2f} MJ/m2/day  "
        f"Rs範囲={filled.solar_mj_m2.min():.2f}-{filled.solar_mj_m2.max():.2f}"
    )
    return frame


def build_flow(src_name: str, out_name: str, label: str) -> pd.Series:
    raw = pd.read_csv(SRC / src_name, encoding="utf-8-sig")
    raw.columns = ["date", "q_m3s"]
    raw["date"] = pd.to_datetime(raw["date"])
    observed = raw["q_m3s"].to_numpy(dtype=float)
    q_day = observed * 86400.0

    frame = pd.DataFrame(
        {
            "day": np.arange(len(raw), dtype=float),
            "b": np.nan,
            "c": np.nan,
            "flow_m3day": q_day,
        }
    )
    # ExtData は NaN を扱えないので、欠測日は前後の実測から線形補間して埋める。
    # 実測がある日だけを NSE に使うためのマスクは別ファイルに出す。
    frame["flow_m3day"] = frame["flow_m3day"].interpolate(limit_direction="both")
    frame.to_csv(OUT / out_name, index=False, header=False, encoding="utf-8")
    missing = int(np.isnan(q_day).sum())
    print(
        f"{label}: {out_name}  {len(frame)}日  実測欠測={missing}日 "
        f"({missing / len(frame) * 100:.1f}%)  "
        f"平均={np.nanmean(q_day):.4g} m3/day ({np.nanmean(observed):.1f} m3/s)  "
        f"最大={np.nanmax(observed):.0f} m3/s"
    )
    return pd.Series(np.isfinite(q_day), name=label)


def main() -> None:
    build_weather("郡山", "郡山2009-2020.csv", "jma_koriyama_2009_2020.xlsx")
    build_weather("丸森", "丸森2009-2020.csv", "jma_marumori_2009_2020.xlsx")
    up = build_flow("福島_日流量.csv", "flow_fukushima_2009_2020_m3day_utf8.csv", "flow_up")
    down = build_flow("舘矢間_日流量.csv", "flow_tateyama_2009_2020_m3day_utf8.csv", "flow_down")

    mask = pd.DataFrame({"day": np.arange(len(up)), "flow_up": up, "flow_down": down})
    mask.to_csv(OUT / "obs_mask_abukuma.csv", index=False, encoding="utf-8")
    both = int((up & down).sum())
    print(f"上下流とも実測がある日: {both} / {len(mask)} ({both / len(mask) * 100:.1f}%)")


if __name__ == "__main__":
    main()
