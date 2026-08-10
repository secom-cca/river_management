# compute_nies_metrics.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ===== 設定 =====
NIES_DIR = Path("data/nies2020")        # CSV の所在
OUT_DIR  = Path("out_metrics")      # 出力先
YEARS    = list(range(2015, 2101))  # 2025–2100
SSPS     = ["126", "245", "585"]    # 対象SSP
VARS     = ["pr", "tas", "tasmax", "tasmin"]
THRESHOLDS = [50, 100, 150, 200]    # 極端降水のしきい値 (mm/day)
ROLLING_WINDOW = 10                 # Gumbel推定の移動窓幅 (年)


def _csv_path(var: str, ssp: str) -> Path:
    return NIES_DIR / f"national_average_{var}_ssp{ssp}.csv"


def _parse_csv(var: str, ssp: str) -> Tuple[pd.DataFrame, List[Dict], List[str], List[int]]:
    """
    national_average_<var>_ssp<ssp>.csv を読み、列名から (model, year) を抽出。
    戻り値: (base_df(index=time), entries, models, years)
    """
    path = _csv_path(var, ssp)
    if not path.exists():
        raise FileNotFoundError(f"{path} が見つかりません")
    df = pd.read_csv(path)
    if "time" not in df.columns:
        df = df.rename(columns={df.columns[0]: "time"})
    base = df.set_index("time")

    pat = re.compile(rf"^{re.escape(var)}_(?P<model>.+?)_ssp{ssp}_.+?_(?P<year>\d{{4}})$")
    entries, models, years = [], set(), set()
    for c in base.columns:
        m = pat.match(str(c))
        if m:
            model = m.group("model"); year = int(m.group("year"))
            entries.append({"model": model, "year": year, "col": c})
            models.add(model); years.add(year)
    return base, entries, sorted(models), sorted(years)


def _extract_one_year(base: pd.DataFrame, entries: List[Dict], model: str, year: int, var: str) -> pd.Series:
    for e in entries:
        if e["model"] == model and e["year"] == year:
            s = pd.to_numeric(base[e["col"]], errors="coerce")
            s.index.name = "doy"
            s.name = var
            return s
    raise KeyError(f"{var}: {model} {year} 列がありません")


def _series_to_dates(year: int, s: pd.Series) -> pd.Series:
    """ day-of-year index (0..365) を日付 index に変換 """
    start = pd.Timestamp(f"{year}-01-01")
    dates = [start + pd.Timedelta(days=int(d)) for d in s.index]
    s = pd.Series(s.values, index=pd.to_datetime(dates), name=s.name)
    return s


def _drop_feb29(s: pd.Series) -> pd.Series:
    """ 閏日の 2/29 を常に削除（年比較を365日で揃える） """
    if not isinstance(s.index, pd.DatetimeIndex):
        return s
    return s[~((s.index.month == 2) & (s.index.day == 29))]


def _choose_5_models(ssp: str) -> List[str]:
    """
    全変数（pr/tas/tasmax/tasmin）に共通して存在するモデルのうち、代表5本を選ぶ。
    優先順を定義し、無ければ残りから先頭順。
    """
    bases, entries, models_sets = {}, {}, []
    for v in VARS:
        b, e, ms, _ = _parse_csv(v, ssp)
        bases[v] = b
        entries[v] = e
        models_sets.append(set(ms))

    common = set.intersection(*models_sets)
    if not common:
        raise RuntimeError(f"SSP{ssp}: 4変数に共通するモデルが見つかりません")

    preferred = ["MIROC6", "MRI-ESM2-0", "IPSL-CM6A-LR", "ACCESS-CM2", "MPI-ESM1-2-HR"]
    chosen = [m for m in preferred if m in common]
    if len(chosen) < 5:
        # 残りを common の辞書順から補完
        rest = [m for m in sorted(common) if m not in chosen]
        chosen += rest[: (5 - len(chosen))]
    return chosen[:5]


def _gumbel_params_from_ams(ams: pd.Series) -> Tuple[float, float]:
    """
    Gumbel(EV1) のモーメント法推定。
    AMS（年最大日降水量）の平均 m, 分散 v から:
      β = sqrt(6 v) / π
      μ = m - γ β （γ=0.57721566）
    """
    gamma = 0.57721566
    m = float(np.mean(ams.values))
    v = float(np.var(ams.values, ddof=0))
    beta = 0.0 if v <= 0 else np.sqrt(6.0 * v) / np.pi
    mu = m - gamma * beta
    return mu, beta


def compute_for_ssp(ssp: str) -> None:
    print(f"== SSP{ssp} ==")

    # それぞれ読み込み（1回だけ）
    bases: Dict[str, pd.DataFrame] = {}
    entries: Dict[str, List[Dict]] = {}
    for v in VARS:
        b, e, _, _ = _parse_csv(v, ssp)
        bases[v] = b; entries[v] = e

    models = _choose_5_models(ssp)
    print("Models:", ", ".join(models))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    gumbel_rows: List[Dict] = []

    for model in models:
        # 年ごとの集計
        ams_by_year: Dict[int, float] = {}

        for y in YEARS:
            try:
                pr  = _extract_one_year(bases["pr"],     entries["pr"],     model, y, "pr")
                tas = _extract_one_year(bases["tas"],    entries["tas"],    model, y, "tas")
                tmx = _extract_one_year(bases["tasmax"], entries["tasmax"], model, y, "tasmax")
                tmn = _extract_one_year(bases["tasmin"], entries["tasmin"], model, y, "tasmin")
            except KeyError:
                # どれか無ければスキップ
                continue

            # 日付化＋2/29削除（比較のため常に365日）
            pr_d  = _drop_feb29(_series_to_dates(y, pr))
            tas_d = _drop_feb29(_series_to_dates(y, tas))
            tmx_d = _drop_feb29(_series_to_dates(y, tmx))
            tmn_d = _drop_feb29(_series_to_dates(y, tmn))

            # 年別集計
            tas_mean    = float(pd.to_numeric(tas_d, errors="coerce").mean())
            tasmax_mean = float(pd.to_numeric(tmx_d, errors="coerce").mean())
            tasmin_mean = float(pd.to_numeric(tmn_d, errors="coerce").mean())
            pr_sum      = float(pd.to_numeric(pr_d,  errors="coerce").sum())  # mm/year

            # 極端降水日数
            ext_counts = {}
            pr_clean = pd.to_numeric(pr_d, errors="coerce").fillna(0.0)
            for thr in THRESHOLDS:
                ext_counts[f"pr_ge{thr}_days"] = int((pr_clean >= thr).sum())

            # AMS：その年の最大日降水
            ams_by_year[y] = float(pr_clean.max())

            row = {
                "ssp": ssp,
                "model": model,
                "year": y,
                "tas_mean": tas_mean,
                "tasmax_mean": tasmax_mean,
                "tasmin_mean": tasmin_mean,
                "pr_sum_mm": pr_sum,
                **ext_counts,
            }
            rows.append(row)

        # Gumbel パラメータ（10年移動窓）
        years_avail = sorted(ams_by_year.keys())
        if len(years_avail) >= ROLLING_WINDOW:
            for i in range(0, len(years_avail) - ROLLING_WINDOW + 1):
                ys = years_avail[i : i + ROLLING_WINDOW]
                ams_window = pd.Series([ams_by_year[yy] for yy in ys], index=ys)
                mu, beta = _gumbel_params_from_ams(ams_window)
                gumbel_rows.append({
                    "ssp": ssp,
                    "model": model,
                    "window_start": ys[0],
                    "window_end": ys[-1],
                    "n_years": len(ys),
                    "gumbel_mu": mu,
                    "gumbel_beta": beta
                })

    # 出力
    df = pd.DataFrame(rows).sort_values(["model", "year"])
    df.to_csv(OUT_DIR / f"annual_metrics_ssp{ssp}_2025-2100_5gcm.csv", index=False, encoding="utf-8")

    gp = pd.DataFrame(gumbel_rows).sort_values(["model", "window_start"])
    gp.to_csv(OUT_DIR / f"annual_gumbel_params_ssp{ssp}_window{ROLLING_WINDOW}.csv", index=False, encoding="utf-8")

    print(f"  -> {OUT_DIR / f'annual_metrics_ssp{ssp}_2025-2100_5gcm.csv'}")
    print(f"  -> {OUT_DIR / f'annual_gumbel_params_ssp{ssp}_window{ROLLING_WINDOW}.csv'}")


def main():
    for ssp in SSPS:
        compute_for_ssp(ssp)
    print("Done.")


if __name__ == "__main__":
    main()
