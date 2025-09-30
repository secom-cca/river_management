# app.py
from __future__ import annotations
import io
import re
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from pysd import load, read_vensim


# =========================
# 設定
# =========================
DEFAULT_MODEL_PY  = Path("River_management_xls.py")
DEFAULT_MODEL_MDL = Path("River_management_xls.mdl")

# モデルの GET XLS DATA が参照する Excel（Vensim 側のファイル名・シート名に合わせる）
INPUT_XLSX_PATH = Path("input.xlsx")
INPUT_SHEET     = "input"

# NIES 未来気候ファイルディレクトリ候補（どちらかに置いてあればOK）
NIES_DIR_CANDIDATES = [Path("data/nies2020"), Path("data/nies")]


# ==== 流域プリセット（代表値セット） ====
PRESETS = {
    "筑後川流域": {
        "initial_dam_capacity": 74_200_000,      # m3
        "upstream_area": 157_585,                # ha
        "downstream_area": 143_951,              # ha
        "forest_area_ratio": 166_000/(198_500*0.9),
        "direct_discharge_ratio": 0.97,
        "current_highwater_discharge": 11_500,   # m3/s
        "paddy_field_ratio": 0.12,
    },
    "長良川流域": {
        "initial_dam_capacity": 8_500_000,      # m3
        "upstream_area": 178_650,               # ha
        "downstream_area": 19_850,              # ha
        "forest_area_ratio": 0.92,
        "direct_discharge_ratio": 0.97,
        "current_highwater_discharge": 8_900,   # m3/s
        "paddy_field_ratio": 0.8,
    },
    "利根川流域（例）": {
        "initial_dam_capacity": 200_000_000,
        "upstream_area": 420_000,
        "downstream_area": 600_000,
        "forest_area_ratio": 0.65,
        "direct_discharge_ratio": 0.96,
        "current_highwater_discharge": 22_000,
        "paddy_field_ratio": 0.10,
    },
}

DEFAULT_RETURN_COLS = [
    "daily_total_gdp",
    "dam_storage",
    "downstream_storage",
    "upstream_storage",
    "river_discharge_downstream",
    "houses_damaged_by_inundation",
    "financial_damage_by_innundation",
    "financial_damage_by_flood",
]

PARAM_SPECS = {
    "daily_precipitation_future_ratio": dict(label="将来降水補正（×）", min=0.5, max=2.0, step=0.01, value=1.0),
    "dam_investment_amount":           dict(label="ダム投資額（円/年）",           min=0, max=1_000_000_000, step=1_000_000, value=0),
    "levee_investment_amount":         dict(label="堤防投資額（円/年）",           min=0, max=100_000_000,  step=1_000_000,  value=0),
    "drainage_investment_amount":      dict(label="排水能力投資額（円/年）",       min=0, max=10_000_000_000, step=100_000_000, value=0),
    "annual_paddy_dam_investment":     dict(label="ため池（圃場）投資額（円/年）",   min=0, max=50_000_000,   step=1_000_000,  value=1_000_000),
    "dam_investment_start_time":       dict(label="ダム投資 開始時期（年）",        min=0, max=11, step=1, value=0),
    "levee_investment_start_time":     dict(label="堤防投資 開始時期（年）",        min=0, max=10, step=1, value=0),
    "eldery_people_ratio":             dict(label="高齢者比率", min=0.0, max=1.0, step=0.01, value=0.6),
    "capacity_building":               dict(label="防災力（避難率係数）", min=0.0, max=1.0, step=0.05, value=0.5),
    "outflow_rate_of_residents":       dict(label="住民流出率（/日）", min=0.0, max=0.1, step=0.0001, value=0.01/365),
    "inflow_rate_of_residents":        dict(label="住民流入率（/日）", min=0.0, max=0.1, step=0.0001, value=0.01/365),
    "ratio_of_paddy_field_in_risky_area": dict(label="リスク域の圃場比率", min=0.0, max=1.0, step=0.01, value=0.01),
    "paddy_field_ratio":                   dict(label="下流域における圃場比率", min=0.0, max=1.0, step=0.01, value=0.12),

    # 地域パラメータ
    "initial_dam_capacity": dict(label="初期ダム容量（m³）", min=0, max=500_000_000, step=100_000, value=74_200_000),
    "upstream_area":        dict(label="上流域面積（ha）",  min=1_000, max=1_000_000, step=100, value=157_585),
    "downstream_area":      dict(label="下流域面積（ha）",  min=1_000, max=1_000_000, step=100, value=143_951),
    "forest_area_ratio":    dict(label="森林面積比（-）",   min=0.0, max=1.0, step=0.001, value=166_000/(198_500*0.9)),
    "direct_discharge_ratio": dict(label="直接流出比（-）", min=0.0, max=1.0, step=0.001, value=1 - 40/1950),
    "current_highwater_discharge": dict(label="計画高水流量（m³/秒）", min=0, max=30_000, step=100, value=11_500),
}


# =========================
# 共通ユーティリティ
# =========================
@lru_cache(maxsize=4)
def _load_model_from_file(model_py: str|None, model_mdl: str|None):
    if model_py and Path(model_py).exists():
        return load(model_py)
    if model_mdl and Path(model_mdl).exists():
        return read_vensim(model_mdl)
    raise FileNotFoundError("モデルファイル（.py/.mdl）が見つかりません。")

def _load_model_fresh(use_py_first: bool, model_py_path: str, model_mdl_path: str):
    py = Path(model_py_path)
    mdl = Path(model_mdl_path)
    if use_py_first and py.exists():
        return load(str(py))
    if (not use_py_first) and mdl.exists():
        return read_vensim(str(mdl))
    if py.exists():
        return load(str(py))
    if mdl.exists():
        return read_vensim(str(mdl))
    raise FileNotFoundError("モデルファイルが見つかりません。")

def _run_simulation(model, params: Dict[str, Any], timestamps: List[float], return_cols: List[str]) -> pd.DataFrame:
    """
    外部データは GET XLS DATA で 'input.xlsx' を参照している前提（data=は使わない）
    """
    try:
        return model.run(params=params, return_timestamps=timestamps, return_columns=return_cols)
    except Exception as e:
        st.warning(f"選択列の一部が見つからない可能性があるため、全量実行して再抽出します。詳細: {e}")
        res = model.run(params=params, return_timestamps=timestamps)
        keep = [c for c in return_cols if c in res.columns]
        if missing := [c for c in return_cols if c not in res.columns]:
            st.warning(f"以下の列はモデルに存在しませんでした: {missing}")
        return res[keep] if keep else res

def _build_model_datetime_index(res: pd.DataFrame, start_date: pd.Timestamp) -> pd.DataFrame:
    res = res.copy()
    res.index = pd.to_datetime([start_date + pd.Timedelta(days=int(t)) for t in res.index])
    res.index.name = "date"
    return res

# 観測CSV（流量）読み込み
def _read_observed_csv(file: io.BytesIO, date_col: str, flow_col: str, unit: str) -> pd.DataFrame:
    df = pd.read_csv(file)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[[date_col, flow_col]].rename(columns={date_col: "date", flow_col: "obs_flow"})
    if unit == "m3/s":
        df["obs_flow"] = df["obs_flow"] * 86400.0  # m3/s → m3/day
    df = df.groupby("date", as_index=False).mean().set_index("date").sort_index()
    return df

# 指標＆ラグ
def _metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    mask = ~(y_true.isna() | y_pred.isna())
    yt = y_true[mask].astype(float)
    yp = y_pred[mask].astype(float)
    if len(yt) == 0:
        return {}
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    mae  = float(np.mean(np.abs(yp - yt)))
    bias = float(np.mean(yp - yt))
    corr = float(np.corrcoef(yt, yp)[0,1]) if len(yt)>1 else np.nan
    nse  = 1.0 - float(np.sum((yp-yt)**2) / np.sum((yt - yt.mean())**2)) if yt.nunique()>1 else np.nan
    return {"RMSE": rmse, "MAE": mae, "Mean Bias": bias, "Pearson r": corr, "NSE": nse}

def _best_lag(y_true: pd.Series, y_pred: pd.Series, max_lag_days: int) -> Tuple[int, float]:
    best_lag, best_r = 0, -np.inf
    for lag in range(-max_lag_days, max_lag_days+1):
        if lag > 0:
            r = y_true.corr(y_pred.shift(lag))
        elif lag < 0:
            r = y_true.shift(-lag).corr(y_pred)
        else:
            r = y_true.corr(y_pred)
        r = -1.0 if pd.isna(r) else float(r)
        if r > best_r:
            best_r, best_lag = r, lag
    return best_lag, best_r


# =========================
# NIES SSP CSV 読み込み＆成形
# =========================
def _nies_csv_path(var: str, ssp_code: str|int) -> Path:
    fname = f"national_average_{var}_ssp{ssp_code}.csv"
    for base in NIES_DIR_CANDIDATES:
        p = base / fname
        if p.exists():
            return p
    # 見つからなければ候補パスを列挙してエラー
    cand = ", ".join(str((base / fname).resolve()) for base in NIES_DIR_CANDIDATES)
    raise FileNotFoundError(f"{fname} が見つかりません。探した場所: {cand}")

def _parse_nies_csv(var: str, ssp_code: str|int) -> Tuple[pd.DataFrame, list[dict], list[str], list[int]]:
    """
    national_average_<var>_ssp<code>.csv を読み込み、列名から model/year を抽出
    返り値: (base_df(index=time), entries, models, years)
    """
    path = _nies_csv_path(var, ssp_code)
    df = pd.read_csv(path)
    if "time" not in df.columns:
        df = df.rename(columns={df.columns[0]: "time"})
    base_df = df.set_index("time")

    # 例: pr_MIROC6_ssp245_r1i1p1f1_2050
    pattern = re.compile(rf'^{re.escape(var)}_(?P<model>.+?)_ssp{ssp_code}_.+?_(?P<year>\d{{4}})$')
    entries, models, years = [], set(), set()
    for c in base_df.columns:
        m = pattern.match(str(c))
        if m:
            model = m.group("model"); year = int(m.group("year"))
            entries.append({"model": model, "year": year, "col": c})
            models.add(model); years.add(year)
    return base_df, entries, sorted(models), sorted(years)

def _extract_one_year(base_df: pd.DataFrame, entries: list[dict], model: str, year: int, var: str) -> pd.Series:
    for e in entries:
        if e["model"] == model and e["year"] == year:
            s = base_df[e["col"]].copy()
            s.index.name = "doy"  # 0..364/365
            s.name = var
            return s
    raise KeyError(f"{var}: {model} {year} の列がありません")

def _drop_feb29_by_date_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df[~((df.index.month == 2) & (df.index.day == 29))]
    return df

def _clean_numeric(series: pd.Series, fallback: pd.Series|None=None) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().all() and fallback is not None:
        s = pd.to_numeric(fallback, errors="coerce")
    s = s.interpolate(limit_direction="both")
    s = s.bfill().ffill().fillna(0)
    s = s.replace([np.inf, -np.inf], 0)
    return s.astype(float)

def build_extdata_multi_year(ssp_code: str|int, model: str, start_year: int, n_years: int) -> pd.DataFrame:
    """
    指定SSP/モデル/開始年/年数の連結テーブル（No., precipitation, temperature, tasmax, tasmin, rsds, date）
    - 閏日は削除して詰める（常に 365*n_years 行）
    - 欠測は補間後に前後詰め、なお残れば 0
    """
    pr_base, pr_ent, _, _     = _parse_nies_csv("pr",     ssp_code)
    tas_base, tas_ent, _, _   = _parse_nies_csv("tas",    ssp_code)
    tmax_base, tmax_ent, _, _ = _parse_nies_csv("tasmax", ssp_code)
    tmin_base, tmin_ent, _, _ = _parse_nies_csv("tasmin", ssp_code)
    rsds_base, rsds_ent, _, _ = _parse_nies_csv("rsds",   ssp_code)

    frames = []
    for y in range(start_year, start_year + n_years):
        pr    = _extract_one_year(pr_base,   pr_ent,   model, y, "precipitation")
        tas   = _extract_one_year(tas_base,  tas_ent,  model, y, "temperature")
        tasmx = _extract_one_year(tmax_base, tmax_ent, model, y, "tasmax")
        tasmn = _extract_one_year(tmin_base, tmin_ent, model, y, "tasmin")
        rsds  = _extract_one_year(rsds_base, rsds_ent, model, y, "rsds")

        start = pd.Timestamp(f"{y}-01-01")
        dates = [start + pd.Timedelta(days=int(d)) for d in pr.index]
        dfy = pd.DataFrame({
            "precipitation": pr.values,
            "temperature":   tas.values,
            "tasmax":        tasmx.values,
            "tasmin":        tasmn.values,
            "rsds":          rsds.values,
        }, index=pd.to_datetime(dates))
        frames.append(_drop_feb29_by_date_index(dfy))

    df = pd.concat(frames, axis=0)
    df.index.name = "date"

    # 数値化＆埋め
    df["temperature"]   = _clean_numeric(df["temperature"])
    df["precipitation"] = _clean_numeric(df["precipitation"])
    df["tasmax"]        = _clean_numeric(df["tasmax"], fallback=df["temperature"])
    df["tasmin"]        = _clean_numeric(df["tasmin"], fallback=df["temperature"])
    df["rsds"]          = _clean_numeric(df["rsds"])

    df = df.reset_index()
    df.insert(0, "No.", np.arange(len(df), dtype=int))
    df = df[["No.", "precipitation", "temperature", "tasmax", "tasmin", "rsds", "date"]]
    return df

def write_input_excel_no_blank(table: pd.DataFrame, out_path: str|Path = INPUT_XLSX_PATH):
    """
    シート名 'input'、空欄なしで出力。
    """
    tbl = table.copy()
    for col in ["precipitation", "temperature", "tasmax", "tasmin", "rsds"]:
        if col not in tbl.columns:
            tbl[col] = 0.0
        tbl[col] = _clean_numeric(tbl[col])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        tbl.to_excel(xw, sheet_name=INPUT_SHEET, index=False)

def _read_input_excel_table(file_or_path: io.BytesIO | str | Path) -> pd.DataFrame:
    """
    input.xlsx の 'input' シートを DataFrame として読み込む
    必須列: No., precipitation, temperature, tasmax, tasmin, rsds, date
    """
    df = pd.read_excel(file_or_path, sheet_name=INPUT_SHEET)
    need = {"precipitation", "temperature", "tasmax", "tasmin", "rsds", "date"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"input.xlsx に必要な列がありません: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    for col in ["precipitation", "temperature", "tasmax", "tasmin", "rsds"]:
        df[col] = _clean_numeric(df[col])
    # 念のためうるう日があれば削除
    df = df.set_index("date")
    df = df[~((df.index.month == 2) & (df.index.day == 29))].reset_index()
    return df


# =========================
# UI
# =========================
st.set_page_config(page_title="River Management Simulator", layout="wide")
st.title("🌊 River Management Simulator (PySD + Streamlit)")
st.caption("AMeDAS を使った過去再現（観測と比較）＋ NIES/SSP × 5 GCM の将来計算。")

with st.sidebar:
    st.header("1) モデル読込")
    use_py_first = st.toggle("変換済み .py を優先する", value=True)
    model_py_path  = st.text_input("モデル .py パス", str(DEFAULT_MODEL_PY))
    model_mdl_path = st.text_input("モデル .mdl パス", str(DEFAULT_MODEL_MDL))

    st.divider()
    st.header("2) AMeDAS 入力（過去再現・観測比較）")
    amedas_xlsx = st.file_uploader("AMeDAS の input.xlsx（シート名 input）", type=["xlsx"])

    st.divider()
    st.header("3) 将来気候（NIES national_average_*.csv）")
    ssp_code = st.selectbox("SSP 選択", options=["119", "126", "245", "585"], index=2)  # 既定: 245
    start_year = st.number_input("開始年", value=2015, step=1, min_value=1900, max_value=2100)
    n_years    = st.number_input("年数（1年以上可）", value=1, step=1, min_value=1, max_value=300)

    st.caption("※ CSV は data/nies2020/ または data/nies/ に配置。例: national_average_pr_ssp245.csv")

    st.divider()
    st.header("4) 流域プリセット＆パラメータ")
    preset_name = st.selectbox("プリセットを選択", list(PRESETS.keys()))
    if "params_ui" not in st.session_state:
        st.session_state["params_ui"] = {}
    if st.button("このプリセットをスライダーへ反映"):
        for k, v in PRESETS[preset_name].items():
            st.session_state["params_ui"][k] = v
        st.success(f"「{preset_name}」の値を反映しました")

    ui_values = {}
    for name, spec in PARAM_SPECS.items():
        default_value = st.session_state["params_ui"].get(name, spec.get("value", 0))
        ui_values[name] = st.slider(
            spec["label"],
            min_value=spec.get("min", 0.0),
            max_value=spec.get("max", 1.0),
            step=spec.get("step", 0.01),
            value=default_value,
            key=f"param_{name}",
            format="%.6f" if spec.get("step", 1) < 1 else "%d"
        )
        st.session_state["params_ui"][name] = ui_values[name]
    params = ui_values

# モデル存在チェック
try:
    _ = _load_model_from_file(
        model_py_path if use_py_first else "",
        model_mdl_path if not use_py_first else "",
    )
except Exception as e:
    st.error(f"モデル読み込みに失敗: {e}")
    st.stop()

# 出力列
with st.container():
    st.header("5) 表示したいモデル出力")
    return_cols = st.multiselect(
        "変数（スネークケース）",
        DEFAULT_RETURN_COLS,
        default=DEFAULT_RETURN_COLS
    )

# 観測データ（流量）
st.divider()
st.header("6) 観測流量（AMeDAS再現との比較用・任意）")
st.caption("CSV 例: date, flow（m3/s or m3/day）— モデルの下流流量と比較します。")
obs_file = st.file_uploader("観測CSVをアップロード", type=["csv"], key="obs")
c1, c2, c3 = st.columns(3)
with c1:
    obs_date_col = st.text_input("日付列名", value="date")
with c2:
    obs_flow_col = st.text_input("流量列名", value="flow")
with c3:
    obs_unit = st.selectbox("観測流量の単位", options=["m3/day", "m3/s"], index=1)
max_lag_days = st.slider("形状比較の許容ラグ（日）", min_value=0, max_value=14, value=5)

# 実行ボタン
run_btn = st.button("▶ AMeDAS再現（観測比較）＋ 5 GCM 将来計算 を実行", type="primary")


# =========================
# 実行
# =========================
if run_btn:
    amedas_result: pd.DataFrame | None = None
    gcm_results: Dict[str, pd.DataFrame] = {}

    # ---------- AMeDAS 再現（観測比較） ----------
    with st.spinner("AMeDAS を用いた再現計算を実施中..."):
        if amedas_xlsx is not None:
            try:
                # AMeDAS input.xlsx を保存してモデルに読ませる
                with open(INPUT_XLSX_PATH, "wb") as f:
                    f.write(amedas_xlsx.getbuffer())

                # テーブル読み込み（開始日・日数を取得）
                am_tbl = _read_input_excel_table(amedas_xlsx)
                start_dt_amedas = pd.to_datetime(am_tbl["date"].iloc[0])
                n_days_amedas = len(am_tbl)
                timestamps_amedas = list(range(n_days_amedas))

                # モデルを新規ロード（毎回）
                model_amedas = _load_model_fresh(use_py_first, str(model_py_path), str(model_mdl_path))

                sim_params_amedas = params.copy()
                for k, v in (("initial_time", 0), ("final_time", n_days_amedas - 1), ("time_step", 1)):
                    if hasattr(model_amedas.components, k):
                        sim_params_amedas[k] = v

                res_amedas = _run_simulation(
                    model_amedas,
                    params=sim_params_amedas,
                    timestamps=timestamps_amedas,
                    return_cols=list(set(return_cols))
                )
                amedas_result = _build_model_datetime_index(res_amedas, start_dt_amedas)
                st.success("AMeDAS 再現のモデル出力を得ました。")

            except Exception as e:
                st.error(f"AMeDAS 再現でエラー: {e}")
                amedas_result = None
        else:
            st.info("AMeDAS の input.xlsx をアップロードすると、再現計算と観測比較が有効になります。")

    # AMeDAS vs 観測 比較
    if amedas_result is not None:
        st.subheader("🆚 AMeDAS 再現結果 × 観測流量 の比較（river_discharge_downstream）")
        if obs_file is not None:
            try:
                obs_df = _read_observed_csv(obs_file, obs_date_col, obs_flow_col, obs_unit)
            except Exception as e:
                st.error(f"観測CSVの読み込みに失敗: {e}")
                obs_df = None
        else:
            obs_df = None

        if obs_df is not None:
            if "river_discharge_downstream" in amedas_result.columns:
                merged = pd.concat(
                    [amedas_result["river_discharge_downstream"].rename("model_flow"), obs_df["obs_flow"]],
                    axis=1
                ).dropna()
                if not merged.empty:
                    lag, r_at_lag = _best_lag(merged["obs_flow"], merged["model_flow"], max_lag_days)
                    st.caption(f"最適ラグ（日）: {lag}（相関 {r_at_lag:.3f}）")

                    plot_target = "model_flow" if lag == 0 else "model_flow_lag"
                    if lag != 0:
                        merged["model_flow_lag"] = merged["model_flow"].shift(lag)
                        merged = merged.dropna()

                    # 指標表示
                    scores = _metrics(merged["obs_flow"], merged[plot_target])
                    if scores:
                        c1, c2, c3, c4, c5 = st.columns(5)
                        c1.metric("RMSE (m³/日)", f"{scores['RMSE']:.2f}")
                        c2.metric("MAE (m³/日)",  f"{scores['MAE']:.2f}")
                        c3.metric("Bias (m³/日)", f"{scores['Mean Bias']:.2f}")
                        c4.metric("相関 r",       f"{scores['Pearson r']:.3f}")
                        c5.metric("NSE",          f"{scores['NSE']:.3f}")

                    # 実値重ね描き
                    st.markdown("**重ね描き（実値）**")
                    show_df = merged[[plot_target, "obs_flow"]].rename(columns={plot_target: "model(AMeDAS)", "obs_flow": "observed"})
                    st.line_chart(show_df, height=320, use_container_width=True)

                    # 標準化（形状比較）
                    st.markdown("**形状比較（標準化：平均0・分散1）**")
                    z = show_df.apply(lambda s: (s - s.mean()) / (s.std() if s.std()!=0 else 1))
                    st.line_chart(z, height=320, use_container_width=True)

                    # ダウンロード
                    with st.expander("📥 AMeDAS再現×観測 のCSVを保存"):
                        out = amedas_result.copy()
                        out["obs_flow"] = obs_df["obs_flow"]
                        out["model_flow"] = amedas_result["river_discharge_downstream"]
                        if lag != 0:
                            out["model_flow_lag"] = out["model_flow"].shift(lag)
                        csv = out.to_csv(index_label="date").encode("utf-8")
                        st.download_button(
                            "AMeDAS再現×観測（CSV）",
                            data=csv,
                            file_name="amedas_baseline_vs_observed.csv",
                            mime="text/csv"
                        )
                else:
                    st.info("AMeDAS 再現と観測の重なり期間がありません。日付の範囲をご確認ください。")
            else:
                st.info("モデル出力に 'river_discharge_downstream' が含まれていません。出力変数の選択をご確認ください。")
        else:
            st.info("観測CSVをアップロードすると、AMeDAS再現との比較が表示されます。")

    # ---------- NIES/SSP × 5 GCM（将来計算） ----------
    with st.spinner("NIES/SSP 入力で 5 GCM の将来計算を実施中..."):
        try:
            _, pr_entries, pr_models, _ = _parse_nies_csv("pr", ssp_code)
        except Exception as e:
            st.exception(e)
            st.stop()

        if len(pr_models) < 5:
            st.warning(f"この SSP に含まれる GCM が 5 未満です: {pr_models}")
        models_to_run = pr_models[:5]

        start_dt_nies = pd.Timestamp(f"{int(start_year)}-01-01")
        for gcm in models_to_run:
            try:
                in_table = build_extdata_multi_year(ssp_code, gcm, int(start_year), int(n_years))
                write_input_excel_no_blank(in_table, INPUT_XLSX_PATH)

                model_gcm = _load_model_fresh(use_py_first, str(model_py_path), str(model_mdl_path))

                n_days = len(in_table)
                timestamps = list(range(n_days))
                sim_params = params.copy()
                for k, v in (("initial_time", 0), ("final_time", n_days - 1), ("time_step", 1)):
                    if hasattr(model_gcm.components, k):
                        sim_params[k] = v

                res = _run_simulation(
                    model_gcm,
                    params=sim_params,
                    timestamps=timestamps,
                    return_cols=list(set(return_cols))
                )
                res = _build_model_datetime_index(res, start_dt_nies)
                gcm_results[gcm] = res

            except Exception as e:
                st.error(f"{gcm} の実行でエラー: {e}")

    if gcm_results:
        st.success("5 GCM の計算が完了しました。")
        st.write("対象 GCM:", ", ".join(gcm_results.keys()))

        # 下流流量の重ね描き
        target_var = "river_discharge_downstream"
        if all((target_var in df.columns) for df in gcm_results.values()):
            st.subheader("📈 下流流量（river_discharge_downstream）— 5 GCM 重ね描き（将来）")
            plot_df = pd.concat(
                [df[target_var].rename(f"{target_var} ({gcm})") for gcm, df in gcm_results.items()],
                axis=1
            )
            st.line_chart(plot_df, height=360, use_container_width=True)

        # 変数ごとのタブ
        st.subheader("📊 変数ごとの時系列（GCM別タブ）")
        for var in return_cols:
            st.markdown(f"**{var}**")
            tabs = st.tabs(list(gcm_results.keys()))
            for (gcm, df), tab in zip(gcm_results.items(), tabs):
                with tab:
                    if var in df.columns:
                        st.line_chart(df[[var]], height=260, use_container_width=True)
                    else:
                        st.info(f"{gcm}: 変数 {var} は出力に存在しません")

        # ダウンロード
        st.subheader("💾 将来計算の結果ダウンロード")
        for gcm, df in gcm_results.items():
            csv_bytes = df.to_csv(index_label="date").encode("utf-8")
            st.download_button(
                f"{gcm} の出力CSVを保存",
                data=csv_bytes,
                file_name=f"simulation_output_{gcm}_ssp{ssp_code}_{start_year}_{int(n_years)}y.csv",
                mime="text/csv"
            )

        if all((target_var in df.columns) for df in gcm_results.values()):
            river_df = pd.concat(
                [df[target_var].rename(gcm) for gcm, df in gcm_results.items()],
                axis=1
            )
            csv_bytes = river_df.to_csv(index_label="date").encode("utf-8")
            st.download_button(
                "river_discharge_downstream（5 GCM横持ち）を保存",
                data=csv_bytes,
                file_name=f"river_discharge_5gcm_ssp{ssp_code}_{start_year}_{int(n_years)}y.csv",
                mime="text/csv"
            )

    # AMeDAS 再現出力の保存
    if amedas_result is not None:
        st.subheader("💾 AMeDAS 再現出力のダウンロード")
        csv_bytes = amedas_result.to_csv(index_label="date").encode("utf-8")
        st.download_button(
            "AMeDAS 再現出力CSVを保存",
            data=csv_bytes,
            file_name=f"simulation_output_AMeDAS_baseline.csv",
            mime="text/csv"
        )


st.divider()
with st.expander("🧩 ヒント & メモ"):
    st.markdown("""
- **AMeDAS 再現**: `input.xlsx`（シート名 `input`、列 `No., precipitation, temperature, tasmax, tasmin, rsds, date`）をアップすると、モデルをその外部データで駆動し、**観測流量 CSV** と比較（RMSE/MAE/Bias/r/NSE、ラグ最適化）します。
- **将来計算**: NIES の `national_average_<var>_ssp<code>.csv` から 5 GCM を自動選択。期間は「開始年＋年数」。閏日は削除して詰め、**空欄は作らず** `input.xlsx` を都度生成します。
- NIES データの場所は `data/nies2020/` または `data/nies/` のどちらかでOK。
- 起動: `pip install streamlit pysd numpy pandas openpyxl` → `streamlit run app.py`
    """)
