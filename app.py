# app.py
from __future__ import annotations
import io
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
DEFAULT_XLS_NAME  = "jma_kurume_2023.xls"   # モデル内 ExtData が参照する既定名

# ==== 流域プリセット（代表値セット） ====
# 必要に応じて値を実測・資料の代表値に差し替えてください
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
        "upstream_area": 178_650,                # ha
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

    # --- 地域パラメータ（新規追加） ---
    # Initial dam capacity (m3)
    "initial_dam_capacity": dict(
        label="初期ダム容量（m³）",
        min=0, max=500_000_000, step=100_000, value=74_200_000
    ),
    # Upstream area (ha)
    "upstream_area": dict(
        label="上流域面積（ha）",
        min=1_000, max=1_000_000, step=100, value=157_585
    ),
    # Downstream area (ha)
    "downstream_area": dict(
        label="下流域面積（ha）",
        min=1_000, max=1_000_000, step=100, value=143_951
    ),
    # Forest area ratio (-)
    "forest_area_ratio": dict(
        label="森林面積比（-）",
        min=0.0, max=1.0, step=0.001, value=166_000/(198_500*0.9)  # ≈0.93
    ),
    # Direct discharge ratio (-)
    "direct_discharge_ratio": dict(
        label="直接流出比（-）",
        min=0.0, max=1.0, step=0.001, value=1 - 40/1950  # ≈0.9795
    ),
    # "Current high-water discharge" (m3/s)
    "current_highwater_discharge": dict(
        label="計画高水流量（m³/秒）",
        min=0, max=30_000, step=100, value=11_500
    ),
}

# =========================
# ユーティリティ
# =========================
@lru_cache(maxsize=4)
def _load_model_from_file(model_py: str|None, model_mdl: str|None):
    if model_py and Path(model_py).exists():
        return load(model_py)
    if model_mdl and Path(model_mdl).exists():
        return read_vensim(model_mdl)
    raise FileNotFoundError("モデルファイル（.py/.mdl）が見つかりません。")

def _ensure_extdata_file(uploaded: io.BytesIO|None, expected_name: str):
    if uploaded is not None:
        with open(expected_name, "wb") as f:
            f.write(uploaded.getbuffer())

def _run_simulation(model, params: Dict[str, Any], timestamps: List[float], return_cols: List[str]) -> pd.DataFrame:
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

def _read_observed_csv(file: io.BytesIO, date_col: str, flow_col: str, unit: str) -> pd.DataFrame:
    df = pd.read_csv(file)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[[date_col, flow_col]].rename(columns={date_col: "date", flow_col: "obs_flow"})
    if unit == "m3/s":
        df["obs_flow"] = df["obs_flow"] * 86400.0  # → m3/day に変換
    # 日単位に整形（重複があれば平均）
    df = df.groupby("date", as_index=False).mean().set_index("date").sort_index()
    return df

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
    """ 相関が最大となるラグ（日）と相関値（形状比較用） """
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
# UI
# =========================
st.set_page_config(page_title="River Management Simulator", layout="wide")
st.title("🌊 River Management Simulator (PySD + Streamlit)")
st.caption("VensimモデルをPySDで実行し、パラメータを触って結果を可視化。観測フローとの形状比較もできます。")

with st.sidebar:
    st.header("1) モデル読込")
    use_py_first = st.toggle("変換済み .py を優先する", value=True)
    model_py_path  = st.text_input("モデル .py パス", str(DEFAULT_MODEL_PY))
    model_mdl_path = st.text_input("モデル .mdl パス", str(DEFAULT_MODEL_MDL))

    st.markdown("**GET XLS DATA（外部データ）**")
    up = st.file_uploader(f"Excel（{DEFAULT_XLS_NAME} として保存）", type=["xls", "xlsx"])

    st.divider()
    st.header("2) シミュレーション時間")
    start_date = st.date_input("モデル開始日（例：2023-01-01）", pd.to_datetime("2023-01-01"))
    init_time = st.number_input("INITIAL TIME (day)", value=0, step=1)
    final_time = st.number_input("FINAL TIME (day)", value=364, step=1, min_value=init_time)
    time_step = st.number_input("TIME STEP (day)", value=1, step=1, min_value=1)
    timestamps = list(range(int(init_time), int(final_time)+1, int(time_step)))

with st.sidebar:
    st.header("3) 流域プリセット")
    preset_name = st.selectbox("プリセットを選択", list(PRESETS.keys()))

    # セッション状態を初期化（初回のみ）
    if "params_ui" not in st.session_state:
        st.session_state["params_ui"] = {}

    # 適用ボタンでスライダー/数値入力へ反映
    if st.button("このプリセットをスライダーへ反映"):
        chosen = PRESETS[preset_name]
        for k, v in chosen.items():
            st.session_state["params_ui"][k] = v
        st.success(f"「{preset_name}」の値を反映しました")

    st.divider()
    st.header("4) パラメータ（上書き）")

    ui_values = {}
    for name, spec in PARAM_SPECS.items():
        label = spec["label"]
        # preset 反映後の値（なければ spec のデフォルト）
        default_value = st.session_state["params_ui"].get(name, spec.get("value", 0))

        # 値域に応じて number_input or slider を使い分け（お好み）
        if "min" in spec and "max" in spec and "step" in spec:
            ui_values[name] = st.slider(
                label,
                min_value=spec["min"],
                max_value=spec["max"],
                step=spec["step"],
                value=default_value,
                key=f"param_{name}",
                format="%.6f" if spec["step"] < 1 else "%d"
            )
        else:
            ui_values[name] = st.slider(label, value=float(default_value), key=f"param_{name}")

        # session_state にも常に反映（プリセット→手動調整の往復を自然に）
        st.session_state["params_ui"][name] = ui_values[name]

    # ここで params を model.run に渡す
    params = ui_values

# モデルロード
try:
    _ensure_extdata_file(up, DEFAULT_XLS_NAME)
    model = _load_model_from_file(
        model_py_path if use_py_first else "",
        model_mdl_path if not use_py_first else "",
    )
except Exception as e:
    st.error(f"モデル読み込みに失敗: {e}")
    st.stop()

# 出力選択
with st.container():
    st.header("4) 出力の選択")
    return_cols = st.multiselect(
        "表示/出力したい変数（スネークケース）",
        DEFAULT_RETURN_COLS,
        default=DEFAULT_RETURN_COLS
    )

# 観測データ
st.divider()
st.header("5) 観測データ（任意）")
st.caption("CSV 例: `date,temperature,precipitation,flow,level`（日付・流量列だけ使います）")
obs_file = st.file_uploader("観測CSVをアップロード", type=["csv"], key="obs")
col1, col2, col3 = st.columns([1,1,1])
with col1:
    obs_date_col = st.text_input("日付列名", value="date")
with col2:
    obs_flow_col = st.text_input("流量列名", value="flow")
with col3:
    obs_unit = st.selectbox("観測流量の単位", options=["m3/day", "m3/s"], index=1)

max_lag_days = st.slider("形状比較の許容ラグ（日）", min_value=0, max_value=14, value=5)

run_btn = st.button("▶ シミュレーションを実行", type="primary")

if run_btn:
    with st.spinner("計算中..."):
        try:
            res = _run_simulation(model, params=params, timestamps=timestamps, return_cols=return_cols)
        except Exception as e:
            st.exception(e)
            st.stop()

    res = _build_model_datetime_index(res, pd.to_datetime(start_date))
    st.success("完了！")
    st.dataframe(res.head(), use_container_width=True)

    # ====== 基本グラフ（選択列ごとにタブ）======
    st.subheader("📈 時系列グラフ（モデル出力）")
    if not res.empty:
        tabs = st.tabs([c for c in res.columns])
        for tab, col in zip(tabs, res.columns):
            with tab:
                st.line_chart(res[[col]], height=300)

    # ====== 観測との比較（river_discharge_downstream）======
    st.divider()
    st.subheader("🆚 観測との比較：river_discharge_downstream")
    if "river_discharge_downstream" not in res.columns:
        st.info("`river_discharge_downstream` を出力に含めてください。")
    else:
        model_series = res["river_discharge_downstream"].rename("model_flow")

        if obs_file is not None:
            try:
                obs_df = _read_observed_csv(obs_file, obs_date_col, obs_flow_col, obs_unit)
            except Exception as e:
                st.error(f"観測CSVの読み込みに失敗: {e}")
                obs_df = None
        else:
            obs_df = None

        if obs_df is not None:
            merged = pd.concat([model_series, obs_df["obs_flow"]], axis=1).dropna()
            st.write("重なり期間:", merged.index.min().date(), "〜", merged.index.max().date())

            # ラグ最適化（形状比較）
            lag, r_at_lag = _best_lag(merged["obs_flow"], merged["model_flow"], max_lag_days)
            st.caption(f"最適ラグ（日）: {lag}（相関 {r_at_lag:.3f}）")
            if lag != 0:
                if lag > 0:   # モデルを遅らせる（観測に対して）
                    merged["model_flow_lag"] = merged["model_flow"].shift(lag)
                else:         # モデルを進める
                    merged["model_flow_lag"] = merged["model_flow"].shift(lag)
                merged = merged.dropna()
                model_plot_col = "model_flow_lag"
            else:
                model_plot_col = "model_flow"

            # 指標
            scores = _metrics(merged["obs_flow"], merged[model_plot_col])
            if scores:
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("RMSE (m³/日)", f"{scores['RMSE']:.2f}")
                c2.metric("MAE (m³/日)", f"{scores['MAE']:.2f}")
                c3.metric("Bias (m³/日)", f"{scores['Mean Bias']:.2f}")
                c4.metric("相関 r", f"{scores['Pearson r']:.3f}")
                c5.metric("NSE", f"{scores['NSE']:.3f}")

            # 生データ重ね描き（同一単位：m3/日）
            st.markdown("**重ね描き（実値）**")
            plot_df = merged[[model_plot_col, "obs_flow"]].rename(columns={model_plot_col: "model", "obs_flow": "observed"})
            st.line_chart(plot_df, height=320, use_container_width=True)

            # 形状比較（標準化）
            st.markdown("**形状比較（標準化：平均0・分散1）**")
            z = plot_df.apply(lambda s: (s - s.mean()) / (s.std() if s.std()!=0 else 1))
            st.line_chart(z, height=320, use_container_width=True)

            with st.expander("📥 結果データをダウンロード"):
                out = res.copy()
                out["obs_flow"] = obs_df["obs_flow"]
                out["model_flow"] = res["river_discharge_downstream"]
                if lag != 0:
                    out["model_flow_lag"] = out["model_flow"].shift(lag)
                csv = out.to_csv(index_label="date").encode("utf-8")
                st.download_button("モデル＋観測（CSV）", data=csv, file_name="model_observed_merged.csv", mime="text/csv")
        else:
            st.info("観測CSVをアップロードすると、重ね描きと精度評価が表示されます。")

    # ====== CSV ダウンロード ======
    st.subheader("💾 モデル結果をCSVで保存")
    csv_model = res.to_csv(index_label="date").encode("utf-8")
    st.download_button("結果をCSVで保存", data=csv_model, file_name="simulation_output.csv", mime="text/csv")

st.divider()
with st.expander("🧩 ヒント & よくあるつまずき"):
    st.markdown("""
- モデル出力の `river_discharge_downstream` は **m³/日** を想定。観測が **m³/秒** のときは単位選択で自動換算します（×86400）。
- モデル時刻 0 日目が **「モデル開始日」** に対応します（既定は 2023-01-01）。必要に応じて変更してください。
- 形状比較は **相関が最大になるラグ（±N日）** を自動探索します。
- 指標: RMSE, MAE, 平均バイアス, Pearson 相関, Nash–Sutcliffe 効率（NSE）。
- 起動: `pip install streamlit pysd numpy pandas xlrd` → `streamlit run app.py`
    """)