from pathlib import Path
from pysd import read_vensim, load

# ---- 入力ファイル設定 ----
MODEL_MDL = Path("River_management_xls_to3.mdl")  # Vensim テキスト
MODEL_PY = Path("River_management_xls_to3.py")    # 変換済み PySD モデル

# ---- モデル読み込み（.py があれば優先、なければ .mdl を読み込んで自動変換）----
if MODEL_PY.exists():
    model = load(MODEL_PY.as_posix())
elif MODEL_MDL.exists():
    model = read_vensim(MODEL_MDL.as_posix())
else:
    raise FileNotFoundError("River_management_xls_to3.py も .mdl も見つかりません。")

# ---- シミュレーション設定（Vensim の CONTROL に合わせる）----
time = list(range(0, 365, 1))

# ---- パラメータ上書き（必要な場合のみ）----
# opt_mode を切り替えると最適化済みパラメータを自動適用
opt_mode = "opt_riv_dis_down"  # None or "opt_riv_dis_down"

params = {
    "daily_precipitation_future_ratio": 1,
    "levee_investment_amount": 0,
    "dam_investment_amount": 0,
}

if opt_mode == "opt_riv_dis_down":
    params.update(
        {
            "upstream_outflow_ratio": 0.0270484,
            "downstream_outflow_ratio": 0.0446371,
            "direct_discharge_ratio": 0.99,
            "upstream_percolation_ratio": 0.303752,
            "downstream_deep_percolation_ratio": 0.251362,
            "upstream_middle_flow_ratio": 0.141604,
            "downstream_percolation_ratio": 0.191405,
            "downstream_middle_flow_ratio": 0.0620332,
            "waterholding_capacity_of_forest": 157.484,
            "upstream_deep_percolation_ratio": 0.922893,
        }
    )

# ---- 取得したい出力（スネークケース）----
return_cols = [
    "day_of_year",
    "year_end_trigger", 
    "heat_stress_sample_trigger",  
    "daily_total_gdp",
    "dam_storage",
    "downstream_storage",
    "upstream_storage",
    "river_discharge_downstream",
    "houses_damaged_by_inundation",
    "financial_damage_by_innundation",
    "financial_damage_by_flood",
    "daily_crop_production",
    "accumulated_crop_production_within_year",
    "yearly_crop_production",
    "yearly_crop_production_per_day",
    "crop_production_cashflow",
    "chalky_kernel_ratio",
    "heat_stress_at_heading_plus_30",
]

# ---- 実行 ----
res = model.run(
    params=params,
    return_timestamps=time,
    return_columns=return_cols,
)

# 年次GDP（合計）を計算して最終行に格納
res = res.copy()
yearly_gdp_total = res["daily_total_gdp"].sum()
res["yearly_gdp_total"] = 0.0
res.loc[res.index[-1], "yearly_gdp_total"] = yearly_gdp_total

print(res[return_cols].head())
print("\nYear-end values (last timestep):")
print(res.iloc[[-1]][["yearly_crop_production", "yearly_gdp_total"]])

#res.to_csv("data/simulation_output_to3.csv", index_label="day")
res.to_csv("data/simulation_output_to3_opt.csv", index_label="day")
