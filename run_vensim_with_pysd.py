# run_vensim_with_pysd.py  ← ファイル名はこのままでOK
from pathlib import Path
from pysd import read_vensim, load

# ---- 入力ファイル設定 ----
MODEL_MDL = Path("River_management_chikugo.mdl")   # Vensim テキスト（任意）
MODEL_PY  = Path("River_management_chikugo.py")    # 変換済み PySD モデル（任意）
EXCEL_FILE = Path("jma_kurume_2023.xls")           # GET XLS DATA で参照される外部データ

# ---- モデル読み込み（.py があれば優先、なければ .mdl を読み込んで自動変換）----
if MODEL_PY.exists():
    model = load(MODEL_PY.as_posix())
elif MODEL_MDL.exists():
    model = read_vensim(MODEL_MDL.as_posix())
else:
    raise FileNotFoundError("River_management_chikugo.py も .mdl も見つかりません。")

# ---- シミュレーション設定（Vensim の CONTROL に合わせる）----
time = list(range(0, 365, 1))

# ---- パラメータ上書き（必要な場合のみ）----
#   ※ すべてスネークケース名。例はデフォルトと同じ値なので実質上書きなし。
params = {
    "daily_precipitation_future_ratio": 1,
    "levee_investment_amount": 0,
    "dam_investment_amount": 0,
}

# ---- 取得したい出力（スネークケース！）----
return_cols = [
    "daily_total_gdp",
    "dam_storage",
    "downstream_storage",
    "upstream_storage",
    "river_discharge_downstream",
    "houses_damaged_by_inundation",
    "financial_damage_by_innundation",
    "financial_damage_by_flood",
]

# ---- 実行 ----
res = model.run(
    params=params,
    return_timestamps=time,
    return_columns=return_cols,
    # data=...  # 通常不要。Excel はモデル側の ExtData が直接読みます
)

print(res.head())
res.to_csv("simulation_output.csv", index_label="day")
