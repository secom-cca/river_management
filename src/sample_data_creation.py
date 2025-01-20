import numpy as np
import pandas as pd

# 設定
np.random.seed(42)
start_year = 2000
end_year = 2100
dates = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31")
days = len(dates)

# 季節ごとの降水確率（基本値）
rain_prob = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5, 6: 0.6,
             7: 0.5, 8: 0.4, 9: 0.3, 10: 0.2, 11: 0.1, 12: 0.2}

# ガンマ分布のパラメータ（基本値）
gamma_shape = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7,
               7: 6, 8: 5, 9: 4, 10: 3, 11: 2, 12: 3}
gamma_scale = {1: 1, 2: 1.5, 3: 2, 4: 2.5, 5: 3, 6: 3.5,
               7: 3, 8: 2.5, 9: 2, 10: 1.5, 11: 1, 12: 1.2}

# 月情報
months = dates.month
years = dates.year

# データ生成
rain_data = []
for date in dates:
    month = date.month
    year = date.year

    # 気候変動の影響を反映
    climate_change_factor = (year - start_year) / (end_year - start_year)  # 0 (2000) → 1 (2100)

    # 降水確率の増加
    adjusted_rain_prob = rain_prob[month] + 0.2 * climate_change_factor  # 最大+20%

    # ガンマ分布パラメータの調整（極端な雨の発生）
    adjusted_shape = gamma_shape[month] * (1 - 0.5 * climate_change_factor)  # 形状パラメータを減少させる
    adjusted_scale = gamma_scale[month] * (1 + 1.0 * climate_change_factor)  # 尺度パラメータを増加させる

    # 降水の有無を判定
    if np.random.rand() < adjusted_rain_prob:
        # ガンマ分布から降水量を生成
        rain_amount = round(np.random.gamma(adjusted_shape, adjusted_scale),1)
    else:
        rain_amount = 0

    rain_data.append(rain_amount)

# DataFrameに変換
df = pd.DataFrame({'Date': dates, 'Rainfall': rain_data})

# CSVに保存
output_filename = '../rainfall/rainfall_data_2000_2100_climate_change.csv'
df.to_csv(output_filename, index=False)

print(f"{output_filename} に気候変動の影響を考慮したデータを保存しました。")
