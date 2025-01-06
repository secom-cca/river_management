import numpy as np
import pandas as pd

# 設定
np.random.seed(42)
start_year = 2000
end_year = 2100
dates = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31")
days = len(dates)

# 季節ごとの降水確率
rain_prob = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5, 6: 0.6,
             7: 0.5, 8: 0.4, 9: 0.3, 10: 0.2, 11: 0.1, 12: 0.2}

# ガンマ分布のパラメータ（季節性を考慮）
gamma_shape = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7,
               7: 6, 8: 5, 9: 4, 10: 3, 11: 2, 12: 3}
gamma_scale = {1: 1, 2: 1.5, 3: 2, 4: 2.5, 5: 3, 6: 3.5,
               7: 3, 8: 2.5, 9: 2, 10: 1.5, 11: 1, 12: 1.2}

# 月情報
months = dates.month

# データ生成
rain_data = []
for month in months:
    if np.random.rand() < rain_prob[month]:  # 降水有無
        rain_amount = np.random.gamma(gamma_shape[month], gamma_scale[month])
    else:
        rain_amount = 0
    rain_data.append(rain_amount)

# DataFrameに変換
df = pd.DataFrame({'Date': dates, 'Rainfall': rain_data})

# CSVに保存
output_filename = 'rainfall_data_2000_2100.csv'
df.to_csv(output_filename, index=False)

print(f"{output_filename} にデータを保存しました。")
