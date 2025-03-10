import pandas as pd

# 入力CSVファイル
input_csv_weather = "data/weather_data_2008_2018.csv"  # 実際のファイルパスを指定
input_csv_radiation = "data/solar_radiation_2008_2018.csv"  # 全天日射量のファイル
output_txt = "data/vensim_lookup_weather.txt"  # 出力ファイルパス

# CSVデータの読み込み
df_weather = pd.read_csv(input_csv_weather)
df_radiation = pd.read_csv(input_csv_radiation)

# 日付から経過日数（タイムステップ）を計算
df_weather['Date'] = pd.to_datetime(df_weather['Date'])
df_weather['Time'] = (df_weather['Date'] - pd.Timestamp("2008-01-01")).dt.days + 1

df_radiation['Date'] = pd.to_datetime(df_radiation['Date'])
df_radiation['Time'] = (df_radiation['Date'] - pd.Timestamp("2008-01-01")).dt.days + 1

# 全天日射量の欠測値（-100）を補正（前後のデータの平均で補完）
df_radiation.loc[df_radiation['全天日射量'] == -100, '全天日射量'] = None
df_radiation['全天日射量'] = df_radiation['全天日射量'].interpolate()

# 各気象データのLOOKUPリストを作成
lookup_precip = [f"({row['Time']},{row['降水量']})" for _, row in df_weather.iterrows()]
lookup_temp_avg = [f"({row['Time']},{row['平均気温']})" for _, row in df_weather.iterrows()]
lookup_temp_max = [f"({row['Time']},{row['最高気温']})" for _, row in df_weather.iterrows()]
lookup_temp_min = [f"({row['Time']},{row['最低気温']})" for _, row in df_weather.iterrows()]
lookup_radiation = [f"({row['Time']},{row['全天日射量']})" for _, row in df_radiation.iterrows()]

# Vensim用LOOKUP関数の形式に変換
lookup_function = f"""
降水量 = WITH LOOKUP(
  {', '.join(lookup_precip)}
)

平均気温 = WITH LOOKUP(
  {', '.join(lookup_temp_avg)}
)

最高気温 = WITH LOOKUP(
  {', '.join(lookup_temp_max)}
)

最低気温 = WITH LOOKUP(
  {', '.join(lookup_temp_min)}
)

全天日射量 = WITH LOOKUP(
  {', '.join(lookup_radiation)}
)
"""

# 結果をテキストファイルに保存
with open(output_txt, "w", encoding="utf-8") as f:
    f.write(lookup_function)

print(f"LOOKUPデータを {output_txt} に保存しました。")
