import pandas as pd

# 月ごとの平均気温データ
monthly_avg_temps = [6, 3.3, 7.2, 13.6, 19.7, 22.8, 28.2, 29.1, 24.8, 18.8, 13.6, 6.9]

# 2000年（うるう年）の1年間の日付
dates = pd.date_range(start="2000-01-01", end="2000-12-31", freq='D')

# 日ごとの気温データの作成
df = pd.DataFrame({"Date": dates})
df["Time"] = (df["Date"] - pd.Timestamp("2000-01-01")).dt.days + 1
df["Temp"] = df["Date"].dt.month.map(lambda m: monthly_avg_temps[m - 1])

# LOOKUPデータを作成
lookup_values = [f"({row['Time']},{row['Temp']})" for _, row in df.iterrows()]

# Vensim用LOOKUP関数の形式に変換
lookup_function = f"Temp = WITH LOOKUP(\n  {', '.join(lookup_values)}\n)"

# ファイルへ保存
output_txt = "temp/daily_temp_2000.txt"
with open(output_txt, "w") as f:
    f.write(lookup_function)