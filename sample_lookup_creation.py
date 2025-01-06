import pandas as pd

# CSVファイルの読み込み
input_csv = 'rainfall_data_2000_2100.csv'  # CSVファイル名
output_txt = 'rainfall_lookup.txt'  # Vensim用のLOOKUPデータ出力ファイル

# CSVデータの読み込み
df = pd.read_csv(input_csv)

# 日付から日数（タイムステップ）を計算
df['Time'] = pd.to_datetime(df['Date']).sub(pd.Timestamp("2000-01-01")).dt.days + 1

# LOOKUP形式の生成
lookup_values = []
for _, row in df.iterrows():
    lookup_values.append(f"({row['Time']},{row['Rainfall']})")

# Vensim用LOOKUP関数の形式に変換
lookup_function = f"Rainfall = WITH LOOKUP(\n  {', '.join(lookup_values)}\n)"

# 結果をテキストファイルに保存
with open(output_txt, 'w') as f:
    f.write(lookup_function)

print(f"LOOKUP形式のデータを {output_txt} に保存しました。")
