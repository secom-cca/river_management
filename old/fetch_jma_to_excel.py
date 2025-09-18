# fetch_jma_to_excel.py
# 使い方:
#   python fetch_jma_to_excel.py --prec-no 82 --block-no 0790 --year 2023 --out jma_kurume_2023.xls
#
# ポイント:
# - block-no が4桁 => AMeDAS (daily_s2.php)
# - block-no が5桁 => 官署    (daily_s1.php)
# - CSVは format=1、CP932、注記行スキップ＆ヘッダ自動検出
# - 出典をメタ情報としてExcelに併記

import argparse
import io
import re
import time
import requests
import pandas as pd
from pathlib import Path

BASE = "https://www.data.jma.go.jp/stats/etrn/view"

def build_endpoint(block_no: str) -> str:
    """block_no の桁数から s1/s2 を自動選択"""
    s = str(block_no)
    if len(s) <= 4:
        # AMeDAS（4桁、先頭0維持）
        s = s.zfill(4)
        return "daily_s2.php", s
    else:
        # 官署（5桁）
        return "daily_s1.php", s

def fetch_month_csv(prec_no: str, block_no: str, year: int, month: int, pause_sec=1.2) -> pd.DataFrame:
    ep, bn = build_endpoint(block_no)
    url = (f"{BASE}/{ep}?prec_no={prec_no}&block_no={bn}"
           f"&year={year}&month={month}&day=&view=p1&format=1")

    # polite headers
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; research; +https://example.org)",
        "Accept": "text/csv, text/plain, */*",
        "Referer": "https://www.data.jma.go.jp/stats/etrn/index.php"
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()

    # JMAのCSVはCP932（Shift_JIS想定）
    text = r.content.decode("cp932", errors="ignore")

    # ヘッダ行（カラム行）を探す（「日」または「年月日」を含み、カンマ区切りの行）
    header_idx = None
    lines = text.splitlines()
    for i, line in enumerate(lines[:50]):  # 上の方にあるはず
        if "," in line and (("日" in line) or ("年月日" in line)):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"ヘッダ行が見つかりませんでした: {url}")

    # データ部を抽出
    csv_text = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(csv_text))

    # 「日」 or 「年月日」などから date を作成
    if "年月日" in df.columns:
        df["date"] = pd.to_datetime(df["年月日"], errors="coerce")
    elif "日" in df.columns:
        # 年・月はURLから補完
        df["日"] = pd.to_numeric(df["日"], errors="coerce")
        df["date"] = pd.to_datetime(
            {"year": [year]*len(df), "month": [month]*len(df), "day": df["日"]}
        )
    else:
        # まれに「日付」などの別表記
        cand = [c for c in df.columns if "日" in c]
        if cand:
            tmp = pd.to_numeric(df[cand[0]], errors="coerce")
            df["date"] = pd.to_datetime(
                {"year": [year]*len(df), "month": [month]*len(df), "day": tmp}
            )
        else:
            raise ValueError(f"日付列が判別できません: {url}")

    # 欠損や特異表記（×, //, ( ) 内注記など）を落として数値化
    def clean_num(s):
        if pd.isna(s): return pd.NA
        s = str(s)
        s = re.sub(r"[^\d\.\-\+]", "", s)  # 数字と.-+以外を削除
        return pd.to_numeric(s, errors="coerce")

    # 欲しい代表列候補名（列名ゆれ対応）
    col_map_candidates = {
        "precipitation": ["降水量の合計(mm)", "降水量合計(mm)", "降水量合計", "降水量(mm)"],
        "tavg": ["平均気温(℃)", "平均気温"],
        "tmin": ["最低気温(℃)", "最低気温"],
        "tmax": ["最高気温(℃)", "最高気温"],
        "sunshine": ["日照時間(時間)", "日照時間"]
    }

    out = pd.DataFrame({"date": df["date"]})
    for key, cands in col_map_candidates.items():
        found = None
        for c in cands:
            if c in df.columns:
                found = c
                break
        if found is not None:
            out[key] = df[found].map(clean_num)
        else:
            out[key] = pd.NA  # 列がない月もある

    # 取得間隔を空ける
    time.sleep(pause_sec)
    return out

def fetch_daily_year_by_codes(prec_no: str, block_no: str, year: int) -> pd.DataFrame:
    frames = []
    for m in range(1, 13):
        try:
            frames.append(fetch_month_csv(str(prec_no), str(block_no), year, m))
        except Exception as e:
            print(f"[WARN] {year}-{m}: {e}")
    if not frames:
        raise RuntimeError("1年分のCSVが1件も取得できませんでした。コードや年を確認してください。")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df

def to_pysd_excel(df: pd.DataFrame, out_path: Path):
    """
    PySDモデル（River_management_chikugo.py）のExtData参照に合わせた
    Excel 'jma' シートを作る。
      - B列: 日付（YYYY/M/D）
      - E列: 平均気温
      - H列: 降水量
      - K列: 日照時間
      - Q列: 最高気温
      - V列: 最低気温
    """
    # 必要列が無ければ NaN のままでもOK（モデル側で補間/ゼロ扱いなら）
    jma = pd.DataFrame()
    jma["B"] = df["date"].dt.strftime("%Y/%-m/%-d") if hasattr(df["date"].dt, "strftime") else df["date"].astype(str)
    jma["E"] = df.get("tavg")
    jma["H"] = df.get("precipitation")
    jma["K"] = df.get("sunshine")
    jma["Q"] = df.get("tmax")
    jma["V"] = df.get("tmin")

    # 出典メモ
    meta = pd.DataFrame({
        "note": [
            "出典：気象庁ホームページ（過去の気象データ検索）",
            "URL: https://www.data.jma.go.jp/stats/etrn/",
            "公共データ利用規約（第1.0版）に準拠して出典を記載しています。"
        ]
    })

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as w:
        jma.to_excel(w, sheet_name="jma", index=False, header=False)
        meta.to_excel(w, sheet_name="about", index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prec-no", required=True, help="例: 82（福岡県）")
    ap.add_argument("--block-no", required=True, help="例: 0790（久留米 AMeDAS） / 47807（官署）")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    print(f"[INFO] 取得先: prec_no={args.prec_no}, block_no={args.block_no}, year={args.year}")
    df = fetch_daily_year_by_codes(args.prec_no, args.block_no, args.year)
    to_pysd_excel(df, args.out)
    print(f"[OK] 保存しました: {args.out}")

if __name__ == "__main__":
    main()
