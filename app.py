# fetch_jma_to_excel.py
# 例:
#   python fetch_jma_to_excel.py --prec-no 82 --block-no 0790 --year 2022 --out jma_kurume_2022.xls
#   python fetch_jma_to_excel.py --prec-no 82 --block-no 0790 --year 2023 --out jma_kurume_2023.xls

import argparse
import io
import os
import re
import time
import requests
import pandas as pd
from pathlib import Path

BASE = "https://www.data.jma.go.jp/stats/etrn/view"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research; river_mgmt)",
    "Accept": "text/csv, text/html, */*",
    "Referer": "https://www.data.jma.go.jp/stats/etrn/index.php",
}

DEBUG_DIR = Path(".debug")
DEBUG_DIR.mkdir(exist_ok=True)

def build_endpoint(block_no: str) -> tuple[str, str]:
    s = str(block_no)
    if len(s) <= 4:
        return "daily_s2.php", s.zfill(4)  # AMeDAS
    return "daily_s1.php", s              # 官署

def is_html(text: str) -> bool:
    t = text.strip().lower()
    return "<html" in t or "<!doctype" in t

def try_format1_csv(ep: str, bn: str, prec_no: str, year: int, month: int, view: str|None):
    # view をつけたり外したりして試す
    view_part = f"&view={view}" if view else ""
    url = (f"{BASE}/{ep}?prec_no={prec_no}&block_no={bn}"
           f"&year={year}&month={month}&day=&format=1{view_part}")
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    text = r.content.decode("cp932", errors="ignore")
    return url, text

def parse_csv_text(text: str) -> pd.DataFrame|None:
    # 先頭〜上位行に「日」または「年月日」を含むカンマ行を探す
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines[:500]):  # 広めに見る
        if "," in line and ("日" in line or "年月日" in line):
            header_idx = i
            break
    if header_idx is None:
        return None
    csv_text = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(csv_text))
    return df

def html_fallback(ep: str, bn: str, prec_no: str, year: int, month: int, view: str|None):
    view_part = f"&view={view}" if view else ""
    url = (f"{BASE}/{ep}?prec_no={prec_no}&block_no={bn}"
           f"&year={year}&month={month}&day=&format=0{view_part}")
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    # 文字コードは自動判定
    content = r.content
    # read_html（lxml）で全テーブルを拾う
    try:
        tables = pd.read_html(io.BytesIO(content), flavor="lxml", displayed_only=False)
    except ValueError:
        tables = []
    if not tables:
        return url, None

    # 「日」or「年月日」を含むテーブルを優先
    cand = None
    for t in tables:
        cols = [str(c) for c in t.columns]
        if any(("日" in c or "年月日" in c) for c in cols):
            cand = t
            break
    if cand is None:
        # 一番列数が多いものを候補に
        cand = max(tables, key=lambda d: d.shape[1])

    return url, cand

def to_daily(df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    # 日付列
    if "年月日" in df.columns:
        df["date"] = pd.to_datetime(df["年月日"], errors="coerce")
    else:
        # 「日」に近い列名を拾う
        day_col = None
        for c in df.columns:
            if "日" in str(c):
                day_col = c
                break
        if day_col is None:
            return pd.DataFrame()
        dd = pd.to_numeric(df[day_col], errors="coerce")
        df["date"] = pd.to_datetime({"year":[year]*len(df), "month":[month]*len(df), "day": dd})

    # 便利関数: 欲しい列を部分一致で探す
    def pick(substr_list):
        for c in df.columns:
            name = str(c)
            if any(s in name for s in substr_list):
                return c
        return None

    # 数値クリーニング
    def clean_num(x):
        if pd.isna(x): return pd.NA
        s = str(x)
        # ※「--」「×」「(注)」などノイズを除去
        s = re.sub(r"[^\d\.\-\+]", "", s)
        return pd.to_numeric(s, errors="coerce")

    out = pd.DataFrame({"date": df["date"]})

    # 降水量
    pcol = pick(["降水量"])
    out["precipitation"] = df[pcol].map(clean_num) if pcol in df.columns else pd.NA

    # 平均気温
    tavg = pick(["平均気温", "平均(気温)"])
    out["tavg"] = df[tavg].map(clean_num) if tavg in df.columns else pd.NA

    # 最高気温
    tmax = pick(["最高気温", "最高(気温)", "最高"])
    out["tmax"] = df[tmax].map(clean_num) if tmax in df.columns else pd.NA

    # 最低気温
    tmin = pick(["最低気温", "最低(気温)", "最低"])
    out["tmin"] = df[tmin].map(clean_num) if tmin in df.columns else pd.NA

    # 日照時間
    sun = pick(["日照時間", "日照"])
    out["sunshine"] = df[sun].map(clean_num) if sun in df.columns else pd.NA

    return out

def fetch_month(prec_no: str, block_no: str, year: int, month: int, pause_sec=1.2) -> pd.DataFrame:
    ep, bn = build_endpoint(block_no)

    # 1) CSVトライ: view=p1 → 未指定 → a1
    for view in ("p1", None, "a1"):
        try:
            url, text = try_format1_csv(ep, bn, str(prec_no), year, month, view)
            if is_html(text):
                # ブロックされてHTMLが返った等
                raise ValueError("CSVでなくHTMLが返却")
            df = parse_csv_text(text)
            if df is not None and not df.empty:
                out = to_daily(df, year, month)
                if not out.empty:
                    time.sleep(pause_sec)
                    return out
        except Exception as e:
            # デバッグ保存
            Path(DEBUG_DIR, f"csv_fail_{year}-{month:02d}_{view or 'noview'}.txt").write_text(
                f"URL={url}\n\n{text[:4000]}", encoding="utf-8", errors="ignore"
            )

    # 2) HTMLフォールバック: view=p1 → 未指定 → a1
    for view in ("p1", None, "a1"):
        try:
            url, tbl = html_fallback(ep, bn, str(prec_no), year, month, view)
            if tbl is None or tbl.empty:
                raise ValueError("HTMLに表が見つからない")
            out = to_daily(tbl, year, month)
            if not out.empty:
                time.sleep(pause_sec)
                return out
        except Exception as e:
            Path(DEBUG_DIR, f"html_fail_{year}-{month:02d}_{view or 'noview'}.html").write_bytes(
                requests.get(url, headers=HEADERS, timeout=30).content
            )

    raise ValueError(f"月データ取得失敗: {year}-{month} (prec_no={prec_no}, block_no={block_no})")

def fetch_daily_year_by_codes(prec_no: str, block_no: str, year: int) -> pd.DataFrame:
    frames = []
    for m in range(1, 13):
        try:
            print(f"[INFO] pull {year}-{m:02d} ...", flush=True)
            frames.append(fetch_month(prec_no, block_no, year, m))
        except Exception as e:
            print(f"[WARN] {year}-{m}: {e}")
    if not frames:
        raise RuntimeError("1年分のCSV/HTMLが1件も取得できませんでした。コードや年を確認してください。")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df

def to_pysd_excel(df: pd.DataFrame, out_path: Path):
    """
    River_management_chikugo.py の ExtData 参照に合わせた Excel 'jma' シートを作る。
      - B列: 日付（YYYY/MM/DD）
      - E列: 平均気温
      - H列: 降水量
      - K列: 日照時間
      - Q列: 最高気温
      - V列: 最低気温
    """
    jma = pd.DataFrame()
    jma["B"] = pd.to_datetime(df["date"]).dt.strftime("%Y/%m/%d")
    jma["E"] = df.get("tavg")
    jma["H"] = df.get("precipitation")
    jma["K"] = df.get("sunshine")
    jma["Q"] = df.get("tmax")
    jma["V"] = df.get("tmin")

    meta = pd.DataFrame({
        "note": [
            "出典：気象庁ホームページ（過去の気象データ検索）",
            "URL: https://www.data.jma.go.jp/stats/etrn/",
            "本ファイルは自動取得データを加工して作成しています。",
        ]
    })

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as w:
        jma.to_excel(w, sheet_name="jma", index=False, header=False)
        meta.to_excel(w, sheet_name="about", index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prec-no", required=True, help="例: 82（福岡県）")
    ap.add_argument("--block-no", required=True, help="例: 0790（久留米 AMeDAS）/ 47807（官署）")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    print(f"[INFO] 取得先: prec_no={args.prec_no}, block_no={args.block_no}, year={args.year}")
    df = fetch_daily_year_by_codes(args.prec_no, args.block_no, args.year)

    # 欠測を 0/前方補間したい場合はここで処理（任意）
    # df["precipitation"] = df["precipitation"].fillna(0)

    to_pysd_excel(df, args.out)
    print(f"[OK] 保存しました: {args.out}")

if __name__ == "__main__":
    main()
