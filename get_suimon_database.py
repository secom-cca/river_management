from __future__ import annotations
import re, os, calendar, pathlib, unicodedata, argparse, sys
from urllib.parse import urljoin
from datetime import date
from time import sleep

import requests
import pandas as pd
from bs4 import BeautifulSoup

# ====== 代表地点 → 観測所ID マッピング ======
# ここに必要な代表地点を追加（例: 「sumimata」→ 墨俣）
REP_TO_ID = {
    "sumimata": "305091285502190",   # 長良川・墨俣（例）
    # "tokyo_xxx": "XXXXXXXXXXXXXXX",
}

WIS_BASES = [
    "https://www1.river.go.jp/cgi-bin",
    "https://www1.river.go.jp/cgi",
]
UA = {"User-Agent": "Mozilla/5.0 (compatible; hydrology-fetch/1.1)"}
DEBUG_SAVE = True  # 失敗時にHTML保存する

def _decode(resp: requests.Response) -> str:
    raw = resp.content
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "euc-jp" in ctype or "euc_jp" in ctype:
        return raw.decode("euc_jp", errors="replace")
    if "shift_jis" in ctype or "ms932" in ctype or "cp932" in ctype:
        return raw.decode("cp932", errors="replace")
    head = raw[:4096].lower()
    if b"charset=euc-jp" in head or b"charset=euc_jp" in head:
        return raw.decode("euc_jp", errors="replace")
    if b"charset=shift_jis" in head or b"charset=ms932" in head or b"charset=cp932" in head:
        return raw.decode("cp932", errors="replace")
    enc = resp.apparent_encoding or resp.encoding or "utf-8"
    try:
        return raw.decode(enc, errors="replace")
    except Exception:
        return raw.decode("utf-8", errors="replace")

def _http_get(url: str, timeout=25) -> str:
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    return _decode(r)

def _follow_frames(html: str, base_url: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    fr = soup.find(["iframe", "frame"])
    if fr and fr.get("src"):
        next_url = urljoin(base_url, fr["src"])
        nxt = _http_get(next_url)
        return nxt, next_url
    meta = soup.find("meta", attrs={"http-equiv": re.compile(r"refresh", re.I)})
    if meta and "content" in meta.attrs:
        m = re.search(r"url\s*=\s*([^;]+)", meta["content"], flags=re.I)
        if m:
            next_url = urljoin(base_url, m.group(1).strip())
            nxt = _http_get(next_url)
            return nxt, next_url
    return html, base_url

def _fetch_year_page(station_id: str, year: int) -> tuple[str, str]:
    params = f"ID={station_id}&KIND=7&KAWABOU=NO&BGNDATE={year}0101&ENDDATE={year}1231"
    last_err = None
    for base in WIS_BASES:
        url = f"{base}/DspWaterData.exe?{params}"
        try:
            html = _http_get(url)
            html2, url2 = _follow_frames(html, url)
            return html2, url2
        except Exception as e:
            last_err = e
    raise RuntimeError(f"年表取得に失敗: ID={station_id}, year={year}, err={last_err}")

def _fetch_csv_text(station_id: str, bgn: date, end: date) -> tuple[str|None, str|None]:
    params = f"ID={station_id}&KIND=7&KAWABOU=NO&BGNDATE={bgn:%Y%m%d}&ENDDATE={end:%Y%m%d}&CSV=1"
    for base in WIS_BASES:
        url = f"{base}/DspWaterData.exe?{params}"
        try:
            txt = _http_get(url)
            txt2, url2 = _follow_frames(txt, url)
            if re.search(r"^\s*\d{4}/\d{1,2}/\d{1,2}\s*,", txt2, flags=re.M):
                return txt2, url2
        except Exception:
            pass
    return None, None

def _to_int_digits_any(s: str) -> int | None:
    if not s:
        return None
    s2 = unicodedata.normalize("NFKC", s)
    m = re.search(r"(\d+)", s2)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None

def _parse_daily_csv_lines(txt: str) -> pd.DataFrame:
    rows = []
    for line in txt.splitlines():
        m = re.match(r"^\s*(\d{4})[/-](\d{1,2})[/-](\d{1,2})\s*[, \t]\s*([^\s,]+)", line)
        if not m:
            continue
        y, mo, d, val = m.groups()
        token = val.strip()
        if token in ("-", "$", "") or ("欠測" in token) or ("休" in token):
            v = pd.NA
        else:
            v = pd.to_numeric(token.replace(",", ""), errors="coerce")
        rows.append({"date": date(int(y), int(mo), int(d)), "flow": v})
    return pd.DataFrame(rows)

def _parse_daily_from_year_table(page: str, year: int) -> pd.DataFrame:
    if "#月" in page and "1日データ" in page:
        rows = []
        for line in page.splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "観測所名" in s:
                continue
            parts = [p.strip() for p in s.split(",")]
            if not parts:
                continue
            month = _to_int_digits_any(parts[0])
            if not month or not (1 <= month <= 12):
                continue
            max_day = calendar.monthrange(year, month)[1]
            for d in range(1, max_day + 1):
                idx = 1 + 2*(d-1)
                if idx >= len(parts):
                    break
                token = parts[idx].strip().replace(" ", "")
                if (token in ("-", "$", "")) or ("欠測" in token) or ("休" in token):
                    v = pd.NA
                else:
                    v = pd.to_numeric(token.replace(",", ""), errors="coerce")
                rows.append({"date": date(year, month, d), "flow": v})
        return pd.DataFrame(rows)

    soup = BeautifulSoup(page, "lxml")
    table = _find_data_table(soup)
    if not table:
        return pd.DataFrame()

    rows = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td","th"])
        if not cells:
            continue
        month = _to_int_digits_any(cells[0].get_text(strip=True))
        if not month or not (1 <= month <= 12):
            continue
        max_day = calendar.monthrange(year, month)[1]
        day, i = 1, 1
        while i < len(cells) and day <= max_day:
            token = cells[i].get_text(strip=True).replace(" ", "")
            if (token in ("-", "$", "")) or ("欠測" in token) or ("休" in token):
                v = pd.NA
            else:
                v = pd.to_numeric(token.replace(",", ""), errors="coerce")
            rows.append({"date": date(year, month, day), "flow": v})
            i += 2
            day += 1
    return pd.DataFrame(rows)

def _iter_months(start: date, end: date):
    cur = date(start.year, start.month, 1)
    while cur <= end:
        last = calendar.monthrange(cur.year, cur.month)[1]
        yield cur, date(cur.year, cur.month, last)
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)

def get_daily_discharge(station_id: str, start: date, end: date, pause_sec: float = 0.3) -> pd.DataFrame:
    if end < start:
        raise ValueError("end は start 以降の日付を指定してください。")

    frames = []
    last_urls = []
    for m0, m1 in _iter_months(start, end):
        b, e = max(m0, start), min(m1, end)
        txt, url_used = _fetch_csv_text(station_id, b, e)
        if txt:
            dfm = _parse_daily_csv_lines(txt)
            if not dfm.empty:
                frames.append(dfm)
        if url_used:
            last_urls.append(url_used)
        sleep(pause_sec)

    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = df[(df["date"] >= start) & (df["date"] <= end)].sort_values("date").reset_index(drop=True)
        return df[["date","flow"]]

    frames = []
    last_page_html = None
    last_page_url = ""
    for y in range(start.year, end.year + 1):
        page, used_url = _fetch_year_page(station_id, y)
        dfd = _parse_dat_link_to_df(page, used_url)
        if not dfd.empty:
            frames.append(dfd)
        else:
            dfy = _parse_daily_from_year_table(page, y)
            if not dfy.empty:
                frames.append(dfy)
        last_page_html, last_page_url = page, used_url
        sleep(pause_sec)

    if not frames:
        if DEBUG_SAVE and last_page_html:
            out = pathlib.Path.cwd() / f"wis_debug_{station_id}_{start.year}_{end.year}.html"
            out.write_text(last_page_html, encoding="utf-8", errors="replace")
            msg_save = f"\nデバッグ保存: {out}"
        else:
            msg_save = ""
        sample = last_urls[-1] if last_urls else f"{WIS_BASES[0]}/DspWaterData.exe?ID={station_id}&KIND=7&KAWABOU=NO&BGNDATE={start:%Y%m%d}&ENDDATE={end:%Y%m%d}&CSV=1"
        raise RuntimeError(
            "日データの抽出に失敗しました。観測所ID/KIND/期間の組合せ、またはページ構成をご確認ください。\n"
            f"手動確認用URL例: {sample}\n"
            f"年表最終URL: {last_page_url}{msg_save}"
        )

    df = pd.concat(frames, ignore_index=True)
    df = df[(df["date"] >= start) & (df["date"] <= end)].sort_values("date").reset_index(drop=True)
    return df[["date","flow"]]

def _find_data_table(soup: BeautifulSoup):
    tables = soup.find_all("table")
    candidate = None
    best_cols = 0
    for t in tables:
        txt = t.get_text(" ", strip=True)
        if ("1日" in txt and "月" in txt) or ("単位" in txt and "m" in txt):
            first_tr = t.find("tr")
            ncols = len(first_tr.find_all(["td","th"])) if first_tr else 0
            if ncols >= best_cols:
                candidate = t
                best_cols = ncols
    if not candidate:
        for t in tables:
            trs = t.find_all("tr")
            if not trs:
                continue
            ncols = max(len(tr.find_all(["td","th"])) for tr in trs)
            if ncols > best_cols:
                candidate = t
                best_cols = ncols
    return candidate

def forward_fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["flow"] = out["flow"].ffill()
    return out

def _parse_dat_link_to_df(page_html: str, year_page_url: str) -> pd.DataFrame:
    soup = BeautifulSoup(page_html, "lxml")
    a = soup.find("a", href=re.compile(r"/dat/(?:dload/)?download/.*\.dat", re.I))
    if not a or not a.get("href"):
        return pd.DataFrame()
    dat_url = urljoin(year_page_url, a["href"])
    txt = _http_get(dat_url)
    df = _parse_daily_csv_lines(txt)
    return df

# ====== ここから追加：CLI とファイル出力 ======

def _parse_year_arg(arg: str) -> list[int]:
    """
    '2015' または '2008-2018' を受け取り、対象年のリストを返す。
    前者は [2015]、後者は [2008, ... , 2018]
    """
    s = arg.strip()
    m = re.fullmatch(r"(\d{4})-(\d{4})", s)
    if m:
        y0, y1 = int(m.group(1)), int(m.group(2))
        if y1 < y0:
            raise ValueError("年の範囲は開始年 <= 終了年で指定してください。")
        return list(range(y0, y1 + 1))
    m = re.fullmatch(r"\d{4}", s)
    if m:
        return [int(s)]
    raise ValueError("年は 'YYYY' または 'YYYY-YYYY' で指定してください。")

def main():
    parser = argparse.ArgumentParser(description="WIS 日流量取得（年ごとCSV出力）")
    parser.add_argument("rep", help="代表地点（REP_TO_ID のキー）")
    parser.add_argument("year", help="年（例: 2015）または 範囲（例: 2008-2018）")
    parser.add_argument("--fill-forward", action="store_true",
                        help="欠測を前日値で補間したCSVも一緒に出力（_filled 付き）")
    parser.add_argument("--pause", type=float, default=0.3, help="アクセス間隔秒（既定0.3）")
    args = parser.parse_args()

    rep = args.rep
    if rep not in REP_TO_ID:
        sys.exit(f"代表地点 '{rep}' が REP_TO_ID に見つかりません。辞書に追加してください。")

    station_id = REP_TO_ID[rep]
    years = _parse_year_arg(args.year)

    for y in years:
        start, end = date(y, 1, 1), date(y, 12, 31)
        print(f"[INFO] fetching {rep} (ID={station_id}) {y} ...")
        df = get_daily_discharge(station_id, start, end, pause_sec=args.pause)

        outpath = pathlib.Path.cwd() / f"flow_{rep}_{y}.csv"
        df.to_csv(outpath, index=False, encoding="utf-8")
        print(f"[OK] saved: {outpath}")

        if args.fill_forward:
            dff = forward_fill_missing(df)
            outpath2 = pathlib.Path.cwd() / f"flow_{rep}_{y}_filled.csv"
            dff.to_csv(outpath2, index=False, encoding="utf-8")
            print(f"[OK] saved: {outpath2}")

if __name__ == "__main__":
    main()
