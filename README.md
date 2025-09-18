# System Dynamics Model for River Resilience

川のレジリエンス（洪水リスク低減策）の評価を目的とした**Vensimモデル**と、その実行・可視化のための周辺ツール一式です。
Vensimでの開発資産を **PySD** や **SDEverywhere** で再利用し、研究・教育・プロトタイピングを素早く回せる構成にしています。

---

## 1. 機能（What’s inside）

* **Vensim モデル**

  * `*.mdl`（Vensimテキスト）と、PySDで実行可能な `*.py` を同梱
* **PySD での実行**

  * `run_vensim_with_pysd.py`：モデルをCLIから実行し、CSV出力
* **Streamlit Webアプリ**

  * `app.py`：ブラウザ上でパラメータを操作・結果可視化・観測データとの比較
* **SDEverywhere**（WASM/JS 変換 & 開発）

  * VensimモデルをWebAssemblyに変換し、ブラウザ/Node.jsで実行可能
* **水門データ収集（WIS）**

  * `get_suimon_database.py`：国土交通省の水文水質データベース（WIS）から**日流量**CSVをスクレイピング

---

## 2. リポジトリ構成（例）

```
.
├─ app.py                         # Streamlitアプリ（PySDでモデル実行・可視化）
├─ run_vensim_with_pysd.py        # CLI実行スクリプト（PySD）
├─ get_suimon_database.py         # 水門/WIS 日流量スクレイパ
├─ River_management_xls.mdl       # Vensimモデル（例1）
├─ River_management_xls.py        # 上記のPySD変換済みファイル（例1）
├─ River_management_chikugo.mdl   # Vensimモデル（例2）
├─ River_management_chikugo.py    # 上記のPySD変換済みファイル（例2）
├─ data/
│   └─ jma_kurume_2023.xls        # モデルが参照する外部Excel（GET XLS DATA）
└─ (SDEverywhere プロジェクト一式)
```

> **メモ**
>
> * `app.py` の既定は `River_management_xls.(py|mdl)` を参照します。
> * `run_vensim_with_pysd.py` は `River_management_chikugo.(py|mdl)` を参照します。
>   変更したい場合は各スクリプト冒頭の定数を書き換えてください。

---

## 3. セットアップ

### 3.1 Python 環境

```bash
# 推奨: 仮想環境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -U pip
pip install streamlit pysd numpy pandas xlrd requests beautifulsoup4 lxml
```

### 3.2 Node.js（SDEverywhere を使う場合）

* Node.js 18+ 推奨
* 既存の SDEverywhere プロジェクトとして利用する場合は `npm install` を実行

---

## 4. 使い方（3通り）

### 4.1 Streamlitでインタラクティブ実行（おすすめ）

```bash
streamlit run app.py
```

* **モデル読み込み**：`River_management_xls.py`（優先）または `River_management_xls.mdl`
* **外部データ（GET XLS DATA）**：`jma_kurume_2023.xls` をアップロードするか、同名で配置
* **プリセット**：筑後川/長良川など代表値をワンクリック適用
* **パラメータ調整**：スライダーで投資額・流域条件等を上書き
* **観測データ比較**：CSV（例: `date,flow`）を読み込み、

  * 単位 `m3/s` → 自動で `m3/day` に変換（×86400）
  * 最大±N日のラグ自動探索で**形状比較**（相関最大化）
  * 指標: RMSE, MAE, Mean Bias, Pearson r, **NSE**
* **出力保存**：モデル結果や観測マージ結果をCSVでダウンロード

**観測CSVの例**

```csv
date,temperature,precipitation,flow,level
2023-01-01,8.2,0.0,120.5,0.8
2023-01-02,8.0,5.1,130.1,0.9
```

アプリでは `date` と `flow` 列のみ使用します（列名はUIで変更可）。

---

### 4.2 CLIでサクッと実行（PySD）

```bash
python run_vensim_with_pysd.py
```

* 取得列はスクリプト内 `return_cols` を変更
* 出力：`data/simulation_output.csv`
* 参照Excel：`data/jma_kurume_2023.xls`（モデルの ExtData が参照）

---

### 4.3 SDEverywhere（WASM/JS化・開発モード）

> このリポジトリには SDEverywhere の**テンプレート由来**の構成・設定が含まれています。
> 既存のSDEプロジェクトとして開発を進められます。

```bash
# 初期化（新規プロジェクト作成時）: 「Default」テンプレートを選択
npm create @sdeverywhere@latest

# ローカル開発（モデル・設定変更を監視し自動ビルド/チェック）
npm run dev
```

* Vensimモデル → **WebAssembly** モジュールに変換
* `config/` のCSVで入出力/ユニット等を設定
* 生成されたコアAPIをJS/TSから呼び出し可能
* シンプルなWebアプリ（jQueryベース）で動作確認

> ライセンスは SDEverywhere の `LICENSE`（MIT）に従います。

---

## 5. モデルとパラメータ

* **モデル開始日**：アプリで指定（既定: `2023-01-01`）。モデル時刻0日目に対応
* **主要出力（例）**：
  `daily_total_gdp`, `dam_storage`, `downstream_storage`,
  `upstream_storage`, `river_discharge_downstream`,
  `houses_damaged_by_inundation`,
  `financial_damage_by_innundation`, `financial_damage_by_flood`
* **代表パラメータ**（一部）

  * 将来降水補正、ダム/堤防/排水/ため池の**投資額**と**開始時期**
  * 高齢者比率、防災力（避難率係数）、住民流出入率
  * 圃場比率、リスク域圃場比率
  * **流域条件**（初期ダム容量、上・下流面積、森林面積比、直接流出比、計画高水流量 など）
* **プリセット**：筑後川・長良川・利根川（例）を同梱。必要に応じて実測代表値に更新してください。

---

## 6. 水門/WIS 日流量データの取得

`get_suimon_database.py` は、WISのフレーム/エンコード差異に頑健な実装です。
代表地点⇔観測所IDは `REP_TO_ID` に追記してください。

### 使い方

```bash
# 2015年のみ
python get_suimon_database.py sumimata 2015

# 2008年～2018年を一括
python get_suimon_database.py sumimata 2008-2018

# 欠測を前日値で補間したCSVも出力
python get_suimon_database.py sumimata 2015 --fill-forward
```

* 出力：`flow_<rep>_<year>.csv`（必要なら `_filled` 版も）
* 失敗時はデバッグHTMLを保存（`wis_debug_*.html`）。`WIS_BASES` を切替トライします。

---

## 7. 外部データ（GET XLS DATA）

* モデル側の **ExtData** が参照する既定名：`jma_kurume_2023.xls`
* `app.py` では**アップローダ**から同名で保存、またはルート/`data/`に配置
* ファイル名を変更する場合は、各スクリプト冒頭の定数を合わせて修正してください

---

## 8. トラブルシューティング

* **「モデルが見つからない」**

  * `*.py` or `*.mdl` のパスを確認（`app.py` サイドバーで指定可）
* **観測と単位が合わない**

  * 観測が `m3/s` の場合、UIで選ぶと自動で `m3/day` に変換されます
* **選択した出力列が存在しない**

  * `app.py` は例外時に**フル実行→存在列のみ再抽出**します（警告を表示）
* **WISスクレイピングが失敗**

  * `REP_TO_ID` のID再確認、期間・種別（`KIND=7`）を見直し
  * 失敗時のメッセージ内URLで手動確認、`WIS_BASES` の別ドメインを試行

---

## 9. 参考（詳細説明）

ドキュメントはこちら：
[https://docs.google.com/document/d/116Xg9WkcorllC6vz6C-agFRKM3BDTerf/edit?usp=drive\_link\&ouid=107470865859100242765\&rtpof=true\&sd=true](https://docs.google.com/document/d/116Xg9WkcorllC6vz6C-agFRKM3BDTerf/edit?usp=drive_link&ouid=107470865859100242765&rtpof=true&sd=true)

---

## 10. ライセンス

* 本リポジトリの SDEverywhere 部分は **MIT License**（`LICENSE` 参照）
* その他のコンテンツのライセンスは各ファイルの記述に従います

---

## 付録：よく使うコマンド早見表

```bash
# 1) Webアプリ
streamlit run app.py

# 2) CLI実行（PySD）
python run_vensim_with_pysd.py

# 3) 観測データ取得（WIS）
python get_suimon_database.py sumimata 2015
python get_suimon_database.py sumimata 2008-2018 --fill-forward

# 4) SDEverywhere 開発
npm install
npm run dev
```

> ご不明点や追加したい流域・観測所IDがあれば `REP_TO_ID`・プリセットを更新してください。
