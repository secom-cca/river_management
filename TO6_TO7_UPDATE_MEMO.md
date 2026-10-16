# River management model: to6 to to7 update memo

更新確認日: 2026-09-11

## 1. 対象ファイル

| 区分 | ファイル | 用途 |
|---|---|---|
| 旧モデル | `River_management_xls_to6.py` | to7との比較元 |
| 現行モデル | `River_management_xls_to7.py` | 全分野を含むPySDモデル本体 |
| 基準実行 | `run_vensim_with_pysd_to7.py` | 単一シナリオの詳細出力 |
| NSE | `calculate_nse_to7.py` | present/baseの上下流NSE計算 |
| パラスタ | `run_vensim_with_pysd_to7_para.py` | 3気候シナリオのLHS |
| 最適化 | `run_vensim_with_pysd_to7_opt.py` | 適応策の差分進化法による探索 |
| 作図 | `plot_para_to7_uncertainty.py`、`plot_opt_to7_results.py`、`plot_opt_to7_patterns.py` | パラスタ・最適化結果の可視化 |

## 2. 住宅被害のイベント単位化

to6では、浸水が継続している間、同じ被害率を世帯数へ日々掛ける構造があり、同一イベントの被害を反復計上する可能性があった。to7では外水・内水ごとにイベント内最大被害率を保持し、最大被害率が更新された分だけ追加被害として計上する。

- 被害率が0の日を1日挟むとイベント終了とし、イベント最大値をリセットする。
- 金銭被害は「イベント開始時の健全なリスク世帯数 × 最大被害率の増分 × 被害原単位」で計上する。
- 過去イベントから未復旧の住宅は健全世帯母数に含めない。
- 重複区域の物理的な被災世帯移動には、外水・内水被害率の大きい方を使い、同じ住宅を二重に被災ストックへ移さない。
- 金銭被害は外水と内水を加算する。重複区域で両方が発生した場合の複合被害を表す意図的な仮定であり、重複割引は行わない。

主な追加変数は次のとおり。

- `flood_event_max_damage_ratio`
- `inundation_event_max_damage_ratio`
- `combined_event_max_damage_ratio`
- `flood_event_exposure_state`
- `inundation_event_exposure_state`
- `incremental_flood_damage_ratio`
- `incremental_inundation_damage_ratio`
- `incremental_flood_damage_fraction`
- `incremental_inundation_damage_fraction`
- `incremental_combined_damage_fraction`

## 3. リスク世帯と住宅復旧

リスク世帯は、外水のみ、内水のみ、外水・内水重複の3ストックで管理する。基準値はユニークな総リスク世帯50,000世帯、重複率50%である。

| 区分 | 初期値 |
|---|---:|
| 外水のみ | 12,500世帯 |
| 内水のみ | 12,500世帯 |
| 重複 | 25,000世帯 |

- 3ストックそれぞれに、自ストックを母数とした自然流入・自然流出を設定した。
- 移住・嵩上げは外水側50%、内水側50%へ配分し、重複区域を含めて各リスク世帯の構成比で配分する。
- 被災世帯も外水のみ、内水のみ、重複の3ストックに分離した。
- 復旧は被災フローを入力とする3次遅延で表現し、基準遅延期間は365日とした。
- `recovery_ratio=0.9`とし、各被災コホートの90%は元のリスク区分へ復帰する。
- 未復旧の10%は被災住宅ストックに残留する。
- 従来の単一`damaged_houses`は、3被災ストックの合計を返す互換用指標に変更した。

## 4. 住宅被害額とGDPの単位

被害原単位を、暫定的な人口当たり日額から、世帯当たりイベント額へ変更した。

```text
flood_damage_per_household      = 30,000,000 Yen/household/event
inundation_damage_per_household = 30,000,000 Yen/household/event
```

- 3,000万円は、2025年6月改正の国土交通省資産評価単価を参考に、家屋と家庭用品を合算して丸めたモデル代表値である。
- 最大被害率0.8の場合、最大2,400万円/世帯/イベントとなる。
- `flood_damage_per_resident`と`innundation_damage_per_resident`は廃止した。
- 正しい綴りの`financial_damage_by_inundation`を追加した。
- 旧出力との互換性のため、`financial_damage_by_innundation`は別名として残した。
- GDP計算では`average_household_size=2.21 person/household`を掛け、世帯数を人口相当に換算する。
- 被災住宅は復旧するまで日次GDPの世帯母数から除外する。

## 5. 水田被害構造

物理的な水田面積と、その年に生産へ利用できる水田面積を分離した。

```text
physical_paddy_field = downstream_area * paddy_field_ratio
```

- `physical_paddy_field`は土地として固定し、田んぼダム容量などの母数に使う。
- `paddy_field`は生産可能面積とし、被災で減少し、復旧で増加する。
- 被害対象は`physical_paddy_field × ratio_of_paddy_field_in_risky_area`のうち、まだ被害を受けていない面積に限定する。
- 水田被害判定は`innundation_level >= 0.3 m`とした。
- 被害発生日ごとに残存するリスク水田の10%を被害化するため、n日継続後の未被害率は`0.9^n`となる。
- 当年被害は年内には回復させず、翌年から回復を開始する。
- 回復率は、90%回復まで2年となる日次幾何回復率から計算する。
- `inundation_flag`による浸水面積と、0.3 m以上の生産被害判定は別指標として維持する。

## 6. 田んぼダム

- 初期田んぼダム率は10%を維持する。
- 追加整備費は`1,500,000 Yen/ha`とした。
- 投資額は年額から日額へ換算し、物理的水田面積100%を上限とする。
- `paddy_dam_ratio`の分母を物理的水田面積とした。
- `paddyfield_storage_capacity`の単位をm3へ整理し、通常水田と田んぼダムの容量を物理的面積から計算する。
- 根拠が不十分だった固定劣化率は無効化し、`degradation_of_paddy_dam=0`とした。
- 現時点の田んぼダムは内水氾濫の発生閾値にのみ作用し、河川流量を遅延・減少させない。

## 7. ダム放流と越流

固定値10,000 m3/dayだった放流能力を、現在のダム容量に比例する値へ変更した。

```text
predischarge_capacity = dam_capacity * predischarge_capacity_ratio
predischarge_capacity_ratio = 0.10 /day
```

- 貯水率60%以下では能動放流を行わない。
- 貯水率60%から90%にかけて、放流能力の0%から100%まで線形に増加させる。
- 放流量は、放流能力と60%貯水率を下回らない範囲の放流可能量の小さい方に制限する。
- `dam_spill`は当日の流入と能動放流後の予測貯水量から計算し、`TIME STEP`で除してm3/dayとした。
- `dam_outflow`は当日に利用可能な水量以下へ制限し、ダム貯水量が負にならないようにした。
- この制御は降雨予測に基づく厳密な事前放流ではなく、貯水率連動型の簡易放流である。

## 8. 森林売上

to6の`managed_plantation_forest_area × 0.02`は日次関数として評価され、年間2%ではなく日々2%相当になっていた。to7では年長で除して日次換算した。

```text
lumbering_area = managed_plantation_forest_area * 0.02 / current_year_length
sales_of_forestry = lumbering_area * 1,010,000
```

- 伐採対象は管理された人工林のみである。
- 年間伐採面積は管理人工林面積の2%である。
- 売上単価は101万円/haである。
- 林業売上は独立した出力指標であり、現時点ではGDPや自治体収入へ接続していない。
- 森林面積固定、森林管理状態、森林機能係数、生物多様性、森林保水機能、土砂災害リスクの基本構造はto6から維持した。

## 9. 維持した農業・気候・水文設定

次の設定はto7で新設したものではなく、to6から維持またはrunnerで継続上書きしている。

- 収量回帰式: `yield_per_10a = 469.68 - 0.0571 * precip_jul_sep + 0.0462 * solar_jul_sep`
- 面積換算: `yearly_total_crop_yield_kg = yield_per_10a * paddy_field_at_harvest * 10`
- 収穫日: 10月1日
- コメ価格基準値: 250 Yen/kg
- present: 降水量倍率1.0、気温加算0.0 degree C
- 2C: 降水量倍率1.1、気温加算1.3 degree C
- 4C: 降水量倍率1.3、気温加算4.1 degree C
- 日田・朝倉の気象入力と、既存の日射量入力
- 2009年から2023年の15年間と、うるう年を含む実カレンダー
- 別途較正した9個の水文パラメータ

## 10. runnerと出力の更新

- 基準実行、NSE、パラスタ、最適化の全runnerを`River_management_xls_to7.py`参照へ変更した。
- 新しいイベント被害、住宅3区分、水田、ダム放流関連の診断列を出力へ追加した。
- 外部データ先頭セルに対する既知のPySD警告だけを抑制し、それ以外の警告は表示する。
- 実行日時を秒単位でファイル名またはフォルダ名へ付け、既存結果を上書きしない。
- パラスタはpresent、2C、4C、LHS 100ケースと各シナリオのbaseを対象とする。
- パラスタで振る仮定は11個で、住宅復旧率、被害原単位、水田日次被害率、ダム放流比率は基準値へ固定する。
- 最適化はpresent、2C、4Cと、balanced、disaster_only、agriculture_only、ecosystem_onlyの4目的パターンを対象とする。
- 現在の最適化は粗い一次探索として`MAXITER=60`、`POPSIZE=5`、`tol=0.01`、1 seed、16 workersで実行する。
- 最適化の最大評価回数は、20変数の場合、1ケース6,100回、12ケース合計73,200回である。
- 最適化途中のsummary、decisions、yearly、最終母集団をチェックポイントへ保存し、後続探索の初期集団に利用できる。

## 11. 検証結果

- present/base、先頭90日除外のNSEは上流0.5105、下流0.5693である。
- 既存の較正済み水文パラメータをto7でも使用している。
- PySD版と独立NumPy水文版について、38出力 × 3シナリオの114比較が設定した数値許容誤差内で一致した。
- NumPy版とPySD版のNSE差は実質ゼロだった。
- NumPy水文版は水文専用であり、農業、住宅、森林、生態系、費用を含む適応策最適化には使用しない。
- 水文だけをNumPy化して残りをPySDで計算するハイブリッド版は約1.01倍にとどまったため、全分野最適化はPySD版を継続する。

## 12. 解釈上の注意

- to6とto7では住宅被害の時間集計、被害原単位、水田被害、ダム放流、林業売上が変わったため、被害額・政策効果・最適化結果を直接比較しない。
- 外水・内水の金銭被害は、重複区域についても意図的に加算する。
- 田んぼダムは内水対策だけを表し、河川流量への流出遅延は未実装である。
- 貯水率連動放流は簡略化された仮定であり、実ダムの操作規則を再現するものではない。
- 住宅被害3,000万円、復旧率90%、復旧遅延365日、水田日次被害率10%は固定された代表値である。
- 降水量は全日に同一倍率を掛け、日射量は気候シナリオに連動させていない。
- 現在のLHS 100ケースと最適化60世代は試行・一次探索用であり、最終的な頑健性確認にはサンプル増加、複数seed、追加世代が必要である。

## 13. 再実行が必要な成果物

to7への更新後は、同じ版と同じ実行日時の結果を使って次を再作成する。

1. present/baseのNSEと診断図
2. 3シナリオのLHSパラメータスタディ
3. 不確実性バンド、レンジバー、相関ヒートマップ
4. 3シナリオ・4目的パターンの適応策最適化
5. 最適化結果図

実行順とコマンドは`TO7_EXECUTION_GUIDE.md`を参照する。
