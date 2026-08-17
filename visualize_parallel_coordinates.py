"""シナリオ比較用パラレル座標ビューア。

results/parameter_study/ 内の para_to7_all_scenarios_*_overall_summary.csv /
*_yearly_summary.csv を読み込み、ブラウザ上で
シナリオ・年・表示パラメータをインタラクティブに切り替えながら
Parallel Coordinates（パラレル座標プロット）で比較できる Dash アプリ。

実行:
    python3 visualize_parallel_coordinates.py
    -> http://127.0.0.1:8050 をブラウザで開く
"""

from __future__ import annotations

import glob
import os

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from basin_config import load_active_basin

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASIN_KEY = load_active_basin()["key"]
DATA_DIR_CANDIDATES = [
    os.path.join(BASE_DIR, "results", "parameter_study", BASIN_KEY),
    os.path.join(BASE_DIR, "data"),  # Legacy output location.
]

META_COLS = ["scenario", "case_id", "varied_parameter", "level"]
CATEGORICAL_AXES = ["scenario", "varied_parameter", "level"]

SCENARIO_ORDER = ["present", "2C", "4C"]
SCENARIO_COLORS = {
    "present": "#2a78d6",  # blue
    "2C": "#eb6834",       # orange
    "4C": "#e34948",       # red
}
CATEGORICAL_PALETTE_CYCLE = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

DEFAULT_AXES = [
    "yield_per_10a",
    "yearly_total_crop_yield_kg",
    "damaged_paddy_field_at_year_end",
    "paddy_field_recovery_years_to_90_percent",
    "river_discharge_downstream_year_max",
    "landslide_disaster_risk_year_sum",
    "financial_damage_by_innundation_year_sum",
    "financial_damage_by_flood_year_sum",
    "municipality_cost_year_sum",
    "yearly_gdp_total",
]


def _find_data_dir() -> str:
    for data_dir in DATA_DIR_CANDIDATES:
        overall = glob.glob(
            os.path.join(data_dir, "para_to7_all_scenarios_*_overall_summary.csv")
        )
        yearly = glob.glob(
            os.path.join(data_dir, "para_to7_all_scenarios_*_yearly_summary.csv")
        )
        if overall and yearly:
            return data_dir
    searched = ", ".join(DATA_DIR_CANDIDATES)
    raise FileNotFoundError(f"No complete parameter-study output set under: {searched}")


DATA_DIR = _find_data_dir()


def _latest(pattern: str) -> str:
    return sorted(glob.glob(os.path.join(DATA_DIR, pattern)))[-1]


OVERALL_CSV = _latest("para_to7_all_scenarios_*_overall_summary.csv")
YEARLY_CSV = _latest("para_to7_all_scenarios_*_yearly_summary.csv")

df_overall = pd.read_csv(OVERALL_CSV)
df_yearly = pd.read_csv(YEARLY_CSV)

YEAR_MIN, YEAR_MAX = int(df_yearly["year"].min()), int(df_yearly["year"].max())


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c != "year" and pd.api.types.is_numeric_dtype(df[c])
    ]


NUMERIC_OVERALL = numeric_columns(df_overall)
NUMERIC_YEARLY = numeric_columns(df_yearly)
ALL_AXIS_OPTIONS = sorted(set(NUMERIC_OVERALL) | set(NUMERIC_YEARLY))
COLOR_OPTIONS = CATEGORICAL_AXES + ALL_AXIS_OPTIONS


def discrete_colorscale(colors: list[str]) -> list[list]:
    n = len(colors)
    scale = []
    for i, c in enumerate(colors):
        scale.append([i / n, c])
        scale.append([(i + 1) / n, c])
    return scale


def continuous_colorscale() -> list[list]:
    n = len(SEQUENTIAL_BLUE)
    return [[i / (n - 1), c] for i, c in enumerate(SEQUENTIAL_BLUE)]


def build_line_spec(df: pd.DataFrame, color_by: str) -> dict:
    if color_by in CATEGORICAL_AXES:
        cats = sorted(df[color_by].dropna().unique().tolist())
        if color_by == "scenario":
            cats = [c for c in SCENARIO_ORDER if c in cats]
            colors = [SCENARIO_COLORS[c] for c in cats]
        else:
            colors = [CATEGORICAL_PALETTE_CYCLE[i % len(CATEGORICAL_PALETTE_CYCLE)] for i in range(len(cats))]
        code_map = {c: i for i, c in enumerate(cats)}
        values = df[color_by].map(code_map).astype(float)
        n = max(len(cats), 1)
        return dict(
            color=values,
            colorscale=discrete_colorscale(colors),
            cmin=0,
            cmax=n,
            showscale=True,
            colorbar=dict(
                title=color_by,
                tickvals=[i + 0.5 for i in range(n)],
                ticktext=cats,
                len=0.75,
            ),
        )
    values = df[color_by].astype(float)
    return dict(
        color=values,
        colorscale=continuous_colorscale(),
        cmin=float(values.min()),
        cmax=float(values.max()),
        showscale=True,
        colorbar=dict(title=color_by, len=0.75),
    )


def build_dimensions(df: pd.DataFrame, axes: list[str]) -> list[dict]:
    dims = []
    for col in axes:
        if col not in df.columns:
            continue
        vals = df[col].astype(float)
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmin == vmax:
            vmin, vmax = vmin - 0.5, vmax + 0.5
        dims.append(dict(range=[vmin, vmax], label=col, values=vals))
    return dims


app = Dash(__name__)
app.title = "シナリオ比較 パラレル座標ビューア"

app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
  :root { color-scheme: light; }
  body {
    margin: 0;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    background: #f9f9f7;
    color: #0b0b0b;
  }
  @media (prefers-color-scheme: dark) {
    body { background: #0d0d0d; color: #ffffff; }
  }
  .panel { background: #fcfcfb; border: 1px solid rgba(11,11,11,0.10); border-radius: 8px; }
  @media (prefers-color-scheme: dark) {
    .panel { background: #1a1a19; border-color: rgba(255,255,255,0.10); }
  }
  label { font-size: 13px; color: #52514e; }
  @media (prefers-color-scheme: dark) {
    label { color: #c3c2b7; }
  }
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""

app.layout = html.Div(
    style={"maxWidth": "1400px", "margin": "0 auto", "padding": "16px 24px"},
    children=[
        html.H2("河川流域管理シナリオ比較：パラレル座標ビューア"),
        html.P(
            f"overall: {os.path.basename(OVERALL_CSV)} / yearly: {os.path.basename(YEARLY_CSV)}",
            style={"fontSize": "12px", "color": "#898781"},
        ),
        html.Div(
            className="panel",
            style={"padding": "16px", "display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
            children=[
                html.Div([
                    html.Label("データ粒度"),
                    dcc.RadioItems(
                        id="mode",
                        options=[
                            {"label": "全期間サマリー (overall)", "value": "overall"},
                            {"label": "年別 (yearly)", "value": "yearly"},
                        ],
                        value="overall",
                        inline=True,
                    ),
                ]),
                html.Div(id="year-slider-wrap", children=[
                    html.Label("年 (year)"),
                    dcc.Slider(
                        id="year-slider",
                        min=YEAR_MIN, max=YEAR_MAX, step=1, value=YEAR_MIN,
                        marks={i: str(i) for i in range(YEAR_MIN, YEAR_MAX + 1)},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ]),
                html.Div([
                    html.Label("シナリオ"),
                    dcc.Checklist(
                        id="scenario-filter",
                        options=[{"label": s, "value": s} for s in SCENARIO_ORDER],
                        value=SCENARIO_ORDER,
                        inline=True,
                    ),
                ]),
                html.Div([
                    html.Label("ケース種別"),
                    dcc.Checklist(
                        id="case-type-filter",
                        options=[
                            {"label": "base（基準値）", "value": "base"},
                            {"label": "lhs（パラメータ変動サンプル）", "value": "lhs_all"},
                        ],
                        value=["base", "lhs_all"],
                        inline=True,
                    ),
                ]),
                html.Div([
                    html.Label("色分け"),
                    dcc.Dropdown(
                        id="color-by",
                        options=[{"label": c, "value": c} for c in COLOR_OPTIONS],
                        value="scenario",
                        clearable=False,
                    ),
                ]),
                html.Div(
                    style={"gridColumn": "1 / span 2"},
                    children=[
                        html.Label("表示する軸（パラメータ、複数選択可）"),
                        dcc.Dropdown(
                            id="axes-select",
                            options=[{"label": c, "value": c} for c in ALL_AXIS_OPTIONS],
                            value=DEFAULT_AXES,
                            multi=True,
                        ),
                    ],
                ),
            ],
        ),
        dcc.Graph(id="parcoords", style={"height": "68vh", "marginTop": "16px"}),
        html.Div(id="row-count", style={"color": "#898781", "fontSize": "13px"}),
    ],
)


@app.callback(
    Output("year-slider-wrap", "style"),
    Input("mode", "value"),
)
def toggle_year_slider(mode):
    return {} if mode == "yearly" else {"display": "none"}


@app.callback(
    Output("axes-select", "options"),
    Output("axes-select", "value"),
    Input("mode", "value"),
)
def refresh_axis_options(mode):
    cols = NUMERIC_YEARLY if mode == "yearly" else NUMERIC_OVERALL
    options = [{"label": c, "value": c} for c in sorted(cols)]
    default = [c for c in DEFAULT_AXES if c in cols]
    return options, default


@app.callback(
    Output("parcoords", "figure"),
    Output("row-count", "children"),
    Input("mode", "value"),
    Input("year-slider", "value"),
    Input("scenario-filter", "value"),
    Input("case-type-filter", "value"),
    Input("color-by", "value"),
    Input("axes-select", "value"),
)
def update_figure(mode, year, scenarios, case_types, color_by, axes):
    df = df_yearly if mode == "yearly" else df_overall
    if mode == "yearly" and year is not None:
        df = df[df["year"] == year]
    if scenarios:
        df = df[df["scenario"].isin(scenarios)]
    if case_types:
        df = df[df["varied_parameter"].isin(case_types)]

    if df.empty or not axes:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text="表示するデータがありません", showarrow=False)],
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig, "0 件"

    line_spec = build_line_spec(df, color_by) if color_by in df.columns else dict(color="#2a78d6")
    dims = build_dimensions(df, axes)

    fig = go.Figure(data=go.Parcoords(line=line_spec, dimensions=dims))
    fig.update_layout(
        margin=dict(l=60, r=60, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color="#0b0b0b"),
    )

    label = "全期間サマリー" if mode == "overall" else f"{year} 年目"
    return fig, f"{label} / {len(df)} 件表示中（軸をドラッグして範囲を絞り込めます）"


if __name__ == "__main__":
    app.run(debug=False, port=8050)
