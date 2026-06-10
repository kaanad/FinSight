import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

# ── Load ───────────────────────────────────────────────────────────────────
CSV = r"C:\incometracker\incomedatasetnew.csv"
df = pd.read_csv(CSV)
df["month_dt"] = pd.to_datetime(df["month"])
df = df.sort_values(["month_dt","user_id"]).reset_index(drop=True)
# Convert month_dt to string for JSON serialisation in Dash
df["month_str"] = df["month_dt"].dt.strftime("%Y-%m-%d")
print(f"[OK] {len(df)} rows | {df['user_id'].nunique()} users | {df['month_dt'].nunique()} months")

EXP = ["rent","food","transport","shopping","entertainment","medical","emi","investment"]

# ── K-Means ────────────────────────────────────────────────────────────────
ua = df.groupby("user_id")[["income","total_expense","savings","saving_ratio"]].mean()
sc = StandardScaler()
km = KMeans(n_clusters=3, random_state=42, n_init=10)
ua["cluster"] = km.fit_predict(sc.fit_transform(ua))
co = sc.inverse_transform(km.cluster_centers_)
od = co[:,2].argsort()
lm = {od[0]:"High Spenders", od[1]:"Balanced", od[2]:"Savers"}
ua["segment"] = ua["cluster"].map(lm)
df = df.merge(ua[["segment"]], left_on="user_id", right_index=True, how="left")

MONTHS = sorted(df["month"].unique())
CITIES = sorted(df["city"].unique())
OCCS   = sorted(df["occupation"].unique())
SEGS   = sorted(df["segment"].unique())

# ── Theme ──────────────────────────────────────────────────────────────────
BG="#0d1117"; CARD="#161b22"; BORDER="#21262d"
CYAN="#58d5c9"; GREEN="#3fb950"; AMBER="#f0a030"
RED="#f85149"; PURPLE="#a371f7"; BLUE="#388bfd"
GOLD="#f5c542"; WHITE="#e6edf3"; MUTED="#8b949e"

def base_layout(**extra):
    d = dict(paper_bgcolor=CARD, plot_bgcolor=CARD,
             font=dict(color=WHITE, family="monospace", size=11),
             margin=dict(l=50, r=30, t=40, b=50),
             xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, color=WHITE),
             yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, color=WHITE))
    d.update(extra)
    return d

CS = {"backgroundColor":CARD,"border":f"1px solid {BORDER}",
      "borderRadius":"8px","padding":"16px","height":"100%"}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "FinSight Analytics"

def kpi(title, val, sub, color, icon):
    return dbc.Card(dbc.CardBody([
        html.P(f"{icon}  {title}",
               style={"color":MUTED,"fontSize":"11px","fontFamily":"monospace","marginBottom":"4px"}),
        html.H3(val, style={"color":WHITE,"fontWeight":"bold","fontFamily":"monospace","marginBottom":"2px"}),
        html.P(sub, style={"color":color,"fontSize":"12px","fontFamily":"monospace","marginBottom":0}),
    ]), style={**CS,"borderTop":f"3px solid {color}"})

app.layout = html.Div(style={"backgroundColor":BG,"minHeight":"100vh","padding":"20px"}, children=[

    dbc.Row([
        dbc.Col([
            html.H2("💰 FinSight Analytics",
                style={"color":WHITE,"fontFamily":"monospace","fontWeight":"bold","marginBottom":"2px"}),
            html.P("Income · Expense · Savings  |  ML-Powered Intelligence",
                style={"color":MUTED,"fontFamily":"monospace","fontSize":"12px"}),
        ], width=5),
        dbc.Col(dbc.Row([
            dbc.Col(dcc.Dropdown(id="dd-city",
                options=[{"label":c,"value":c} for c in CITIES],
                value=None, placeholder="🏙 All Cities", multi=True,
                clearable=True), width=4),
            dbc.Col(dcc.Dropdown(id="dd-occ",
                options=[{"label":o,"value":o} for o in OCCS],
                value=None, placeholder="💼 All Occupations", multi=True,
                clearable=True), width=4),
            dbc.Col(dcc.Dropdown(id="dd-seg",
                options=[{"label":s,"value":s} for s in SEGS],
                value=None, placeholder="🎯 All Segments", multi=True,
                clearable=True), width=4),
        ]), width=7),
    ], className="mb-3"),

    dbc.Card(dbc.CardBody([
        html.P("📅 Month Range",
               style={"color":MUTED,"fontFamily":"monospace","fontSize":"11px","marginBottom":"6px"}),
        dcc.RangeSlider(id="sl-month", min=0, max=len(MONTHS)-1,
            value=[0, len(MONTHS)-1],
            marks={i:{"label":m,"style":{"color":MUTED,"fontSize":"9px","fontFamily":"monospace"}}
                   for i,m in enumerate(MONTHS)},
            tooltip={"placement":"bottom","always_visible":False})
    ]), style={**CS,"marginBottom":"14px"}),

    html.Div(id="kpi-row", className="mb-3"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("📊 GROSS MARGIN — Income vs Expense",
                   style={"color":MUTED,"fontFamily":"monospace","fontSize":"11px","marginBottom":"4px"}),
            dcc.Graph(id="ch-gm", style={"height":"300px"}, config={"displayModeBar":False})
        ]), style=CS), width=8),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("🍩 EXPENSE BREAKDOWN",
                   style={"color":MUTED,"fontFamily":"monospace","fontSize":"11px","marginBottom":"4px"}),
            dcc.Graph(id="ch-donut", style={"height":"300px"}, config={"displayModeBar":False})
        ]), style=CS), width=4),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("📈 SAVINGS TREND + FORECAST",
                   style={"color":MUTED,"fontFamily":"monospace","fontSize":"11px","marginBottom":"4px"}),
            dcc.Graph(id="ch-trend", style={"height":"260px"}, config={"displayModeBar":False})
        ]), style=CS), width=5),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("⚡ SAVINGS RATE GAUGE",
                   style={"color":MUTED,"fontFamily":"monospace","fontSize":"11px","marginBottom":"4px"}),
            dcc.Graph(id="ch-gauge", style={"height":"260px"}, config={"displayModeBar":False})
        ]), style=CS), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("🤖 ML SEGMENTS",
                   style={"color":MUTED,"fontFamily":"monospace","fontSize":"11px","marginBottom":"4px"}),
            dcc.Graph(id="ch-seg", style={"height":"260px"}, config={"displayModeBar":False})
        ]), style=CS), width=4),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("🏙 SAVINGS BY CITY",
                   style={"color":MUTED,"fontFamily":"monospace","fontSize":"11px","marginBottom":"4px"}),
            dcc.Graph(id="ch-city", style={"height":"240px"}, config={"displayModeBar":False})
        ]), style=CS), width=4),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("💼 SAVINGS BY OCCUPATION",
                   style={"color":MUTED,"fontFamily":"monospace","fontSize":"11px","marginBottom":"4px"}),
            dcc.Graph(id="ch-occ", style={"height":"240px"}, config={"displayModeBar":False})
        ]), style=CS), width=4),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("⚠️ ANOMALY DETECTION",
                   style={"color":MUTED,"fontFamily":"monospace","fontSize":"11px","marginBottom":"4px"}),
            dcc.Graph(id="ch-anom", style={"height":"240px"}, config={"displayModeBar":False})
        ]), style={**CS,"borderColor":RED}), width=4),
    ], className="mb-3"),

    dbc.Card(dbc.CardBody([
        html.P("🔬 USER SCATTER — Income vs Savings (hover for details)",
               style={"color":MUTED,"fontFamily":"monospace","fontSize":"11px","marginBottom":"4px"}),
        dcc.Graph(id="ch-scatter", style={"height":"260px"}, config={"displayModeBar":False})
    ]), style={**CS,"marginBottom":"14px"}),

    html.P("ML: K-Means Clustering · Linear Regression Forecast · Z-Score Anomaly Detection",
           style={"color":MUTED,"fontFamily":"monospace","fontSize":"10px","textAlign":"center","marginTop":"8px"}),
])

@app.callback(
    Output("kpi-row","children"),
    Output("ch-gm","figure"),
    Output("ch-donut","figure"),
    Output("ch-trend","figure"),
    Output("ch-gauge","figure"),
    Output("ch-seg","figure"),
    Output("ch-city","figure"),
    Output("ch-occ","figure"),
    Output("ch-anom","figure"),
    Output("ch-scatter","figure"),
    Input("dd-city","value"),
    Input("dd-occ","value"),
    Input("dd-seg","value"),
    Input("sl-month","value"),
)
def update(city, occ, seg, mrange):
    m0 = MONTHS[mrange[0]]
    m1 = MONTHS[mrange[1]]
    d  = df[(df["month"] >= m0) & (df["month"] <= m1)].copy()
    if city: d = d[d["city"].isin(city)]
    if occ:  d = d[d["occupation"].isin(occ)]
    if seg:  d = d[d["segment"].isin(seg)]

    print(f"[Filter] {len(d)} rows after filters | city={city} occ={occ} seg={seg} months={m0}→{m1}")

    empty = go.Figure()
    empty.update_layout(**base_layout())

    if d.empty:
        msg = kpi("No Data","—","Clear filters to see data", RED,"⚠️")
        return dbc.Row([dbc.Col(msg)]), empty,empty,empty,empty,empty,empty,empty,empty,empty

    # monthly aggregation — use string dates to avoid serialisation issues
    mo = (d.groupby("month")[["income","total_expense","savings","saving_ratio"]]
           .mean().reset_index().sort_values("month"))
    mo["month_dt"] = pd.to_datetime(mo["month"])
    xs = mo["month"].tolist()   # string x-axis: "2025-01", "2025-02" …

    # ── KPIs ──
    latest = mo.iloc[-1]
    prev   = mo.iloc[-2] if len(mo) > 1 else mo.iloc[-1]
    chg    = (latest["savings"] - prev["savings"]) / max(abs(prev["savings"]), 1) * 100
    kpis = dbc.Row([
        dbc.Col(kpi("Avg Income",     f"₹{int(mo['income'].mean()):,}",        "Monthly average",                     GREEN,  "💵"), width=3),
        dbc.Col(kpi("Avg Expense",    f"₹{int(mo['total_expense'].mean()):,}", "Monthly average",                     RED,    "🛒"), width=3),
        dbc.Col(kpi("Latest Savings", f"₹{int(latest['savings']):,}",
                    f"{'+' if chg>=0 else ''}{chg:.1f}% vs prev month",         GREEN if chg>=0 else RED,              "💰"), width=3),
        dbc.Col(kpi("Avg Saving Rate",f"{mo['saving_ratio'].mean()*100:.1f}%", "Target: 30%",                         AMBER,  "📊"), width=3),
    ])

    # ── Gross Margin ──
    fg = go.Figure()
    fg.add_bar(x=xs, y=mo["income"].tolist(),        name="Income",  marker_color=CYAN,  opacity=0.85)
    fg.add_bar(x=xs, y=mo["total_expense"].tolist(), name="Expense", marker_color=GREEN, opacity=0.75)
    sr_scaled = (mo["saving_ratio"] * mo["income"].max()).tolist()
    fg.add_scatter(x=xs, y=sr_scaled, name="Saving %", yaxis="y2",
                   line=dict(color=GOLD, width=2), mode="lines+markers", marker_size=6)
    fg.update_layout(**base_layout(
        barmode="group",
        yaxis=dict(tickprefix="₹", gridcolor=BORDER, color=WHITE),
        yaxis2=dict(overlaying="y", side="right", showgrid=False,
                    tickformat=".0%", range=[0, mo["income"].max()], color=GOLD),
        legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)", font_color=WHITE),
    ))

    # ── Expense Donut ──
    em = d[EXP].mean()
    fd = go.Figure(go.Pie(
        labels=[c.title() for c in EXP], values=em.tolist(), hole=0.55,
        marker=dict(colors=[CYAN,GREEN,AMBER,BLUE,PURPLE,RED,GOLD,MUTED],
                    line=dict(color=CARD, width=2)),
        textinfo="percent+label", textfont_size=9,
    ))
    fd.update_layout(**base_layout(
        showlegend=False,
        annotations=[dict(text=f"₹{int(em.sum()):,}", x=0.5, y=0.5,
                          font=dict(size=13,color=WHITE), showarrow=False)],
    ))

    # ── Savings Trend + Forecast ──
    Xt  = np.arange(len(mo)).reshape(-1,1)
    reg = LinearRegression().fit(Xt, mo["savings"].values)
    Xf  = np.arange(len(mo), len(mo)+3).reshape(-1,1)
    fsv = reg.predict(Xf).tolist()
    fdt = pd.date_range(mo["month_dt"].iloc[-1]+pd.DateOffset(months=1), periods=3, freq="MS")
    fxs = [d.strftime("%Y-%m") for d in fdt]
    z   = np.abs(stats.zscore(mo["savings"])) if len(mo) > 2 else np.zeros(len(mo))
    an  = mo[z > 1.5]
    ft  = go.Figure()
    ft.add_scatter(x=xs, y=mo["savings"].tolist(), name="Actual",
                   line=dict(color=CYAN, width=2), mode="lines+markers", marker_size=6)
    ft.add_scatter(x=fxs, y=fsv, name="Forecast",
                   line=dict(color=CYAN, width=2, dash="dash"),
                   mode="lines+markers", marker=dict(size=8, symbol="diamond", color=CYAN))
    if not an.empty:
        ft.add_scatter(x=an["month"].tolist(), y=an["savings"].tolist(),
                       name="⚠ Anomaly", mode="markers",
                       marker=dict(color=RED, size=12, symbol="triangle-up"))
    ft.update_layout(**base_layout(
        yaxis=dict(tickprefix="₹", gridcolor=BORDER, color=WHITE),
        legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)", font_color=WHITE),
    ))

    # ── Gauge ──
    rate = mo["saving_ratio"].mean() * 100
    fg2  = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(rate,1),
        number=dict(suffix="%", font=dict(color=WHITE, size=32)),
        delta=dict(reference=30, increasing=dict(color=GREEN), decreasing=dict(color=RED)),
        gauge=dict(
            axis=dict(range=[0,60], tickcolor=MUTED, tickfont=dict(color=MUTED,size=9)),
            bar=dict(color=CYAN, thickness=0.25),
            bgcolor=CARD, bordercolor=BORDER,
            steps=[dict(range=[0,15],  color="#2d1b1b"),
                   dict(range=[15,30], color="#2d2a1b"),
                   dict(range=[30,60], color="#1b2d1e")],
            threshold=dict(line=dict(color=AMBER,width=3), thickness=0.75, value=30),
        ),
        title=dict(text="Avg Saving Rate", font=dict(color=MUTED,size=11)),
    ))
    fg2.update_layout(**base_layout())

    # ── Segment Donut ──
    sc2 = d.drop_duplicates("user_id")["segment"].value_counts()
    fs  = go.Figure(go.Pie(
        labels=sc2.index.tolist(), values=sc2.values.tolist(), hole=0.5,
        marker=dict(colors=[GREEN,RED,AMBER], line=dict(color=CARD,width=2)),
        textinfo="label+percent", textfont_size=9,
    ))
    fs.update_layout(**base_layout(showlegend=False,
        annotations=[dict(text="Users", x=0.5, y=0.5,
                          font=dict(size=12,color=MUTED), showarrow=False)]))

    # ── City Bar ──
    cv = d.groupby("city")["savings"].mean().sort_values()
    fc = go.Figure(go.Bar(
        x=cv.values.tolist(), y=cv.index.tolist(), orientation="h",
        marker_color=[CYAN if v==cv.max() else BLUE for v in cv.values],
        text=[f"₹{int(v):,}" for v in cv.values],
        textposition="outside", textfont=dict(color=WHITE,size=9),
    ))
    fc.update_layout(**base_layout(
        xaxis=dict(tickprefix="₹", gridcolor=BORDER, color=WHITE),
        yaxis=dict(gridcolor=BORDER, color=WHITE),
    ))

    # ── Occupation Bar ──
    ov = d.groupby("occupation")["savings"].mean().sort_values()
    fo = go.Figure(go.Bar(
        x=ov.values.tolist(), y=ov.index.tolist(), orientation="h",
        marker_color=[GOLD if v==ov.max() else PURPLE for v in ov.values],
        text=[f"₹{int(v):,}" for v in ov.values],
        textposition="outside", textfont=dict(color=WHITE,size=9),
    ))
    fo.update_layout(**base_layout(
        xaxis=dict(tickprefix="₹", gridcolor=BORDER, color=WHITE),
        yaxis=dict(gridcolor=BORDER, color=WHITE),
    ))

    # ── Anomaly ──
    mo2 = d.groupby("month")["savings"].mean().reset_index().sort_values("month")
    z2  = np.abs(stats.zscore(mo2["savings"])) if len(mo2) > 2 else np.zeros(len(mo2))
    a2  = mo2[z2 > 1.5]; n2 = mo2[z2 <= 1.5]
    fa  = go.Figure()
    fa.add_scatter(x=n2["month"].tolist(), y=n2["savings"].tolist(),
                   name="Normal", mode="lines+markers",
                   line=dict(color=CYAN,width=1.5), marker_size=5)
    if not a2.empty:
        fa.add_scatter(x=a2["month"].tolist(), y=a2["savings"].tolist(),
                       name="⚠ Anomaly", mode="markers",
                       marker=dict(color=RED, size=12, symbol="triangle-up"))
    fa.update_layout(**base_layout(
        paper_bgcolor="#1a0e0e",
        yaxis=dict(tickprefix="₹", gridcolor=BORDER, color=WHITE),
        legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)", font_color=WHITE),
    ))

    # ── Scatter ──
    usc = d.groupby(["user_id","name","occupation","city","segment"])[
          ["income","savings","saving_ratio"]].mean().reset_index()
    fsc = px.scatter(usc, x="income", y="savings", color="segment", symbol="occupation",
        hover_data={"name":True,"city":True,"saving_ratio":":.1%"},
        color_discrete_map={"Savers":GREEN,"High Spenders":RED,"Balanced":AMBER},
        size="saving_ratio", size_max=14)
    fsc.update_layout(**base_layout(
        xaxis=dict(tickprefix="₹", gridcolor=BORDER, color=WHITE, title="Income"),
        yaxis=dict(tickprefix="₹", gridcolor=BORDER, color=WHITE, title="Savings"),
        legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)", font_color=WHITE),
    ))

    return kpis, fg, fd, ft, fg2, fs, fc, fo, fa, fsc


if __name__ == "__main__":
    print("\n" + "="*52)
    print("  FinSight Analytics — ready!")
    print("  Open browser: http://127.0.0.1:8050")
    print("="*52 + "\n")
    app.run(debug=False, host="127.0.0.1", port=8050)
