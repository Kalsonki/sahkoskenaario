"""
charts.py – Plotly-kuvaajat sähköhintaskenaarioille, markkinadynamiikalle ja riskianalyysille.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from model.scenarios import (
    RegressionResult, ScenarioResult,
    SCENARIO_LABELS, SCENARIO_COLORS, SCENARIO_NAMES,
    START_YEAR, END_YEAR,
)

MONTH_LABELS_FI = {
    1: "Tam", 2: "Hel", 3: "Maa", 4: "Huh", 5: "Tou", 6: "Kes",
    7: "Hei", 8: "Elo", 9: "Syy", 10: "Lok", 11: "Mar", 12: "Jou",
}
MONTH_LABELS_FULL = {
    1: "Tammikuu", 2: "Helmikuu", 3: "Maaliskuu", 4: "Huhtikuu",
    5: "Toukokuu",  6: "Kesäkuu",  7: "Heinäkuu",  8: "Elokuu",
    9: "Syyskuu",  10: "Lokakuu", 11: "Marraskuu", 12: "Joulukuu",
}

_LAYOUT_BASE = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=60, r=20, t=80, b=60),
)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Muuntaa hex-värikoodin (#RRGGBB) rgba-muotoon Plotlyn fillcolor-parametria varten."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _date_label(year: int, month: int) -> str:
    return f"{year}-{month:02d}"


# ══════════════════════════════════════════════════════════════════════════════
# DATA-ANALYYSI -KUVAAJAT (Excel-fundamenttidata)
# ══════════════════════════════════════════════════════════════════════════════

def fundamental_time_series(fundamental_df: pd.DataFrame) -> go.Figure:
    """Aikasarjakaavio tunnistetusta fundamenttidatasta."""
    numeric_cols = [c for c in fundamental_df.columns if c != "date"
                    and pd.api.types.is_numeric_dtype(fundamental_df[c])]
    if not numeric_cols or "date" not in fundamental_df.columns:
        fig = go.Figure()
        fig.update_layout(title="Ei piirrettävää dataa", **_LAYOUT_BASE)
        return fig

    priority = ["price_fi", "gas_price", "co2_price", "consumption",
                "wind_capacity", "hydro_production", "nuclear_production"]
    ordered = [c for c in priority if c in numeric_cols]
    ordered += [c for c in numeric_cols if c not in ordered]

    col_labels = {
        "price_fi":           "Sähkön spot-hinta (€/MWh)",
        "consumption":        "Kulutus (MWh)",
        "wind_capacity":      "Tuulivoima (MW)",
        "hydro_production":   "Vesivoima (MWh)",
        "nuclear_production": "Ydinvoima (MWh)",
        "gas_price":          "Kaasun hinta",
        "co2_price":          "CO₂-päästöoikeus",
    }

    fig = go.Figure()
    colors_cycle = ["#1565C0", "#2E7D32", "#B71C1C", "#F57C00", "#6A1B9A", "#00796B", "#AD1457"]

    for i, col in enumerate(ordered[:6]):
        label = col_labels.get(col, col)
        fig.add_trace(go.Scatter(
            x=fundamental_df["date"],
            y=fundamental_df[col],
            name=label,
            line=dict(color=colors_cycle[i % len(colors_cycle)], width=1.8),
            hovertemplate=f"%{{x|%Y-%m}}: %{{y:.2f}}<extra>{label}</extra>",
        ))

    fig.update_layout(
        title="Historiallinen fundamenttidata",
        xaxis_title="Aika",
        yaxis_title="Arvo",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=450,
        xaxis=dict(gridcolor="#E0E0E0"),
        yaxis=dict(gridcolor="#E0E0E0"),
        **_LAYOUT_BASE,
    )
    return fig


def correlation_heatmap(fundamental_df: pd.DataFrame) -> go.Figure:
    """Korrelaatiomatriisi: mitkä muuttujat selittävät hintaa eniten."""
    numeric_cols = [c for c in fundamental_df.columns if c != "date"
                    and pd.api.types.is_numeric_dtype(fundamental_df[c])]
    if len(numeric_cols) < 2:
        fig = go.Figure()
        fig.update_layout(title="Liian vähän sarakkeita korrelaatiomatriisiin", **_LAYOUT_BASE)
        return fig

    col_labels = {
        "price_fi":           "Spot-hinta",
        "consumption":        "Kulutus",
        "wind_capacity":      "Tuulivoima",
        "hydro_production":   "Vesivoima",
        "nuclear_production": "Ydinvoima",
        "gas_price":          "Kaasun hinta",
        "co2_price":          "CO₂",
    }

    corr = fundamental_df[numeric_cols].corr()
    labels = [col_labels.get(c, c) for c in corr.columns]

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Korrelaatio"),
    ))
    fig.update_layout(
        title="Korrelaatiomatriisi – muuttujien väliset yhteydet",
        height=420,
        margin=dict(l=120, r=20, t=80, b=100),
    )
    return fig


def regression_coef_chart(regression: RegressionResult) -> go.Figure:
    """Pylväsdiagrammi regressiomallin kertoimista."""
    col_labels = {
        "wind_capacity":      "Tuulivoima",
        "hydro_production":   "Vesivoima",
        "nuclear_production": "Ydinvoima",
        "gas_price":          "Kaasun hinta",
        "co2_price":          "CO₂",
        "month_sin":          "Kausitrendi (sin)",
        "month_cos":          "Kausitrendi (cos)",
    }

    if not regression.coef:
        fig = go.Figure()
        fig.update_layout(title="Ei regressiokertoimia saatavilla", **_LAYOUT_BASE)
        return fig

    features = list(regression.coef.keys())
    coefs = list(regression.coef.values())
    labels = [col_labels.get(f, f) for f in features]
    colors = ["#2E7D32" if c >= 0 else "#B71C1C" for c in coefs]

    fig = go.Figure(go.Bar(
        x=labels,
        y=coefs,
        marker_color=colors,
        hovertemplate="%{x}: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Regressiomallin kertoimet (R² = {regression.r2:.3f})",
        xaxis_title="Piirre",
        yaxis_title="Kerroin (standardisoitu)",
        height=380,
        yaxis=dict(gridcolor="#E0E0E0", zeroline=True, zerolinecolor="#555"),
        **_LAYOUT_BASE,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# HINTASKENAARIO-KUVAAJAT
# ══════════════════════════════════════════════════════════════════════════════

def price_scenario_chart(
    scenario_results: dict[str, ScenarioResult],
    historical_df: pd.DataFrame,
    visible_scenarios: Optional[list[str]] = None,
) -> go.Figure:
    """
    Viivakaavio: kolme skenaariopolkua 2025–2035 + historiallinen vertailu.
    Varjostettu alue = P10–P90 epävarmuusvyöhyke.
    """
    if visible_scenarios is None:
        visible_scenarios = SCENARIO_NAMES

    fig = go.Figure()

    if not historical_df.empty:
        hist = historical_df.sort_values(["year", "month"])
        hist["label"] = hist.apply(lambda r: _date_label(int(r.year), int(r.month)), axis=1)
        fig.add_trace(go.Scatter(
            x=hist["label"],
            y=hist["price_eur_mwh"],
            name="Historiallinen (2015–2024)",
            line=dict(color="#757575", width=1.5, dash="dot"),
            hovertemplate="%{x}: %{y:.1f} €/MWh<extra>Historiallinen</extra>",
        ))

    for scenario in SCENARIO_NAMES:
        if scenario not in visible_scenarios:
            continue
        result = scenario_results[scenario]
        df = result.monthly_prices.sort_values(["year", "month"])
        df["label"] = df.apply(lambda r: _date_label(int(r.year), int(r.month)), axis=1)
        color = result.color

        fig.add_trace(go.Scatter(
            x=pd.concat([df["label"], df["label"].iloc[::-1]]),
            y=pd.concat([df["p90"], df["p10"].iloc[::-1]]),
            fill="toself",
            fillcolor=_hex_to_rgba(color, 0.13),
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
            name=f"{result.label} (alue)",
        ))

        fig.add_trace(go.Scatter(
            x=df["label"],
            y=df["p50"],
            name=result.label,
            line=dict(color=color, width=2),
            hovertemplate="%{x}: %{y:.1f} €/MWh<extra>" + result.label + "</extra>",
        ))

    fig.update_layout(
        title="Sähköhinnan skenaariopolut 2025–2035",
        xaxis_title="Kuukausi",
        yaxis_title="Hinta (€/MWh)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=480,
        xaxis=dict(
            tickangle=-45,
            tickmode="array",
            tickvals=[f"{y}-01" for y in range(START_YEAR, END_YEAR + 1)],
            ticktext=[str(y) for y in range(START_YEAR, END_YEAR + 1)],
            gridcolor="#E0E0E0",
        ),
        yaxis=dict(gridcolor="#E0E0E0"),
        **_LAYOUT_BASE,
    )
    return fig


def price_percentile_paths(
    scenario_results: dict[str, ScenarioResult],
    scenario: str = "perus",
) -> go.Figure:
    """
    Riskianalyysikaavio: P5, P25, P50, P75, P95 persentiilipolut yhdelle skenaariolle.
    P95-alue korostetaan punaisella.
    """
    result = scenario_results.get(scenario)
    if result is None:
        fig = go.Figure()
        fig.update_layout(title="Skenaario ei saatavilla", **_LAYOUT_BASE)
        return fig

    df = result.monthly_prices.sort_values(["year", "month"])
    df["label"] = df.apply(lambda r: _date_label(int(r.year), int(r.month)), axis=1)
    color = result.color

    fig = go.Figure()

    # P5–P95 äärivyöhyke (punainen korostus)
    if "p5" in df.columns and "p95" in df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([df["label"], df["label"].iloc[::-1]]),
            y=pd.concat([df["p95"], df["p5"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(183,28,28,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=True,
            name="P5–P95 (äärialue)",
            hoverinfo="skip",
        ))

    # P25–P75 vyöhyke
    if "p25" in df.columns and "p75" in df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([df["label"], df["label"].iloc[::-1]]),
            y=pd.concat([df["p75"], df["p25"].iloc[::-1]]),
            fill="toself",
            fillcolor=_hex_to_rgba(color, 0.18),
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=True,
            name="P25–P75",
            hoverinfo="skip",
        ))

    # Persentiiliviivat
    for pct, dash, width, label_suffix in [
        ("p95", "dot",   1.2, "P95 (pahimmat 5%)"),
        ("p75", "dash",  1.0, "P75"),
        ("p50", "solid", 2.0, "P50 (mediaani)"),
        ("p25", "dash",  1.0, "P25"),
        ("p5",  "dot",   1.2, "P5 (parhaat 5%)"),
    ]:
        if pct not in df.columns:
            continue
        pct_color = "#B71C1C" if pct == "p95" else "#2E7D32" if pct == "p5" else color
        fig.add_trace(go.Scatter(
            x=df["label"],
            y=df[pct],
            name=label_suffix,
            line=dict(color=pct_color, width=width, dash=dash),
            hovertemplate=f"%{{x}}: %{{y:.1f}} €/MWh<extra>{label_suffix}</extra>",
        ))

    sc_label = SCENARIO_LABELS.get(scenario, scenario)
    fig.update_layout(
        title=f"Hintariskin persentiilipolut – {sc_label}",
        xaxis_title="Kuukausi",
        yaxis_title="Hinta (€/MWh)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=480,
        xaxis=dict(
            tickangle=-45,
            tickmode="array",
            tickvals=[f"{y}-01" for y in range(START_YEAR, END_YEAR + 1)],
            ticktext=[str(y) for y in range(START_YEAR, END_YEAR + 1)],
            gridcolor="#E0E0E0",
        ),
        yaxis=dict(gridcolor="#E0E0E0"),
        **_LAYOUT_BASE,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MARKKINADYNAMIIKKA-KUVAAJAT
# ══════════════════════════════════════════════════════════════════════════════

def tornado_chart(sensitivity_df: pd.DataFrame) -> go.Figure:
    """
    Tornado-kaavio: muuttujien hinnavaikutus €/MWh perusskenaariossa vuonna 2030.

    sensitivity_df: DataFrame muuttuja, vaikutus_matala, vaikutus_korkea, arvo_matala, arvo_korkea
    """
    fig = go.Figure()

    labels = sensitivity_df["muuttuja"].tolist()

    # Negatiiviset vaikutukset (hintaa laskevat)
    neg_vals = [min(r["vaikutus_matala"], r["vaikutus_korkea"]) for _, r in sensitivity_df.iterrows()]
    pos_vals = [max(r["vaikutus_matala"], r["vaikutus_korkea"]) for _, r in sensitivity_df.iterrows()]

    fig.add_trace(go.Bar(
        y=labels,
        x=neg_vals,
        orientation="h",
        name="Hintaa laskeva",
        marker_color="#2E7D32",
        hovertemplate="%{y}: %{x:.1f} €/MWh<extra>Laskeva</extra>",
    ))

    fig.add_trace(go.Bar(
        y=labels,
        x=pos_vals,
        orientation="h",
        name="Hintaa nostava",
        marker_color="#B71C1C",
        hovertemplate="%{y}: %{x:.1f} €/MWh<extra>Nostava</extra>",
    ))

    fig.update_layout(
        title="Markkinamuuttujien hinnavaikutus 2030 – perusskenaario",
        xaxis_title="Hintavaikutus (€/MWh)",
        barmode="overlay",
        height=420,
        xaxis=dict(gridcolor="#E0E0E0", zeroline=True, zerolinecolor="#333", zerolinewidth=1.5),
        yaxis=dict(gridcolor="#E0E0E0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **_LAYOUT_BASE,
    )
    return fig


def datacenter_growth_chart(dc_df: pd.DataFrame) -> go.Figure:
    """Datakeskusten TWh-kasvukäyrä vuosittain."""
    colors = ["#B71C1C" if row["capped"] else "#1565C0" for _, row in dc_df.iterrows()]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dc_df["year"],
        y=dc_df["twh"],
        marker_color=colors,
        hovertemplate="%{x}: %{y:.2f} TWh<extra>Datakeskukset</extra>",
        name="Datakeskukset TWh",
    ))

    if dc_df["capped"].any():
        cap_year = dc_df[dc_df["capped"]]["year"].min()
        fig.add_vline(
            x=cap_year - 0.5,
            line_dash="dash",
            line_color="#B71C1C",
            annotation_text="50 TWh katto",
            annotation_position="top right",
        )

    fig.update_layout(
        title="Datakeskusten sähkönkulutus Suomessa 2025–2035",
        xaxis_title="Vuosi",
        yaxis_title="Kulutus (TWh/vuosi)",
        height=360,
        yaxis=dict(gridcolor="#E0E0E0"),
        xaxis=dict(dtick=1, gridcolor="#E0E0E0"),
        showlegend=False,
        **_LAYOUT_BASE,
    )
    return fig


def interconnect_hintaero_chart(
    scenario_results: dict[str, ScenarioResult],
    max_hintaero: float,
) -> go.Figure:
    """
    Hintaero FI vs arvioitu Pohjoismaat-referenssi kuukausittain
    sekä siirtoyhteyksien sallima maksimaalinen hintaero.
    """
    fig = go.Figure()
    color_map = {"perus": "#1565C0", "matala": "#2E7D32", "korkea": "#B71C1C"}

    for scenario in ["matala", "perus", "korkea"]:
        result = scenario_results.get(scenario)
        if result is None:
            continue
        df = result.monthly_prices.sort_values(["year", "month"])
        df["label"] = df.apply(lambda r: _date_label(int(r.year), int(r.month)), axis=1)
        # Arvioitu Nord Pool referenssihinta = FI hinta / (1 + hydro/muut tekijät)
        # Yksinkertaistus: referenssi ≈ FI P50 * 0.92
        df["nordpool_ref"] = df["p50"] * 0.92
        df["hintaero"] = df["p50"] - df["nordpool_ref"]
        color = color_map.get(scenario, "#555")
        fig.add_trace(go.Scatter(
            x=df["label"],
            y=df["hintaero"],
            name=SCENARIO_LABELS.get(scenario, scenario),
            line=dict(color=color, width=1.5),
            hovertemplate="%{x}: %{y:.1f} €/MWh<extra>" + SCENARIO_LABELS.get(scenario, scenario) + "</extra>",
        ))

    # Siirtoyhteyksien sallima maksimihintaero
    labels_all = [f"{y}-01" for y in range(START_YEAR, END_YEAR + 1)]
    fig.add_hline(
        y=max_hintaero,
        line_dash="dash",
        line_color="#E65100",
        annotation_text=f"Max sallittu hintaero: {max_hintaero:.0f} €/MWh",
        annotation_position="top right",
    )

    fig.update_layout(
        title="FI–Pohjoismaat hintaero ja siirtoyhteyksien kapasiteettiraja",
        xaxis_title="Kuukausi",
        yaxis_title="Hintaero (€/MWh)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=400,
        xaxis=dict(
            tickangle=-45,
            tickmode="array",
            tickvals=[f"{y}-01" for y in range(START_YEAR, END_YEAR + 1)],
            ticktext=[str(y) for y in range(START_YEAR, END_YEAR + 1)],
            gridcolor="#E0E0E0",
        ),
        yaxis=dict(gridcolor="#E0E0E0"),
        **_LAYOUT_BASE,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# RISKIANALYYSI-KUVAAJAT
# ══════════════════════════════════════════════════════════════════════════════

def hedge_comparison_chart(hedge_results: list, vol_mwh: float) -> go.Figure:
    """
    Pylväsdiagrammi: hedgausstrategioiden vertailu P50 vs P95 kustannuksin.
    """
    from model.risk import HedgeResult

    labels = [h.strategy_name for h in hedge_results]
    cost_p50 = [h.annual_cost_p50 / 1000 for h in hedge_results]
    cost_p95 = [h.annual_cost_p95 / 1000 for h in hedge_results]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="P50-skenaario (todennäköisin)",
        x=labels,
        y=cost_p50,
        marker_color="#1565C0",
        hovertemplate="%{x}: %{y:.1f} k€<extra>P50</extra>",
    ))
    fig.add_trace(go.Bar(
        name="P95-skenaario (pahin 5%)",
        x=labels,
        y=cost_p95,
        marker_color="#B71C1C",
        hovertemplate="%{x}: %{y:.1f} k€<extra>P95</extra>",
    ))

    fig.update_layout(
        title=f"Hedgausstrategioiden kustannusvertailu (volyymi {vol_mwh:,.0f} MWh/v)",
        xaxis_title="Strategia",
        yaxis_title="Vuosikustannus (k€/v)",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=420,
        yaxis=dict(gridcolor="#E0E0E0"),
        **_LAYOUT_BASE,
    )
    return fig


def efficient_frontier_chart(frontier_df: pd.DataFrame) -> go.Figure:
    """
    Efficient frontier -kaavio: hedge-kustannus vs CVaR 95%.
    x = lisäkustannus vs spot P50 (€/v), y = CVaR 95% (€/v)
    """
    type_colors = {
        "Kiinteä":  "#1565C0",
        "Collar":   "#2E7D32",
        "Termiini": "#F57C00",
    }

    fig = go.Figure()

    for tyyppi, grp in frontier_df.groupby("tyyppi"):
        color = type_colors.get(tyyppi, "#757575")
        fig.add_trace(go.Scatter(
            x=grp["hedge_cost"] / 1000,
            y=grp["cvar_95"] / 1000,
            mode="lines+markers",
            name=tyyppi,
            marker=dict(color=color, size=6),
            line=dict(color=color, width=1.5),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Lisäkustannus: %{x:.0f} k€/v<br>"
                "CVaR 95%: %{y:.0f} k€/v<extra></extra>"
            ),
            text=grp["strategia"],
        ))

    # Merkitse spot-piste (0, spot_cvar)
    spot_row = frontier_df[frontier_df["strategia"] == "Kiinteä 0%"]
    if not spot_row.empty:
        fig.add_trace(go.Scatter(
            x=[0],
            y=[float(spot_row["cvar_95"].iloc[0]) / 1000],
            mode="markers",
            marker=dict(color="#B71C1C", size=12, symbol="diamond"),
            name="Täysi spot",
            hovertemplate="Täysi spot<br>CVaR 95%: %{y:.0f} k€/v<extra></extra>",
        ))

    fig.update_layout(
        title="Efficient frontier – suojauksen kustannus vs riski (CVaR 95%)",
        xaxis_title="Lisäkustannus vs spot P50 (k€/vuosi)",
        yaxis_title="CVaR 95% (k€/vuosi)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=450,
        xaxis=dict(gridcolor="#E0E0E0"),
        yaxis=dict(gridcolor="#E0E0E0"),
        **_LAYOUT_BASE,
    )
    return fig


def stress_test_chart(stress_tests: list) -> go.Figure:
    """Pylväsdiagrammi: stressitestien hintapiikki vs baseline."""
    from model.risk import StressTest

    names = [t.name for t in stress_tests]
    baseline = [t.baseline_price for t in stress_tests]
    spike = [t.price_spike for t in stress_tests]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Normaali hinta (perus P50)",
        x=names,
        y=baseline,
        marker_color="#1565C0",
        hovertemplate="%{x}: %{y:.1f} €/MWh<extra>Normaali</extra>",
    ))
    fig.add_trace(go.Bar(
        name="Stressiskenaario",
        x=names,
        y=[s - b for s, b in zip(spike, baseline)],
        base=baseline,
        marker_color="#B71C1C",
        hovertemplate="%{x}: %{y:.1f} €/MWh lisää<extra>Stressi</extra>",
    ))

    fig.update_layout(
        title="Stressitestit – hintapiikki vs perusskenaario",
        xaxis_title="Stressiskenaario",
        yaxis_title="Hinta (€/MWh)",
        barmode="stack",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=400,
        yaxis=dict(gridcolor="#E0E0E0"),
        **_LAYOUT_BASE,
    )
    return fig


def hedge_annual_cost_chart(
    hedge_df: pd.DataFrame,
    strategy_label: str,
    color: str = "#1565C0",
) -> go.Figure:
    """
    Viivakaavio: valitun hedgauksen efektiivinen hinta vs spot P50 vuosittain.
    """
    fig = go.Figure()

    # Spot P50
    fig.add_trace(go.Scatter(
        x=hedge_df["year"],
        y=hedge_df["spot_p50"],
        name="Spot P50",
        line=dict(color="#757575", width=1.5, dash="dot"),
        hovertemplate="%{x}: %{y:.1f} €/MWh<extra>Spot P50</extra>",
    ))

    # Hedgattu P10–P90 nauha
    fig.add_trace(go.Scatter(
        x=pd.concat([hedge_df["year"], hedge_df["year"].iloc[::-1]]),
        y=pd.concat([hedge_df["eff_price_p90"], hedge_df["eff_price_p10"].iloc[::-1]]),
        fill="toself",
        fillcolor=_hex_to_rgba(color, 0.15),
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Hedgattu P50
    fig.add_trace(go.Scatter(
        x=hedge_df["year"],
        y=hedge_df["eff_price_p50"],
        name=strategy_label,
        line=dict(color=color, width=2.5),
        hovertemplate="%{x}: %{y:.1f} €/MWh<extra>" + strategy_label + "</extra>",
    ))

    fig.update_layout(
        title=f"Efektiivinen hinta – {strategy_label} vs spot",
        xaxis_title="Vuosi",
        yaxis_title="Hinta (€/MWh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=380,
        xaxis=dict(dtick=1, gridcolor="#E0E0E0"),
        yaxis=dict(gridcolor="#E0E0E0"),
        **_LAYOUT_BASE,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# KUUKAUSIANALYYSI-KUVAAJAT
# ══════════════════════════════════════════════════════════════════════════════

def monthly_heatmap(scenario_results: dict[str, ScenarioResult], scenario: str = "perus") -> go.Figure:
    """Heatmap: vuosi × kuukausi, arvo = P50-hinta €/MWh."""
    result = scenario_results.get(scenario)
    if result is None:
        fig = go.Figure()
        fig.update_layout(title="Skenaario ei saatavilla", **_LAYOUT_BASE)
        return fig

    df = result.monthly_prices[["year", "month", "p50"]].copy()
    pivot = df.pivot(index="year", columns="month", values="p50")

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[MONTH_LABELS_FI[m] for m in pivot.columns],
        y=pivot.index.astype(str).tolist(),
        colorscale="RdYlGn_r",
        hovertemplate="Vuosi: %{y}, Kuukausi: %{x}<br>Hinta: %{z:.1f} €/MWh<extra></extra>",
        colorbar=dict(title="€/MWh"),
    ))
    fig.update_layout(
        title=f"Kuukausittainen hintakartta – {SCENARIO_LABELS.get(scenario, scenario)}",
        xaxis_title="Kuukausi",
        yaxis_title="Vuosi",
        height=400,
        margin=dict(l=60, r=20, t=80, b=60),
    )
    return fig


def monthly_avg_bar(scenario_results: dict[str, ScenarioResult], scenario: str = "perus") -> go.Figure:
    """Pylväsdiagrammi: kuukausikeskihinnat (kaikki vuodet) valitulle skenaariolle."""
    result = scenario_results.get(scenario)
    if result is None:
        fig = go.Figure()
        fig.update_layout(title="Skenaario ei saatavilla", **_LAYOUT_BASE)
        return fig

    avg = result.monthly_prices.groupby("month")["p50"].mean().reset_index()
    avg["label"] = avg["month"].map(MONTH_LABELS_FI)

    bar_colors = [
        "#B71C1C" if m in [1, 2, 12] else
        "#2E7D32" if m in [6, 7, 8] else "#1565C0"
        for m in avg["month"]
    ]

    fig = go.Figure(go.Bar(
        x=avg["label"],
        y=avg["p50"],
        marker_color=bar_colors,
        hovertemplate="%{x}: %{y:.1f} €/MWh<extra></extra>",
    ))
    fig.update_layout(
        title=f"Kuukausikeskiarvot – {SCENARIO_LABELS.get(scenario, scenario)}",
        xaxis_title="Kuukausi",
        yaxis_title="Hinta (€/MWh)",
        height=360,
        yaxis=dict(gridcolor="#E0E0E0"),
        showlegend=False,
        **_LAYOUT_BASE,
    )
    return fig
