"""
report.py – Tekstiyhteenveto ja PDF-raportin generointi ReportLab-kirjastolla.
"""

import io
from datetime import date
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from model.scenarios import SCENARIO_LABELS, ScenarioResult, ScenarioParams


# ── Tyylit ───────────────────────────────────────────────────────────────────

def _h1() -> ParagraphStyle:
    return ParagraphStyle(
        "Header",
        parent=getSampleStyleSheet()["Heading1"],
        fontSize=18, textColor=colors.HexColor("#1B5E20"),
        spaceAfter=6, alignment=TA_CENTER,
    )


def _h2() -> ParagraphStyle:
    return ParagraphStyle(
        "SubHeader",
        parent=getSampleStyleSheet()["Heading2"],
        fontSize=13, textColor=colors.HexColor("#2E7D32"),
        spaceBefore=12, spaceAfter=4,
    )


def _body() -> ParagraphStyle:
    return ParagraphStyle(
        "Body",
        parent=getSampleStyleSheet()["Normal"],
        fontSize=10, leading=14, spaceAfter=6,
    )


# ── Yhteenvetoteksti ─────────────────────────────────────────────────────────

def generate_summary_text(
    scenario_results: dict[str, ScenarioResult],
    params: ScenarioParams,
    risk_summary: dict[str, Any] | None = None,
    data_notes: list[str] | None = None,
) -> str:
    """
    Generoi automaattinen tekstiyhteenveto suomeksi markkinaskenaarioiden
    ja riskianalyysin tuloksista.
    """
    # Vuosikeskihinnat P50 per skenaario
    def avg_p50(scenario: str) -> float:
        r = scenario_results.get(scenario)
        if r is None:
            return 0.0
        return float(r.annual_prices["p50"].mean())

    m_avg = avg_p50("matala")
    p_avg = avg_p50("perus")
    k_avg = avg_p50("korkea")

    # 2025 vs 2035 kehitys
    def price_year(scenario: str, year: int) -> float:
        r = scenario_results.get(scenario)
        if r is None:
            return 0.0
        sub = r.annual_prices[r.annual_prices["year"] == year]
        return float(sub["p50"].values[0]) if not sub.empty else 0.0

    p_2025 = price_year("perus", 2025)
    p_2035 = price_year("perus", 2035)

    # Kulutuskasvu
    total_growth = params.electrification_twh + params.ev_twh
    dc_final = params.datacenter_base_twh * ((1 + params.datacenter_growth_pct / 100) ** 10)
    dc_final = min(dc_final, 50.0)

    text = (
        f"Analyysi kattaa vuodet 2025–2035 kolmella skenaariolla. "
        f"Perusskenaariossa FI spot-hinta on keskimäärin {p_avg:.1f} €/MWh "
        f"(matala {m_avg:.1f} €/MWh, korkea {k_avg:.1f} €/MWh). "
        f"Vuodesta 2025 ({p_2025:.1f} €/MWh) vuoteen 2035 ({p_2035:.1f} €/MWh) "
        f"hinta {'nousee' if p_2035 > p_2025 else 'laskee'} perusskenaariossa "
        f"{abs(p_2035 - p_2025):.1f} €/MWh. "
        f"Sähköistyminen ja sähköautot lisäävät kulutusta {total_growth:.0f} TWh "
        f"vuoteen 2035 mennessä. "
        f"Datakeskusten kulutuksen ennustetaan kasvavan "
        f"{params.datacenter_base_twh:.1f} → {dc_final:.1f} TWh. "
    )

    if risk_summary:
        rc = risk_summary.get("risk_class", "")
        cvar = risk_summary.get("cvar_95", 0.0)
        vol = risk_summary.get("volatiliteetti", 0.0)
        text += (
            f"Riskitaso on {rc} (volatiliteetti {vol:.1f} €/MWh, "
            f"CVaR 95% = {cvar:.1f} €/MWh). "
            f"{risk_summary.get('suositus_teksti', '')} "
        )

    if data_notes:
        text += "Dataoletukset: " + "; ".join(data_notes[:3]) + "."

    return text


# ── PDF-rakentaja ────────────────────────────────────────────────────────────

def build_pdf_report(
    scenario_results: dict[str, ScenarioResult],
    params: ScenarioParams,
    n_simulations: int,
    r2: float = 0.0,
    risk_summary: dict[str, Any] | None = None,
    hedge_results: list | None = None,
    data_notes: list[str] | None = None,
) -> bytes:
    """
    Luo PDF-raportin markkinaanalyysin tuloksista.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )

    story = []

    # Otsikko
    story.append(Paragraph("Sähkönhintaskenaariot 2025–2035", _h1()))
    story.append(Paragraph(
        f"Suomen sähkömarkkinan analyysi | {date.today().strftime('%d.%m.%Y')}",
        ParagraphStyle("sub", parent=_body(), alignment=TA_CENTER, textColor=colors.grey),
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1B5E20"), spaceAfter=12))
    story.append(Spacer(1, 0.3 * cm))

    # Markkinaparametrit
    story.append(Paragraph("Markkinaparametrit", _h2()))

    nuclear_map = {
        "nykytaso": "Nykytaso ≈ 4.4 GW",
        "ol3_hanhikivi": "OL3 + Hanhikivi korvaava ≈ 5.4 GW",
        "uusi_voimala": "Uusi voimala ≈ 6 GW",
        "smr": "SMR ≈ 4.9 GW",
    }
    hydro_map = {"normaali": "Normaali", "kuiva": "Kuiva vuosi", "märkä": "Märkä vuosi"}

    param_rows = [
        ["Parametri", "Arvo"],
        ["Tuulivoiman lisäkapasiteetti", f"{params.wind_fi_gw:.1f} GW"],
        ["Aurinkoenergian kasvu", f"{params.solar_fi_gw:.1f} GW"],
        ["Ydinvoima FI", nuclear_map.get(params.nuclear_fi, params.nuclear_fi)],
        ["Vesivoima Pohjoismaat", hydro_map.get(params.hydro_nordic, params.hydro_nordic)],
        ["Kaasun hinta", f"{params.gas_price_mwh:.0f} €/MWh"],
        ["CO₂-hinta", f"{params.co2_price_t:.0f} €/t"],
        ["Sähköistyminen + LPs", f"{params.electrification_twh:.0f} TWh"],
        ["Sähköautot", f"{params.ev_twh:.1f} TWh"],
        ["Datakeskukset lähtötaso", f"{params.datacenter_base_twh:.1f} TWh"],
        ["Datakeskusten kasvuvauhti", f"{params.datacenter_growth_pct:.0f} %/v"],
        ["Monte Carlo -simulaatioita", str(n_simulations)],
        ["Tarkastelujakso", "2025–2035"],
    ]
    if r2 > 0:
        param_rows.append(["Regressiomallin R²", f"{r2:.3f}"])

    _add_table(story, param_rows,
               col_widths=[9 * cm, 7 * cm],
               header_color="#1B5E20",
               row_alt_color="#F1F8E9",
               grid_color="#C8E6C9")
    story.append(Spacer(1, 0.4 * cm))

    # Vuosikeskihinnat per skenaario
    story.append(Paragraph("Vuosikeskihinnat 2025–2035 (P50, €/MWh)", _h2()))
    price_rows = [["Vuosi", "Matala", "Perus", "Korkea"]]
    years = list(range(2025, 2036))
    for year in years:
        row = [str(year)]
        for sc in ["matala", "perus", "korkea"]:
            r = scenario_results.get(sc)
            if r is not None:
                sub = r.annual_prices[r.annual_prices["year"] == year]
                val = f"{sub['p50'].values[0]:.1f}" if not sub.empty else "–"
            else:
                val = "–"
            row.append(val)
        price_rows.append(row)
    _add_table(story, price_rows,
               col_widths=[3 * cm, 4 * cm, 4 * cm, 4 * cm],
               header_color="#1565C0",
               row_alt_color="#E3F2FD",
               grid_color="#90CAF9",
               align_right_from=1)
    story.append(Spacer(1, 0.4 * cm))

    # Riskianalyysi
    if risk_summary:
        story.append(Paragraph("Riskianalyysi", _h2()))
        story.append(Paragraph(risk_summary.get("suositus_teksti", ""), _body()))
        risk_rows = [
            ["Mittari", "Arvo"],
            ["Riskitaso", risk_summary.get("risk_class", "–").capitalize()],
            ["Volatiliteetti (€/MWh)", f"{risk_summary.get('volatiliteetti', 0):.1f}"],
            ["CVaR 95% (€/MWh)", f"{risk_summary.get('cvar_95', 0):.1f}"],
            ["Suositeltu strategia", risk_summary.get("paras_strategia", "–")],
        ]
        _add_table(story, risk_rows,
                   col_widths=[9 * cm, 7 * cm],
                   header_color="#B71C1C",
                   row_alt_color="#FFEBEE",
                   grid_color="#FFCDD2")
        story.append(Spacer(1, 0.4 * cm))

    # Hedgausstrategioiden vertailu
    if hedge_results:
        story.append(Paragraph("Hedgausstrategioiden vertailu", _h2()))
        hedge_rows = [["Strategia", "Hinta P50\n(€/MWh)", "Hinta P95\n(€/MWh)", "Riskivähennys\n(%)"]]
        for h in hedge_results:
            hedge_rows.append([
                h.strategy_name,
                f"{h.effective_price_p50:.1f}",
                f"{h.effective_price_p95:.1f}",
                f"{h.risk_reduction_ratio:.1f}%",
            ])
        _add_table(story, hedge_rows,
                   col_widths=[7 * cm, 3 * cm, 3 * cm, 3 * cm],
                   header_color="#37474F",
                   row_alt_color="#ECEFF1",
                   grid_color="#B0BEC5",
                   align_right_from=1)
        story.append(Spacer(1, 0.4 * cm))

    # Automaattinen yhteenveto
    story.append(Paragraph("Automaattinen yhteenveto", _h2()))
    summary = generate_summary_text(scenario_results, params, risk_summary, data_notes)
    story.append(Paragraph(summary, _body()))
    story.append(Spacer(1, 0.5 * cm))

    # Dataoletukset
    if data_notes:
        story.append(Paragraph("Käytetyt dataoletukset", _h2()))
        for note in data_notes:
            story.append(Paragraph(f"• {note}", _body()))
        story.append(Spacer(1, 0.3 * cm))

    # Vastuuvapauslauseke
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey, spaceAfter=6))
    story.append(Paragraph(
        "Tämä raportti on laadittu Monte Carlo -simulaation ja tilastollisten mallien pohjalta. "
        "Laskelmat eivät ole takuita tulevista hinnoista. "
        "Käytä tuloksia suuntaa-antavana apuvälineenä energiasuunnittelussa. "
        "Tiedot käsitellään vain paikallisesti — data ei lähde koneeltasi.",
        ParagraphStyle("disc", parent=_body(), fontSize=8, textColor=colors.grey),
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def _add_table(
    story: list,
    rows: list,
    col_widths: list,
    header_color: str,
    row_alt_color: str,
    grid_color: str,
    align_right_from: int = None,
) -> None:
    """Apufunktio tyylitellyn taulukon lisäämiseen."""
    t = Table(rows, colWidths=col_widths)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(row_alt_color)]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor(grid_color)),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]
    if align_right_from is not None:
        style.append(("ALIGN", (align_right_from, 0), (-1, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    story.append(t)
