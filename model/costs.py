"""
costs.py – Yrityksen sähkökustannusten laskenta skenaarioiden pohjalta.
"""

import numpy as np
import pandas as pd
from typing import Optional

from model.scenarios import ScenarioResult, SCENARIO_NAMES


def build_consumption_profile(
    annual_mwh: float,
    distribution: str,
    custom_weights: Optional[dict[int, float]] = None,
) -> dict[int, float]:
    """
    Rakentaa kuukausittaisen kulutusprofiilin (MWh per kuukausi).

    distribution: 'tasainen' | 'talvi' | 'kesä' | 'mukautettu'
    custom_weights: {kuukausi: %-osuus} jos distribution=='mukautettu'
    """
    base = annual_mwh / 12.0

    if distribution == "tasainen":
        return {m: base for m in range(1, 13)}

    if distribution == "talvi":
        winter = {11, 12, 1, 2}
        weights = {m: (1.5 if m in winter else 1.0) for m in range(1, 13)}
        total = sum(weights.values())
        return {m: annual_mwh * weights[m] / total for m in range(1, 13)}

    if distribution == "kesä":
        summer = {6, 7, 8}
        weights = {m: (1.5 if m in summer else 1.0) for m in range(1, 13)}
        total = sum(weights.values())
        return {m: annual_mwh * weights[m] / total for m in range(1, 13)}

    if distribution == "mukautettu" and custom_weights:
        total_pct = sum(custom_weights.values()) or 1.0
        return {m: annual_mwh * (custom_weights.get(m, 0) / total_pct) for m in range(1, 13)}

    return {m: base for m in range(1, 13)}


def apply_contract_price(
    spot_price: float,
    contract_type: str,
    fixed_share: float = 0.0,
    fixed_price: float = 0.0,
) -> float:
    """
    Laskee efektiivisen hinnan sopimustyypistä riippuen.

    contract_type: 'spot' | 'osittain_kiintea' | 'kiintea'
    fixed_share: 0–1 kiinteän osuus (osittain kiinteässä)
    fixed_price: kiinteä hintakomponentti €/MWh
    """
    if contract_type == "kiintea":
        return fixed_price
    if contract_type == "osittain_kiintea":
        return (1.0 - fixed_share) * spot_price + fixed_share * fixed_price
    return spot_price  # spot


def calculate_costs(
    scenario_results: dict[str, ScenarioResult],
    annual_mwh: float,
    distribution: str,
    contract_type: str,
    fixed_share: float = 0.0,
    fixed_price: float = 0.0,
    custom_weights: Optional[dict[int, float]] = None,
) -> pd.DataFrame:
    """
    Laskee kuukausittaiset kustannukset kaikille skenaarioille.

    Palauttaa DataFrame:n sarakkeilla:
    scenario, year, month, consumption_mwh,
    price_p10, price_p50, price_p90,
    eff_price_p10, eff_price_p50, eff_price_p90,
    cost_p10, cost_p50, cost_p90
    """
    profile = build_consumption_profile(annual_mwh, distribution, custom_weights)
    records = []

    for scenario_name, result in scenario_results.items():
        for _, row in result.monthly_prices.iterrows():
            month = int(row["month"])
            consumption = profile.get(month, annual_mwh / 12)

            eff_p10 = apply_contract_price(row["p10"], contract_type, fixed_share, fixed_price)
            eff_p50 = apply_contract_price(row["p50"], contract_type, fixed_share, fixed_price)
            eff_p90 = apply_contract_price(row["p90"], contract_type, fixed_share, fixed_price)

            records.append({
                "scenario": scenario_name,
                "year": int(row["year"]),
                "month": month,
                "consumption_mwh": consumption,
                "price_p10": row["p10"],
                "price_p50": row["p50"],
                "price_p90": row["p90"],
                "eff_price_p10": eff_p10,
                "eff_price_p50": eff_p50,
                "eff_price_p90": eff_p90,
                "cost_p10": consumption * eff_p10,
                "cost_p50": consumption * eff_p50,
                "cost_p90": consumption * eff_p90,
            })

    return pd.DataFrame(records)


def annual_costs(cost_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregoi kuukausittaiset kustannukset vuositasolle.
    Palauttaa: scenario, year, cost_p10, cost_p50, cost_p90
    """
    return (
        cost_df.groupby(["scenario", "year"])[["cost_p10", "cost_p50", "cost_p90"]]
        .sum()
        .reset_index()
    )


def cumulative_costs(annual_df: pd.DataFrame) -> pd.DataFrame:
    """Laskee kumulatiiviset kustannukset vuosittain per skenaario."""
    frames = []
    for scenario in SCENARIO_NAMES:
        sub = annual_df[annual_df["scenario"] == scenario].sort_values("year").copy()
        sub["cum_p10"] = sub["cost_p10"].cumsum()
        sub["cum_p50"] = sub["cost_p50"].cumsum()
        sub["cum_p90"] = sub["cost_p90"].cumsum()
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def risk_exposure(annual_df: pd.DataFrame) -> pd.DataFrame:
    """Laskee riskialttiuden: Korkea P50 miinus Matala P50 per vuosi."""
    korkea = annual_df[annual_df["scenario"] == "korkea"][["year", "cost_p50"]].rename(
        columns={"cost_p50": "korkea"}
    )
    matala = annual_df[annual_df["scenario"] == "matala"][["year", "cost_p50"]].rename(
        columns={"cost_p50": "matala"}
    )
    merged = korkea.merge(matala, on="year")
    merged["riski_eur"] = merged["korkea"] - merged["matala"]
    return merged


def optimization_savings(
    scenario_results: dict[str, ScenarioResult],
    annual_mwh: float,
    shift_fraction: float = 0.10,
    scenario_name: str = "perus",
) -> dict[str, float]:
    """
    Laskee säästöpotentiaalin siirtämällä talvikulutusta kesälle.

    shift_fraction: osuus talvesta siirretään (oletus 10%)
    Palauttaa: siirretty_mwh, säästö_eur_vuosi, talvi_hinta, kesä_hinta
    """
    result = scenario_results.get(scenario_name)
    if result is None:
        return {"siirretty_mwh": 0.0, "säästö_eur_vuosi": 0.0, "talvi_hinta": 0.0, "kesä_hinta": 0.0}

    monthly = result.monthly_prices.copy()

    # Käytetään vuoden 2027 mediaanihintoja
    ref_year = 2027
    ref_rows = monthly[monthly["year"] == ref_year]
    if ref_rows.empty:
        ref_rows = monthly

    ref = ref_rows.set_index("month")["p50"]

    winter_months = [12, 1, 2]
    summer_months = [6, 7, 8]

    talvi_hinta = float(ref[[m for m in winter_months if m in ref.index]].mean()) if any(m in ref.index for m in winter_months) else 70.0
    kesä_hinta  = float(ref[[m for m in summer_months if m in ref.index]].mean()) if any(m in ref.index for m in summer_months) else 50.0

    siirretty_mwh = annual_mwh * shift_fraction * (3 / 12)
    hintaero = talvi_hinta - kesä_hinta
    säästö = siirretty_mwh * hintaero

    return {
        "siirretty_mwh":    round(siirretty_mwh, 1),
        "säästö_eur_vuosi": round(max(säästö, 0.0), 0),
        "talvi_hinta":      round(talvi_hinta, 1),
        "kesä_hinta":       round(kesä_hinta, 1),
    }
