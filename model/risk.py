"""
risk.py – Riskianalyysi ja hedgausstrategiat sähkönhintamallin tulosten pohjalta.

Laskee Monte Carlo -tuloksista:
  - VaR 95%, CVaR 95%, volatiliteetti, hintapiikkiriski
  - Hedgausstrategioiden vertailu (spot, kiinteä, collar, termiini, yhdistelmä)
  - Stressitestit (energiakriisi, kuiva talvi, ydinvoimahäiriö, datakeskusboom)
  - Optimal hedge -suositus ja efficient frontier
"""

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from model.scenarios import (
    ScenarioResult, ScenarioParams, SCENARIO_NAMES,
    START_YEAR, END_YEAR, compute_market_adjustments,
)

# ── Dataluokat ────────────────────────────────────────────────────────────────

@dataclass
class RiskMetrics:
    """Riskimittarit yhdelle skenaariolle."""
    scenario: str
    var_95: float            # VaR 95% vuosihinta €/MWh
    cvar_95: float           # CVaR 95% vuosihinta €/MWh
    max_monthly_price: float # Korkein kuukausihinta (P95 yli kaikkien kuukausien)
    volatility: float        # Kuukausittainen hintojen hajonta (P50-perusteella)
    spike_prob: float        # Arvio: P(jokin kk > 150 €/MWh) per vuosi
    annual_cost_p50: float   # Vuosikustannus P50-skenaariossa €
    annual_cost_var95: float # Kustannus VaR 95% -tasolla €
    annual_cost_cvar95: float# Kustannus CVaR 95% -tasolla €


@dataclass
class HedgeParams:
    """Hedgausstrategian parametrit."""
    strategy: str = "spot"          # spot / kiintea_100 / osittainen_5050 / collar / termiini / yhdistelma
    fixed_price: float = 65.0       # €/MWh (kiinteä tai termiinihinta)
    floor_price: float = 40.0       # collar floor €/MWh
    cap_price: float = 120.0        # collar cap €/MWh
    hedge_pct: float = 0.70         # termiiniosuus (0–1)
    forward_premium: float = 0.05   # termiinipreemio (0–0.15)
    spot_pct: float = 0.30          # yhdistelmä: spot-osuus (0–1)
    collar_pct: float = 0.40        # yhdistelmä: collar-osuus (0–1)
    vol_mwh: float = 10_000.0       # vuosivolyymi MWh
    dist_type: str = "tasainen"     # tasainen / talvi / kesä


@dataclass
class HedgeResult:
    """Yhden hedgausstrategian tulos."""
    strategy_name: str
    effective_price_p50: float       # €/MWh P50-skenaariossa
    effective_price_p95: float       # €/MWh pahimmassa 5%:ssa
    annual_cost_p50: float           # €/vuosi P50-skenaariossa
    annual_cost_p95: float           # €/vuosi pahimmassa 5%:ssa
    hedge_cost_vs_spot_p50: float    # €/vuosi lisäkustannus vs spot P50 (+ = kalliimpaa)
    hedge_benefit_vs_spot_p95: float # €/vuosi säästö vs spot P95 (+ = säästää)
    risk_reduction_ratio: float      # CVaR-parannus % vs spot
    cvar_95: float                   # CVaR 95% kustannus €/vuosi


@dataclass
class StressTest:
    """Yksi stressitestiskenaario."""
    name: str
    description: str
    price_spike: float           # huippuhinta €/MWh
    baseline_price: float        # perushintataso €/MWh
    price_increase_pct: float    # hinnannousu %
    duration_months: int         # arvioitu kesto
    annual_cost_impact: float    # kustannusvaikutus €/vuosi (vol_mwh tasolla)
    best_hedge: str              # paras suojausstrategia


# ── Riskimittarien laskenta ───────────────────────────────────────────────────

def calculate_risk_metrics(
    scenario_result: ScenarioResult,
    vol_mwh: float = 10_000.0,
) -> RiskMetrics:
    """
    Laskee riskimittarit yhdelle skenaariolle annual_sim_matrixista.

    annual_sim_matrix: shape (n_sim, n_years) — vuosittaiset avg hinnat per simulaatio.
    """
    sim = scenario_result.annual_sim_matrix  # (n_sim, n_years)

    # Vuosikeskiarvot per simulaatio (keskiarvo yli vuosien)
    avg_annual = sim.mean(axis=1)  # (n_sim,)

    var_95  = float(np.percentile(avg_annual, 95))
    above   = avg_annual[avg_annual >= var_95]
    cvar_95 = float(above.mean()) if len(above) > 0 else var_95

    # Korkein kuukausihinta P95 kuvaajasta
    mp = scenario_result.monthly_prices
    max_monthly = float(mp["p95"].max()) if "p95" in mp.columns else float(mp["p90"].max())

    # Volatiliteetti P50-kuukausisarjasta
    volatility = float(mp["p50"].std()) if "p50" in mp.columns else 0.0

    # Hintapiikkiriski: P95-arvo jonakin kuukautena > 150 €/MWh
    p95_col = "p95" if "p95" in mp.columns else "p90"
    spike_months = int((mp[p95_col] > 150.0).sum())
    spike_prob = spike_months / len(mp) if len(mp) > 0 else 0.0

    # Kustannukset vol_mwh tasolla
    p50_price = float(np.percentile(avg_annual, 50))
    annual_cost_p50   = p50_price   * vol_mwh
    annual_cost_var95 = var_95      * vol_mwh
    annual_cost_cvar95 = cvar_95    * vol_mwh

    return RiskMetrics(
        scenario=scenario_result.name,
        var_95=round(var_95, 2),
        cvar_95=round(cvar_95, 2),
        max_monthly_price=round(max_monthly, 2),
        volatility=round(volatility, 2),
        spike_prob=round(spike_prob, 4),
        annual_cost_p50=round(annual_cost_p50, 0),
        annual_cost_var95=round(annual_cost_var95, 0),
        annual_cost_cvar95=round(annual_cost_cvar95, 0),
    )


# ── Hedgausstrategioiden laskenta ─────────────────────────────────────────────

def _apply_hedge(prices: np.ndarray, hp: HedgeParams) -> np.ndarray:
    """
    Laskee efektiivisen hinnan strategiasta riippuen.

    prices: (n_sim,) array vuotuisista keskihinnoista €/MWh
    Palauttaa: (n_sim,) efektiiviset hinnat
    """
    s = hp.strategy

    if s == "spot":
        return prices.copy()

    if s == "kiintea_100":
        return np.full_like(prices, hp.fixed_price)

    if s == "osittainen_5050":
        return 0.5 * prices + 0.5 * hp.fixed_price

    if s == "collar":
        return np.clip(prices, hp.floor_price, hp.cap_price)

    if s == "termiini":
        # Termiini: hedge_pct kiinnitetty termiinihintaan (spot × (1+premium) perusteella),
        # loppuosa spot-hintaan
        fwd = np.mean(prices) * (1.0 + hp.forward_premium)
        return hp.hedge_pct * fwd + (1.0 - hp.hedge_pct) * prices

    if s == "yhdistelma":
        forward_pct = max(1.0 - hp.spot_pct - hp.collar_pct, 0.0)
        fwd = np.mean(prices) * (1.0 + hp.forward_premium)
        collar_prices = np.clip(prices, hp.floor_price, hp.cap_price)
        return hp.spot_pct * prices + hp.collar_pct * collar_prices + forward_pct * fwd

    return prices.copy()


def _strategy_label(strategy: str) -> str:
    labels = {
        "spot":          "Täysi spot",
        "kiintea_100":   "Kiinteä hinta 100%",
        "osittainen_5050": "Osittainen 50/50",
        "collar":        "Collar-strategia",
        "termiini":      "Termiinisuojaus 12kk",
        "yhdistelma":    "Yhdistelmästrategia",
    }
    return labels.get(strategy, strategy)


def calculate_all_hedges(
    scenario_result: ScenarioResult,
    hedge_params: HedgeParams,
    reference_strategy: str = "spot",
) -> list[HedgeResult]:
    """
    Laskee kaikki viisi vakiostrategiaa vertailua varten.
    Käyttää scenarion annual_sim_matrixia (vuosittaiset avg hinnat).

    Palauttaa listan HedgeResult-olioita.
    """
    sim = scenario_result.annual_sim_matrix  # (n_sim, n_years)
    # Käytä koko periodin (2025–2035) keskiarvoa per simulaatio
    avg_annual = sim.mean(axis=1)

    # Spot-referenssi CVaR:lle
    spot_eff = avg_annual.copy()
    spot_p50  = float(np.percentile(spot_eff, 50))
    spot_p95  = float(np.percentile(spot_eff, 95))
    above_spot = spot_eff[spot_eff >= spot_p95]
    spot_cvar = float(above_spot.mean()) if len(above_spot) > 0 else spot_p95

    results = []
    strategies = ["spot", "kiintea_100", "osittainen_5050", "collar", "termiini"]

    for strat in strategies:
        hp_copy = copy.copy(hedge_params)
        hp_copy.strategy = strat
        eff = _apply_hedge(avg_annual, hp_copy)

        p50  = float(np.percentile(eff, 50))
        p95  = float(np.percentile(eff, 95))
        above = eff[eff >= p95]
        cvar = float(above.mean()) if len(above) > 0 else p95

        cost_p50 = p50 * hedge_params.vol_mwh
        cost_p95 = p95 * hedge_params.vol_mwh

        hedge_cost = (p50 - spot_p50) * hedge_params.vol_mwh
        hedge_benefit = (spot_p95 - p95) * hedge_params.vol_mwh
        risk_red = (spot_cvar - cvar) / spot_cvar * 100 if spot_cvar > 0 else 0.0

        results.append(HedgeResult(
            strategy_name=_strategy_label(strat),
            effective_price_p50=round(p50, 2),
            effective_price_p95=round(p95, 2),
            annual_cost_p50=round(cost_p50, 0),
            annual_cost_p95=round(cost_p95, 0),
            hedge_cost_vs_spot_p50=round(hedge_cost, 0),
            hedge_benefit_vs_spot_p95=round(hedge_benefit, 0),
            risk_reduction_ratio=round(risk_red, 1),
            cvar_95=round(cvar * hedge_params.vol_mwh, 0),
        ))

    return results


def calculate_active_hedge(
    scenario_result: ScenarioResult,
    hedge_params: HedgeParams,
) -> pd.DataFrame:
    """
    Laskee valitun hedgausstrategian vuosittaiset efektiiviset hinnat.

    Palauttaa DataFrame: year, spot_p50, eff_price_p50, eff_price_p10, eff_price_p90
    """
    sim = scenario_result.annual_sim_matrix  # (n_sim, n_years)
    years = list(range(START_YEAR, END_YEAR + 1))
    rows = []

    for yi, year in enumerate(years):
        annual_col = sim[:, yi]
        eff = _apply_hedge(annual_col, hedge_params)
        rows.append({
            "year": year,
            "spot_p50":      float(np.percentile(annual_col, 50)),
            "eff_price_p10": float(np.percentile(eff, 10)),
            "eff_price_p50": float(np.percentile(eff, 50)),
            "eff_price_p90": float(np.percentile(eff, 90)),
        })

    return pd.DataFrame(rows)


# ── Stressitestit ─────────────────────────────────────────────────────────────

def run_stress_tests(base_params: ScenarioParams, vol_mwh: float = 10_000.0) -> list[StressTest]:
    """
    Ajaa neljä stressiskenaariota ja palauttaa arviot hintavaikutuksista.
    Käyttää deterministisiä laskelmia (ei full MC).
    """
    STRESS_YEAR = 2027
    BASE_PRICE_PERUS = 62.0

    def stressed_price(p: ScenarioParams) -> float:
        return BASE_PRICE_PERUS * compute_market_adjustments(p, STRESS_YEAR)

    baseline = stressed_price(base_params)

    results: list[StressTest] = []

    # ── Stressiskenaario 1: Energiakriisi 2021-tyylinen ──────────────────────
    p1 = copy.copy(base_params)
    p1.gas_price_mwh = base_params.gas_price_mwh * 3.0   # +200%
    p1.hydro_nordic = "kuiva"                             # vesivoima -20%
    p1.electrification_twh = base_params.electrification_twh + 4.25  # kulutus +5%
    sp1 = stressed_price(p1)
    # Paras hedge: collar (suojaa huipulta)
    results.append(StressTest(
        name="Energiakriisi 2021-tyylinen",
        description="Kaasun hinta +200%, vesivoima –25%, kulutus +5%",
        price_spike=round(sp1, 1),
        baseline_price=round(baseline, 1),
        price_increase_pct=round((sp1 / baseline - 1) * 100, 1),
        duration_months=6,
        annual_cost_impact=round((sp1 - baseline) * vol_mwh * (6 / 12), 0),
        best_hedge="Collar-strategia (cap 120 €/MWh)",
    ))

    # ── Stressiskenaario 2: Kuiva pohjoismaistalvi ───────────────────────────
    p2 = copy.copy(base_params)
    p2.hydro_nordic = "kuiva"
    # Tyyni talvi → tuulivoima -15%: approksimoidaan vähentämällä kapasiteettia
    p2.wind_fi_gw = max(base_params.wind_fi_gw - base_params.wind_fi_gw * 0.15, 0)
    p2.nuclear_se = "yksi_kiinni"  # Ruotsin ydinvoima osittain kiinni
    sp2 = stressed_price(p2)
    results.append(StressTest(
        name="Kuiva pohjoismaistalvi",
        description="Vesivoima –30%, tuulivoima –15% (tyyni talvi), SE ydinvoima rajoitettu",
        price_spike=round(sp2, 1),
        baseline_price=round(baseline, 1),
        price_increase_pct=round((sp2 / baseline - 1) * 100, 1),
        duration_months=3,
        annual_cost_impact=round((sp2 - baseline) * vol_mwh * (3 / 12), 0),
        best_hedge="Termiinisuojaus 12kk rullaava (70%)",
    ))

    # ── Stressiskenaario 3: Ydinvoiman alasajo ───────────────────────────────
    p3 = copy.copy(base_params)
    # OL3 tekninen vika: simuloidaan nykytasona + SE reaktori kiinni
    p3.nuclear_fi = "nykytaso"
    p3.nuclear_se = "yksi_kiinni"
    # Lisäksi kapasiteetti simuloidaan alemmaksi — käytetään mukautettua kerrointa
    sp3 = stressed_price(p3) * 1.12  # +12% lisäpainetta OL3-viasta
    results.append(StressTest(
        name="Ydinvoiman alasajo",
        description="OL3 tekninen vika + Ruotsin reaktori kiinni (6–12 kk)",
        price_spike=round(sp3, 1),
        baseline_price=round(baseline, 1),
        price_increase_pct=round((sp3 / baseline - 1) * 100, 1),
        duration_months=9,
        annual_cost_impact=round((sp3 - baseline) * vol_mwh * (9 / 12), 0),
        best_hedge="Kiinteä hinta 100% tai termiinisuojaus",
    ))

    # ── Stressiskenaario 4: Datakeskusboom ───────────────────────────────────
    p4 = copy.copy(base_params)
    p4.datacenter_growth_pct = 45.0   # räjähdysmäinen kasvu
    p4.datacenter_base_twh = base_params.datacenter_base_twh
    p4.electrification_twh = base_params.electrification_twh + 20.0  # +20 TWh
    sp4 = stressed_price(p4)
    results.append(StressTest(
        name="Datakeskusboom",
        description="Datakeskusten kulutus +20 TWh odottamatta nopeasti (2027–2029)",
        price_spike=round(sp4, 1),
        baseline_price=round(baseline, 1),
        price_increase_pct=round((sp4 / baseline - 1) * 100, 1),
        duration_months=36,
        annual_cost_impact=round((sp4 - baseline) * vol_mwh, 0),
        best_hedge="Collar-strategia tai pitkän termiinin suojaus",
    ))

    return results


# ── Optimal hedge -suositus ───────────────────────────────────────────────────

def get_hedge_recommendation(
    risk_metrics: dict[str, RiskMetrics],
    hedge_results: list[HedgeResult] | None = None,
) -> dict[str, Any]:
    """
    Analysoi riskitason ja antaa hedgaussuosituksen.

    Palauttaa: {risk_class, suositus_teksti, paras_strategia}
    """
    perus = risk_metrics.get("perus")
    if perus is None:
        return {"risk_class": "tuntematon", "suositus_teksti": "Ei riittävästi dataa.", "paras_strategia": "spot"}

    # Luokittele riskitaso
    if perus.volatility < 12.0:
        risk_class = "matala"
    elif perus.volatility < 22.0:
        risk_class = "kohtalainen"
    else:
        risk_class = "korkea"

    # Etsi paras strategia CVaR-minimoinnin perusteella
    paras = "collar"
    if hedge_results:
        sorted_h = sorted(hedge_results, key=lambda h: h.cvar_95)
        if sorted_h:
            paras = sorted_h[0].strategy_name

    if risk_class == "matala":
        teksti = (
            f"Matalariskinen markkinatilanne (volatiliteetti {perus.volatility:.1f} €/MWh). "
            "Suositellaan: 70% spot + 30% termiinisuojaus. "
            "Täysi spot on kohtuullinen vaihtoehto, mutta lyhyt termiinisuojaus antaa turvaa yllätyksille."
        )
    elif risk_class == "kohtalainen":
        teksti = (
            f"Kohtalainen riskitaso (volatiliteetti {perus.volatility:.1f} €/MWh, "
            f"CVaR 95% = {perus.cvar_95:.1f} €/MWh). "
            "Suositellaan: Collar-strategia (floor 40 €/MWh, cap 120 €/MWh) "
            "tai 50/50 kiinteä + spot. Tasapainottaa kustannuksen ja riskin."
        )
    else:
        teksti = (
            f"Korkea riskitaso (volatiliteetti {perus.volatility:.1f} €/MWh, "
            f"CVaR 95% = {perus.cvar_95:.1f} €/MWh, "
            f"hintapiikkiriski {perus.spike_prob*100:.1f}%). "
            "Suositellaan: Collar-strategia tai kiinteä hinta 70–100%. "
            "CVaR-minimoinnin perusteella paras strategia: "
            f"{paras}."
        )

    return {
        "risk_class": risk_class,
        "suositus_teksti": teksti,
        "paras_strategia": paras,
        "volatiliteetti": perus.volatility,
        "cvar_95": perus.cvar_95,
    }


# ── Efficient frontier ────────────────────────────────────────────────────────

def compute_efficient_frontier(
    scenario_result: ScenarioResult,
    vol_mwh: float,
    fixed_price: float = 65.0,
    floor_price: float = 40.0,
    cap_price: float = 120.0,
    forward_premium: float = 0.05,
) -> pd.DataFrame:
    """
    Laskee efficient frontier -kuvaajan datan:
    x = suojauksen lisäkustannus vs spot P50 (€/vuosi)
    y = CVaR 95% (€/vuosi)

    Vaihtelee hedge_pct 0..100% ja collar cap 80..200 €/MWh.
    Palauttaa DataFrame: strategia, hedge_cost, cvar_95
    """
    sim = scenario_result.annual_sim_matrix
    avg_annual = sim.mean(axis=1)

    spot_p50 = float(np.percentile(avg_annual, 50))
    spot_p95 = float(np.percentile(avg_annual, 95))
    above_spot = avg_annual[avg_annual >= spot_p95]
    spot_cvar = float(above_spot.mean()) if len(above_spot) > 0 else spot_p95

    rows = []

    base_hp = HedgeParams(
        fixed_price=fixed_price,
        floor_price=floor_price,
        cap_price=cap_price,
        forward_premium=forward_premium,
        vol_mwh=vol_mwh,
    )

    # Kiinteä hinta eri osuuksilla
    for pct in np.linspace(0, 1.0, 21):
        hp = copy.copy(base_hp)
        hp.strategy = "kiintea_100" if pct == 1.0 else "spot" if pct == 0.0 else "osittainen_5050"
        if hp.strategy == "osittainen_5050":
            # Emulate partial hedge at pct%
            eff = pct * fixed_price + (1 - pct) * avg_annual
        elif hp.strategy == "kiintea_100":
            eff = np.full_like(avg_annual, fixed_price)
        else:
            eff = avg_annual.copy()
        p95 = float(np.percentile(eff, 95))
        above = eff[eff >= p95]
        cvar = float(above.mean()) if len(above) > 0 else p95
        hedge_cost = (float(np.percentile(eff, 50)) - spot_p50) * vol_mwh
        rows.append({
            "strategia": f"Kiinteä {pct*100:.0f}%",
            "hedge_cost": round(hedge_cost, 0),
            "cvar_95": round(cvar * vol_mwh, 0),
            "tyyppi": "Kiinteä",
        })

    # Collar eri cap-tasoilla
    for cap in np.arange(80, 210, 20):
        eff = np.clip(avg_annual, floor_price, float(cap))
        p95 = float(np.percentile(eff, 95))
        above = eff[eff >= p95]
        cvar = float(above.mean()) if len(above) > 0 else p95
        hedge_cost = (float(np.percentile(eff, 50)) - spot_p50) * vol_mwh
        rows.append({
            "strategia": f"Collar cap={cap:.0f}",
            "hedge_cost": round(hedge_cost, 0),
            "cvar_95": round(cvar * vol_mwh, 0),
            "tyyppi": "Collar",
        })

    # Termiinisuojaus eri hedge-osuuksilla
    fwd = float(np.mean(avg_annual)) * (1.0 + forward_premium)
    for pct in np.linspace(0, 1.0, 11):
        eff = pct * fwd + (1 - pct) * avg_annual
        p95 = float(np.percentile(eff, 95))
        above = eff[eff >= p95]
        cvar = float(above.mean()) if len(above) > 0 else p95
        hedge_cost = (float(np.percentile(eff, 50)) - spot_p50) * vol_mwh
        rows.append({
            "strategia": f"Termiini {pct*100:.0f}%",
            "hedge_cost": round(hedge_cost, 0),
            "cvar_95": round(cvar * vol_mwh, 0),
            "tyyppi": "Termiini",
        })

    return pd.DataFrame(rows)


def build_risk_metrics_table(risk_metrics: dict[str, RiskMetrics]) -> pd.DataFrame:
    """Rakentaa riskimittarit-taulukon kolmelle skenaariolle vertailua varten."""
    rows = []
    labels = {"matala": "Matala", "perus": "Perus", "korkea": "Korkea"}
    for sc in ["matala", "perus", "korkea"]:
        m = risk_metrics.get(sc)
        if m is None:
            continue
        rows.append({
            "Skenaario": labels.get(sc, sc),
            "VaR 95% (€/MWh)": m.var_95,
            "CVaR 95% (€/MWh)": m.cvar_95,
            "Max kk-hinta (€/MWh)": m.max_monthly_price,
            "Volatiliteetti (€/MWh)": m.volatility,
            "Hintapiikkiriski (%)": round(m.spike_prob * 100, 1),
        })
    return pd.DataFrame(rows)
