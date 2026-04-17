"""
scenarios.py – Skenaariomalli ja Monte Carlo -simulaatio sähköhinnoille 2025–2035.

Laaja markkinamalli: FI tuulivoima/aurinko/ydinvoima, kulutuskasvu, datakeskukset,
pohjoismaiset vesivarat, Ruotsin ydinvoima, siirtoyhteydet, kaasun hinta, CO2.
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Vakiot ───────────────────────────────────────────────────────────────────

MONTH_FACTORS = {
    1: 1.38, 2: 1.30, 3: 1.12, 4: 0.95, 5: 0.85, 6: 0.80,
    7: 0.82, 8: 0.88, 9: 0.95, 10: 1.05, 11: 1.22, 12: 1.40,
}

SCENARIO_NAMES  = ["matala", "perus", "korkea"]
SCENARIO_LABELS = {
    "matala": "Matala (optimistinen)",
    "perus":  "Perus (todennäköisin)",
    "korkea": "Korkea (riskiskenaario)",
}
SCENARIO_COLORS = {
    "matala": "#2E7D32",
    "perus":  "#1565C0",
    "korkea": "#B71C1C",
}

START_YEAR = 2025
END_YEAR   = 2035
FI_BASE_CONSUMPTION_TWH = 85.0   # Suomen nykyinen sähkönkulutus TWh/vuosi

# ── Markkinaparametrien optiot (avain: (nimi_UI, hintavaikutus)) ─────────────

NUCLEAR_FI_OPTIONS: dict[str, tuple[str, float]] = {
    "nykytaso":      ("Nykytaso (OL1+OL2+OL3 ≈ 4.4 GW)", 0.0),
    "ol3_hanhikivi": ("OL3 täydellä teholla + Hanhikivi korvaava (~5.4 GW)", -0.045),
    "uusi_voimala":  ("Uusi ydinvoimala rakennettu (~6 GW)", -0.075),
    "smr":           ("SMR-reaktori (+0.5 GW) ≈ 4.9 GW", -0.022),
}

NUCLEAR_SE_OPTIONS: dict[str, tuple[str, float]] = {
    "normaali":     ("Normaali", 0.0),
    "yksi_kiinni":  ("Yksi reaktori kiinni", 0.08),
    "laajennukset": ("Laajennukset käynnissä", -0.05),
}

HYDRO_OPTIONS: dict[str, tuple[str, float]] = {
    "normaali": ("Normaali", 0.0),
    "kuiva":    ("Kuiva vuosi (–20%)", 0.18),
    "märkä":    ("Märkä vuosi (+15%)", -0.12),
}

INTERCONNECT_FI_EE_OPTIONS: dict[str, tuple[str, int]] = {
    "nykytaso":   ("Nykytaso (1 000 MW)", 1000),
    "estlink3":   ("Estlink 3 rakennettu (2 000 MW)", 2000),
    "rajoitettu": ("Rajoitettu kapasiteetti (500 MW, häiriö)", 500),
}

INTERCONNECT_FI_SE_OPTIONS: dict[str, tuple[str, int]] = {
    "nykytaso":    ("Nykytaso (~2 200 MW)", 2200),
    "laajennettu": ("Laajennettu (+500 MW = 2 700 MW)", 2700),
    "rajoitettu":  ("Rajoitettu (ruuhkatilanne ~1 540 MW)", 1540),
}

INTERCONNECT_NO_OPTIONS: dict[str, tuple[str, float]] = {
    "normaali":    ("Normaali", 1.0),
    "rajoitettu":  ("Rajoitettu", 0.7),
    "laajennettu": ("Laajennettu", 1.3),
}

# Perushinnat per skenaario (€/MWh) ennen markkinavaikutuksia
_BASE_PRICES = {"matala": 42.0, "perus": 62.0, "korkea": 92.0}
# Suhteellinen hajonta per skenaario
_BASE_STDS   = {"matala": 0.10, "perus": 0.15, "korkea": 0.22}


# ── Dataluokat ────────────────────────────────────────────────────────────────

@dataclass
class ScenarioParams:
    """Kaikkien markkinaparametrien kokoelma skenaarioajoon."""
    n_simulations: int = 500
    seed: int = 2025
    crisis_probability: float = 0.10

    # Suomen uusiutuva energia
    wind_fi_gw: float = 5.0         # Tuulivoiman lisäkapasiteetti 2025–2035, GW
    solar_fi_gw: float = 1.5        # Aurinkoenergian kasvu, GW

    # Suomen ydinvoima
    nuclear_fi: str = "nykytaso"

    # Pohjoismaiset markkinat
    hydro_nordic: str = "normaali"
    nuclear_se: str = "normaali"

    # Polttoainehinnat
    gas_price_mwh: float = 40.0     # €/MWh
    co2_price_t: float = 70.0       # €/t CO2

    # Kulutuskasvu Suomi
    electrification_twh: float = 8.0    # sähköistyminen (teollisuus + LPs)
    ev_twh: float = 1.0                 # sähköautot

    # Datakeskukset
    datacenter_base_twh: float = 2.0
    datacenter_growth_pct: float = 8.0  # % per vuosi

    # Siirtoyhteydet
    interconnect_fi_se: str = "nykytaso"
    interconnect_fi_ee: str = "nykytaso"
    interconnect_no: str = "normaali"


@dataclass
class RegressionResult:
    """Regressiomallin tulos fundamenttidatasta."""
    r2: float = 0.0
    coef: dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    used_features: list[str] = field(default_factory=list)
    base_price_adjustment: float = 0.0
    seasonal_factors: dict[int, float] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    """Yhden skenaarion tulokset."""
    name: str
    label: str
    color: str
    monthly_prices: pd.DataFrame    # year, month, p5, p10, p25, p50, p75, p90, p95
    annual_prices: pd.DataFrame     # year, p5, p10, p25, p50, p75, p90, p95
    annual_sim_matrix: np.ndarray   # shape (n_sim, n_years): vuosittaiset avg hinnat


# ── Markkinavaikutusten laskenta ──────────────────────────────────────────────

def compute_market_adjustments(params: ScenarioParams, year: int) -> float:
    """
    Laskee markkinaparametrien kokonaisvaikutuksen hintakertoimena
    suhteessa perustilanteeseen (1.0 = ei muutosta).

    Julkinen funktio: käytetään myös riskianalyysissä ja herkkyysanalyysissä.
    """
    years = max(year - START_YEAR, 0)  # 0..10
    factor = 1.0

    # 1. Tuulivoima + aurinkoenergia (lineaarinen kasvu 10v aikana)
    wind_new_twh  = params.wind_fi_gw  * 2.8 * (years / 10.0)
    solar_new_twh = params.solar_fi_gw * 0.9 * (years / 10.0)
    renewable_pct = (wind_new_twh + solar_new_twh) / FI_BASE_CONSUMPTION_TWH
    factor *= max(1.0 + (-0.40) * renewable_pct, 0.50)

    # 2. Ydinvoima Suomi (täysi vaikutus vuoteen 2030 mennessä)
    _, nfi_adj = NUCLEAR_FI_OPTIONS.get(params.nuclear_fi, ("", 0.0))
    ramp_nfi = min(years / 5.0, 1.0)
    factor *= (1.0 + nfi_adj * ramp_nfi)

    # 3. Pohjoismaiset vesivarat — välitön vaikutus
    _, hydro_adj = HYDRO_OPTIONS.get(params.hydro_nordic, ("", 0.0))
    # Norjan siirtopotentiaali skaalaa hydrovaikutusta
    _, no_factor = INTERCONNECT_NO_OPTIONS.get(params.interconnect_no, ("", 1.0))
    effective_hydro = hydro_adj * no_factor
    factor *= (1.0 + effective_hydro)

    # 4. Ruotsin ydinvoima
    _, nse_adj = NUCLEAR_SE_OPTIONS.get(params.nuclear_se, ("", 0.0))
    factor *= (1.0 + nse_adj)

    # 5. Kaasun hinta (referenssi 40 €/MWh, joustavuus 0.30)
    gas_pct = (params.gas_price_mwh - 40.0) / 40.0
    factor *= (1.0 + 0.30 * gas_pct)

    # 6. CO2-hinta (referenssi 70 €/t, joustavuus 0.20)
    co2_pct = (params.co2_price_t - 70.0) / 70.0
    factor *= (1.0 + 0.20 * co2_pct)

    # 7. Kulutuskasvu: sähköistyminen + sähköautot (joustavuus 0.30)
    total_growth_twh = params.electrification_twh + params.ev_twh
    consumption_pct = total_growth_twh * (years / 10.0) / FI_BASE_CONSUMPTION_TWH
    factor *= (1.0 + 0.30 * consumption_pct)

    # 8. Datakeskukset (joustavuus 0.30)
    dc_twh = params.datacenter_base_twh * ((1 + params.datacenter_growth_pct / 100) ** years)
    dc_twh = min(dc_twh, 50.0)
    dc_growth_pct = max(dc_twh - params.datacenter_base_twh, 0.0) / FI_BASE_CONSUMPTION_TWH
    factor *= (1.0 + 0.30 * dc_growth_pct)

    # 9. Siirtoyhteydet: suurempi kapasiteetti → pienempi hinnoille painetta
    _, fi_se_mw = INTERCONNECT_FI_SE_OPTIONS.get(params.interconnect_fi_se, ("", 2200))
    _, fi_ee_mw = INTERCONNECT_FI_EE_OPTIONS.get(params.interconnect_fi_ee, ("", 1000))
    total_ic_mw = fi_se_mw + fi_ee_mw
    ic_adj = (total_ic_mw - 3200) / 1000.0 * (-0.015)  # +1 GW → -1.5%
    factor *= (1.0 + ic_adj)

    return max(factor, 0.20)


def compute_max_hintaero(params: ScenarioParams) -> float:
    """
    Laskee maksimaalisen FI–Pohjoismaat hintaeron (€/MWh)
    siirtoyhteyksien kapasiteetin perusteella.

    max_hintaero = 500 / kokonaiskapasiteetti_GW
    """
    _, fi_se_mw = INTERCONNECT_FI_SE_OPTIONS.get(params.interconnect_fi_se, ("", 2200))
    _, fi_ee_mw = INTERCONNECT_FI_EE_OPTIONS.get(params.interconnect_fi_ee, ("", 1000))
    total_gw = (fi_se_mw + fi_ee_mw) / 1000.0
    return 500.0 / max(total_gw, 0.1)


def compute_variable_sensitivities(params: ScenarioParams, base_year: int = 2030) -> pd.DataFrame:
    """
    Laskee kunkin markkinamuuttujan hinnavaikutuksen yksi-kerrallaan-menetelmällä
    perusskenaariossa. Käytetään tornado-kuvaajaan.

    Palauttaa DataFrame: muuttuja, vaikutus_matala, vaikutus_korkea,
                          arvo_matala, arvo_korkea
    """
    base_price_perus = 62.0
    base_factor = compute_market_adjustments(params, base_year)
    base_result = base_price_perus * base_factor

    def impact(varied: ScenarioParams) -> float:
        return base_price_perus * compute_market_adjustments(varied, base_year) - base_result

    def vary(field_name: str, value: Any) -> ScenarioParams:
        p = copy.copy(params)
        setattr(p, field_name, value)
        return p

    rows = [
        {
            "muuttuja":       "Kaasun hinta",
            "vaikutus_matala": impact(vary("gas_price_mwh", 20.0)),
            "vaikutus_korkea": impact(vary("gas_price_mwh", 80.0)),
            "arvo_matala":    "20 €/MWh",
            "arvo_korkea":    "80 €/MWh",
        },
        {
            "muuttuja":       "CO₂-hinta",
            "vaikutus_matala": impact(vary("co2_price_t", 40.0)),
            "vaikutus_korkea": impact(vary("co2_price_t", 120.0)),
            "arvo_matala":    "40 €/t",
            "arvo_korkea":    "120 €/t",
        },
        {
            "muuttuja":       "Vesivoima Pohjoismaat",
            "vaikutus_matala": impact(vary("hydro_nordic", "märkä")),
            "vaikutus_korkea": impact(vary("hydro_nordic", "kuiva")),
            "arvo_matala":    "Märkä vuosi",
            "arvo_korkea":    "Kuiva vuosi",
        },
        {
            "muuttuja":       "Ruotsin ydinvoima",
            "vaikutus_matala": impact(vary("nuclear_se", "laajennukset")),
            "vaikutus_korkea": impact(vary("nuclear_se", "yksi_kiinni")),
            "arvo_matala":    "Laajennukset",
            "arvo_korkea":    "Reaktori kiinni",
        },
        {
            "muuttuja":       "Tuulivoima FI (lisäys)",
            "vaikutus_matala": impact(vary("wind_fi_gw", 0.0)),
            "vaikutus_korkea": impact(vary("wind_fi_gw", 15.0)),
            "arvo_matala":    "0 GW",
            "arvo_korkea":    "15 GW",
        },
        {
            "muuttuja":       "Ydinvoima FI",
            "vaikutus_matala": impact(vary("nuclear_fi", "nykytaso")),
            "vaikutus_korkea": impact(vary("nuclear_fi", "uusi_voimala")),
            "arvo_matala":    "Nykytaso",
            "arvo_korkea":    "Uusi voimala",
        },
        {
            "muuttuja":       "Aurinkoenergia FI",
            "vaikutus_matala": impact(vary("solar_fi_gw", 0.0)),
            "vaikutus_korkea": impact(vary("solar_fi_gw", 5.0)),
            "arvo_matala":    "0 GW",
            "arvo_korkea":    "5 GW",
        },
        {
            "muuttuja":       "Datakeskusten kasvu",
            "vaikutus_matala": impact(vary("datacenter_growth_pct", 0.0)),
            "vaikutus_korkea": impact(vary("datacenter_growth_pct", 30.0)),
            "arvo_matala":    "0 %/v",
            "arvo_korkea":    "30 %/v",
        },
        {
            "muuttuja":       "Sähköistyminen + LPs",
            "vaikutus_matala": impact(vary("electrification_twh", 0.0)),
            "vaikutus_korkea": impact(vary("electrification_twh", 30.0)),
            "arvo_matala":    "0 TWh",
            "arvo_korkea":    "30 TWh",
        },
        {
            "muuttuja":       "Siirtoyhteydet FI–SE",
            "vaikutus_matala": impact(vary("interconnect_fi_se", "rajoitettu")),
            "vaikutus_korkea": impact(vary("interconnect_fi_se", "laajennettu")),
            "arvo_matala":    "Rajoitettu",
            "arvo_korkea":    "Laajennettu",
        },
    ]

    df = pd.DataFrame(rows)
    df["spread"] = df["vaikutus_korkea"] - df["vaikutus_matala"]
    # Järjestä suurimmasta vaikutuksesta pienimpään (tornado-järjestys)
    df = df.reindex(df["spread"].abs().sort_values(ascending=True).index)
    return df.reset_index(drop=True)


def compute_datacenter_projection(params: ScenarioParams) -> pd.DataFrame:
    """Laskee datakeskusten TWh-kasvukäyrän vuosittain."""
    rows = []
    for year in range(START_YEAR, END_YEAR + 1):
        years = year - START_YEAR
        twh = params.datacenter_base_twh * ((1 + params.datacenter_growth_pct / 100) ** years)
        capped = twh >= 50.0
        twh = min(twh, 50.0)
        rows.append({"year": year, "twh": round(twh, 2), "capped": capped})
    return pd.DataFrame(rows)


def compute_consumption_growth(params: ScenarioParams) -> dict[str, float]:
    """Laskee kulutuskasvun yhteenvedon."""
    total_growth = params.electrification_twh + params.ev_twh
    return {
        "nyky_twh": FI_BASE_CONSUMPTION_TWH,
        "kasvu_twh": total_growth,
        "ennuste_2035_twh": FI_BASE_CONSUMPTION_TWH + total_growth,
    }


def compute_impact_breakdown(
    params: ScenarioParams,
    ref_years: tuple[int, ...] = (2030, 2035),
) -> dict:
    """
    Erittelee kolmen päämuuttujan (kulutuskasvu, datakeskukset, tuulivoima)
    €/MWh-vaikutuksen perusskenaariossa annetuille vuosille.

    Käyttää yksi-kerrallaan-menetelmää: asettaa kunkin muuttujan nollaan
    ja laskee erotuksen täyden mallin hintaan.

    Palauttaa sanakirjan:
    {
        vuosi: {
            "base_price":          float,   perusskenaario täysillä parametreilla
            "kulutus_impact":      float,   €/MWh (positiivinen = hinnannousu)
            "datacenter_impact":   float,
            "wind_impact":         float,   (negatiivinen = hinnanalennus)
            "total_growth_twh":    float,
            "dc_kasvu_twh":        float,
            "wind_re_twh":         float,
        },
        ...
    }
    """
    REF_BASE = _BASE_PRICES["perus"]  # 62 €/MWh
    result: dict[int, dict] = {}

    for year in ref_years:
        full_factor = compute_market_adjustments(params, year)
        full_price  = REF_BASE * full_factor

        # ── Kulutuskasvu: vertaa nykyiseen vs. nolla ─────────────────────
        p0 = copy.copy(params)
        p0.electrification_twh = 0.0
        p0.ev_twh = 0.0
        f0 = compute_market_adjustments(p0, year)
        kulutus_impact = (full_factor - f0) * REF_BASE

        # ── Datakeskukset: nollakasvuvauhti (lähtötaso pysyy vakiona) ────
        p1 = copy.copy(params)
        p1.datacenter_growth_pct = 0.0
        f1 = compute_market_adjustments(p1, year)
        dc_impact = (full_factor - f1) * REF_BASE

        # ── Tuulivoima + aurinko: nolla lisäkapasiteetti ─────────────────
        p2 = copy.copy(params)
        p2.wind_fi_gw  = 0.0
        p2.solar_fi_gw = 0.0
        f2 = compute_market_adjustments(p2, year)
        wind_impact = (full_factor - f2) * REF_BASE  # negatiivinen

        # Laskennalliset määrät kyseiselle vuodelle
        years_from_start = max(year - START_YEAR, 0)
        wind_re_twh = (
            params.wind_fi_gw  * 2.8 * (years_from_start / 10.0)
            + params.solar_fi_gw * 0.9 * (years_from_start / 10.0)
        )
        dc_twh_now = min(
            params.datacenter_base_twh * ((1 + params.datacenter_growth_pct / 100) ** years_from_start),
            50.0,
        )
        dc_kasvu_twh = max(dc_twh_now - params.datacenter_base_twh, 0.0)

        result[year] = {
            "base_price":        round(full_price, 1),
            "kulutus_impact":    round(kulutus_impact, 1),
            "datacenter_impact": round(dc_impact, 1),
            "wind_impact":       round(wind_impact, 1),
            "total_growth_twh":  round(params.electrification_twh + params.ev_twh, 1),
            "dc_kasvu_twh":      round(dc_kasvu_twh, 1),
            "wind_re_twh":       round(wind_re_twh, 1),
        }

    return result


# ── Regressiomallin kalibrointi ───────────────────────────────────────────────

def calibrate_regression(fundamental_df: pd.DataFrame) -> RegressionResult:
    """
    Sovittaa lineaarisen regressiomallin historialliseen fundamenttidataan.
    Palauttaa RegressionResult-olion mallin kertoimilla ja R².
    Jos hintadata puuttuu tai data on liian niukkaa, palautetaan tyhjä tulos.
    """
    result = RegressionResult()

    if fundamental_df.empty or "price_fi" not in fundamental_df.columns:
        logger.info("Hintadataa ei löydy — ei kalibrointia")
        return result

    df = fundamental_df.copy().dropna(subset=["price_fi"])
    if len(df) < 10:
        logger.warning("Liian vähän dataa kalibrointiin (%d riviä)", len(df))
        return result

    feature_cols = []
    for col in ["wind_capacity", "hydro_production", "nuclear_production", "gas_price", "co2_price"]:
        if col in df.columns and df[col].notna().sum() > 5:
            feature_cols.append(col)

    if "date" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
        feature_cols += ["month_sin", "month_cos"]

    if not feature_cols:
        logger.info("Ei käyttökelpoisia piirteitä regressioon")
        return result

    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler

        X = df[feature_cols].fillna(df[feature_cols].mean())
        y = df["price_fi"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LinearRegression()
        model.fit(X_scaled, y)

        r2 = model.score(X_scaled, y)
        coef = {feat: float(c) for feat, c in zip(feature_cols, model.coef_)}

        result.r2 = round(r2, 3)
        result.coef = coef
        result.intercept = float(model.intercept_)
        result.used_features = feature_cols
        result.base_price_adjustment = float(y.mean()) - 62.0

        if "date" in df.columns:
            seasonal = df.groupby(df["date"].dt.month)["price_fi"].mean()
            mean_price = seasonal.mean()
            if mean_price > 0:
                result.seasonal_factors = {m: float(p / mean_price) for m, p in seasonal.items()}

        logger.info("Regressio kalibroitu: R²=%.3f, piirteet=%s", r2, feature_cols)
    except ImportError:
        logger.warning("scikit-learn ei ole asennettu — ei kalibrointia")
    except Exception as e:
        logger.warning("Regressio epäonnistui: %s", e)

    return result


# ── Monte Carlo -simulaatio ───────────────────────────────────────────────────

def run_monte_carlo(
    params: ScenarioParams,
    regression: RegressionResult | None = None,
    progress_callback: Any = None,
    timeout_seconds: float = 30.0,
) -> dict[str, ScenarioResult]:
    """
    Ajaa täysin vektorisoidun Monte Carlo -simulaation kolmelle skenaariolle.

    Kaikki sisäkkäiset Python for-silmukat on poistettu:
    - Vuosikohtainen otanta: np.random.normal(means, stds, (n_sim, n_years))
    - Kuukausikohtainen otanta: broadcasting (n_sim, n_years, 12)
    - Persentiilit: np.percentile(..., axis=0) koko matriisille kerralla

    progress_callback(fraction: float, viesti: str) – kutsutaan jokaisen skenaarion jälkeen.
    timeout_seconds – jos ylittyy, palautetaan osittaiset tulokset varoituksella.

    Palauttaa dict {skenaarionimi: ScenarioResult} jossa
    monthly_prices sisältää p5/p10/p25/p50/p75/p90/p95.
    """
    import time as _time

    reg = regression if regression is not None else RegressionResult()
    rng = np.random.default_rng(params.seed)
    results: dict[str, ScenarioResult] = {}

    n_sim   = params.n_simulations
    n_years = END_YEAR - START_YEAR + 1  # 11
    n_months = 12

    years_arr  = np.arange(START_YEAR, END_YEAR + 1, dtype=np.int32)   # (11,)
    months_arr = np.arange(1, 13, dtype=np.int32)                       # (12,)

    # Kausikerroin-vektori (12,) – käytetään regressiosta tai vakioista
    seasonal = reg.seasonal_factors if reg.seasonal_factors else {}
    mf_arr = np.array(
        [seasonal.get(m, MONTH_FACTORS[m]) for m in range(1, 13)],
        dtype=np.float64,
    )  # (12,)

    pctiles = [5, 10, 25, 50, 75, 90, 95]
    n_pct   = len(pctiles)

    # Indeksit DataFrame-rakentamista varten (rakennetaan kerran)
    years_rep   = np.repeat(years_arr, n_months)    # (n_years*12,) = (132,)
    months_tile = np.tile(months_arr, n_years)       # (n_years*12,) = (132,)

    t_start = _time.monotonic()

    for sc_idx, scenario in enumerate(SCENARIO_NAMES):
        # ── Tarkista timeout ennen jokaista skenaariota ──────────────────────
        elapsed = _time.monotonic() - t_start
        if elapsed > timeout_seconds and results:
            logger.warning(
                "Monte Carlo timeout %.1fs ylitetty skenaariossa '%s' — "
                "palautetaan osittaiset tulokset (%d/%d skenaariota).",
                timeout_seconds, scenario, sc_idx, len(SCENARIO_NAMES),
            )
            break

        base    = _BASE_PRICES[scenario]
        std_rel = _BASE_STDS[scenario]
        adj     = reg.base_price_adjustment

        # ── Vuosittaiset markkinavaikutukset (11 skalaaria — nopea) ──────────
        # Ainut jäljellä oleva Python-silmukka: vain 11 kierrosta
        market_factors = np.array(
            [compute_market_adjustments(params, int(y)) for y in years_arr],
            dtype=np.float64,
        )  # (n_years,)

        means = np.maximum((base + adj) * market_factors, 15.0)  # (n_years,)
        stds  = means * std_rel                                    # (n_years,)

        # ── Vuositason otanta — täysin vektorisoitu ──────────────────────────
        # annual_samples: (n_sim, n_years)
        annual_samples = rng.normal(
            loc=means,                      # broadcast: (n_years,) → (n_sim, n_years)
            scale=stds,
            size=(n_sim, n_years),
        )

        # Kriisitapahtumat korkea-skenaariossa — vektorisoitu maski
        if scenario == "korkea" and params.crisis_probability > 0:
            crisis_mask = rng.random((n_sim, n_years)) < params.crisis_probability
            boosts      = rng.uniform(1.3, 1.8, (n_sim, n_years))
            annual_samples = np.where(crisis_mask, annual_samples * boosts, annual_samples)

        annual_samples = np.maximum(annual_samples, 1.0)  # (n_sim, n_years)

        # ── Kuukausitason otanta — täysin vektorisoitu ───────────────────────
        # monthly_base: (n_sim, n_years, 12)
        #   annual_samples[:, :, None] * mf_arr[None, None, :]
        monthly_base = annual_samples[:, :, np.newaxis] * mf_arr[np.newaxis, np.newaxis, :]

        # Kuukausihajonta: stds (n_years,) → (1, n_years, 1) broadcast
        noise = rng.normal(
            loc=0.0,
            scale=(stds * 0.12)[np.newaxis, :, np.newaxis],
            size=(n_sim, n_years, n_months),
        )
        monthly_samples = np.maximum(monthly_base + noise, 0.5)  # (n_sim, n_years, 12)

        # ── Persentiilit — yksi numpy-kutsu per taso ─────────────────────────
        # annual_pcts:  (n_pct, n_years)
        # monthly_pcts: (n_pct, n_years, 12)
        annual_pcts  = np.percentile(annual_samples,  pctiles, axis=0)
        monthly_pcts = np.percentile(monthly_samples, pctiles, axis=0)

        # ── Rakenna DataFramet — ei silmukoita rivien yli ───────────────────
        annual_df = pd.DataFrame(
            {f"p{p}": annual_pcts[i] for i, p in enumerate(pctiles)},
        )
        annual_df.insert(0, "year", years_arr)

        monthly_pcts_flat = monthly_pcts.reshape(n_pct, -1)  # (n_pct, n_years*12)
        monthly_df = pd.DataFrame(
            {f"p{p}": monthly_pcts_flat[i] for i, p in enumerate(pctiles)},
        )
        monthly_df.insert(0, "year",  years_rep)
        monthly_df.insert(1, "month", months_tile)

        results[scenario] = ScenarioResult(
            name=scenario,
            label=SCENARIO_LABELS[scenario],
            color=SCENARIO_COLORS[scenario],
            monthly_prices=monthly_df,
            annual_prices=annual_df,
            annual_sim_matrix=annual_samples.astype(np.float32),
        )

        # ── Edistymisraportti ────────────────────────────────────────────────
        if progress_callback is not None:
            fraction = (sc_idx + 1) / len(SCENARIO_NAMES)
            progress_callback(fraction, SCENARIO_LABELS[scenario])

        logger.debug(
            "Skenaario '%s' valmis: %.0f ms",
            scenario, (_time.monotonic() - t_start) * 1000,
        )

    return results


def scenarios_to_dataframe(results: dict[str, ScenarioResult]) -> pd.DataFrame:
    """
    Yhdistää kaikkien skenaarioiden kuukausitulokset yhdeksi DataFrameksi.
    Sarakkeet: year, month, scenario, p5, p10, p25, p50, p75, p90, p95
    """
    frames = []
    for name, result in results.items():
        df = result.monthly_prices.copy()
        df["scenario"] = name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
