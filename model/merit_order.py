"""
merit_order.py – Merit order -malli Suomen sähkömarkkinoille.

Marginaalikustannusjärjestys:
Tuuli → Aurinko → Ydinvoima → Vesivoima → CHP → Tuonti → Kaasu
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class MeritOrderParams:
    gas_price_mwh: float = 40.0      # €/MWh
    co2_price_t: float   = 70.0      # €/t
    water_level: str     = "normaali" # normaali / kuiva / märkä
    nordpool_ref: float  = 55.0      # Nord Pool viitehinta €/MWh
    month: int           = 1

# Marginaalikustannukset €/MWh per tuotantomuoto
MARGINAL_BASE = {
    "tuuli":     (0.0,  3.0),
    "aurinko":   (0.0,  2.0),
    "ydinvoima": (8.0, 12.0),
    "tuonti":    None,  # lasketaan nordpool_ref + 2
    "chp":       None,  # lasketaan kaasukaavalla
    "kaasu":     None,  # lasketaan kaasukaavalla
}

# Vesivoiman opportunity cost €/MWh per vesivarantotaso
HYDRO_OC = {
    "märkä":    (0.0,  2.0),
    "normaali": (0.0,  5.0),
    "kuiva":    (15.0, 40.0),
}

# Kausikerroin vesivoiman kustannukseen
HYDRO_SEASONAL = {
    1:1.75, 2:1.80, 3:1.60, 4:0.40, 5:0.35, 6:0.85,
    7:0.90, 8:0.95, 9:0.75, 10:0.70, 11:0.65, 12:1.70
}

def _hydro_marginal(water_level: str, month: int) -> float:
    """Laskee vesivoiman marginaalikustannuksen €/MWh."""
    lo, hi = HYDRO_OC.get(water_level, (0.0, 5.0))
    base = (lo + hi) / 2.0
    seasonal = HYDRO_SEASONAL.get(month, 1.0)
    mult = {"märkä": 0.4, "normaali": 1.0, "kuiva": 1.8}.get(water_level, 1.0)
    return round(base * seasonal * mult, 2)

def _chp_marginal(gas_price_mwh: float, co2_price_t: float) -> float:
    return round(gas_price_mwh * 0.45 + co2_price_t * 0.35, 2)

def _gas_marginal(gas_price_mwh: float, co2_price_t: float) -> float:
    return round(gas_price_mwh * 0.55 + co2_price_t * 0.45, 2)

@dataclass
class MeritOrderSlice:
    """Yksi kapasiteettisiivuke merit order -käyrällä."""
    source: str
    capacity_mw: float
    marginal_cost: float
    cumulative_mw: float = 0.0

def build_merit_order(
    params: MeritOrderParams,
    capacity_mw: Dict[str, float],
) -> List[MeritOrderSlice]:
    """
    Rakentaa merit order -käyrän annetulle kapasiteetti-sanakirjalle.

    capacity_mw: {"tuuli": X, "aurinko": Y, "ydinvoima": Z,
                  "vesivoima": V, "chp": C, "tuonti": T, "kaasu": G}
    Palauttaa järjestetyn listan MeritOrderSlice-olioita.
    """
    hydro_mc  = _hydro_marginal(params.water_level, params.month)
    chp_mc    = _chp_marginal(params.gas_price_mwh, params.co2_price_t)
    import_mc = params.nordpool_ref + 2.0
    gas_mc    = _gas_marginal(params.gas_price_mwh, params.co2_price_t)

    source_costs = {
        "tuuli":     2.0,
        "aurinko":   1.5,
        "ydinvoima": 10.0,
        "vesivoima": hydro_mc,
        "chp":       chp_mc,
        "tuonti":    import_mc,
        "kaasu":     gas_mc,
    }

    slices = []
    for source, cap in capacity_mw.items():
        if cap > 0:
            slices.append(MeritOrderSlice(
                source=source,
                capacity_mw=float(cap),
                marginal_cost=source_costs.get(source, 50.0),
            ))

    slices.sort(key=lambda s: s.marginal_cost)

    cumulative = 0.0
    for s in slices:
        cumulative += s.capacity_mw
        s.cumulative_mw = round(cumulative, 0)

    return slices

def calculate_market_price(
    month: int,
    capacity_mw: Dict[str, float],
    demand_mw: float,
    water_level: str = "normaali",
    gas_price: float = 40.0,
    co2_price: float = 70.0,
    nordpool_ref: float = 55.0,
) -> Tuple[float, str, float]:
    """
    Laskee markkinahinnan merit order -mallilla.

    Palauttaa: (markkinahinta_eur_mwh, rajakustannustuotantomuoto, kapasiteettiylijaama_mw)
    """
    params = MeritOrderParams(
        gas_price_mwh=gas_price,
        co2_price_t=co2_price,
        water_level=water_level,
        nordpool_ref=nordpool_ref,
        month=month,
    )
    slices = build_merit_order(params, capacity_mw)

    if not slices:
        return nordpool_ref, "tuntematon", 0.0

    # Löydä rajakustannustuotantomuoto
    marginal_source = slices[-1].source
    marginal_cost   = slices[-1].marginal_cost
    total_cap       = slices[-1].cumulative_mw

    # Jos kysyntä ylittää tarjonnan, hinta = viimeisin + preemio
    surplus_mw = total_cap - demand_mw
    if surplus_mw < 0:
        # Ylikuormitus → preemio
        premium = abs(surplus_mw) / demand_mw * 50.0
        price = marginal_cost + premium
    else:
        # Löydä oikea rajakustannus
        price = marginal_cost
        for s in slices:
            if s.cumulative_mw >= demand_mw:
                price = s.marginal_cost
                marginal_source = s.source
                break

    return round(price, 2), marginal_source, round(surplus_mw, 0)

def merit_order_time_series(
    months: range,
    base_capacity: Dict[str, float],
    demand_mw: float,
    gas_price: float = 40.0,
    co2_price: float = 70.0,
    water_level: str = "normaali",
) -> pd.DataFrame:
    """Laskee merit order -hinnan jokaiselle kuukaudelle."""
    records = []
    for month in months:
        price, source, surplus = calculate_market_price(
            month, base_capacity, demand_mw, water_level, gas_price, co2_price
        )
        records.append({
            "month": month,
            "price_eur_mwh": price,
            "marginal_source": source,
            "surplus_mw": surplus,
        })
    return pd.DataFrame(records)

def merit_order_to_df(slices: List[MeritOrderSlice]) -> pd.DataFrame:
    """Muuntaa MeritOrderSlice-listan DataFrameksi kuvaajaa varten."""
    return pd.DataFrame([{
        "source": s.source,
        "capacity_mw": s.capacity_mw,
        "marginal_cost": s.marginal_cost,
        "cumulative_mw": s.cumulative_mw,
    } for s in slices])

SOURCE_COLORS = {
    "tuuli":     "#2196F3",
    "aurinko":   "#FFC107",
    "ydinvoima": "#9C27B0",
    "vesivoima": "#00BCD4",
    "chp":       "#FF9800",
    "tuonti":    "#4CAF50",
    "kaasu":     "#F44336",
}
