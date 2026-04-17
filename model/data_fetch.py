"""
data_fetch.py – Datan haku Excel-tiedostosta sekä synteettinen fallback.

Pääfunktiot:
  - load_fundamental_data(filepath)  → standardoitu DataFrame Excel-tiedostosta
  - load_historical_prices()         → kuukausittaiset hinnat (synteettinen)
  - generate_synthetic_prices()      → synteettinen historiallinen data
"""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model.data_inspect import inspect_excel, paras_valilehti

logger = logging.getLogger(__name__)


# ── Vakiot ───────────────────────────────────────────────────────────────────

# Standardi-sarakenimet ja niiden tunnistusavainsanat
_COLUMN_MAP = {
    "date":               ["date", "pvm", "päivä", "aika", "time", "timestamp", "period", "vuosi", "year", "month"],
    "price_fi":           ["hinta", "price", "spot", "electricity", "sahkö", "sähkö", "eur_mwh", "€/mwh"],
    "consumption":        ["kulutus", "consumption", "demand", "käyttö", "load"],
    "wind_capacity":      ["tuulivoima", "wind", "tuuli", "wind_cap"],
    "hydro_production":   ["vesivoima", "hydro", "vesisähkö", "hydro_prod"],
    "nuclear_production": ["ydinvoima", "nuclear", "ydin", "nuclear_prod"],
    "gas_price":          ["kaasu", "gas", "lng", "ttf", "nbp", "gas_price"],
    "co2_price":          ["co2", "hiili", "carbon", "päästö", "emission", "eua"],
}


def _best_column_match(columns: list[str], keywords: list[str]) -> str | None:
    """Löytää parhaiten avainsanoja vastaavan sarakkeen (ei-case-sensitiivinen)."""
    for col in columns:
        col_lower = col.lower()
        for kw in keywords:
            if kw in col_lower:
                return col
    return None


def _to_datetime_safe(series: pd.Series) -> pd.Series:
    """
    Yrittää muuntaa sarake datetime-muotoon usealla tavalla.
    Palauttaa alkuperäisen sarakkeen jos muunnos ei onnistu.
    """
    for fmt in [None, "%Y-%m-%d", "%d.%m.%Y", "%Y/%m/%d", "%m/%d/%Y"]:
        try:
            return pd.to_datetime(series, format=fmt, errors="raise")
        except Exception:
            continue
    # Kokeile ymd-numerosarjaa (esim. 20150101)
    try:
        return pd.to_datetime(series.astype(str), format="%Y%m%d", errors="raise")
    except Exception:
        pass
    return series


def _normalize_to_monthly(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Normalisoi aikasarjan kuukausitasolle.

    Jos data on jo kuukausitasolla, palauttaa sellaisenaan.
    Jos päivätaso tai tunnitaso, aggregoi kuukausikeskiarvoiksi.
    """
    df = df.copy()
    df[date_col] = _to_datetime_safe(df[date_col])

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        logger.warning("Päivämääräsaraketta ei voitu muuntaa: %s", date_col)
        return df

    df["_year"] = df[date_col].dt.year
    df["_month"] = df[date_col].dt.month

    # Tunnista frekvenssi: jos enemmän kuin 13 riviä per vuosi, kyseessä kuukausitasoa tarkempi data
    rows_per_year = df.groupby("_year").size().median()
    if rows_per_year <= 13:
        # Jo kuukausitasolla tai karkeampaa — käytä sellaisenaan
        df["date"] = pd.to_datetime(df[["_year", "_month"]].assign(day=1))
        df = df.drop(columns=[date_col, "_year", "_month"], errors="ignore")
        return df

    # Aggregoi numeeristen sarakkeiden kuukausikeskiarvot
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ("_year", "_month")]
    agg = df.groupby(["_year", "_month"])[numeric_cols].mean().reset_index()
    agg["date"] = pd.to_datetime(agg[["_year", "_month"]].rename(columns={"_year": "year", "_month": "month"}).assign(day=1))
    agg = agg.drop(columns=["_year", "_month"], errors="ignore")
    return agg


def load_fundamental_data(filepath: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Lataa fundamenttidata Excel-tiedostosta ja standardisoi sarakkeet.

    Vaiheet:
    1. Kutsuu inspect_excel() rakenteen tunnistamiseksi
    2. Valitsee parhaan välilehden
    3. Lataa ja normalisoi data
    4. Nimeää sarakkeet standardinimiiksi
    5. Käsittelee puuttuvat arvot

    Palauttaa:
        (df, meta)
        df:   DataFrame standardisarakkeilla (date, price_fi, consumption, ...)
        meta: sanakirja, jossa inspect-tulos + löydetyt sarakkeet + oletukset
    """
    filepath = Path(filepath)
    meta: dict[str, Any] = {"oletukset": [], "löydetyt_sarakkeet": {}, "inspect": {}}

    # Vaihe 1: tarkista tiedosto
    inspect = inspect_excel(filepath)
    meta["inspect"] = inspect

    if "virhe" in inspect:
        logger.error("Excel-tiedoston avaus epäonnistui: %s", inspect["virhe"])
        return pd.DataFrame(), meta

    best_sheet = paras_valilehti(inspect)
    if best_sheet is None:
        logger.error("Yhtään sopivaa välilehteä ei löydy")
        return pd.DataFrame(), meta

    meta["käytetty_välilehti"] = best_sheet
    sheet_info = inspect["sheets"][best_sheet]
    columns = sheet_info["columns"]

    # Vaihe 2: lataa välilehti
    try:
        raw = pd.read_excel(filepath, sheet_name=best_sheet, engine="openpyxl")
        raw = raw.dropna(how="all").dropna(axis=1, how="all")
        raw.columns = raw.columns.astype(str)
    except Exception as e:
        logger.error("Välilehden lataus epäonnistui: %s", e)
        meta["oletukset"].append(f"Välilehden lataus epäonnistui: {e}")
        return pd.DataFrame(), meta

    raw_columns = list(raw.columns)

    # Vaihe 3: tunnista sarakkeet
    found: dict[str, str] = {}  # standardinimi -> alkuperäinen sarakenimi
    for std_name, keywords in _COLUMN_MAP.items():
        match = _best_column_match(raw_columns, keywords)
        if match:
            found[std_name] = match

    meta["löydetyt_sarakkeet"] = {k: v for k, v in found.items() if k != "date"}

    # Vaihe 4: päivämäärä — pakollinenkenttä tai generoidaan
    date_col = found.get("date")
    if date_col and date_col in raw.columns:
        df = _normalize_to_monthly(raw, date_col)
    else:
        # Kokeile löytyykö vuosi+kuukausi-sarakkeet erikseen
        year_col = _best_column_match(raw_columns, ["year", "vuosi"])
        month_col = _best_column_match(raw_columns, ["month", "kuukausi"])
        if year_col and month_col:
            raw["_date_gen"] = pd.to_datetime(
                raw[[year_col, month_col]].rename(columns={year_col: "year", month_col: "month"}).assign(day=1)
            )
            df = raw.copy()
            df["date"] = df["_date_gen"]
            df = df.drop(columns=["_date_gen"], errors="ignore")
            meta["oletukset"].append("Päivämäärä rakennettu vuosi+kuukausi-sarakkeista.")
        else:
            logger.warning("Päivämääräsaraketta ei löydy — generoidaan kuukausijärjestys")
            df = raw.copy()
            df["date"] = pd.date_range(start="2015-01", periods=len(df), freq="MS")
            meta["oletukset"].append(
                "Päivämääräsaraketta ei löydetty. Käytetään automaattista kuukausijärjestystä alkaen 2015-01."
            )

    # Vaihe 5: nimeä tunnistetut sarakkeet standardinimiiksi
    rename_map: dict[str, str] = {}
    for std_name, orig_col in found.items():
        if std_name == "date":
            continue
        if orig_col in df.columns:
            rename_map[orig_col] = std_name

    df = df.rename(columns=rename_map)

    # Ilmoita puuttuvat standardisarakkeet oletuksina
    all_std = [k for k in _COLUMN_MAP if k != "date"]
    for std in all_std:
        if std not in df.columns:
            meta["oletukset"].append(
                f"'{std}' ei löydy datasta — käytetään synteettistä oletusarvoa mallissa."
            )

    # Vaihe 6: varmista date-sarake ja lajittele
    if "date" not in df.columns:
        meta["oletukset"].append("Päivämääräsaraketta ei pystytty luomaan.")
        return df, meta

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Vaihe 7: muunna numeerinen data ja interpoloi puuttuvat
    for col in df.select_dtypes(include="object").columns:
        if col == "date":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    logger.info(
        "Fundamenttidata ladattu: %d riviä, sarakkeet: %s",
        len(df), list(df.columns)
    )
    return df, meta


# ── Synteettinen historiallinen data ─────────────────────────────────────────

def generate_synthetic_prices(start_year: int = 2015, end_year: int = 2024) -> pd.DataFrame:
    """
    Generoi realistisen synteettisen kuukausittaisen hintadatan vuosille 2015–2024.

    Mallintaa oikeat suomalaiset hintatasot:
    - 2015–2020: 30–45 €/MWh, normaali kausivaihtelu
    - 2021–2022: energiakriisi, piikkihinnat
    - 2023–2024: normalisoituminen OL3:n myötä
    """
    rng = np.random.default_rng(42)

    annual_base = {
        2015: 31.0, 2016: 33.0, 2017: 37.0, 2018: 44.0, 2019: 40.0,
        2020: 28.0, 2021: 72.0, 2022: 140.0, 2023: 55.0, 2024: 45.0,
    }
    month_factors = {
        1: 1.35, 2: 1.30, 3: 1.10, 4: 0.95, 5: 0.85, 6: 0.80,
        7: 0.82, 8: 0.88, 9: 0.95, 10: 1.05, 11: 1.20, 12: 1.38,
    }
    crisis_boost = {
        (2021, 12): 2.5, (2022, 1): 3.2, (2022, 2): 2.8,
        (2022, 8): 3.5, (2022, 9): 4.0, (2022, 10): 3.0,
        (2022, 11): 2.5, (2022, 12): 2.2,
    }

    records = []
    for year in range(start_year, end_year + 1):
        base = annual_base.get(year, 45.0)
        for month in range(1, 13):
            mf = month_factors[month]
            boost = crisis_boost.get((year, month), 1.0)
            noise = rng.normal(1.0, 0.08)
            price = max(base * mf * boost * noise, 0.0)
            records.append({"year": year, "month": month, "price_eur_mwh": round(price, 2)})

    return pd.DataFrame(records)


def load_historical_prices() -> pd.DataFrame:
    """
    Lataa historiallinen hintadata.

    Käyttää synteettistä dataa (realistinen fallback ilman API-avainta).
    Palauttaa DataFrame:n sarakkeilla: year, month, price_eur_mwh
    """
    logger.info("Käytetään synteettistä historiallista dataa")
    return generate_synthetic_prices()
