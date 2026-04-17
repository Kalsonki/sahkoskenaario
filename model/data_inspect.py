"""
data_inspect.py – Excel-tiedoston automaattinen tunnistus ja rakenteiden analyysi.
"""

import re
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Avainsanalistat tunnistusta varten
# ---------------------------------------------------------------------------

_PAIVAYS_PATTERNS = [
    "date", "pvm", "päivä", "aika", "time", "vuosi", "year",
    "kuukausi", "month", "timestamp", "period",
]

_HINTA_PATTERNS = [
    "hinta", "price", "spot", "€", "eur", "mwh", "kwh",
    "cost", "rate", "tariff", "electricity",
]

_KULUTUS_PATTERNS = [
    "kulutus", "consumption", "demand", "käyttö", "use",
    "load", "energy_use", "kulutus_mwh",
]

_KAPASITEETTI_PATTERNS = [
    "kapasiteetti", "capacity", "mw", "gw", "installed",
    "teho", "power", "cap",
]

_TUOTANTO_PATTERNS = [
    "tuulivoima", "wind", "hydro", "nuclear", "ydinvoima",
    "vesivoima", "solar", "aurinko", "tuuli", "production",
    "tuotanto", "generation",
]

_KAASU_PATTERNS = ["kaasu", "gas", "lng", "ttf", "nbp"]

_CO2_PATTERNS = ["co2", "hiili", "carbon", "päästö", "emission", "eua"]


def _match_patterns(col: str, patterns: list[str]) -> bool:
    """Tarkistaa, löytyykö jokin hakusanoista sarakkeen nimestä."""
    col_lower = col.lower()
    return any(p in col_lower for p in patterns)


def _detect_column_roles(columns: list[str]) -> dict[str, list[str]]:
    """
    Ryhmittelee sarakkeet roolien mukaan (päivämäärä, hinta, kulutus jne.).

    Palauttaa dict: rooli -> lista sarakkeiden nimiä.
    """
    roles: dict[str, list[str]] = {
        "paivays": [],
        "hinta": [],
        "kulutus": [],
        "kapasiteetti": [],
        "tuotanto": [],
        "kaasu": [],
        "co2": [],
        "muut": [],
    }
    for col in columns:
        if _match_patterns(col, _PAIVAYS_PATTERNS):
            roles["paivays"].append(col)
        elif _match_patterns(col, _CO2_PATTERNS):
            roles["co2"].append(col)
        elif _match_patterns(col, _KAASU_PATTERNS):
            roles["kaasu"].append(col)
        elif _match_patterns(col, _HINTA_PATTERNS):
            roles["hinta"].append(col)
        elif _match_patterns(col, _KULUTUS_PATTERNS):
            roles["kulutus"].append(col)
        elif _match_patterns(col, _TUOTANTO_PATTERNS):
            roles["tuotanto"].append(col)
        elif _match_patterns(col, _KAPASITEETTI_PATTERNS):
            roles["kapasiteetti"].append(col)
        else:
            roles["muut"].append(col)
    return roles


def inspect_excel(filepath: str | Path) -> dict[str, Any]:
    """
    Analysoi Excel-tiedoston rakenne ilman varsinaista datan latausta.

    Palauttaa sanakirjan, jossa jokaisesta välilehdestä:
      - sheet_names: kaikki välilehdet
      - sheets: dict välilehdistä, sisältäen:
          * columns: sarakkeiden nimet
          * dtypes: sarakkeiden tietotyypit
          * n_rows: rivimäärä
          * preview: ensimmäiset 3 riviä (dict-lista)
          * roles: automaattisesti tunnistetut sarake-roolit
          * tunnistettu_data: ihmisluettava yhteenveto

    Virhetilanteessa palauttaa {'virhe': str, 'sheets': {}}.
    """
    filepath = Path(filepath)
    result: dict[str, Any] = {"tiedosto": filepath.name, "sheets": {}}

    try:
        xl = pd.ExcelFile(filepath, engine="openpyxl")
    except Exception as e:
        return {"virhe": f"Tiedostoa ei voitu avata: {e}", "sheets": {}}

    result["sheet_names"] = xl.sheet_names

    for sheet in xl.sheet_names:
        try:
            df = xl.parse(sheet)
        except Exception as e:
            result["sheets"][sheet] = {"virhe": str(e)}
            continue

        # Poista täysin tyhjät rivit ja sarakkeet
        df = df.dropna(how="all").dropna(axis=1, how="all")

        columns = list(df.columns.astype(str))
        roles = _detect_column_roles(columns)

        # Tunnistettu data -yhteenveto
        tunnistettu: list[str] = []
        if roles["paivays"]:
            tunnistettu.append(f"Aikasarja: {', '.join(roles['paivays'])}")
        if roles["hinta"]:
            tunnistettu.append(f"Hintatiedot: {', '.join(roles['hinta'])}")
        if roles["kulutus"]:
            tunnistettu.append(f"Kulutustiedot: {', '.join(roles['kulutus'])}")
        if roles["tuotanto"]:
            tunnistettu.append(f"Tuotantotiedot: {', '.join(roles['tuotanto'])}")
        if roles["kapasiteetti"]:
            tunnistettu.append(f"Kapasiteettitiedot: {', '.join(roles['kapasiteetti'])}")
        if roles["kaasu"]:
            tunnistettu.append(f"Kaasun hinta: {', '.join(roles['kaasu'])}")
        if roles["co2"]:
            tunnistettu.append(f"CO2-päästöoikeus: {', '.join(roles['co2'])}")

        result["sheets"][sheet] = {
            "columns": columns,
            "dtypes": {c: str(t) for c, t in zip(df.columns.astype(str), df.dtypes)},
            "n_rows": len(df),
            "preview": df.head(3).astype(str).to_dict(orient="records"),
            "roles": roles,
            "tunnistettu_data": tunnistettu if tunnistettu else ["Ei tunnistettu automaattisesti"],
        }

    return result


def paras_valilehti(inspect_result: dict[str, Any]) -> str | None:
    """
    Valitsee parhaan välilehden datanlatausta varten.

    Priorisoi välilehti, jossa on eniten tunnistettuja sarakkeita.
    Palauttaa None jos tuloksessa ei ole yhtään välilehteä.
    """
    sheets = inspect_result.get("sheets", {})
    if not sheets:
        return None

    def pisteet(sheet_info: dict) -> int:
        roles = sheet_info.get("roles", {})
        return sum(
            len(v)
            for k, v in roles.items()
            if k != "muut" and isinstance(v, list)
        )

    return max(sheets.keys(), key=lambda s: pisteet(sheets[s]))
