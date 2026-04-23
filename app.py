"""
app.py – Streamlit-pääsovellus: Sähkönhintaskenaariot 2025–2038.
Käynnistys: streamlit run app.py
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Sivuasetukset (ennen muita st-kutsuja) ────────────────────────────────────
st.set_page_config(
    page_title="Sähkönhintaskenaariot 2025–2038",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Salasanasuojaus ───────────────────────────────────────────────────────────

def check_password() -> bool:
    """Tarkistaa salasanan ja palauttaa True jos kirjautunut sisään."""
    if st.session_state.get("authenticated"):
        return True
    st.title("⚡ Sähkönhintaskenaariot 2025–2038")
    st.markdown("### Kirjaudu sisään")
    password = st.text_input("Salasana", type="password", key="pw_input")
    if st.button("Kirjaudu"):
        if password == st.secrets.get("APP_PASSWORD", "demo1234"):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Väärä salasana")
    st.caption("Vihje: oletussalasana on `demo1234` (muuta .streamlit/secrets.toml)")
    return False

if not check_password():
    st.stop()

# ── Tyylimääritykset ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stMetricValue { font-size: 1.8rem !important; }
    .block-container { padding-top: 1rem; }
    div[data-testid="metric-container"] {
        background-color: #F9FBE7;
        border: 1px solid #C5E1A5;
        border-radius: 8px;
        padding: 12px;
    }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Sähkönhintaskenaariot 2025–2038")
st.caption("Suomen sähkömarkkinan analyysi | Monte Carlo -simulaatio | Merit Order | Riskianalyysi")

# ── Tuonti malleista ──────────────────────────────────────────────────────────
from model.data_fetch import load_fundamental_data, load_historical_prices
from model.data_inspect import inspect_excel
from model.scenarios import (
    RegressionResult, ScenarioParams, ScenarioResult,
    NUCLEAR_FI_OPTIONS, NUCLEAR_SE_OPTIONS, HYDRO_OPTIONS,
    INTERCONNECT_FI_EE_OPTIONS, INTERCONNECT_FI_SE_OPTIONS, INTERCONNECT_NO_OPTIONS,
    FI_BASE_CONSUMPTION_TWH, FI_CONSUMPTION_BREAKDOWN_2025, FI_MONTHLY_CONSUMPTION_2025,
    START_YEAR, END_YEAR,
    SCENARIO_NAMES, SCENARIO_LABELS, SCENARIO_COLORS,
    calibrate_regression, run_monte_carlo, scenarios_to_dataframe,
    compute_variable_sensitivities, compute_datacenter_projection,
    compute_consumption_growth, compute_max_hintaero,
    compute_impact_breakdown,
)
from model.risk import (
    HedgeParams, calculate_risk_metrics, calculate_all_hedges,
    calculate_active_hedge, run_stress_tests, get_hedge_recommendation,
    compute_efficient_frontier, build_risk_metrics_table,
)
from model.capacity import (
    CapacityParams, calculate_monthly_capacity, calculate_capacity_margin,
    capacity_time_series, find_critical_months, NUCLEAR_OPTIONS_MW,
)
from model.merit_order import (
    MeritOrderParams, MeritOrderSlice, build_merit_order,
    calculate_market_price, merit_order_time_series, merit_order_to_df,
    SOURCE_COLORS,
)
from ui.charts import (
    correlation_heatmap, datacenter_growth_chart, efficient_frontier_chart,
    fundamental_time_series, hedge_annual_cost_chart, hedge_comparison_chart,
    interconnect_hintaero_chart, monthly_avg_bar, monthly_heatmap,
    price_percentile_paths, price_scenario_chart, regression_coef_chart,
    stress_test_chart, tornado_chart,
)
from ui.report import build_pdf_report, generate_summary_text


# ── Kuukausinimitykset ────────────────────────────────────────────────────────
MONTH_NAMES_FI = {
    1: "Tammikuu", 2: "Helmikuu", 3: "Maaliskuu", 4: "Huhtikuu",
    5: "Toukokuu", 6: "Kesäkuu",  7: "Heinäkuu",  8: "Elokuu",
    9: "Syyskuu", 10: "Lokakuu", 11: "Marraskuu", 12: "Joulukuu",
}
MONTH_SHORT_FI = {
    1: "Tam", 2: "Hel", 3: "Maa", 4: "Huh", 5: "Tou", 6: "Kes",
    7: "Hei", 8: "Elo", 9: "Syy", 10: "Lok", 11: "Mar", 12: "Jou",
}


# ── Välimuistifunktiot ────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Ladataan historiallista dataa...")
def get_historical_data() -> pd.DataFrame:
    return load_historical_prices()


@st.cache_data(show_spinner=False, max_entries=30)
def run_scenarios_cached(
    params_key: tuple,
    _params: ScenarioParams,
    _regression: RegressionResult,
    timeout_seconds: float = 30.0,
) -> dict[str, ScenarioResult]:
    """
    Välimuistitettu Monte Carlo -ajo. Cache invalidoituu kun params_key muuttuu.
    _params ja _regression: underscore-prefix → Streamlit EI käytä niitä hashaukseen.
    """
    return run_monte_carlo(_params, _regression, timeout_seconds=timeout_seconds)


@st.cache_data(show_spinner="Luetaan Excel-tiedostoa...", max_entries=3)
def load_excel_cached(file_bytes: bytes, filename: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    suffix = Path(filename).suffix or ".xlsx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        df, meta = load_fundamental_data(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return df, meta


# ══════════════════════════════════════════════════════════════════════════════
# SIVUPALKKI
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("Parametrit")

    # ── Datatiedosto ──────────────────────────────────────────────────────────
    st.subheader("Datatiedosto")
    uploaded_file = st.file_uploader(
        "Lataa Excel-tiedosto (.xlsx)",
        type=["xlsx", "xls"],
        help="Tiedosto käsitellään vain paikallisesti — data ei lähde koneeltasi.",
    )

    fundamental_df: pd.DataFrame = pd.DataFrame()
    excel_meta: dict[str, Any] = {}
    regression: RegressionResult = RegressionResult()
    has_excel = False

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        with st.spinner("Analysoidaan Excel-tiedostoa..."):
            fundamental_df, excel_meta = load_excel_cached(file_bytes, uploaded_file.name)

        if not fundamental_df.empty:
            has_excel = True
            with st.spinner("Kalibroidaan regressiomalli..."):
                regression = calibrate_regression(fundamental_df)

            with st.expander("Tunnistettu data", expanded=True):
                found_cols = excel_meta.get("löydetyt_sarakkeet", {})
                sheet = excel_meta.get("käytetty_välilehti", "?")
                st.caption(f"Välilehti: **{sheet}** | {len(fundamental_df)} kuukautta")
                col_display = {
                    "price_fi":           "Spot-hinta",
                    "consumption":        "Kulutus",
                    "wind_capacity":      "Tuulivoima",
                    "hydro_production":   "Vesivoima",
                    "nuclear_production": "Ydinvoima",
                    "gas_price":          "Kaasun hinta",
                    "co2_price":          "CO₂-hinta",
                }
                for std, label in col_display.items():
                    if std in found_cols:
                        st.markdown(f"✅ {label}: `{found_cols[std]}`")
                    else:
                        st.markdown(f"🟡 {label}: synteettinen oletus")
                if regression.r2 > 0:
                    r2_pct = regression.r2 * 100
                    color = "green" if r2_pct > 60 else "orange"
                    st.markdown(
                        f"**R² = :{color}[{r2_pct:.1f}%]** "
                        f"({len(regression.used_features)} piirrettä)"
                    )
        else:
            st.error(
                "Excel-tiedoston lataus epäonnistui. "
                + excel_meta.get("inspect", {}).get("virhe", "Tuntematon virhe.")
            )
    else:
        st.info("Ei ladattua tiedostoa — käytetään synteettistä dataa.", icon="ℹ️")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SUOMEN SÄHKÖMARKKINAN KEHITYS
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Suomen sähkömarkkinan kehitys")

    with st.expander("Kulutuksen kasvu", expanded=True):
        electrification_twh = st.slider(
            "Perussähköistyminen (teollisuus + lämpöpumput)",
            0, 30, 8, 1,
            help="Sähköistymisen tuoma kulutuskasvu vuoteen 2035 mennessä (TWh)",
        )
        ev_twh = st.slider(
            "Sähköautot",
            0, 6, 1, 1,
            help="Sähköautojen lisäkulutus vuoteen 2035 (TWh)",
        )
        total_growth = electrification_twh + ev_twh
        st.caption(
            f"Kulutus kasvaa **{total_growth} TWh** → "
            f"yhteensä **{FI_BASE_CONSUMPTION_TWH + total_growth:.0f} TWh** vuoteen 2035"
        )

    with st.expander("Datakeskukset", expanded=False):
        datacenter_base = st.slider(
            "Datakeskusten kulutus 2025 (lähtötaso, TWh)",
            0.5, 10.0, 3.0, 0.5,
        )
        datacenter_growth = st.slider(
            "Datakeskusten vuosikasvuvauhti (%/v)",
            0, 50, 32, 1,
        )
        dc_final = datacenter_base * ((1 + datacenter_growth / 100) ** 10)
        dc_capped = dc_final >= 50.0
        dc_final_disp = min(dc_final, 50.0)
        st.caption(
            f"Datakeskukset kasvavat **{datacenter_base:.1f} TWh → "
            f"{dc_final_disp:.1f} TWh** vuoteen 2035"
        )
        if dc_capped:
            st.info(f"Datakeskuskulutus saavuttaa 50 TWh tavoitetason vuoteen 2035 mennessä.", icon="⚡")

    with st.expander("Tuulivoima ja aurinkoenergia", expanded=False):
        wind_fi_gw = st.slider(
            "Tuulivoiman lisäkapasiteetti 2025–2035 (GW)",
            0.0, 15.0, 5.0, 0.5,
            help="Suomen nykykapasiteetti ~7 GW. Tässä lisärakentaminen.",
        )
        solar_fi_gw = st.slider(
            "Aurinkoenergian kasvu (GW)",
            0.0, 5.0, 1.5, 0.5,
        )
        new_re_twh = wind_fi_gw * 2.8 + solar_fi_gw * 0.9
        st.caption(
            f"Uusiutuva tuotanto kasvaa arviolta **+{new_re_twh:.0f} TWh/vuosi** vuoteen 2035"
        )

    # Hintavaikutusten pikakatsaus
    _preview_params = ScenarioParams(
        wind_fi_gw=wind_fi_gw,
        solar_fi_gw=solar_fi_gw,
        electrification_twh=electrification_twh,
        ev_twh=ev_twh,
        datacenter_base_twh=datacenter_base,
        datacenter_growth_pct=datacenter_growth,
    )
    _impact = compute_impact_breakdown(_preview_params, ref_years=(2030, 2035))

    with st.expander("Hintavaikutukset 2030 / 2035", expanded=False):
        for _yr in (2030, 2035):
            _d = _impact[_yr]
            st.markdown(f"**Vuosi {_yr}** — perusskenaario P50: **{_d['base_price']:.1f} €/MWh**")

            def _fmt(v: float) -> str:
                sign = "+" if v >= 0 else ""
                return f"{sign}{v:.1f} €/MWh"

            st.caption(
                f"Kulutuskasvu ({_d['total_growth_twh']:.0f} TWh): **{_fmt(_d['kulutus_impact'])}**  \n"
                f"Datakeskukset (+{_d['dc_kasvu_twh']:.1f} TWh): **{_fmt(_d['datacenter_impact'])}**  \n"
                f"Tuulivoima (+{_d['wind_re_twh']:.0f} TWh/v): **{_fmt(_d['wind_impact'])}**"
            )
            if _yr == 2030:
                st.divider()

    with st.expander("Ydinvoima Suomi", expanded=False):
        nuclear_fi_keys = list(NUCLEAR_FI_OPTIONS.keys())
        nuclear_fi_labels = [NUCLEAR_FI_OPTIONS[k][0] for k in nuclear_fi_keys]
        nuclear_fi_idx = st.selectbox(
            "Suomen ydinvoimakapasiteetti 2035",
            options=range(len(nuclear_fi_keys)),
            format_func=lambda i: nuclear_fi_labels[i],
            index=0,
        )
        nuclear_fi = nuclear_fi_keys[nuclear_fi_idx]

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # POHJOISMAISET MARKKINAT
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Pohjoismaiset markkinat")

    with st.expander("Vesivoima ja Ruotsi", expanded=True):
        hydro_keys = list(HYDRO_OPTIONS.keys())
        hydro_labels = [HYDRO_OPTIONS[k][0] for k in hydro_keys]
        hydro_idx = st.selectbox(
            "Vesivarannot (Norja + Ruotsi)",
            options=range(len(hydro_keys)),
            format_func=lambda i: hydro_labels[i],
        )
        hydro_nordic = hydro_keys[hydro_idx]

        nuclear_se_keys = list(NUCLEAR_SE_OPTIONS.keys())
        nuclear_se_labels = [NUCLEAR_SE_OPTIONS[k][0] for k in nuclear_se_keys]
        nuclear_se_idx = st.selectbox(
            "Ruotsin ydinvoima",
            options=range(len(nuclear_se_keys)),
            format_func=lambda i: nuclear_se_labels[i],
        )
        nuclear_se = nuclear_se_keys[nuclear_se_idx]

    with st.expander("Polttoainehinnat", expanded=True):
        gas_price = st.slider(
            "Kaasun hinta (€/MWh)",
            20, 80, 40, 5,
            help="Euroopan kaasun referenssihinta (TTF)",
        )
        co2_price = st.slider(
            "Päästöoikeuden hinta ETS (€/t CO₂)",
            40, 120, 70, 5,
        )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SIIRTOYHTEYDET
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Siirtoyhteydet")

    with st.expander("Siirtoyhteyksien kapasiteetti", expanded=False):
        fi_ee_keys = list(INTERCONNECT_FI_EE_OPTIONS.keys())
        fi_ee_labels = [INTERCONNECT_FI_EE_OPTIONS[k][0] for k in fi_ee_keys]
        fi_ee_idx = st.selectbox(
            "EstLink (FI–EE)",
            options=range(len(fi_ee_keys)),
            format_func=lambda i: fi_ee_labels[i],
        )
        interconnect_fi_ee = fi_ee_keys[fi_ee_idx]

        fi_se_keys = list(INTERCONNECT_FI_SE_OPTIONS.keys())
        fi_se_labels = [INTERCONNECT_FI_SE_OPTIONS[k][0] for k in fi_se_keys]
        fi_se_idx = st.selectbox(
            "Fennoskan / FI–SE -yhteys",
            options=range(len(fi_se_keys)),
            format_func=lambda i: fi_se_labels[i],
        )
        interconnect_fi_se = fi_se_keys[fi_se_idx]

        no_keys = list(INTERCONNECT_NO_OPTIONS.keys())
        no_labels = [INTERCONNECT_NO_OPTIONS[k][0] for k in no_keys]
        no_idx = st.selectbox(
            "Norja-yhteys (Ruotsin kautta)",
            options=range(len(no_keys)),
            format_func=lambda i: no_labels[i],
        )
        interconnect_no = no_keys[no_idx]

        tmp_params_ic = ScenarioParams(
            interconnect_fi_se=interconnect_fi_se,
            interconnect_fi_ee=interconnect_fi_ee,
        )
        max_hintaero_preview = compute_max_hintaero(tmp_params_ic)
        st.caption(f"Max FI–Pohjoismaat hintaero: **{max_hintaero_preview:.0f} €/MWh**")

        st.markdown("**Naapurialueiden hintataso suhteessa FI:hin**")
        se3_price_relative = st.slider(
            "SE3 (Tukholma) hinta % FI:stä",
            60, 120, 92, 1,
            help="SE3 on historiallisesti 5–15% halvempi kuin FI (FI-SE EPAD). 100% = pariteetti.",
        ) / 100.0
        se1_price_relative = st.slider(
            "SE1 (Luulaja) hinta % FI:stä",
            40, 110, 78, 1,
            help="Pohjois-Ruotsin vesivoima — usein 15–30% halvempi kuin FI.",
        ) / 100.0
        ee_price_relative = st.slider(
            "EE (Viro) hinta % FI:stä",
            70, 130, 100, 1,
            help="Viro on usein lähellä FI-tasoa. Ennen venäläinen kaasu teki siitä halvemman.",
        ) / 100.0

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # KAPASITEETTIMALLI
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Kapasiteettimalli")

    with st.expander("Kapasiteettiparametrit", expanded=False):
        wind_total_gw = st.slider(
            "Tuulivoima yhteensä 2025 (GW, asennettu)",
            3.0, 20.0, 7.0, 0.5,
            help="Suomen kokonaistuulivoimakapasiteetti (ei lisärakentaminen)",
        )
        solar_cap_gw = st.slider(
            "Aurinkovoima yhteensä (GW)",
            0.5, 10.0, 1.5, 0.5,
        )
        _, fi_se_mw_cap = INTERCONNECT_FI_SE_OPTIONS.get(interconnect_fi_se, ("", 2200))
        _, fi_ee_mw_cap = INTERCONNECT_FI_EE_OPTIONS.get(interconnect_fi_ee, ("", 1000))

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # MONTE CARLO
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("Monte Carlo -asetukset", expanded=False):
        n_simulations = st.selectbox(
            "Simulaatioita",
            options=[100, 500, 1000],
            index=1,
            format_func=lambda x: f"{x} ajoa",
        )
        crisis_prob = st.slider(
            "Energiakriisin todennäköisyys (%)",
            0, 30, 10, 1,
            help="Korkean skenaarion kriisitapahtuma",
        ) / 100.0

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # HEDGAUSSTRATEGIA
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Hedgausstrategia")

    hedge_strategy = st.selectbox(
        "Strategia",
        options=["spot", "kiintea_100", "osittainen_5050", "collar", "termiini", "yhdistelma"],
        format_func=lambda x: {
            "spot":            "Täysi spot (ei hedgausta)",
            "kiintea_100":     "Kiinteä hinta 100%",
            "osittainen_5050": "Osittainen kiinteä 50/50",
            "collar":          "Collar-strategia (floor + cap)",
            "termiini":        "Termiinisuojaus 12kk rullaava",
            "yhdistelma":      "Yhdistelmä (spot + collar + termiini)",
        }[x],
    )

    fixed_price  = 65.0
    floor_price  = 40.0
    cap_price    = 120.0
    hedge_pct    = 0.70
    fwd_premium  = 0.05
    spot_pct_h   = 0.30
    collar_pct_h = 0.40

    if hedge_strategy in ("kiintea_100", "osittainen_5050"):
        fixed_price = st.number_input(
            "Kiinteä hinta (€/MWh)", 20.0, 200.0, 65.0, 1.0
        )

    if hedge_strategy == "collar":
        floor_price = st.slider("Floor-hinta (€/MWh)", 30, 60, 40, 1)
        cap_price   = st.slider("Cap-hinta (€/MWh)", 80, 200, 120, 5)

    if hedge_strategy == "termiini":
        hedge_pct   = st.slider("Suojausaste (%)", 0, 100, 70, 5) / 100.0
        fwd_premium = st.slider("Termiinipreemio (%)", 0, 15, 5, 1) / 100.0

    if hedge_strategy == "yhdistelma":
        spot_pct_h   = st.slider("Spot-osuus (%)", 0, 100, 30, 5) / 100.0
        max_collar   = max(0, int((1 - spot_pct_h) * 100))
        collar_pct_h = st.slider("Collar-osuus (%)", 0, max_collar, min(40, max_collar), 5) / 100.0
        fwd_pct_h    = 1.0 - spot_pct_h - collar_pct_h
        st.caption(f"Termiini-osuus: **{fwd_pct_h*100:.0f}%**")
        floor_price  = st.slider("Collar floor (€/MWh)", 30, 60, 40, 1)
        cap_price    = st.slider("Collar cap (€/MWh)", 80, 200, 120, 5)
        fwd_premium  = st.slider("Termiinipreemio (%)", 0, 15, 5, 1) / 100.0

    st.markdown("**Volyymi**")
    vol_mwh = st.slider(
        "Vuosivolyymi (MWh)",
        100, 500_000, 10_000, 500,
        help="Hedgattava vuosittainen sähkövolyymi",
    )
    dist_type = st.radio(
        "Kulutusjakauma",
        options=["tasainen", "talvi", "kesä"],
        format_func=lambda x: {
            "tasainen": "Tasainen ympäri vuoden",
            "talvi":    "Talvipainotus",
            "kesä":     "Kesäpainotus",
        }[x],
    )

    st.divider()
    st.caption("⚡ Sähkönhintaskenaariot v4.0")


# ── Parametrit koottuna ───────────────────────────────────────────────────────

scenario_params = ScenarioParams(
    n_simulations=n_simulations,
    crisis_probability=crisis_prob,
    wind_fi_gw=wind_fi_gw,
    solar_fi_gw=solar_fi_gw,
    nuclear_fi=nuclear_fi,
    hydro_nordic=hydro_nordic,
    nuclear_se=nuclear_se,
    gas_price_mwh=gas_price,
    co2_price_t=co2_price,
    electrification_twh=electrification_twh,
    ev_twh=ev_twh,
    datacenter_base_twh=datacenter_base,
    datacenter_growth_pct=datacenter_growth,
    interconnect_fi_se=interconnect_fi_se,
    interconnect_fi_ee=interconnect_fi_ee,
    interconnect_no=interconnect_no,
    se3_price_relative=se3_price_relative,
    se1_price_relative=se1_price_relative,
    ee_price_relative=ee_price_relative,
)

hedge_params = HedgeParams(
    strategy=hedge_strategy,
    fixed_price=fixed_price,
    floor_price=floor_price,
    cap_price=cap_price,
    hedge_pct=hedge_pct,
    forward_premium=fwd_premium,
    spot_pct=spot_pct_h,
    collar_pct=collar_pct_h,
    vol_mwh=vol_mwh,
    dist_type=dist_type,
)

# Kapasiteettiparametrit
capacity_params = CapacityParams(
    nuclear_fi=nuclear_fi,
    wind_fi_total_gw=wind_total_gw,
    solar_fi_gw=solar_cap_gw,
    interconnect_fi_se_mw=float(fi_se_mw_cap),
    interconnect_fi_ee_mw=float(fi_ee_mw_cap),
)

max_hintaero = compute_max_hintaero(scenario_params)


# ── Laskenta välimuistilla ────────────────────────────────────────────────────

params_key = (
    n_simulations, crisis_prob,
    wind_fi_gw, solar_fi_gw, nuclear_fi,
    hydro_nordic, nuclear_se,
    gas_price, co2_price,
    electrification_twh, ev_twh,
    datacenter_base, datacenter_growth,
    interconnect_fi_se, interconnect_fi_ee, interconnect_no,
    tuple(sorted(fundamental_df.columns.tolist())) if not fundamental_df.empty else (),
    len(fundamental_df),
)

if "last_params" not in st.session_state or st.session_state.last_params != params_key:
    hist_df = get_historical_data()

    _progress_bar = st.progress(0.0, text="Aloitetaan Monte Carlo -simulaatio...")

    try:
        import time as _t
        _progress_bar.progress(0.15, text="Lasketaan matala-skenaariota...")
        _t0 = _t.monotonic()

        scenario_results = run_scenarios_cached(
            params_key,
            scenario_params,
            regression if has_excel else RegressionResult(),
            timeout_seconds=30.0,
        )

        _elapsed = _t.monotonic() - _t0
        _laskettu = len(scenario_results)

        if _elapsed < 0.05:
            _progress_bar.progress(1.0, text=f"Välimuisti ({_laskettu}/3 skenaariota)")
        else:
            _progress_bar.progress(1.0, text=f"Valmis — {_laskettu}/3 skenaariota ({_elapsed:.2f} s)")

        if _laskettu < 3:
            st.warning(
                f"Timeout (30 s) ylittyi — vain {_laskettu}/3 skenaariota laskettu. "
                "Vähennä simulaatioiden määrää tai yksinkertaista parametreja.",
            )

    except Exception as e:
        _progress_bar.empty()
        st.error(f"Skenaariomalli epäonnistui: {e}")
        st.stop()

    st.session_state.update({
        "last_params":      params_key,
        "hist_df":          hist_df,
        "scenario_results": scenario_results,
    })

hist_df          = st.session_state["hist_df"]
scenario_results = st.session_state["scenario_results"]

# Riskimittarit (lasketaan aina — nopea)
risk_metrics = {
    sc: calculate_risk_metrics(scenario_results[sc], vol_mwh)
    for sc in ["matala", "perus", "korkea"]
    if sc in scenario_results
}
risk_summary  = get_hedge_recommendation(risk_metrics)

# Hedgausanalyysi
hedge_results = calculate_all_hedges(scenario_results["perus"], hedge_params)
hedge_df      = calculate_active_hedge(scenario_results["perus"], hedge_params)

# Strategiatunnus → etiketti
strategy_label_map = {
    "spot":            "Täysi spot",
    "kiintea_100":     "Kiinteä hinta 100%",
    "osittainen_5050": "Osittainen 50/50",
    "collar":          "Collar-strategia",
    "termiini":        "Termiinisuojaus",
    "yhdistelma":      "Yhdistelmästrategia",
}
active_label = strategy_label_map.get(hedge_strategy, hedge_strategy)


# ══════════════════════════════════════════════════════════════════════════════
# VÄLILEHDET
# ══════════════════════════════════════════════════════════════════════════════

_tab_labels = [
    "Hintaskenaariot",
    "Markkinadynamiikka",
    "Riskianalyysi",
    "Kuukausianalyysi",
    "Raportti",
]
if has_excel:
    _tab_labels = ["Data-analyysi"] + _tab_labels

tabs = st.tabs(_tab_labels)
tab_offset = 1 if has_excel else 0


# ══════════════════════════════════════════════════════════════════════════════
# VÄLILEHTI 0 – DATA-ANALYYSI (vain kun Excel ladattu)
# ══════════════════════════════════════════════════════════════════════════════
if has_excel:
    with tabs[0]:
        st.subheader("Historiallinen fundamenttidata")
        st.caption(
            f"Tiedosto: **{uploaded_file.name}** | "
            f"Välilehti: **{excel_meta.get('käytetty_välilehti', '?')}** | "
            f"{len(fundamental_df)} kuukautta"
        )

        try:
            st.plotly_chart(fundamental_time_series(fundamental_df), use_container_width=True)
        except Exception as e:
            st.error(f"Aikasarjakaavio epäonnistui: {e}")

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Korrelaatiomatriisi")
            try:
                st.plotly_chart(correlation_heatmap(fundamental_df), use_container_width=True)
            except Exception as e:
                st.error(f"Korrelaatiomatriisi epäonnistui: {e}")

        with col_right:
            st.subheader("Regressiomallin kertoimet")
            if regression.r2 > 0:
                st.success(
                    f"Malli selittää **{regression.r2*100:.1f}%** "
                    f"historiallisesta hintavaihtelusta (R² = {regression.r2:.3f})"
                )
                try:
                    st.plotly_chart(regression_coef_chart(regression), use_container_width=True)
                except Exception as e:
                    st.error(f"Kerroinkuvaaja epäonnistui: {e}")

                with st.expander("Regressiokertoimet taulukossa"):
                    coef_df = pd.DataFrame(
                        {"Piirre": list(regression.coef.keys()),
                         "Kerroin": list(regression.coef.values())}
                    ).sort_values("Kerroin", ascending=False)
                    coef_df["Kerroin"] = coef_df["Kerroin"].round(4)
                    st.dataframe(coef_df, use_container_width=True, hide_index=True)
            else:
                st.info(
                    "Regressiomallia ei voitu kalibroida. "
                    "Tarvitaan vähintään hintasarake ja jokin selittävä muuttuja.",
                    icon="ℹ️",
                )

        with st.expander("Näytä raakadata (ensimmäiset 20 riviä)"):
            st.dataframe(fundamental_df.head(20), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# VÄLILEHTI 1 – HINTASKENAARIOT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[tab_offset]:
    st.subheader("Sähköhinnan skenaariopolut 2025–2035")

    if has_excel and regression.r2 > 0:
        st.success(
            f"Skenaariot kalibroitu omalla datalla (R² = {regression.r2:.3f}).",
            icon="✅",
        )
    else:
        st.info(
            "Käytetään markkinaparametrien oletuksia. "
            "Lataa Excel kalibroinnin aktivoimiseksi.",
            icon="ℹ️",
        )

    col_cb1, col_cb2, col_cb3 = st.columns(3)
    show_matala = col_cb1.checkbox("Matala (optimistinen)", value=True)
    show_perus  = col_cb2.checkbox("Perus (todennäköisin)", value=True)
    show_korkea = col_cb3.checkbox("Korkea (riskiskenaario)", value=True)

    visible = (
        (["matala"] if show_matala else []) +
        (["perus"]  if show_perus  else []) +
        (["korkea"] if show_korkea else [])
    )

    try:
        fig_price = price_scenario_chart(scenario_results, hist_df, visible)
        st.plotly_chart(fig_price, use_container_width=True)
    except Exception as e:
        st.error(f"Kuvaajan piirto epäonnistui: {e}")

    st.caption(
        "Viivat: P50-mediaani. Varjostusalue: P10–P90 epävarmuusvyöhyke. "
        "Pisteviivainen: historiallinen 2015–2024 (synteettinen)."
    )

    st.subheader("Vuosikeskihinnat per skenaario (€/MWh, P50)")
    ann_price = {
        sc: result.annual_prices.set_index("year")["p50"]
        for sc, result in scenario_results.items()
    }
    price_table = pd.DataFrame(ann_price).rename(columns={
        "matala": "Matala", "perus": "Perus", "korkea": "Korkea"
    })
    price_table.index.name = "Vuosi"

    try:
        st.dataframe(
            price_table.style.format("{:.1f}").background_gradient(cmap="RdYlGn_r", axis=None),
            use_container_width=True,
        )
    except Exception:
        st.dataframe(price_table.round(1), use_container_width=True)

    # Avainmittarit
    st.divider()
    st.subheader("Avainmittarit 2025–2035 (perusskenaario)")
    perus_r = scenario_results.get("perus")
    if perus_r is not None:
        p_2025 = float(perus_r.annual_prices[perus_r.annual_prices["year"] == 2025]["p50"].values[0])
        p_2035 = float(perus_r.annual_prices[perus_r.annual_prices["year"] == 2035]["p50"].values[0])
        p_avg  = float(perus_r.annual_prices["p50"].mean())
        p_max  = float(perus_r.monthly_prices["p95"].max())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Hinta 2025 (P50)", f"{p_2025:.1f} €/MWh")
        c2.metric("Hinta 2035 (P50)", f"{p_2035:.1f} €/MWh", f"{p_2035 - p_2025:+.1f} €/MWh")
        c3.metric("Keskihinta 2025–35", f"{p_avg:.1f} €/MWh")
        c4.metric("Korkein kk-hinta P95", f"{p_max:.1f} €/MWh")


# ══════════════════════════════════════════════════════════════════════════════
# VÄLILEHTI 2 – MARKKINADYNAMIIKKA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[tab_offset + 1]:
    st.subheader("Markkinadynamiikka – Suomi 2025–2035")

    # ── OSIO 1: Tornado + Datakeskukset ──────────────────────────────────────
    col_t1, col_t2 = st.columns([1.5, 1])

    with col_t1:
        st.markdown("### Muuttujien hinnavaikutus (tornado)")
        try:
            sens_df = compute_variable_sensitivities(scenario_params)
            st.plotly_chart(tornado_chart(sens_df), use_container_width=True)
            st.caption(
                "Pylväät näyttävät, kuinka paljon muuttujan ääriarvot muuttavat "
                "FI-hintaa perusskenaariossa vuonna 2030 (€/MWh)."
            )
        except Exception as e:
            st.error(f"Tornado-kaavio epäonnistui: {e}")

    with col_t2:
        st.markdown("### Datakeskusten kasvu")
        try:
            dc_df = compute_datacenter_projection(scenario_params)
            st.plotly_chart(datacenter_growth_chart(dc_df), use_container_width=True)
            dc_2035 = dc_df[dc_df["year"] == 2035]["twh"].values[0]
            st.metric(
                "Datakeskukset 2035",
                f"{dc_2035:.1f} TWh",
                f"+{dc_2035 - datacenter_base:.1f} TWh lähtötasosta",
            )
        except Exception as e:
            st.error(f"Datakeskuskaavio epäonnistui: {e}")

    st.divider()

    # ── OSIO 2: Merit Order -kaavio ───────────────────────────────────────────
    st.markdown("### Merit Order -käyrä")
    st.caption(
        "Merit order -käyrä näyttää tuotantomuotojen järjestyksen marginaalikustannuksen mukaan. "
        "Markkinahinta määräytyy siitä tuotantomuodosta, jota tarvitaan kysynnän täyttämiseen."
    )

    mo_col1, mo_col2 = st.columns([3, 1])
    with mo_col2:
        mo_month = st.slider(
            "Kuukausi merit orderille",
            1, 12, 1,
            format="%d",
            key="mo_month_slider",
        )
        st.caption(f"Valittu: **{MONTH_NAMES_FI[mo_month]}**")
        mo_water = st.selectbox(
            "Vesivarat",
            options=["normaali", "kuiva", "märkä"],
            format_func=lambda x: {"normaali": "Normaali", "kuiva": "Kuiva", "märkä": "Märkä"}[x],
            key="mo_water_select",
        )
        mo_demand_mw = st.slider(
            "Kysyntä (MW)",
            2000, 14000, 8000, 100,
            key="mo_demand_slider",
        )

    with mo_col1:
        try:
            # Rakenna kapasiteetti merit orderille
            mo_cap_dict = {
                "tuuli":     wind_total_gw * 1000 * {
                    1:0.35,2:0.35,3:0.35,4:0.28,5:0.28,6:0.22,
                    7:0.22,8:0.22,9:0.32,10:0.32,11:0.32,12:0.35
                }.get(mo_month, 0.28),
                "aurinko":   solar_cap_gw * 1000 * {
                    1:0.02,2:0.04,3:0.08,4:0.12,5:0.16,6:0.18,
                    7:0.17,8:0.15,9:0.10,10:0.05,11:0.02,12:0.01
                }.get(mo_month, 0.05),
                "ydinvoima": NUCLEAR_OPTIONS_MW.get(nuclear_fi, 2500) * 0.90,
                "vesivoima": {  # Fingrid historia 2015–2024 (MW)
                    1:1484,2:1688,3:1622,4:1589,5:1957,6:1544,
                    7:1594,8:1369,9:1412,10:1588,11:1511,12:1596
                }.get(mo_month, 1500) * 0.6,
                "chp":       {
                    1:1484,2:1688,3:1622,4:1589,5:1957,6:1544,
                    7:1594,8:1369,9:1412,10:1588,11:1511,12:1596
                }.get(mo_month, 1500) * 0.4,
                "tuonti":    float(fi_se_mw_cap + fi_ee_mw_cap) * 0.7,
                "kaasu":     1500.0,
            }

            mo_params = MeritOrderParams(
                gas_price_mwh=float(gas_price),
                co2_price_t=float(co2_price),
                water_level=mo_water,
                nordpool_ref=55.0,
                month=mo_month,
            )
            mo_slices = build_merit_order(mo_params, mo_cap_dict)
            mo_df = merit_order_to_df(mo_slices)

            # Laske markkinahinta
            mo_price, mo_source, mo_surplus = calculate_market_price(
                mo_month, mo_cap_dict, mo_demand_mw,
                mo_water, float(gas_price), float(co2_price),
            )

            # Piirrä stepwise merit order -kaavio
            fig_mo = go.Figure()

            cum_prev = 0.0
            for _, row in mo_df.iterrows():
                src = row["source"]
                color = SOURCE_COLORS.get(src, "#9E9E9E")
                cum_now = row["cumulative_mw"]
                mc = row["marginal_cost"]

                # Väri rgba-muodossa
                hex_c = color.lstrip("#")
                r_c = int(hex_c[0:2], 16)
                g_c = int(hex_c[2:4], 16)
                b_c = int(hex_c[4:6], 16)
                fill_rgba = f"rgba({r_c},{g_c},{b_c},0.7)"

                # Step-bar kahdella pisteellä (vasempi reunasta, oikea reunaan)
                fig_mo.add_trace(go.Scatter(
                    x=[cum_prev, cum_prev, cum_now, cum_now],
                    y=[0, mc, mc, 0],
                    fill="toself",
                    fillcolor=fill_rgba,
                    line=dict(color=color, width=1.5),
                    name=src,
                    hovertemplate=(
                        f"<b>{src}</b><br>"
                        f"Kapasiteetti: {row['capacity_mw']:.0f} MW<br>"
                        f"Marginaalikustannus: {mc:.1f} €/MWh<br>"
                        f"Kumulatiivinen: {cum_now:.0f} MW<extra></extra>"
                    ),
                    showlegend=True,
                ))
                cum_prev = cum_now

            # Kysynnän viiva
            fig_mo.add_vline(
                x=mo_demand_mw,
                line_dash="dash",
                line_color="#E53935",
                line_width=2,
                annotation_text=f"Kysyntä {mo_demand_mw:,} MW",
                annotation_position="top left",
            )

            # Markkinahintaviiva
            fig_mo.add_hline(
                y=mo_price,
                line_dash="dot",
                line_color="#FF6F00",
                line_width=2,
                annotation_text=f"Markkinahinta {mo_price:.1f} €/MWh ({mo_source})",
                annotation_position="bottom right",
            )

            fig_mo.update_layout(
                title=f"Merit order – {MONTH_NAMES_FI[mo_month]} | Vesi: {mo_water}",
                xaxis_title="Kumulatiivinen kapasiteetti (MW)",
                yaxis_title="Marginaalikustannus (€/MWh)",
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=420,
                margin=dict(l=60, r=20, t=80, b=60),
                xaxis=dict(gridcolor="#E0E0E0"),
                yaxis=dict(gridcolor="#E0E0E0"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig_mo, use_container_width=True)

            # Markkinahintatulos
            surplus_text = f"+{mo_surplus:.0f} MW ylijäämä" if mo_surplus >= 0 else f"{mo_surplus:.0f} MW alijäämä"
            col_mo_a, col_mo_b, col_mo_c = st.columns(3)
            col_mo_a.metric("Merit order -hinta", f"{mo_price:.1f} €/MWh")
            col_mo_b.metric("Rajakustannusmuoto", mo_source)
            col_mo_c.metric("Kapasiteettitilanne", surplus_text)

        except Exception as e:
            st.error(f"Merit order -kaavio epäonnistui: {e}")

    st.divider()

    # ── OSIO 3: Kapasiteettimalli ─────────────────────────────────────────────
    st.markdown("### Kapasiteetti vs kysyntä 2025–2035")

    try:
        cap_ts = capacity_time_series(capacity_params, 2025, 2035)

        # Laske kysyntä kuukausittain (MW huipputeho)
        base_demand_mw = (FI_BASE_CONSUMPTION_TWH + total_growth / 2) * 1e6 / 8760
        monthly_demand_factor_cap = {
            1:1.35,2:1.30,3:1.10,4:0.95,5:0.85,6:0.80,
            7:0.82,8:0.88,9:0.95,10:1.05,11:1.22,12:1.38
        }
        cap_ts["demand_mw"] = cap_ts["month"].map(
            lambda m: base_demand_mw * monthly_demand_factor_cap.get(m, 1.0) * 1.6
        )
        cap_ts["label"] = cap_ts.apply(
            lambda r: f"{int(r['year'])}-{int(r['month']):02d}", axis=1
        )
        cap_ts["surplus"] = cap_ts["total_mw"] + cap_ts["interconnect_mw"] - cap_ts["demand_mw"]

        fig_cap = go.Figure()

        # Tuotantomuodot pinottuna
        cap_colors = {
            "nuclear_mw":   "rgba(156,39,176,0.8)",
            "wind_mw":      "rgba(33,150,243,0.8)",
            "solar_mw":     "rgba(255,193,7,0.8)",
            "hydro_chp_mw": "rgba(0,188,212,0.8)",
        }
        cap_labels = {
            "nuclear_mw":   "Ydinvoima",
            "wind_mw":      "Tuulivoima",
            "solar_mw":     "Aurinkovoima",
            "hydro_chp_mw": "Vesivoima + CHP",
        }

        for col_key in ["nuclear_mw", "wind_mw", "solar_mw", "hydro_chp_mw"]:
            fig_cap.add_trace(go.Bar(
                x=cap_ts["label"],
                y=cap_ts[col_key],
                name=cap_labels[col_key],
                marker_color=cap_colors[col_key],
                hovertemplate=f"%{{x}}: %{{y:.0f}} MW<extra>{cap_labels[col_key]}</extra>",
            ))

        # Siirtoyhteydet
        fig_cap.add_trace(go.Scatter(
            x=cap_ts["label"],
            y=cap_ts["total_mw"] + cap_ts["interconnect_mw"],
            name="Tuotanto + siirtoyhteydet",
            line=dict(color="rgba(0,150,136,1)", width=2, dash="dash"),
            hovertemplate="%{x}: %{y:.0f} MW<extra>Kokonaistarjonta</extra>",
        ))

        # Kysyntä
        fig_cap.add_trace(go.Scatter(
            x=cap_ts["label"],
            y=cap_ts["demand_mw"],
            name="Huipputehokysyntä",
            line=dict(color="rgba(229,57,53,1)", width=2.5),
            hovertemplate="%{x}: %{y:.0f} MW<extra>Kysyntä</extra>",
        ))

        fig_cap.update_layout(
            title="Kapasiteetti vs kysyntä 2025–2035 (kuukausittain)",
            xaxis_title="Kuukausi",
            yaxis_title="Teho (MW)",
            barmode="stack",
            height=480,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=60, r=20, t=80, b=60),
            xaxis=dict(
                tickangle=-45,
                tickmode="array",
                tickvals=[f"{y}-01" for y in range(2025, 2036)],
                ticktext=[str(y) for y in range(2025, 2036)],
                gridcolor="#E0E0E0",
            ),
            yaxis=dict(gridcolor="#E0E0E0"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_cap, use_container_width=True)

        # Kriittiset kuukaudet
        crit_df = find_critical_months(capacity_params, FI_BASE_CONSUMPTION_TWH + total_growth / 2)
        if not crit_df.empty:
            st.warning(
                f"Löydettiin {len(crit_df)} kapasiteettialijäämäkuukautta 2025–2035.",
                icon="⚠️",
            )
            crit_display = crit_df.copy()
            crit_display["Kuukausi"] = crit_display["month"].map(
                lambda m: MONTH_NAMES_FI.get(m, str(m))
            )
            crit_display = crit_display.rename(columns={
                "year": "Vuosi",
                "surplus_mw": "Alijäämä (MW)",
                "utilization_pct": "Käyttöaste (%)",
                "premium_eur_mwh": "Hintapreemio (€/MWh)",
            })
            st.dataframe(
                crit_display[["Vuosi", "Kuukausi", "Alijäämä (MW)", "Käyttöaste (%)", "Hintapreemio (€/MWh)"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("Ei kapasiteettialijäämäkuukausia valituilla parametreilla.", icon="✅")

    except Exception as e:
        st.error(f"Kapasiteettikaavio epäonnistui: {e}")

    st.divider()

    # ── Kulutuskomponentit 2025 (Fingrid-data) ────────────────────────────────
    st.markdown("### Sähkönkulutuksen rakenne 2025 (Fingrid)")
    col_pie, col_monthly = st.columns(2)

    with col_pie:
        labels = list(FI_CONSUMPTION_BREAKDOWN_2025.keys())
        values = list(FI_CONSUMPTION_BREAKDOWN_2025.values())
        colors = ["#1565C0","#42A5F5","#26C6DA","#66BB6A","#FFA726","#EF5350","#9E9E9E"]
        fig_pie = go.Figure(go.Pie(
            labels=labels, values=values,
            marker=dict(colors=colors),
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:.1f} TWh<extra></extra>",
        ))
        fig_pie.update_layout(
            title=f"Kulutus yhteensä {sum(values):.1f} TWh",
            height=350, showlegend=False,
            margin=dict(t=40, b=0, l=0, r=0),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_monthly:
        months_fi = ["Tam","Hel","Maa","Huh","Tou","Kes","Hei","Elo","Syy","Lok","Mar","Jou"]
        monthly_vals = [FI_MONTHLY_CONSUMPTION_2025[m] for m in range(1, 13)]
        fig_bar = go.Figure(go.Bar(
            x=months_fi, y=monthly_vals,
            marker_color="#1565C0",
            hovertemplate="%{x}: %{y:.2f} TWh<extra></extra>",
        ))
        fig_bar.update_layout(
            title="Kuukausittainen kulutus 2025 (TWh)",
            yaxis_title="TWh", height=350,
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor="#E0E0E0"),
            margin=dict(t=40, b=0, l=0, r=0),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ── OSIO 4: Kulutuskasvun yhteenveto + siirtoyhteydet ────────────────────
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("### Kulutuskasvun yhteenveto")
        growth_info = compute_consumption_growth(scenario_params)
        st.metric("Kulutus 2025 (nykytaso)", f"{growth_info['nyky_twh']:.0f} TWh")
        st.metric(
            "Kulutus 2035 (ennuste)",
            f"{growth_info['ennuste_2035_twh']:.0f} TWh",
            f"+{growth_info['kasvu_twh']:.0f} TWh",
        )
        st.progress(min(growth_info["kasvu_twh"] / 40.0, 1.0))
        st.caption(
            f"Sähköistyminen: **{electrification_twh} TWh** | "
            f"Sähköautot: **{ev_twh} TWh** | "
            f"Datakeskukset: **+{dc_final_disp - datacenter_base:.1f} TWh**"
        )

        st.markdown("#### Hintavaikutus – parametrien vaikutus perusskenaarioon")
        _impact2 = compute_impact_breakdown(scenario_params, ref_years=(2030, 2035))
        _cols_imp = st.columns(2)
        for _ci, _yr in enumerate((2030, 2035)):
            _d = _impact2[_yr]
            with _cols_imp[_ci]:
                st.markdown(f"**Vuosi {_yr}**")
                st.metric("Perusskenaario P50", f"{_d['base_price']:.1f} €/MWh")

                def _sign_str(v: float) -> str:
                    return f"+{v:.1f}" if v >= 0 else f"{v:.1f}"

                st.markdown(
                    f"| Tekijä | Vaikutus |\n"
                    f"|--------|----------|\n"
                    f"| Kulutuskasvu ({_d['total_growth_twh']:.0f} TWh) "
                    f"| **{_sign_str(_d['kulutus_impact'])} €/MWh** |\n"
                    f"| Datakeskukset (+{_d['dc_kasvu_twh']:.1f} TWh) "
                    f"| **{_sign_str(_d['datacenter_impact'])} €/MWh** |\n"
                    f"| Tuulivoima (+{_d['wind_re_twh']:.0f} TWh/v) "
                    f"| **{_sign_str(_d['wind_impact'])} €/MWh** |"
                )

    with col_m2:
        st.markdown("### Siirtoyhteydet – hintaero FI vs Pohjoismaat")
        _, fi_se_mw_disp = INTERCONNECT_FI_SE_OPTIONS.get(interconnect_fi_se, ("", 2200))
        _, fi_ee_mw_disp = INTERCONNECT_FI_EE_OPTIONS.get(interconnect_fi_ee, ("", 1000))
        total_ic_gw = (fi_se_mw_disp + fi_ee_mw_disp) / 1000.0
        st.metric(
            "Max sallittu hintaero FI–Pohjoismaat",
            f"{max_hintaero:.0f} €/MWh",
            help="500 / kokonaiskapasiteetti_GW",
        )
        st.metric(
            "Kokonaiskapasiteetti",
            f"{total_ic_gw:.1f} GW",
            f"FI–SE: {fi_se_mw_disp:,} MW + FI–EE: {fi_ee_mw_disp:,} MW",
        )
        try:
            fig_ic = interconnect_hintaero_chart(scenario_results, max_hintaero)
            st.plotly_chart(fig_ic, use_container_width=True)
        except Exception as e:
            st.error(f"Hintaerokaavio epäonnistui: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# VÄLILEHTI 3 – RISKIANALYYSI
# ══════════════════════════════════════════════════════════════════════════════
with tabs[tab_offset + 2]:
    st.subheader("Riskianalyysi ja hedgausstrategiat")

    # ── OSIO 1: Hintariski ilman hedgausta ───────────────────────────────────
    st.markdown("### Osio 1 – Hintariski ilman hedgausta")

    risk_scenario = st.selectbox(
        "Skenaario riskianalyysiin",
        options=["matala", "perus", "korkea"],
        format_func=lambda x: {"matala": "Matala", "perus": "Perus", "korkea": "Korkea"}[x],
        index=1,
        key="risk_scenario_select",
    )

    try:
        fig_pct = price_percentile_paths(scenario_results, risk_scenario)
        st.plotly_chart(fig_pct, use_container_width=True)
    except Exception as e:
        st.error(f"Persentiilipolut epäonnistui: {e}")

    rm = risk_metrics.get(risk_scenario)
    if rm:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("VaR 95% (€/MWh)", f"{rm.var_95:.1f}")
        c2.metric("CVaR 95% (€/MWh)", f"{rm.cvar_95:.1f}")
        c3.metric("Max kk-hinta (€/MWh)", f"{rm.max_monthly_price:.1f}")
        c4.metric("Volatiliteetti (€/MWh)", f"{rm.volatility:.1f}")
        c5.metric(
            "Hintapiikkiriski",
            f"{rm.spike_prob * 100:.1f}%",
            help="Todennäköisyys että jokin kuukausi ylittää 150 €/MWh",
        )

    st.markdown("#### Riskimittarit kaikille skenaarioille")
    risk_table = build_risk_metrics_table(risk_metrics)
    try:
        st.dataframe(
            risk_table.style.format({
                "VaR 95% (€/MWh)": "{:.1f}",
                "CVaR 95% (€/MWh)": "{:.1f}",
                "Max kk-hinta (€/MWh)": "{:.1f}",
                "Volatiliteetti (€/MWh)": "{:.1f}",
                "Hintapiikkiriski (%)": "{:.1f}",
            }).background_gradient(
                cmap="RdYlGn_r",
                subset=["CVaR 95% (€/MWh)", "Hintapiikkiriski (%)"],
            ),
            use_container_width=True,
            hide_index=True,
        )
    except Exception:
        st.dataframe(risk_table, use_container_width=True, hide_index=True)

    st.divider()

    # ── OSIO 2: Hedgausstrategioiden vertailu ────────────────────────────────
    st.markdown("### Osio 2 – Hedgausstrategioiden vertailu")
    st.caption(f"Perusskenaario | Volyymi: {vol_mwh:,.0f} MWh/v")

    try:
        fig_hedge = hedge_comparison_chart(hedge_results, vol_mwh)
        st.plotly_chart(fig_hedge, use_container_width=True)
    except Exception as e:
        st.error(f"Hedgausvertailu epäonnistui: {e}")

    hedge_table = pd.DataFrame([{
        "Strategia":                      h.strategy_name,
        "Hinta P50 (€/MWh)":             h.effective_price_p50,
        "Hinta P95 (€/MWh)":             h.effective_price_p95,
        "Kustannus P50 (k€/v)":          round(h.annual_cost_p50 / 1000, 1),
        "Kustannus P95 (k€/v)":          round(h.annual_cost_p95 / 1000, 1),
        "Lisäkustannus vs spot P50 (k€)": round(h.hedge_cost_vs_spot_p50 / 1000, 1),
        "Säästö vs spot P95 (k€)":       round(h.hedge_benefit_vs_spot_p95 / 1000, 1),
        "Riskivähennys (%)":             h.risk_reduction_ratio,
    } for h in hedge_results])

    try:
        st.dataframe(
            hedge_table.style
            .highlight_min(subset=["Kustannus P95 (k€/v)"], color="#C8E6C9")
            .highlight_max(subset=["Riskivähennys (%)"], color="#C8E6C9")
            .format({
                "Hinta P50 (€/MWh)": "{:.1f}",
                "Hinta P95 (€/MWh)": "{:.1f}",
                "Kustannus P50 (k€/v)": "{:.1f}",
                "Kustannus P95 (k€/v)": "{:.1f}",
                "Lisäkustannus vs spot P50 (k€)": "{:+.1f}",
                "Säästö vs spot P95 (k€)": "{:+.1f}",
                "Riskivähennys (%)": "{:.1f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
    except Exception:
        st.dataframe(hedge_table, use_container_width=True, hide_index=True)

    try:
        fig_h_annual = hedge_annual_cost_chart(hedge_df, active_label, color="#1565C0")
        st.plotly_chart(fig_h_annual, use_container_width=True)
    except Exception as e:
        st.error(f"Hedgauksen efektiivinen hinta -kaavio epäonnistui: {e}")

    st.divider()

    # ── OSIO 3: Stressitestit ─────────────────────────────────────────────────
    st.markdown("### Osio 3 – Stressitestit")

    with st.spinner("Ajetaan stressitestejä..."):
        stress_tests = run_stress_tests(scenario_params, vol_mwh)

    try:
        fig_stress = stress_test_chart(stress_tests)
        st.plotly_chart(fig_stress, use_container_width=True)
    except Exception as e:
        st.error(f"Stressitestikaavio epäonnistui: {e}")

    stress_table = pd.DataFrame([{
        "Skenaario":              t.name,
        "Kuvaus":                 t.description,
        "Hintapiikki (€/MWh)":   t.price_spike,
        "Nousua (%)":             t.price_increase_pct,
        "Kesto (kk)":             t.duration_months,
        "Kustannusvaikutus (k€)": round(t.annual_cost_impact / 1000, 1),
        "Paras hedge":            t.best_hedge,
    } for t in stress_tests])

    try:
        st.dataframe(
            stress_table.style.format({
                "Hintapiikki (€/MWh)": "{:.1f}",
                "Nousua (%)": "{:.1f}",
                "Kustannusvaikutus (k€)": "{:.1f}",
            }).background_gradient(
                cmap="RdYlGn_r",
                subset=["Hintapiikki (€/MWh)"],
            ),
            use_container_width=True,
            hide_index=True,
        )
    except Exception:
        st.dataframe(stress_table, use_container_width=True, hide_index=True)

    st.divider()

    # ── OSIO 4: Efficient frontier ja suositus ────────────────────────────────
    st.markdown("### Osio 4 – Optimaalinen hedgaussuositus")

    st.info(risk_summary.get("suositus_teksti", ""), icon="💡")

    col_ef1, col_ef2 = st.columns([2, 1])
    with col_ef1:
        try:
            frontier_df = compute_efficient_frontier(
                scenario_results["perus"],
                vol_mwh,
                fixed_price=fixed_price,
                floor_price=floor_price,
                cap_price=cap_price,
                forward_premium=fwd_premium,
            )
            fig_ef = efficient_frontier_chart(frontier_df)
            st.plotly_chart(fig_ef, use_container_width=True)
        except Exception as e:
            st.error(f"Efficient frontier epäonnistui: {e}")

    with col_ef2:
        st.markdown("**Riskiparametrit (perusskenaario)**")
        pm = risk_metrics.get("perus")
        if pm:
            st.metric("Riskitaso", risk_summary.get("risk_class", "–").capitalize())
            st.metric("VaR 95%", f"{pm.var_95:.1f} €/MWh")
            st.metric("CVaR 95%", f"{pm.cvar_95:.1f} €/MWh")
            st.metric("Volatiliteetti", f"{pm.volatility:.1f} €/MWh")
        st.markdown("**Paras strategia (CVaR-min)**")
        st.success(risk_summary.get("paras_strategia", "–"))


# ══════════════════════════════════════════════════════════════════════════════
# VÄLILEHTI 4 – KUUKAUSIANALYYSI
# ══════════════════════════════════════════════════════════════════════════════
with tabs[tab_offset + 3]:
    st.subheader("Kuukausittainen hinta-analyysi")

    hm_scenario = st.selectbox(
        "Skenaario heatmapille",
        options=["matala", "perus", "korkea"],
        format_func=lambda x: {"matala": "Matala", "perus": "Perus", "korkea": "Korkea"}[x],
        index=1,
        key="hm_scenario_select",
    )

    try:
        st.plotly_chart(monthly_heatmap(scenario_results, hm_scenario), use_container_width=True)
    except Exception as e:
        st.error(f"Heatmap epäonnistui: {e}")

    col_l, col_r = st.columns(2)

    with col_l:
        try:
            st.plotly_chart(monthly_avg_bar(scenario_results, hm_scenario), use_container_width=True)
        except Exception as e:
            st.error(f"Kuukausipylväs epäonnistui: {e}")

    with col_r:
        st.subheader("Kausi-optimointisuositus")
        ref = scenario_results["perus"].monthly_prices
        month_avg = ref.groupby("month")["p50"].mean()

        st.write("**Kalleimmat kuukaudet (perus):**")
        for m, p in month_avg.nlargest(3).items():
            st.write(f"- {MONTH_NAMES_FI[m]}: {p:.1f} €/MWh")

        st.write("**Halvimmat kuukaudet (perus):**")
        for m, p in month_avg.nsmallest(3).items():
            st.write(f"- {MONTH_NAMES_FI[m]}: {p:.1f} €/MWh")

        talvi_hinta = float(month_avg[[12, 1, 2]].mean())
        kesa_hinta  = float(month_avg[[6, 7, 8]].mean())
        diff = talvi_hinta - kesa_hinta

        if diff > 5:
            st.success(
                f"Kulutuksen siirto talvesta kesälle kannattaa:\n\n"
                f"- Talven P50-hinta: {talvi_hinta:.1f} €/MWh\n"
                f"- Kesän P50-hinta: {kesa_hinta:.1f} €/MWh\n"
                f"- Hintaero: {diff:.1f} €/MWh"
            )
        else:
            st.info("Talven ja kesän hintaero on pieni valituissa parametreissa.")

    st.divider()

    # ── Kuukausittainen merit order -aikasarja ────────────────────────────────
    st.markdown("### Kuukausittainen merit order -hinta (vuosi 2025)")
    st.caption(
        "Merit order -laskenta kullekin kuukaudelle nykyisillä kapasiteetti- ja kysyntäparametreilla."
    )

    try:
        mo_base_cap = {
            "tuuli":     wind_total_gw * 1000 * 0.28,
            "aurinko":   solar_cap_gw * 1000 * 0.08,
            "ydinvoima": NUCLEAR_OPTIONS_MW.get(nuclear_fi, 2500) * 0.90,
            "vesivoima": 1200.0,
            "chp":       800.0,
            "tuonti":    float(fi_se_mw_cap + fi_ee_mw_cap) * 0.7,
            "kaasu":     1500.0,
        }
        # Kausikorjattu kapasiteetti
        wind_cf_map = {
            1:0.35,2:0.35,3:0.35,4:0.28,5:0.28,6:0.22,
            7:0.22,8:0.22,9:0.32,10:0.32,11:0.32,12:0.35
        }
        solar_cf_map = {
            1:0.02,2:0.04,3:0.08,4:0.12,5:0.16,6:0.18,
            7:0.17,8:0.15,9:0.10,10:0.05,11:0.02,12:0.01
        }
        hydro_mw_map = {  # Fingrid historia 2015–2024, kuukausikeskiarvot (MW)
            1:1484,2:1688,3:1622,4:1589,5:1957,6:1544,
            7:1594,8:1369,9:1412,10:1588,11:1511,12:1596
        }
        demand_factor_map = {
            1:1.35,2:1.30,3:1.10,4:0.95,5:0.85,6:0.80,
            7:0.82,8:0.88,9:0.95,10:1.05,11:1.22,12:1.38
        }
        dc_twh_2030 = datacenter_base * ((1 + datacenter_growth / 100) ** 5)
        total_consumption_2030 = FI_BASE_CONSUMPTION_TWH + (electrification_twh + ev_twh) * 0.5 + max(dc_twh_2030 - datacenter_base, 0)
        avg_demand_mo = total_consumption_2030 * 1e6 / 8760

        mo_monthly_records = []
        for mo_m in range(1, 13):
            mo_m_cap = {
                "tuuli":     wind_total_gw * 1000 * wind_cf_map[mo_m],
                "aurinko":   solar_cap_gw * 1000 * solar_cf_map[mo_m],
                "ydinvoima": NUCLEAR_OPTIONS_MW.get(nuclear_fi, 2500) * 0.90,
                "vesivoima": hydro_mw_map[mo_m] * 0.6,
                "chp":       hydro_mw_map[mo_m] * 0.4,
                "tuonti":    float(fi_se_mw_cap + fi_ee_mw_cap) * 0.7,
                "kaasu":     1500.0,
            }
            mo_demand_m = avg_demand_mo * demand_factor_map[mo_m] * 1.6
            mo_p, mo_src, mo_sur = calculate_market_price(
                mo_m, mo_m_cap, mo_demand_m,
                hydro_nordic, float(gas_price), float(co2_price),
            )
            mo_monthly_records.append({
                "month": mo_m,
                "month_label": MONTH_SHORT_FI[mo_m],
                "mo_price": mo_p,
                "marginal_source": mo_src,
                "surplus_mw": mo_sur,
            })

        mo_monthly_df = pd.DataFrame(mo_monthly_records)

        # Kaavio
        bar_colors_mo = [
            SOURCE_COLORS.get(src, "#9E9E9E")
            for src in mo_monthly_df["marginal_source"]
        ]

        fig_mo_monthly = go.Figure()
        fig_mo_monthly.add_trace(go.Bar(
            x=mo_monthly_df["month_label"],
            y=mo_monthly_df["mo_price"],
            marker_color=bar_colors_mo,
            hovertemplate=(
                "%{x}: %{y:.1f} €/MWh"
                "<br>Rajakustannusmuoto: " +
                mo_monthly_df["marginal_source"].astype(str) +
                "<extra></extra>"
            ).tolist(),
            name="Merit order -hinta",
        ))

        # Lisää spot-hinta P50 vertailuksi
        perus_monthly_avg = (
            scenario_results["perus"].monthly_prices
            .groupby("month")["p50"]
            .mean()
            .reset_index()
        )
        fig_mo_monthly.add_trace(go.Scatter(
            x=perus_monthly_avg["month"].map(MONTH_SHORT_FI),
            y=perus_monthly_avg["p50"],
            name="Spot P50 (perusskenaario)",
            line=dict(color="#1565C0", width=2, dash="dot"),
            hovertemplate="%{x}: %{y:.1f} €/MWh<extra>Spot P50</extra>",
        ))

        fig_mo_monthly.update_layout(
            title="Kuukausittainen merit order -hinta vs spot P50",
            xaxis_title="Kuukausi",
            yaxis_title="Hinta (€/MWh)",
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=380,
            margin=dict(l=60, r=20, t=80, b=60),
            yaxis=dict(gridcolor="#E0E0E0"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_mo_monthly, use_container_width=True)

        # Taulukko
        mo_table = mo_monthly_df.rename(columns={
            "month_label": "Kuukausi",
            "mo_price": "Merit order -hinta (€/MWh)",
            "marginal_source": "Rajakustannusmuoto",
            "surplus_mw": "Kapasiteettiylijäämä (MW)",
        })
        st.dataframe(
            mo_table[["Kuukausi", "Merit order -hinta (€/MWh)", "Rajakustannusmuoto", "Kapasiteettiylijäämä (MW)"]].style.format({
                "Merit order -hinta (€/MWh)": "{:.1f}",
                "Kapasiteettiylijäämä (MW)": "{:.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    except Exception as e:
        st.error(f"Kuukausittainen merit order epäonnistui: {e}")

    st.divider()

    # ── Kulutuskäyrä 2025–2035 ────────────────────────────────────────────────
    st.markdown("### Sähkönkulutuksen kehitys 2025–2035")
    st.caption("Kulutusennuste perustuu sivupalkin parametreihin.")

    years_range = list(range(2025, 2036))
    base = FI_BASE_CONSUMPTION_TWH
    elec_growth = electrification_twh + ev_twh

    consumption_base  = [base] * len(years_range)
    consumption_elec  = [base + elec_growth * (y - 2025) / 10 for y in years_range]
    consumption_total = [
        base
        + elec_growth * (y - 2025) / 10
        + max(datacenter_base * ((1 + datacenter_growth / 100) ** (y - 2025)) - datacenter_base, 0)
        for y in years_range
    ]

    fig_cons = go.Figure()
    fig_cons.add_trace(go.Scatter(
        x=years_range, y=consumption_base,
        name="Peruskulutus (ei kasvua)",
        line=dict(color="#9E9E9E", width=1, dash="dot"),
        hovertemplate="%{x}: %{y:.1f} TWh<extra>Peruskulutus</extra>",
    ))
    fig_cons.add_trace(go.Scatter(
        x=years_range, y=consumption_elec,
        name="+ Sähköistyminen & sähköautot",
        line=dict(color="#1976D2", width=2),
        hovertemplate="%{x}: %{y:.1f} TWh<extra>Sähköistyminen</extra>",
    ))
    fig_cons.add_trace(go.Scatter(
        x=years_range, y=consumption_total,
        name="+ Datakeskukset (kokonaisennuste)",
        line=dict(color="#E53935", width=2),
        fill="tonexty",
        fillcolor="rgba(229,57,53,0.10)",
        hovertemplate="%{x}: %{y:.1f} TWh<extra>Kokonaisennuste</extra>",
    ))
    fig_cons.update_layout(
        xaxis_title="Vuosi",
        yaxis_title="Kulutus (TWh/vuosi)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=360,
        margin=dict(l=60, r=20, t=40, b=60),
        yaxis=dict(gridcolor="#E0E0E0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_cons, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# VÄLILEHTI 5 – RAPORTTI
# ══════════════════════════════════════════════════════════════════════════════
with tabs[tab_offset + 4]:
    st.subheader("Analyysiyhteenveto ja lataukset")

    data_notes = (
        excel_meta.get("oletukset", [])
        if has_excel
        else ["Käytetty synteettinen historiallinen data."]
    )

    try:
        summary = generate_summary_text(
            scenario_results, scenario_params, risk_summary, data_notes
        )
        st.info(summary)
    except Exception as e:
        st.error(f"Yhteenvedon generointi epäonnistui: {e}")

    if has_excel:
        st.success(
            f"**Käytetty oma data:** {uploaded_file.name} | "
            f"R² = {regression.r2:.3f} | "
            f"Välilehti: {excel_meta.get('käytetty_välilehti', '?')}",
            icon="✅",
        )
    else:
        st.warning(
            "Skenaariot perustuvat markkinaparametreihin ja synteettiseen dataan. "
            "Lataa Excel-tiedosto sivupalkista kalibroinnin aktivoimiseksi.",
            icon="⚠️",
        )

    st.divider()
    st.subheader("Lataa data ja raportit")

    col_d1, col_d2, col_d3 = st.columns(3)

    with col_d1:
        try:
            sc_csv = scenarios_to_dataframe(scenario_results)
            st.download_button(
                "Lataa scenarios_data.csv",
                data=sc_csv.to_csv(index=False, float_format="%.2f"),
                file_name="scenarios_data.csv",
                mime="text/csv",
                help="Kuukausittaiset hinnat P5–P95 kaikille skenaarioille",
            )
        except Exception as e:
            st.error(f"Skenaariotiedosto: {e}")

    with col_d2:
        try:
            risk_csv = build_risk_metrics_table(risk_metrics)
            st.download_button(
                "Lataa risk_analysis.csv",
                data=risk_csv.to_csv(index=False, float_format="%.2f"),
                file_name="risk_analysis.csv",
                mime="text/csv",
                help="Riskimittarit kaikille skenaarioille",
            )
        except Exception as e:
            st.error(f"Riskitiedosto: {e}")

    with col_d3:
        try:
            pdf_bytes = build_pdf_report(
                scenario_results=scenario_results,
                params=scenario_params,
                n_simulations=n_simulations,
                r2=regression.r2,
                risk_summary=risk_summary,
                hedge_results=hedge_results,
                data_notes=data_notes,
            )
            st.download_button(
                "Lataa PDF-raportti",
                data=pdf_bytes,
                file_name="sahkoskenaario_raportti.pdf",
                mime="application/pdf",
                help="Täydellinen PDF-raportti",
            )
        except Exception as e:
            st.error(f"PDF-raportti: {e}")

    # Merit order -yhteenveto
    st.divider()
    st.subheader("Merit order -hintayhteenveto (kaikki kuukaudet)")
    try:
        mo_all_cap_avg = {
            "tuuli":     wind_total_gw * 1000 * 0.28,
            "aurinko":   solar_cap_gw * 1000 * 0.06,
            "ydinvoima": NUCLEAR_OPTIONS_MW.get(nuclear_fi, 2500) * 0.90,
            "vesivoima": 1200.0,
            "chp":       800.0,
            "tuonti":    float(fi_se_mw_cap + fi_ee_mw_cap) * 0.7,
            "kaasu":     1500.0,
        }
        mo_ts = merit_order_time_series(
            range(1, 13),
            mo_all_cap_avg,
            FI_BASE_CONSUMPTION_TWH * 1e6 / 8760 * 1.4,
            float(gas_price),
            float(co2_price),
            hydro_nordic,
        )
        mo_ts["month_label"] = mo_ts["month"].map(MONTH_SHORT_FI)
        mo_ts_display = mo_ts.rename(columns={
            "month_label": "Kuukausi",
            "price_eur_mwh": "Merit order -hinta (€/MWh)",
            "marginal_source": "Rajakustannusmuoto",
            "surplus_mw": "Ylijäämä (MW)",
        })
        st.dataframe(
            mo_ts_display[["Kuukausi", "Merit order -hinta (€/MWh)", "Rajakustannusmuoto", "Ylijäämä (MW)"]].style.format({
                "Merit order -hinta (€/MWh)": "{:.1f}",
                "Ylijäämä (MW)": "{:.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
    except Exception as e:
        st.error(f"Merit order -yhteenveto: {e}")

    st.divider()
    with st.expander("Markkinaparametrit"):
        nuclear_fi_label = NUCLEAR_FI_OPTIONS.get(nuclear_fi, ("–",))[0]
        nuclear_se_label = NUCLEAR_SE_OPTIONS.get(nuclear_se, ("–",))[0]
        hydro_label      = HYDRO_OPTIONS.get(hydro_nordic, ("–",))[0]
        fi_se_label      = INTERCONNECT_FI_SE_OPTIONS.get(interconnect_fi_se, ("–",))[0]
        fi_ee_label      = INTERCONNECT_FI_EE_OPTIONS.get(interconnect_fi_ee, ("–",))[0]

        param_info = {
            "Tuulivoiman lisäkapasiteetti":   f"{wind_fi_gw:.1f} GW",
            "Tuulivoima yhteensä (kapasiteetti)": f"{wind_total_gw:.1f} GW",
            "Aurinkoenergian kasvu":          f"{solar_fi_gw:.1f} GW",
            "Aurinkovoima yhteensä":          f"{solar_cap_gw:.1f} GW",
            "Ydinvoima FI":                  nuclear_fi_label,
            "Vesivoima Pohjoismaat":          hydro_label,
            "Ruotsin ydinvoima":              nuclear_se_label,
            "Kaasun hinta":                  f"{gas_price} €/MWh",
            "CO₂-hinta":                     f"{co2_price} €/t",
            "Sähköistyminen + LPs":          f"{electrification_twh} TWh",
            "Sähköautot":                    f"{ev_twh} TWh",
            "Datakeskukset (lähtötaso)":      f"{datacenter_base:.1f} TWh",
            "Datakeskusten kasvuvauhti":      f"{datacenter_growth} %/v",
            "FI–SE siirtoyhteys":            fi_se_label,
            "FI–EE siirtoyhteys":            fi_ee_label,
            "Energiakriisin todennäköisyys": f"{crisis_prob*100:.0f}%",
            "Monte Carlo -ajoja":            str(n_simulations),
            "Hedgausstrategia":              active_label,
            "Hedgausvolyymi":               f"{vol_mwh:,.0f} MWh/v",
            "Excel ladattu":                "Kyllä" if has_excel else "Ei",
            "Regressio R²":                 f"{regression.r2:.3f}" if regression.r2 > 0 else "–",
        }
        param_df = pd.DataFrame.from_dict(param_info, orient="index", columns=["Arvo"])
        param_df["Arvo"] = param_df["Arvo"].astype(str)
        st.table(param_df)
