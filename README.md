# ⚡ Sähkönhintaskenaariot 2025–2035

Yrityksen energiakustannusten suunnittelutyökalu — Monte Carlo -simulaatio,
Merit Order -malli ja riskianalyysi Suomen sähkömarkkinoille 2025–2035.

---

## Pika-aloitus

### 1. Asenna riippuvuudet

```bash
pip install -r requirements.txt
```

tai Windowsilla:

```bash
py -m pip install -r requirements.txt
```

### 2. Aseta salasana

Muokkaa tiedostoa `.streamlit/secrets.toml`:

```toml
APP_PASSWORD = "oma_salasanasi"
```

Oletussalasana ilman secrets.toml-tiedostoa: `demo1234`

### 3. (Valinnainen) ENTSO-E API-avain

```bash
export ENTSOE_API_KEY="sinun_avaimesi"
```

Hanki ilmainen avain: https://transparency.entsoe.eu/usrm/user/createPublicUser

Ilman avainta työkalu käyttää realistista synteettistä dataa automaattisesti.

### 4. Käynnistä sovellus

```bash
streamlit run app.py
```

tai Windowsilla:

```bash
py -m streamlit run app.py
```

### 5. Avaa selaimessa

```
http://localhost:8501
```

---

## Tiedostorakenne

```
sähköskenaario/
├── app.py                    # Streamlit-pääsovellus (salasanasuojaus, 6 välilehteä)
├── model/
│   ├── capacity.py           # Tuotantokapasiteettimalli (MW per kuukausi)
│   ├── merit_order.py        # Merit order -malli (marginaalikustannusjärjestys)
│   ├── scenarios.py          # Monte Carlo -skenaariomalli (vektorisoitu)
│   ├── costs.py              # Kustannuslaskenta ja optimointianalyysi
│   ├── risk.py               # VaR/CVaR, stressitestit, hedgausstrategiat
│   ├── data_fetch.py         # Datanhaku (ENTSO-E + synteettinen fallback)
│   └── data_inspect.py       # Excel-tiedoston automaattinen tunnistus
├── ui/
│   ├── charts.py             # Plotly-kuvaajat
│   └── report.py             # PDF-raportin generointi (ReportLab)
├── .streamlit/
│   └── secrets.toml          # Salasana — EI GitHubiin
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Ominaisuudet

### Merit Order -malli
- Tuotantomuotojen marginaalikustannusjärjestys: Tuuli → Aurinko → Ydinvoima → Vesivoima → CHP → Tuonti → Kaasu
- Vesivoiman opportunity cost vaihtelee vesivarantotilanteen mukaan (normaali / kuiva / märkä)
- Interaktiivinen kuukausikohtainen merit order -käyrä

### Kapasiteettimalli
- Kuukausittainen tuotantokapasiteetti per tuotantomuoto (MW)
- Kapasiteettiylijäämä/-alijäämä ja siirtoyhteysrajoitteet
- Kriittiset kuukaudet (top 5 riskikuukautta 2025–2035)

### Monte Carlo -simulaatio
- 3 skenaariota: Matala / Perus / Korkea
- 100–1 000 ajoa, kaikki numpy-vektorioperaatioina (< 50 ms)
- Persentiilit: P5 / P10 / P25 / P50 / P75 / P90 / P95
- Markkinavaikutukset: tuulivoima, ydinvoima, vesivoima, kaasu, CO₂, datakeskukset

### Riskianalyysi
- VaR 95%, CVaR 95%, volatiliteetti, hintapiikkiriski
- 4 automaattista stressitestiä (energiakriisi, kuiva talvi, ydinvika, datakeskusboom)
- Hedgausstrategiat: spot, kiinteä, collar, termiini, yhdistelmä
- Efficient frontier -analyysi

### Excel-tiedoston käsittely
- Automaattinen sarakkeiden tunnistus avainsanoilla
- Regressiomalli fundamenttidatasta (R²-arvo)
- Toimii myös osittaisella datalla (puuttuvat sarakkeet saavat synteettisen oletuksen)

---

## Välilehdet

| Välilehti | Sisältö |
|-----------|---------|
| 🔬 Data-analyysi | Historiallinen data, korrelaatiomatriisi, regressiokertoimet |
| 📈 Hintaskenaariot | 3 skenaariopolkua + epävarmuusvyöhyke + historia |
| ⚡ Markkinadynamiikka | Merit order -käyrä, kapasiteetti vs. kysyntä, tornado, hintaerot |
| 🛡️ Riskianalyysi | VaR/CVaR, hedgausvertailu, stressitestit, efficient frontier |
| 📅 Kuukausianalyysi | Heatmap, kuukausikeskiarvot, optimointisuositus |
| 📄 Raportti | Automaattinen yhteenveto, CSV- ja PDF-lataukset |

---

## Streamlit Cloud -julkaisu

1. Pushaa koodi GitHubiin (`.gitignore` estää `secrets.toml`-tiedoston pushauksen)
2. Mene [share.streamlit.io](https://share.streamlit.io)
3. Valitse repository ja `app.py`
4. Aseta **Settings → Secrets**: `APP_PASSWORD = "salasanasi"`

---

## Skenaariokuvaukset

| Skenaario | Kuvaus | Tavoitetaso |
|-----------|--------|-------------|
| **Matala** | Nopea tuulivoimakasvu (+6 GW), OL3 luotettava, kaasu halpa | 30–50 €/MWh |
| **Perus** | Trendikehitys jatkuu, +3 GW tuuli, maltillinen sähköistyminen | 50–80 €/MWh |
| **Korkea** | Kuiva vuosi, energiakriisi tai rakentamisen viivästyminen | 80–150 €/MWh |

---

## Tietoturva

- Excel-tiedosto käsitellään vain paikallisessa muistissa
- Data ei lähde koneeltasi mihinkään
- Salasana tallennetaan vain `.streamlit/secrets.toml`-tiedostoon (ei koodiin)
- `.gitignore` estää `secrets.toml`-tiedoston päätymisen versionhallintaan
