# 💧 VannVakt — Water Cut Intelligence

**Multi-signal water breakthrough and production performance detector.**

Water breakthrough is one of the leading causes of well abandonment on the Norwegian Continental Shelf. 
VannVakt detects it early — tracking five signals simultaneously across historical and live production data.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vannvakt.streamlit.app/)

---

## What It Does

VannVakt monitors oil wells for water breakthrough — the point where a well starts producing more water 
than oil. Left undetected, this drives up lifting costs and kills well economics.

The app tracks **five signals simultaneously:**

| Signal | What It Catches |
|--------|----------------|
| Water cut (primary) | Direct breakthrough detection |
| Oil decline | Production loss confirmation |
| GOR (Gas-Oil Ratio) | Reservoir pressure drop |
| Field baseline deviation | Well vs field average |
| Trend acceleration | Rate of change, not just level |


---

## Two Analysis Modes

The app auto-detects which mode to use based on data density.

### Historical Mode — Volve NCS (2007–2016)
- 298 clean monthly records across 6 producer wells
- Full trend analysis, GOR, 6-month linear forecast
- Dual-signal breakthrough detection (sudden + sustained)
- Field baseline comparison

### Live Snapshot Mode — Florida DEP (Jan–Feb 2026)
- Current production snapshot, two months of data
- Month-on-month change detection
- Daily rate normalisation (bbl/day = monthly ÷ DaysOn)
- 65 shut-in wells excluded and reported transparently

---

## Breakthrough Detection Logic

### Historical (Volve)

**Sudden breakthrough** — all three must be true:
- Water cut rises ≥ 10% in one month
- Water cut already above 20%
- Oil production declining

**Sustained breakthrough:**
- 3-month rolling average water cut ≥ 30%

![Oil vs Water Cut](volve%20figs/vannvakt_oil_vs_wc.png)


Thresholds are defined in `BREAKTHROUGH_CONFIG` and separated from logic for independent review.

![Oil vs Water Cut](volve%20figs/vannvakt_watercut.png)

### Live Snapshot (Florida)

**Sustained flag:** water cut ≥ 80%

**Dual-signal flag** — all three must be true:
- Water cut rose ≥ 1% month-on-month
- Water cut already above 70%
- Daily oil rate declining

  
![VannVakt Livesnapshot](volve%20figs/vannvakt_livesnapshot.png)

---

## Dashboard

![VannVakt Diagnostic](volve%20figs/vannvakt_diagnostic.png)

---

## Tech Stack

- **Python** — data processing and analytics
- **Streamlit** — dashboard and UI
- **Plotly** — interactive charts
- **Pandas / NumPy** — data engineering
- **Anthropic API** — agentic Q&A layer (optional)

---

## Project Structure

```
VannVakt/
├── vannvakt_app.py                        # Main Streamlit app
├── cleaned_vannvakt_data.csv              # Volve NCS clean data (2007–2016)
├── volve_well_meta.csv                    # Well metadata
├── 2026_WELL_PRODUCTION_DATA_REPORT_...   # Florida DEP 2026 data
├── volve_data_cleaning.ipynb              # Data cleaning notebook
├── requirements.txt                       # Python dependencies
├── .streamlit/                            # Streamlit config
├── .gitignore                             # Excludes .env and raw xlsx
└── README.md
```

---

## Run Locally

```bash
git clone https://github.com/Anuri-ops/VannVakt
cd VannVakt
pip install -r requirements.txt
```

Create a `.env` file (optional — only needed for the chat feature):
```
ANTHROPIC_API_KEY=your_key_here
```

Run the app:
```bash
streamlit run vannvakt_app.py
```

The app runs fully without an API key. The agentic chat layer is disabled if no key is found.

---

## Data Sources

- **Volve NCS** — Equinor open dataset, Norwegian Continental Shelf. Released under the Equinor Open Data Licence.
- **Florida DEP** — Florida Department of Environmental Protection, 2026 Well Production Data Report. Public domain.

---

## Demo

▶ [Watch the demo on YouTube](https://youtu.be/NEKRpJQf_C8)

---

## Author

**Anuri Nwagbara** — Subsurface & Data Engineer  
[LinkedIn](https://linkedin.com/in/anuri-nwagbara) · [GitHub](https://github.com/Anuri-ops)
