# VannVakt — Technical Methodology

**Multi-signal water breakthrough and production performance detector**  
Author: Anuri Nwagbara | GitHub: github.com/Anuri-ops  
Data: Volve NCS (Equinor open) + Florida DEP 2026 (public)

---

## 1. Problem Statement

Water breakthrough — the encroachment of formation water into a producing well — is one of the primary drivers of production decline in mature oil fields. Early detection allows engineers to consider remedial options (water shut-off, zonal isolation, workover) before the well becomes uneconomic.

A single threshold on water cut alone generates false positives. VannVakt uses a dual-signal approach: a sudden jump must coincide with declining oil production to be classified as breakthrough, rather than measurement noise or normal field variation.

---

## 2. Datasets

### 2.1 Volve NCS — Historical Mode
- **Source:** Equinor open dataset, Norwegian Continental Shelf, Block 15/9
- **Field:** Volve oil field, Hugin Formation (Upper Jurassic)
- **Drive mechanism:** Water drive
- **Wells:** 6 deviated producer wells
  - 15/9-F-1 C (spud 2013, TD 4,079 m)
  - 15/9-F-11 (spud 2013, TD 4,758 m)
  - 15/9-F-12 (spud 2007, TD 3,496 m)
  - 15/9-F-14 (spud 2007, TD 3,730 m)
  - 15/9-F-15 D (spud 2013, TD 4,671 m)
  - 15/9-F-5 (spud 2007, TD 3,762 m)
- **Period:** 2007–2016 (9 years of monthly production)
- **Clean file:** `cleaned_vannvakt_data.csv` (298 active rows)
- **Required columns:** Date, Wellbore name, Year, Month, Oil, Gas, Water, On Stream, Water_Cut

### 2.2 Florida DEP — Live Snapshot Mode
- **Source:** Florida Department of Environmental Protection, 2026 Well Production Data Report
- **File:** `2026_WELL_PRODUCTION_DATA_REPORT_WEBSITE.xlsx`
- **Sheets loaded:** JANUARY (31 days), FEBRUARY (28 days)
- **Period:** January–February 2026 (two months only)
- **Field maturity:** Late life — average water cut above 90%
- **Shut-in wells:** 65 excluded (DaysOn = 0), reported transparently in UI
- **Required columns:** OilField, Lease_Id, WellName, Operator, APINumber, County, Oil, Gas, Water, DaysOn

---

## 3. Schema Validation

Every dataset is validated on load before any analysis runs. The app halts with a clear error message if validation fails; warnings are surfaced in a collapsible expander without halting.

**Checks performed:**
- All required columns are present
- Numeric columns contain numeric data (non-numeric values flagged as warnings, set to 0)
- Values within expected ranges:
  - Water_Cut ∈ [0.0, 1.0]
  - Oil, Gas, Water ≥ 0
  - Month ∈ [1, 12]
  - DaysOn ∈ [0, 31]
- Dataset is not empty
- Well name column has no all-null rows

---

## 4. Mode Detection

The app auto-selects analysis mode based on data density per well:

```python
median_points_per_well = df.groupby("Wellbore name").size().median()
mode = "historical" if median_points_per_well > 12 else "live"
```

Wells with more than 12 data points are treated as time-series datasets (historical). Two-month snapshots fall into live mode automatically.

---

## 5. Data Cleaning & Derived Metrics

### 5.1 Volve (cleaned separately in `volve_data_cleaning.ipynb` — app assumes clean input)

In-app derived columns computed after loading:

```python
WC_Change  = Water_Cut.diff()                              # per well group
WC_Rolling = Water_Cut.rolling(3, min_periods=1).mean()   # per well group
Oil_Change = Oil.diff()                                    # per well group
GOR        = Gas / Oil  (where Oil > 0, else NaN)
```

### 5.2 Florida DEP (cleaned in-app inside `load_florida()`)

- OilField forward-filled (merged cells in source Excel)
- Rows where WellName is null removed
- Numeric columns coerced; non-numeric set to 0
- Shut-in wells (DaysOn = 0) separated before analysis
- Partial month flag: `DaysOn > 0` AND `DaysOn < days_in_month − 2`

Daily rate normalisation (applied where DaysOn > 0):
```python
Oil_Daily   = Oil / DaysOn
Water_Daily = Water / DaysOn
Gas_Daily   = Gas / DaysOn
```

Water cut (ratio-based — not affected by operating days):
```python
Water_Cut = Water / (Oil + Water)   # where total > 0, else 0
```

GOR (Florida, only where Gas column sum > 0):
```python
GOR = Gas_Daily / Oil_Daily   # where Oil_Daily > 0, else NaN
```

Month-on-month change (per well group):
```python
WC_Change        = Water_Cut.diff()
Oil_Change_Daily = Oil_Daily.diff()
```

---

## 6. Breakthrough Detection

Thresholds are defined in `BREAKTHROUGH_CONFIG` and separated from logic for independent review.

### 6.1 Historical Mode — Volve NCS

Calibrated from observed natural variation in stable Volve wells (< 8%/month).

**Sudden Breakthrough** — all three conditions must be true simultaneously:
```
WC_Change  >= 0.10   (water cut rose ≥ 10% in one month)
Water_Cut  >= 0.20   (already above 20% — filters noise at low levels)
Oil_Change <  0      (oil production is declining)
```

**Sustained Breakthrough:**
```
WC_Rolling >= 0.30   (3-month rolling average ≥ 30%)
```

**Classification:**
```
Sudden_BT AND Sustained_BT  →  "Sudden + Sustained"
Sudden_BT only               →  "Sudden"
Sustained_BT only            →  "Sustained"
Neither                      →  None
```

### 6.2 Live Snapshot Mode — Florida DEP

Adjusted for late field life. Florida wells average > 90% water cut — applying Volve thresholds would produce zero detections.

**Sustained Flag:**
```
Water_Cut >= 0.80
```

**Dual-Signal Flag** — all three must be true:
```
WC_Change          >= 0.01   (rose ≥ 1% month-on-month)
Water_Cut          >= 0.70   (already a high water producer)
Oil_Change_Daily   <  0      (daily oil rate is declining)
```

Same classification logic as historical applies.

---

## 7. Linear Forecast (Historical Mode Only)

Requires minimum 6 data points per well. Uses the last 6 months to reflect recent trajectory rather than full 9-year history.

```python
recent["t"] = np.arange(len(recent))
slope = np.polyfit(recent["t"], recent["Water_Cut"], deg=1)
future_WC = last_WC + slope[0] * period
future_WC = min(future_WC, 1.0)        # capped at 100%
```

Projected 6 months forward on monthly steps (`freq="MS"`). Explicitly labelled in the UI as straight-line extrapolation — not a reservoir simulation.

---

## 8. Well Health Score

Composite score 0–100. Based on the most recent row per well, with 3-record lookback for trend signals. Clamped to [0, 100].

| Condition | Penalty |
|-----------|---------|
| Water cut > 95% | −40 |
| Water cut > 90% | −30 |
| Water cut > 80% | −20 |
| Water cut > 60% | −10 |
| Breakthrough = Sudden + Sustained | −25 |
| Breakthrough = Sudden | −15 |
| Breakthrough = Sustained | −10 |
| 3-month avg WC_Change > 5%/month | −15 |
| 3-month avg WC_Change > 1%/month | −5 |
| 3-month avg oil change < 0 | −10 |

Oil column used: `Oil_Change_Daily` (live mode) or `Oil_Change` (historical mode).

Rating labels: 🔴 Critical (< 40), 🟠 Watch (< 70), 🟢 Stable (≥ 70).

---

## 9. Data Confidence Score

Separate from health score. Measures data reliability for a given well. Clamped to [0, 100].

| Condition | Penalty |
|-----------|---------|
| Partial month (DaysOn < days_in_month − 2) | −20 |
| DaysOn < 20 | −15 |
| GOR is NaN | −10 |

Labels: ✅ High (≥ 90), ⚠ Medium (≥ 70), ⚠ Low (< 70).

---

## 10. Agentic Q&A Layer

Requires `ANTHROPIC_API_KEY`. App runs fully without a key — chat is disabled with a sidebar warning. Model: `claude-sonnet-4-5`, max 500 tokens, up to 5 tool-call iterations per question.

**Four tools registered:**

| Tool | Returns |
|------|---------|
| `well_status` | Water cut, WC change, oil rate, GOR, breakthrough type, status label for a named well |
| `critical_wells` | All wells where Water_Cut > 80% or Breakthrough_Type is not null, sorted descending |
| `data_quality` | Active well count, shut-in exclusions, partial month records, threshold values |
| `field_summary` | Active well count, shut-in count, high WC count, average WC, highest WC well |

System prompt is mode-aware: historical mode enables trend and GOR discussion; live mode restricts to current state and month-on-month change only.

---

## 11. Dashboard Structure

| Section | Historical | Live |
|---------|-----------|------|
| Mode badge + context note | ✅ | ✅ |
| Engineering logic expander | ✅ | ✅ |
| Data quality banners | — | ✅ (partial months + shut-in) |
| Field Overview KPIs | 4 metrics | 5 metrics (+ shut-in count) |
| Well Analysis — Chart 1 | WC trend + rolling avg + forecast | WC bar chart (Jan vs Feb) |
| Well Analysis — Chart 2 | Oil rate vs WC (dual axis) | Oil rate vs WC (dual bar) |
| GOR chart | ✅ if GOR available and sum > 0 | ✅ if GOR available and sum > 0 |
| All Wells bubble chart | ✅ | ✅ |
| Forecast section | ✅ if ≥ 6 data points | — |
| Diagnosis cards + actions | ✅ | ✅ |
| Well Health Score table | ✅ | ✅ |
| All Wells Summary table | ✅ | ✅ |
| Ask VannVakt chat | ✅ if API key present | ✅ if API key present |

---

## 12. Key Design Decisions

**Why separate thresholds for Volve and Florida?**
Florida wells are in late field life with average water cut above 90%. The Volve sudden threshold (10% jump, 20% floor) would produce zero detections at that baseline. Live thresholds are calibrated to what is meaningful at high baseline water cut.

**Why exclude shut-in wells rather than impute?**
DaysOn = 0 means no production and no water cut signal. Imputing a value would introduce noise. Excluded from analysis, but reported transparently in the UI.

**Why normalise Florida volumes to daily rates?**
Monthly volumes from wells with different operating days are not directly comparable. Dividing by DaysOn produces bbl/day comparable across wells. Water cut is ratio-based and requires no normalisation.

**Why a 3-month rolling average for sustained breakthrough?**
A single high reading can be a data anomaly or measurement spike. Three consecutive months above threshold indicates genuine sustained water production.

**Why is the forecast a straight line?**
With only 2 data points in live mode and highly variable history in Volve, a reservoir simulation cannot be calibrated from this data alone. The linear forecast is explicitly labelled as a direction indicator, not a prediction.

**Why is mode detection automatic?**
Removes the burden from the user. The threshold (> 12 data points per well) correctly separates 9-year monthly Volve records from 2-month Florida snapshots, and generalises to any future dataset loaded into the app.
