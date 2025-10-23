
# Greenhouse Ops Mini-Demo

**Purpose.** A tiny, forward-deployed style demo that ingests greenhouse-like data, flags bad sensors, and produces a same-day irrigation recommendation per zone. It also compares water use against a naive schedule.

**What this is not.** A field-calibrated ET model. Assumptions are simple and noted below so reviewers can reason about them quickly.

## Quickstart

```bash
pip install -r requirements.txt
python src/run_demo.py
```

Outputs:
- `data/irrigation_recommendations.csv` with nightly mm by zone.
- `figures/soil_moisture_outliers_*.png` outlier visualization.
- `figures/irrigation_naive_vs_rec_*.png` naive vs recommended.
- A KPI summarizing modeled water savings vs a naive 5 mm every other night schedule.

## Method (short)

1. **Synthetic data.** Seven days, hourly resolution, two zones. Air temperature, RH, wind, solar proxy, soil VWC, and a naive irrigation schedule are generated with realistic daily cycles and noise. A few soil VWC outliers simulate bad sensors.
2. **Anomaly detection.** A Hampel filter (rolling median + MAD) marks outliers in soil moisture.
3. **ET proxy.** A bounded dryness term, normalized solar and temperature, and wind combine into an hourly evapotranspiration proxy, scaled so peak daytime is ~0.2 mm/hr. This is for demo only.
4. **Decision at 18:00.** For each zone-day, take the median of the last 6 hours of non-outlier soil VWC, convert deficit to mm across a 200 mm root zone, add projected night ET, divide by 0.85 efficiency, cap at 12 mm.
5. **KPI.** Compare recommended totals to a naive 5 mm every other night schedule and report percent reduction.

## Key assumptions (explicit)

- Target soil moisture (VWC): **0.25**
- Effective root depth: **200 mm**
- Irrigation efficiency: **0.85**
- Rain forecast: **0 mm** (placeholder)
- VWC-to-mm conversion: **illustrative** (see `src/run_demo.py`), treat as a tunable parameter

These numbers are placeholders for a demo. In a real deployment, they would be replaced by site-specific calibration or Foundry ontology parameters.

## Results (example)
See `figures/` for plots and `data/irrigation_recommendations.csv`. A sample aggregated KPI is printed at the end of a run and saved as a table. In this synthetic dataset, the recommended policy reduces total applied water by approximately **71.8%** relative to the naive schedule.

## Files

- `src/run_demo.py` — end-to-end script: ingest, detect outliers, compute recommendations, export artifacts.
- `data/synthetic_greenhouse_data.csv` — generated sample data.
- `data/irrigation_recommendations.csv` — nightly recommendations.
- `figures/` — plots.

