
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def hampel(series, window_size=9, n_sigmas=3.0):
    s = pd.Series(series).copy()
    rolling_median = s.rolling(window=window_size, center=True, min_periods=1).median()
    diff = (s - rolling_median).abs()
    mad = diff.rolling(window=window_size, center=True, min_periods=1).median()
    mad = mad.replace(0, mad[mad>0].min() if (mad>0).any() else 1e-6)
    threshold = n_sigmas * 1.4826 * mad
    return diff > threshold

def eto_proxy_mm_per_hr(air_temp_C, rel_humidity_pct, solar_irradiance_wm2, wind_mps):
    dryness = np.clip(1.0 - rel_humidity_pct/100.0, 0.0, 1.0)
    solar_norm = np.clip(solar_irradiance_wm2 / 800.0, 0, 1.5)
    t_norm = np.clip(air_temp_C / 30.0, 0, 1.5)
    index = dryness * (0.6*solar_norm + 0.3*t_norm + 0.1*np.clip(wind_mps/3.0, 0, 1.5))
    return 0.2 * index

def main():
    base = Path(__file__).resolve().parents[1]
    data_dir = base / "data"
    fig_dir = base / "figures"
    fig_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_dir / "synthetic_greenhouse_data.csv", parse_dates=["timestamp"])
    df["soil_outlier"] = False
    for z, dfg in df.groupby("zone"):
        mask = df["zone"] == z
        out = hampel(df.loc[mask, "soil_moisture_vwc"], window_size=9, n_sigmas=3.0)
        df.loc[mask, "soil_outlier"] = out.values

    df["eto_mm_hr"] = eto_proxy_mm_per_hr(df["air_temp_C"], df["rel_humidity_pct"], df["solar_irradiance_wm2"], df["wind_mps"])
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour

    TARGET_VWC = 0.25
    ROOT_DEPTH_MM = 200.0
    EFFICIENCY = 0.85
    RAIN_FORECAST_MM = 0.0
    WATER_HOLDING_COEFF = 1.0

    recs = []
    for z, dfg in df.groupby("zone"):
        for day, dfd in dfg.groupby("date"):
            last6 = dfd[(dfd["hour"]>=12) & (dfd["hour"]<=18)]
            sm_series = last6.loc[~last6["soil_outlier"], "soil_moisture_vwc"]
            if sm_series.empty:
                current_vwc = dfd.loc[dfd["hour"]==18, "soil_moisture_vwc"].mean()
            else:
                current_vwc = sm_series.median()
            eto_tonight = dfd[(dfd["hour"]>=18) | (dfd["hour"]<=6)]["eto_mm_hr"].sum()
            deficit_vwc = max(0.0, TARGET_VWC - current_vwc)
            mm_needed = deficit_vwc * (ROOT_DEPTH_MM/100.0) * (100.0 * WATER_HOLDING_COEFF)
            mm_rec = max(0.0, mm_needed + eto_tonight - RAIN_FORECAST_MM) / EFFICIENCY
            mm_rec = float(np.clip(mm_rec, 0.0, 12.0))
            recs.append({"date": day, "zone": z, "current_vwc": current_vwc, "eto_tonight_mm": float(eto_tonight), "irrigation_rec_mm": mm_rec})

    rec_df = pd.DataFrame(recs)
    rec_df.to_csv(data_dir / "irrigation_recommendations.csv", index=False)

    # KPI
    naive_totals = (
        df.groupby(["zone", "date"])["irrigation_applied_mm_naive"]
          .sum()
          .reset_index()
          .rename(columns={"irrigation_applied_mm_naive":"naive_mm"})
    )
    kpi = rec_df.merge(naive_totals, on=["zone", "date"], how="left").fillna({"naive_mm": 0.0})
    kpi["savings_mm"] = kpi["naive_mm"] - kpi["irrigation_rec_mm"]
    totals = kpi.groupby("zone")[["naive_mm", "irrigation_rec_mm", "savings_mm"]].sum().reset_index()
    totals["pct_reduction"] = np.where(
        totals["naive_mm"]>0, 100.0 * totals["savings_mm"]/totals["naive_mm"], 0.0
    )
    print("\\n=== KPI: Water Use vs Naive (last 7 days) ===")
    print(totals.to_string(index=False, float_format=lambda x: f"{x:0.2f}"))

    # Plots
    for z in df["zone"].unique():
        dfg = df[df["zone"] == z]
        plt.figure(figsize=(10,4))
        plt.plot(dfg["timestamp"], dfg["soil_moisture_vwc"], label="Soil VWC")
        plt.scatter(dfg.loc[dfg["soil_outlier"], "timestamp"], dfg.loc[dfg["soil_outlier"], "soil_moisture_vwc"], marker="x", label="Outliers")
        plt.title(f"Soil Moisture with Outliers — {z}")
        plt.xlabel("Time")
        plt.ylabel("VWC (0–1)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(base / "figures" / f"soil_moisture_outliers_{z}.png", dpi=160)
        plt.close()

    for z in df["zone"].unique():
        dfz = kpi[kpi["zone"] == z]
        plt.figure(figsize=(8,4))
        plt.bar(dfz["date"].astype(str), dfz["naive_mm"], alpha=0.6, label="Naive (mm)")
        plt.plot(dfz["date"].astype(str), dfz["irrigation_rec_mm"], label="Recommended (mm)")
        plt.title(f"Nightly Irrigation: Naive vs Recommended — {z}")
        plt.xlabel("Date")
        plt.ylabel("mm")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(base / "figures" / f"irrigation_naive_vs_rec_{z}.png", dpi=160)
        plt.close()

if __name__ == "__main__":
    main()
