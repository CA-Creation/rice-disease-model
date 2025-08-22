# save as forecast_from_rice_only.py  (or paste into your existing script)

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import timedelta

# ---------- helpers ----------
def ensure_dir(path="outputs"):
    os.makedirs(path, exist_ok=True); return path

def canonicalize(col: str) -> str:
    c = (col or "").strip().lower().replace("Â°", "°")
    for ch in ["(",")","%","/","-","_"]:
        c = c.replace(ch," ")
    return " ".join(c.split())

def find_date_column(df: pd.DataFrame):
    for c in df.columns:
        s = pd.to_datetime(df[c], errors="coerce")
        if s.notna().mean() > 0.95:
            return c
    return None

def feature_map():
    return {
        "temperature": ["temperature °c","temperature c","avg temperature","average temperature","maximum temperature","minimum temperature","temp","tavg","tmean"],
        "humidity": ["humidity","relative humidity","avg relative humidity","average relative humidity","rh"],
        "rainfall": ["rainfall mm","total rainfall","rainfall","precipitation","rain","ppt","prcp"],
        "wind_speed": ["wind speed m s","average wind speed","wind speed","wind"],
        "ph": ["ph level","average ph","ph","maximam ph","maximum ph","minimum ph","min ph"],
        "soil_moisture": ["soil moisture","avg soil moisture","soil moisture percent","sm"],
        "nitrogen": ["nitrogen content mg kg","nitrogen","n"],
        "potassium": ["potassium content mg kg","potassium","k"],
        "salinity": ["sanility","salinity","average sanility","maximum sanility","minimum sanility","ec"]
    }

def find_columns(df: pd.DataFrame, wanted: dict) -> dict:
    canon = {canonicalize(c): c for c in df.columns}
    out = {}
    for key, opts in wanted.items():
        for o in opts:
            if o in canon:
                out[key] = canon[o]; break
    return out

def build_risk_index(df_features: pd.DataFrame) -> pd.Series:
    # min-max normalize to [0,1]
    X = df_features.copy()
    for c in X.columns:
        s = pd.to_numeric(X[c], errors="coerce").replace([np.inf,-np.inf], np.nan)
        if s.notna().sum()==0: X[c] = 0.0
        else:
            vmin, vmax = s.min(), s.max()
            X[c] = 0.0 if (pd.isna(vmin) or pd.isna(vmax) or vmax==vmin) else (s-vmin)/(vmax-vmin)

    # heuristic weights — humidity & rainfall strongest, then temperature
    weights = {
        "humidity": 0.28, "rainfall": 0.28, "temperature": 0.18,
        "soil_moisture": 0.10, "wind_speed": 0.06,
        "salinity": 0.03, "ph": 0.03, "nitrogen": 0.02, "potassium": 0.02
    }
    used = [c for c in X.columns if c in weights]
    if not used:
        base = X.select_dtypes(include=[np.number]).mean(axis=1).fillna(0.0)
        return base.clip(0,1)
    wsum = sum(weights[c] for c in used)
    risk = sum(weights[c]*X[c] for c in used) / (wsum if wsum>0 else 1.0)
    return risk.rolling(3, min_periods=1).mean().clip(0,1)

def seasonal_naive_forecast(hist_df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    last_year = hist_df.tail(365)["risk_prob"].to_numpy()
    if len(last_year) < 365:
        last_year = np.resize(last_year, 365)
    reps = int(np.ceil(horizon_days/365))
    vals = np.tile(last_year, reps)[:horizon_days]
    start = hist_df["Date"].max() + pd.Timedelta(days=1)
    fut_dates = pd.date_range(start, periods=horizon_days, freq="D")
    return pd.DataFrame({"Date":fut_dates, "risk_prob":vals})

def bootstrap_uncertainty(hist_df, fut_df, n_boot=200):
    history = hist_df.sort_values("Date").reset_index(drop=True)
    if len(history) < 366:
        residuals = history["risk_prob"].diff().dropna().values
    else:
        residuals = (history["risk_prob"] - history["risk_prob"].shift(365)).dropna().values
    if residuals.size==0: residuals = np.array([0.0])
    rng = np.random.default_rng(42)
    base = fut_df["risk_prob"].values
    sims = [np.clip(base + rng.choice(residuals, size=len(base), replace=True), 0,1) for _ in range(n_boot)]
    sims = np.vstack(sims)
    q05 = np.quantile(sims, 0.05, axis=0)
    q50 = np.quantile(sims, 0.50, axis=0)
    q95 = np.quantile(sims, 0.95, axis=0)
    return pd.DataFrame({"Date":fut_df["Date"], "q05":q05, "q50":q50, "q95":q95})

def add_season_cols(df: pd.DataFrame):
    df = df.copy()
    d = pd.to_datetime(df["Date"])
    season = []; season_year = []
    for dt in d:
        m, y = dt.month, dt.year
        if m >= 10 or m <= 3:
            season.append("Maha"); season_year.append(y if m>=10 else y-1)
        else:
            season.append("Yala"); season_year.append(y)
    df["season"] = season; df["season_year"] = season_year
    return df

def find_low_risk_windows(df, threshold=0.40, min_days=21):
    s = df[["Date","risk_prob"]].sort_values("Date").reset_index(drop=True)
    mask = s["risk_prob"] < threshold
    runs, start_i = [], None
    for i, ok in enumerate(mask):
        if ok and start_i is None: start_i = i
        elif not ok and start_i is not None:
            runs.append((start_i, i-1)); start_i = None
    if start_i is not None: runs.append((start_i, len(s)-1))
    out = []
    for a,b in runs:
        start, end = s.loc[a,"Date"], s.loc[b,"Date"]
        L = (end-start).days + 1
        if L >= min_days:
            seg = s.iloc[a:b+1]
            out.append({"start_date":start, "end_date":end, "length_days":L, "mean_risk":float(seg["risk_prob"].mean())})
    return pd.DataFrame(out).sort_values(["length_days","mean_risk"], ascending=[False, True])

# ---------- main pipeline using only rice_disease_data.csv ----------
def forecast_and_research_from_rice_only(csv_path="rice_disease_data.csv", out_dir="outputs",
                                         low_threshold=0.40, low_min_days=21):
    out_dir = ensure_dir(out_dir)
    df = pd.read_csv(csv_path, encoding="ISO-8859-1")

    date_col = find_date_column(df)
    if date_col is None:
        raise ValueError("Could not detect a valid Date column in rice_disease_data.csv")
    df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.loc[df["Date"].notna()].sort_values("Date").reset_index(drop=True)

    fmap = find_columns(df, feature_map())
    feat_df = pd.DataFrame({k: pd.to_numeric(df[v], errors="coerce") for k,v in fmap.items()})
    risk = build_risk_index(feat_df)
    hist_df = pd.DataFrame({"Date": df["Date"], "risk_prob": risk})
    hist_df.to_csv(os.path.join(out_dir, "historical_risk_from_weather.csv"), index=False)

    # ---- 5-year forecast (robust seasonal-naive) ----
    horizon_days = max((pd.Timestamp(hist_df["Date"].max()) + pd.DateOffset(years=5)
                        - pd.Timestamp(hist_df["Date"].max())).days, 5*365+1)
    fut_df = seasonal_naive_forecast(hist_df, horizon_days=horizon_days)
    fut_df.to_csv(os.path.join(out_dir, "forecast_risk_5y_daily.csv"), index=False)

    # ---- uncertainty (bootstrap) ----
    unc = bootstrap_uncertainty(hist_df, fut_df, n_boot=200)
    unc.to_csv(os.path.join(out_dir, "forecast_5y_uncertainty.csv"), index=False)

    # ---- research Qs: low-risk windows ----
    low = find_low_risk_windows(fut_df, threshold=low_threshold, min_days=low_min_days)
    low.to_csv(os.path.join(out_dir, "forecast_5y_low_risk_windows.csv"), index=False)

    # ---- research Qs: monthly & seasonal trends (LINE charts) ----
    # Monthly line
    fm = fut_df.copy()
    fm["ym"] = fm["Date"].dt.to_period("M").dt.to_timestamp("M")
    monthly = fm.groupby("ym")["risk_prob"].mean().reset_index()
    monthly.to_csv(os.path.join(out_dir, "forecast_5y_monthly_mean.csv"), index=False)
    plt.figure(figsize=(12,4)); plt.plot(monthly["ym"], monthly["risk_prob"], marker="o")
    plt.title("Monthly Mean Risk — Next 5 Years"); plt.xlabel("Month"); plt.ylabel("Mean risk")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "forecast_5y_monthly_mean_line.png"), dpi=150); plt.close()

    # Season-year (Maha vs Yala) lines
    seas = add_season_cols(fut_df).groupby(["season_year","season"], as_index=False)["risk_prob"].mean()
    pv = seas.pivot(index="season_year", columns="season", values="risk_prob").sort_index()
    plt.figure(figsize=(10,4))
    for col in pv.columns:
        plt.plot(pv.index, pv[col], marker="o", label=col)
    plt.title("Average Risk by Season-Year (Maha vs Yala) — 5Y Forecast")
    plt.xlabel("Season year"); plt.ylabel("Mean risk"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "forecast_5y_season_year_lines.png"), dpi=150); plt.close()

    # Historical vs forecast + rolling lines
    plt.figure(figsize=(12,5))
    plt.plot(hist_df["Date"], hist_df["risk_prob"], label="Historical")
    plt.plot(fut_df["Date"], fut_df["risk_prob"], label="Forecast", linestyle="--")
    plt.title("Disease Risk: Historical & 5Y Forecast"); plt.xlabel("Date"); plt.ylabel("Risk")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "line_hist_vs_forecast.png"), dpi=150); plt.close()

    comb = pd.concat([hist_df.assign(part="hist"), fut_df.assign(part="fut")], ignore_index=True).sort_values("Date")
    comb["roll7"] = comb["risk_prob"].rolling(7, min_periods=1).mean()
    comb["roll30"] = comb["risk_prob"].rolling(30, min_periods=1).mean()
    plt.figure(figsize=(12,5))
    plt.plot(comb["Date"], comb["roll7"], label="7-day rolling")
    plt.plot(comb["Date"], comb["roll30"], label="30-day rolling")
    plt.title("Rolling Mean Risk (7d & 30d)"); plt.xlabel("Date"); plt.ylabel("Rolling mean risk")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "line_rolling_7_30.png"), dpi=150); plt.close()

    # ---- concise summary for your paper ----
    summary = {
        "first_forecast_date": str(fut_df["Date"].min().date()),
        "last_forecast_date": str(fut_df["Date"].max().date()),
        "forecast_mean_risk": float(fut_df["risk_prob"].mean()),
        "forecast_max_risk": float(fut_df["risk_prob"].max()),
        "forecast_min_risk": float(fut_df["risk_prob"].min()),
        "n_low_risk_windows": int(len(low)),
        "low_risk_threshold": float(low_threshold),
        "low_risk_min_days": int(low_min_days),
    }
    with open(os.path.join(out_dir, "research_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Research Summary (rice_disease_data.csv only) ===")
    for k,v in summary.items(): print(f"{k}: {v}")
    print("Saved:\n - outputs/forecast_risk_5y_daily.csv\n - outputs/forecast_5y_uncertainty.csv"
          "\n - outputs/forecast_5y_low_risk_windows.csv\n - outputs/forecast_5y_monthly_mean.csv"
          "\n - multiple line charts under outputs/")

    return {"hist_df":hist_df, "fut_df":fut_df, "uncertainty":unc, "low_windows":low, "monthly":monthly, "summary":summary}

if __name__ == "__main__":
    forecast_and_research_from_rice_only("rice_disease_data.csv", out_dir="outputs", low_threshold=0.40, low_min_days=21)
