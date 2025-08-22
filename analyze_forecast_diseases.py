"""
Rice Disease Risk Research from rice_forecast_5yearswithdiseases.csv
--------------------------------------------------------------------
Inputs:
- rice_forecast_5yearswithdiseases.csv
  (must include Date, Disease_Presence_Prob, Predicted_Disease_Type, and climate features)

Outputs (saved to ./outputs):
- metrics_short_term.csv, metrics_long_term.csv
- comparison_short_term.png, comparison_long_term.png   (LSTM uni/multi vs Seasonal-Naive) [line graphs]
- training_loss_univariate.png, training_loss_multivariate.png       [line graphs]
- rolling_rmse_60d_uni_vs_multi.png                                  [line graph]
- feature_ablation_rolling_mae_60d.png                               [line graph per ablation run]
- planting_windows_all.csv (low-risk windows overall)
- planting_windows_by_disease.csv (low-risk windows per disease)
- risk_with_threshold.png (overall daily risk + threshold)            [line graph]
- risk_with_seasons.png (daily risk with Maha/Yala vlines)            [line graph]
- scenarios_5y_multivariate.png (baseline vs wetter/drier/warmer/cooler) [line graph]
- seasonal_patterns_by_disease.png (monthly means by disease type)    [line graph]
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional deep learning
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except Exception:
    HAS_TF = False

# ------------------------- Helpers -------------------------

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def canonicalize(col: str) -> str:
    c = (col or "").strip().lower().replace("Â°", "°")
    for ch in ["(", ")", "%", "/", "-", "_"]:
        c = c.replace(ch, " ")
    return " ".join(c.split())

def get_feature_map():
    return {
        "temperature": ["temperature °c","temperature c","avg temperature","average temperature",
                        "maximum temperature","minimum temperature","temp","tavg","tmin","tmax"],
        "humidity": ["humidity","relative humidity","avg relative humidity","average relative humidity","rh"],
        "rainfall": ["rainfall mm","total rainfall","rainfall","precipitation","rain"],
        "wind_speed": ["wind speed m s","average wind speed","wind speed","windspeed","wind"],
        "ph": ["ph level","average ph","ph","maximum ph","minimum ph"],
        "soil_moisture": ["soil moisture","avg soil moisture","soil moisture percent","soil moisture %"],
        "nitrogen": ["nitrogen content mg kg","nitrogen","n"],
        "potassium": ["potassium content mg kg","potassium","k"],
        "salinity": ["sanility","salinity","average salinity","maximum salinity","minimum salinity","ec"]
    }

def find_columns(df: pd.DataFrame, wanted: dict) -> dict:
    canon = {canonicalize(c): c for c in df.columns}
    out = {}
    for key, opts in wanted.items():
        for o in opts:
            if o in canon:
                out[key] = canon[o]
                break
    return out

def as_datetime_sorted(df, date_col="Date"):
    d = pd.to_datetime(df[date_col], errors="coerce")
    if d.isna().any():
        raise ValueError("Unparseable dates detected.")
    df = df.copy()
    df[date_col] = d
    return df.sort_values(date_col).reset_index(drop=True)

def train_val_split_time(df, frac_val=0.2):
    n = len(df)
    k = max(1, int(n*(1-frac_val)))
    return df.iloc[:k].reset_index(drop=True), df.iloc[k:].reset_index(drop=True)

def seasonal_naive(series: np.ndarray, horizon: int, season: int = 365) -> np.ndarray:
    if len(series) < season:
        season = max(1, min(season, len(series)))
    pattern = series[-season:]
    reps = int(np.ceil(horizon / season))
    return np.tile(pattern, reps)[:horizon]

def rolling_rmse(y_true, y_pred, window=7):
    err = (y_true - y_pred)**2
    out = pd.Series(err).rolling(window, min_periods=1).mean().pow(0.5).values
    return out

def add_season_cols(df, date_col="Date"):
    df = df.copy()
    d = pd.to_datetime(df[date_col])
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    # Maha ~ Oct-Mar; Yala ~ Apr-Sep
    season, season_year = [], []
    for dt in d:
        m, y = dt.month, dt.year
        if m >= 10 or m <= 3:
            season.append("Maha")
            season_year.append(y if m >= 10 else y-1)
        else:
            season.append("Yala")
            season_year.append(y)
    df["season"] = season
    df["season_year"] = season_year
    return df

def ensure_line_plot(x, y, label=None):
    plt.plot(x, y, label=label)

# ------------------------- LSTM Models -------------------------

def build_lstm(input_steps: int, n_features: int):
    model = Sequential([
        LSTM(64, activation="tanh", input_shape=(input_steps, n_features)),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def make_supervised(X: np.ndarray, y: np.ndarray, seq_len: int):
    # X shape: (T, F), y shape: (T,)
    xs, ys = [], []
    for i in range(len(y) - seq_len):
        xs.append(X[i:i+seq_len, :])
        ys.append(y[i+seq_len])
    return np.array(xs), np.array(ys)

def fit_lstm(series_df: pd.DataFrame, risk_col: str, feat_cols: list, seq_len=28,
             val_frac=0.2, tag="univariate"):
    """
    Returns: model, hist (training history), scalers dict, arrays needed for forecasting
    """
    if not HAS_TF:
        raise RuntimeError("TensorFlow is not installed; cannot train LSTM.")
    D = series_df.copy()
    y = D[risk_col].astype(float).values.reshape(-1,1)  # (T,1)

    if feat_cols:
        X = D[feat_cols].astype(float).values  # (T,F)
        # Scale features + target to [0,1] by min-max
        x_min, x_max = X.min(axis=0), X.max(axis=0)
        x_den = np.where((x_max - x_min) > 0, (x_max - x_min), 1.0)
        Xn = (X - x_min) / x_den
        y_min, y_max = y.min(), y.max()
        y_den = (y_max - y_min) if y_max > y_min else 1.0
        yn = (y - y_min) / y_den
        Z = np.concatenate([yn, Xn], axis=1)  # first column is normalized risk
        n_features = Z.shape[1]
        Xseq, Y = make_supervised(Z, yn.ravel(), seq_len)
        # Split time-wise
        n = len(Y)
        k = max(1, int(n*(1-val_frac)))
        Xtr, Xva = Xseq[:k], Xseq[k:]
        Ytr, Yva = Y[:k], Y[k:]
        model = build_lstm(seq_len, n_features)
        es = EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=0)
        h = model.fit(Xtr, Ytr, validation_data=(Xva, Yva),
                      epochs=200, batch_size=32, verbose=0, callbacks=[es])
        scalers = {"x_min":x_min, "x_den":x_den, "y_min":y_min, "y_den":y_den, "feat_cols":feat_cols}
        return model, h, scalers
    else:
        # Univariate: only risk column
        y_min, y_max = y.min(), y.max()
        y_den = (y_max - y_min) if y_max > y_min else 1.0
        yn = (y - y_min) / y_den
        Z = yn  # (T,1)
        n_features = 1
        # build supervised
        Xseq = []
        Y = []
        for i in range(len(yn)-seq_len):
            Xseq.append(yn[i:i+seq_len, :])
            Y.append(yn[i+seq_len, 0])
        Xseq, Y = np.array(Xseq), np.array(Y)
        n = len(Y)
        k = max(1, int(n*(1-val_frac)))
        Xtr, Xva = Xseq[:k], Xseq[k:]
        Ytr, Yva = Y[:k], Y[k:]
        model = build_lstm(seq_len, n_features)
        es = EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=0)
        h = model.fit(Xtr, Ytr, validation_data=(Xva, Yva),
                      epochs=200, batch_size=32, verbose=0, callbacks=[es])
        scalers = {"y_min":y_min, "y_den":y_den, "feat_cols":[]}
        return model, h, scalers

def forecast_lstm(series_df: pd.DataFrame, risk_col: str, feat_cols: list,
                  model, scalers, seq_len=28, horizon=60, scenario=None):
    """
    scenario: optional dict to perturb features during forecast, e.g.
      {"rainfall_scale":1.15, "temperature_add":+1.0}
    """
    D = series_df.copy()
    y = D[risk_col].astype(float).values.reshape(-1,1)

    if feat_cols:
        X = D[feat_cols].astype(float).values
        # Apply perturbations for the future steps as we roll forward
        x_min, x_den = scalers["x_min"], scalers["x_den"]
        y_min, y_den = scalers["y_min"], scalers["y_den"]

        # normalized full past
        Zpast = np.concatenate([ (y - y_min) / (y_den if y_den>0 else 1.0),
                                 (X - x_min) / np.where(x_den>0, x_den, 1.0)], axis=1)

        # seed: last seq_len rows
        buf = Zpast[-seq_len:, :].copy()
        preds = []
        for t in range(horizon):
            xinp = buf.reshape(1, seq_len, Zpast.shape[1])
            yhat_n = model.predict(xinp, verbose=0)[0,0]  # normalized
            # build feature row for next step
            if feat_cols:
                # by default repeat last normalized features
                feat_next = buf[-1, 1:].copy()

                # apply scenario adjustments in *original* space, then re-normalize
                if scenario:
                    # denormalize to original
                    orig_feats = feat_next * x_den + x_min
                    # rainfall scale
                    if "rainfall_scale" in scenario and "rainfall" in feat_cols:
                        r_idx = feat_cols.index("rainfall")
                        orig_feats[r_idx] = orig_feats[r_idx] * float(scenario["rainfall_scale"])
                    # temperature add
                    if "temperature_add" in scenario and "temperature" in feat_cols:
                        t_idx = feat_cols.index("temperature")
                        orig_feats[t_idx] = orig_feats[t_idx] + float(scenario["temperature_add"])
                    # humidity scale
                    if "humidity_scale" in scenario and "humidity" in feat_cols:
                        h_idx = feat_cols.index("humidity")
                        orig_feats[h_idx] = orig_feats[h_idx] * float(scenario["humidity_scale"])
                    # re-normalize
                    feat_next = (orig_feats - x_min) / np.where(x_den>0, x_den, 1.0)

                next_row = np.concatenate([[yhat_n], feat_next])
            else:
                next_row = np.array([yhat_n])

            preds.append(yhat_n*y_den + y_min)
            # roll window
            buf = np.vstack([buf[1:], next_row])

        return np.clip(np.array(preds).ravel(), 0, 1)
    else:
        # univariate
        y_min, y_den = scalers["y_min"], scalers["y_den"]
        yn = (y - y_min) / (y_den if y_den>0 else 1.0)
        buf = yn[-seq_len:, :].copy()
        preds = []
        for _ in range(horizon):
            xinp = buf.reshape(1, seq_len, 1)
            yhat_n = model.predict(xinp, verbose=0)[0,0]
            preds.append(yhat_n*y_den + y_min)
            buf = np.vstack([buf[1:], [[yhat_n]]])
        return np.clip(np.array(preds).ravel(), 0, 1)

# ------------------------- Windows & Seasons -------------------------

def find_low_risk_windows(df, date_col="Date", risk_col="Disease_Presence_Prob", threshold=0.40, min_days=21):
    s = df[[date_col, risk_col]].sort_values(date_col).reset_index(drop=True)
    mask = s[risk_col] < threshold
    runs, start_i = [], None
    for i, ok in enumerate(mask):
        if ok and start_i is None:
            start_i = i
        elif not ok and start_i is not None:
            runs.append((start_i, i-1)); start_i = None
    if start_i is not None:
        runs.append((start_i, len(s)-1))
    out = []
    for a,b in runs:
        start, end = s.loc[a, date_col], s.loc[b, date_col]
        length = (end - start).days + 1
        if length >= min_days:
            seg = s.iloc[a:b+1]
            out.append({"start_date":start, "end_date":end, "length_days":length, "mean_risk":float(seg[risk_col].mean())})
    return pd.DataFrame(out).sort_values(["length_days","mean_risk"], ascending=[False, True])

# ------------------------- Visualization (LINE only) -------------------------

def save_line(x, ys: dict, title, xlabel, ylabel, path, vlines=None, legend=True):
    plt.figure(figsize=(12,5))
    for label, y in ys.items():
        ensure_line_plot(x, y, label=label)
    if vlines:
        for xv, vlabel in vlines:
            plt.axvline(x=xv, linestyle="--", alpha=0.5)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    if legend: plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

# ------------------------- Main Research Pipeline -------------------------

def run_research(
    csv_path="rice_forecast_5yearswithdiseases.csv",
    date_col="Date",
    risk_col="Disease_Presence_Prob",
    disease_col="Predicted_Disease_Type",
    seq_len=28,
    short_horizon=60,
    long_horizon_days=5*365
):
    df = pd.read_csv(csv_path, encoding="ISO-8859-1")
    df = as_datetime_sorted(df, date_col)
    # pick features present
    fmap = find_columns(df, get_feature_map())
    feat_cols = list(fmap.keys())  # canonical keys we'll use in model
    feat_orig_cols = [fmap[k] for k in feat_cols]

    # Assemble modeling frame
    M = df[[date_col, risk_col] + feat_orig_cols + [disease_col]].copy()
    M = M.rename(columns={fmap.get(k,k): k for k in feat_cols})
    # Clean
    for c in [risk_col] + feat_cols:
        M[c] = pd.to_numeric(M[c], errors="coerce")
    M = M.dropna(subset=[risk_col]).reset_index(drop=True)
    M = M.fillna(method="ffill").fillna(method="bfill")

    # ----------------- Short-term (60d) vs Long-term (5y) -----------------
    # Use first part as "training history", test on trailing windows inside the file
    if len(M) < seq_len + short_horizon + 30:
        raise ValueError("Time series too short for 60-day evaluation. Provide more rows.")

    # Short-term test segment: last 60 days
    short_test = M.iloc[-short_horizon:].copy()
    short_train = M.iloc[:-short_horizon].copy()

    # Long-term test segment: last N days (min(long_horizon_days, available-...))
    long_h = min(long_horizon_days, len(M) - (seq_len + 30))
    long_test = M.iloc[-long_h:].copy()
    long_train = M.iloc[:-long_h].copy()

    metrics_short = []
    metrics_long = []

    # ---------- Train Univariate & Multivariate LSTMs ----------
    if not HAS_TF:
        raise RuntimeError("TensorFlow not available. Please install tensorflow to run LSTM modeling.")

    # Univariate on full history (for forecasting later)
    uni_model, uni_hist, uni_scalers = fit_lstm(M, risk_col, feat_cols=[], seq_len=seq_len, val_frac=0.2, tag="uni")
    # Multivariate on full history
    multi_model, multi_hist, multi_scalers = fit_lstm(M, risk_col, feat_cols=feat_cols, seq_len=seq_len, val_frac=0.2, tag="multi")

    # Save training loss curves (impact of features proxy)
    save_line(np.arange(len(uni_hist.history["loss"])),
              {"train_loss":uni_hist.history["loss"], "val_loss":uni_hist.history["val_loss"]},
              "LSTM Univariate – Training/Validation Loss",
              "Epoch", "MSE loss",
              os.path.join(OUT_DIR, "training_loss_univariate.png"))

    save_line(np.arange(len(multi_hist.history["loss"])),
              {"train_loss":multi_hist.history["loss"], "val_loss":multi_hist.history["val_loss"]},
              "LSTM Multivariate – Training/Validation Loss",
              "Epoch", "MSE loss",
              os.path.join(OUT_DIR, "training_loss_multivariate.png"))

    # ---------- Short-term comparison (60d) ----------
    # Forecast 60 days based on history just before the window
    def forecast_from_split(train_df, test_df, horizon):
        base_df = train_df.copy()
        # LSTM uni/multi
        f_uni = forecast_lstm(base_df, risk_col, [], uni_model, uni_scalers, seq_len=seq_len, horizon=horizon)
        f_multi = forecast_lstm(base_df, risk_col, feat_cols, multi_model, multi_scalers, seq_len=seq_len, horizon=horizon)
        # Seasonal-naive
        s_naive = seasonal_naive(base_df[risk_col].values, horizon, season=365)
        return f_uni, f_multi, s_naive

    s_uni, s_multi, s_naive = forecast_from_split(short_train, short_test, short_horizon)
    y_short = short_test[risk_col].values
    rmse = lambda a,b: float(np.sqrt(np.mean((a-b)**2)))
    mae  = lambda a,b: float(np.mean(np.abs(a-b)))

    metrics_short.append({
        "horizon_days": short_horizon,
        "model": "LSTM_univariate", "RMSE": rmse(y_short, s_uni), "MAE": mae(y_short, s_uni)
    })
    metrics_short.append({
        "horizon_days": short_horizon,
        "model": "LSTM_multivariate", "RMSE": rmse(y_short, s_multi), "MAE": mae(y_short, s_multi)
    })
    metrics_short.append({
        "horizon_days": short_horizon,
        "model": "Seasonal-Naive", "RMSE": rmse(y_short, s_naive), "MAE": mae(y_short, s_naive)
    })
    pd.DataFrame(metrics_short).to_csv(os.path.join(OUT_DIR, "metrics_short_term.csv"), index=False)

    save_line(short_test[date_col].values,
              {"Actual": y_short, "LSTM-univariate": s_uni, "LSTM-multivariate": s_multi, "Seasonal-Naive": s_naive},
              f"Short-term ({short_horizon}d) – Actual vs Forecasts",
              "Date", "Risk",
              os.path.join(OUT_DIR, "comparison_short_term.png"))

    # rolling RMSE line (uni vs multi)
    r_uni = rolling_rmse(y_short, s_uni, window=7)
    r_multi = rolling_rmse(y_short, s_multi, window=7)
    save_line(short_test[date_col].values,
              {"RollingRMSE-Uni": r_uni, "RollingRMSE-Multi": r_multi},
              "Short-term 60d – Rolling RMSE (7-day) LSTM Uni vs Multi",
              "Date", "RMSE",
              os.path.join(OUT_DIR, "rolling_rmse_60d_uni_vs_multi.png"))

    # ---------- Long-term comparison (up to 5y) ----------
    l_uni, l_multi, l_naive = forecast_from_split(long_train, long_test, len(long_test))
    y_long = long_test[risk_col].values

    metrics_long.append({
        "horizon_days": len(long_test),
        "model": "LSTM_univariate", "RMSE": rmse(y_long, l_uni), "MAE": mae(y_long, l_uni)
    })
    metrics_long.append({
        "horizon_days": len(long_test),
        "model": "LSTM_multivariate", "RMSE": rmse(y_long, l_multi), "MAE": mae(y_long, l_multi)
    })
    metrics_long.append({
        "horizon_days": len(long_test),
        "model": "Seasonal-Naive", "RMSE": rmse(y_long, l_naive), "MAE": mae(y_long, l_naive)
    })
    pd.DataFrame(metrics_long).to_csv(os.path.join(OUT_DIR, "metrics_long_term.csv"), index=False)

    save_line(long_test[date_col].values,
              {"Actual": y_long, "LSTM-univariate": l_uni, "LSTM-multivariate": l_multi, "Seasonal-Naive": l_naive},
              f"Long-term ({len(long_test)}d) – Actual vs Forecasts",
              "Date", "Risk",
              os.path.join(OUT_DIR, "comparison_long_term.png"))

    # ---------- Feature Impact via Ablations (60d window) ----------
    # Drop one feature at a time and forecast short_horizon, plot cumulative MAE over the 60 days
    ab_lines = {}
    base_mae_cum = np.cumsum(np.abs(y_short - s_multi))
    ab_lines["Full (multi)"] = base_mae_cum
    for f in feat_cols:
        drop_cols = [c for c in feat_cols if c != f]
        # retrain multivariate with features minus f
        tmp_model, _, tmp_scalers = fit_lstm(M, risk_col, feat_cols=drop_cols, seq_len=seq_len, val_frac=0.2, tag=f"ablate_{f}")
        yhat = forecast_lstm(short_train, risk_col, drop_cols, tmp_model, tmp_scalers, seq_len=seq_len, horizon=short_horizon)
        ab_lines[f"-{f}"] = np.cumsum(np.abs(y_short - yhat))
    save_line(np.arange(short_horizon), ab_lines,
              "Feature Impact (60d) – Cumulative Absolute Error",
              "Day", "Cumulative MAE",
              os.path.join(OUT_DIR, "feature_ablation_rolling_mae_60d.png"), legend=True)

    # ----------------- Best Planting Windows & Seasons -----------------
    # Overall windows
    windows = find_low_risk_windows(M, date_col=date_col, risk_col=risk_col, threshold=0.40, min_days=21)
    windows.to_csv(os.path.join(OUT_DIR, "planting_windows_all.csv"), index=False)

    # Per disease type windows (evaluate risk restricted to rows of that type)
    rows = []
    for dname, g in M.groupby(disease_col):
        w = find_low_risk_windows(g, date_col=date_col, risk_col=risk_col, threshold=0.40, min_days=21)
        if not w.empty:
            ww = w.copy()
            ww.insert(0, "disease_type", dname)
            rows.append(ww)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(os.path.join(OUT_DIR, "planting_windows_by_disease.csv"), index=False)

    # Daily risk + threshold (line graph)
    save_line(M[date_col].values,
              {"Risk": M[risk_col].values, "Threshold(0.40)": np.full(len(M), 0.40)},
              "Daily Disease Risk with Threshold",
              "Date", "Risk",
              os.path.join(OUT_DIR, "risk_with_threshold.png"))

    # Risk with Maha/Yala vertical lines
    S = add_season_cols(M, date_col)
    # mark season starts
    season_start_dates = []
    for (sy, sn), grp in S.groupby(["season_year","season"]):
        d0 = grp[date_col].min()
        season_start_dates.append((d0, f"{sn} {sy}"))
    save_line(M[date_col].values, {"Risk": M[risk_col].values},
              "Daily Disease Risk with Maha/Yala Season Boundaries",
              "Date", "Risk",
              os.path.join(OUT_DIR, "risk_with_seasons.png"),
              vlines=[(d, lab) for d, lab in season_start_dates],
              legend=True)

    # ----------------- Scenario Robustness (5-year) -----------------
    # Using multivariate model: perturb rainfall/temperature/humidity during forecasting
    # Baseline forecast of long_h days from the long_train split start
    base_fore = forecast_lstm(long_train, risk_col, feat_cols, multi_model, multi_scalers, seq_len=seq_len, horizon=len(long_test))
    scenarios = {
        "Baseline": {},
        "Wetter(+15% rain)": {"rainfall_scale":1.15},
        "Drier(-10% rain)": {"rainfall_scale":0.90},
        "Warmer(+1°C temp)": {"temperature_add":1.0},
        "Cooler(-0.5°C temp)": {"temperature_add":-0.5},
        "Humid(+5% RH)": {"humidity_scale":1.05},
    }
    lines = {"Actual": y_long}
    for name, sc in scenarios.items():
        yhat = forecast_lstm(long_train, risk_col, feat_cols, multi_model, multi_scalers, seq_len=seq_len, horizon=len(long_test), scenario=sc)
        lines[name] = yhat
    save_line(long_test[date_col].values, lines,
              "Scenario Robustness – Multivariate LSTM (Long-term window)",
              "Date", "Risk",
              os.path.join(OUT_DIR, "scenarios_5y_multivariate.png"))

    # ----------------- Seasonal Patterns by Disease Type -----------------
    MM = M.copy()
    MM["ym"] = MM[date_col].dt.to_period("M").dt.to_timestamp("M")
    monthly = MM.groupby(["ym", disease_col])[risk_col].mean().reset_index()
    # Convert to wide per disease type and plot each as a line over months
    dtypes = list(monthly[disease_col].unique())
    ymap = {}
    idx = pd.to_datetime(sorted(monthly["ym"].unique()))
    for dtp in dtypes:
        series = monthly[monthly[disease_col]==dtp].set_index("ym")[risk_col].reindex(idx, fill_value=np.nan).interpolate().values
        ymap[dtp] = series
    save_line(idx, ymap,
              "Seasonal Patterns – Monthly Mean Risk by Disease Type",
              "Month", "Mean Risk",
              os.path.join(OUT_DIR, "seasonal_patterns_by_disease.png"))

    print("All outputs written to:", os.path.abspath(OUT_DIR))

# ------------------------- Entry -------------------------

if __name__ == "__main__":
    run_research()
