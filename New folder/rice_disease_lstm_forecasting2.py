"""
Rice Disease Research Pipeline — Finalized (No Straight Lines in Graphs)
------------------------------------------------------------------------
All plots use scatter points (no connecting straight line segments).
The uncertainty chart shows only a shaded band (no median line).

Usage
-----
python rice_disease_research_pipeline_final_nolines.py

Or import and run:
from rice_disease_research_pipeline_final_nolines import run_research_pipeline
results = run_research_pipeline(out_dir="outputs")

Inputs (same as your v2 code):
- Rice_Deseases_Factors_550.csv (with Disease Presence, Disease Type, and agro-weather factors)
- rice_disease_data.csv (daily weather features + Date)

Outputs (key files in out_dir)
- historical_risk.csv
- forecast_risk_5y_daily.csv (day-by-day for 5 years)
- forecast_5y_uncertainty.csv (q05/q50/q95)
- forecast_5y_scenarios.csv (baseline + scenario columns)
- forecast_5y_low_risk_windows.csv (good periods with less disease)
- forecast_5y_disease_types.csv (daily per-disease) & _monthly.csv
- backtest_metrics.csv, backtest_rmse_line_*.png, backtest_final_fold_pred_line.png (now scatter)
- MANY additional scatter-based charts
"""

import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from datetime import timedelta

# ===== Optional libs =====
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except Exception:
    HAS_TF = False

try:
    import statsmodels.api as sm
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_SM = True
except Exception:
    HAS_SM = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ========================= Utils =========================

def ensure_output_dir(path="outputs"):
    os.makedirs(path, exist_ok=True)
    return path

def canonicalize(col: str) -> str:
    c = (col or "").strip().lower().replace("Â°", "°")
    for ch in ["(", ")", "%", "/", "-", "_"]:
        c = c.replace(ch, " ")
    return " ".join(c.split())

def add_season_cols(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    d = pd.to_datetime(df[date_col])
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    season = []
    season_year = []
    for dt in d:
        m = dt.month
        y = dt.year
        if m >= 10 or m <= 3:
            season.append("Maha")
            sy = y if m >= 10 else y - 1
            season_year.append(sy)
        else:
            season.append("Yala")
            season_year.append(y)
    df["season"] = season
    df["season_year"] = season_year
    return df

def infer_binary(series):
    s = series.copy()
    if s.dtype.kind in "biufc":
        return (s > 0).astype(int)
    s = s.astype(str).str.strip().str.lower()
    mapping = {
        "yes":1, "y":1, "true":1, "present":1, "1":1, "pos":1, "positive":1,
        "no":0,  "n":0, "false":0, "absent":0,  "0":0, "neg":0, "negative":0
    }
    return s.map(mapping).fillna(0).astype(int)

def get_common_feature_map():
    return {
        "temperature": ["temperature °c","temperature c","avg temperature","average temperature","maximum temperature","minimum temperature"],
        "humidity": ["humidity","relative humidity","avg relative humidity","average relative humidity"],
        "rainfall": ["rainfall mm","total rainfall","rainfall","precipitation"],
        "wind_speed": ["wind speed m s","average wind speed","wind speed"],
        "ph": ["ph level","average ph","ph","maximam ph","maximum ph","minimum ph","min ph"],
        "soil_moisture": ["soil moisture","soil moisture ","avg soil moisture","soil moisture percent"],
        "nitrogen": ["nitrogen content mg kg","nitrogen"],
        "potassium": ["potassium content mg kg","potassium"],
        "salinity": ["sanility","salinity","average sanility","maximum sanility","minimum sanility"]
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

# ---------- Plotting helper (NO STRAIGHT LINES) ----------
def scatter_series(x, y, label=None, s=10, alpha=0.9):
    """Scatter-only renderer to avoid any straight connecting lines."""
    plt.scatter(x, y, s=s, alpha=alpha, label=label)

# ========================= Data Loading =========================

def load_datasets(factors_path="Rice_Deseases_Factors_550.csv", 
                  weather_path="rice_forecast_5years.csv"):
    fdf = pd.read_csv(factors_path)
    wdf = pd.read_csv(weather_path, encoding="ISO-8859-1")

    fcols = {canonicalize(c): c for c in fdf.columns}
    wcols = {canonicalize(c): c for c in wdf.columns}

    def find_col_like(cands, cmap):
        for c in cands:
            if c in cmap: return cmap[c]
        return None

    presence_col = find_col_like(["disease presence"], fcols)
    dtype_col    = find_col_like(["disease type"], fcols)
    date_col     = find_col_like(["date"], wcols)
    if presence_col is None:
        raise ValueError("Missing 'Disease Presence' in factors dataset")
    if dtype_col is None:
        raise ValueError("Missing 'Disease Type' in factors dataset")
    if date_col is None:
        raise ValueError("Missing 'Date' in rice_disease_data.csv")

    wanted = get_common_feature_map()
    fmap = find_columns(fdf, wanted)
    wmap = find_columns(wdf, wanted)
    common = sorted(set(fmap).intersection(set(wmap)))
    if not common:
        raise ValueError("No common feature columns between datasets")

    fX = pd.DataFrame({k: pd.to_numeric(fdf[fmap[k]], errors="coerce") for k in common})
    wX = pd.DataFrame({k: pd.to_numeric(wdf[wmap[k]], errors="coerce") for k in common})

    y_presence = infer_binary(fdf[presence_col])
    y_type_raw = fdf[dtype_col].astype(str).str.strip()

    w_dates = pd.to_datetime(wdf[date_col], errors="coerce")
    if w_dates.isna().any():
        raise ValueError("Some 'Date' values could not be parsed")

    fX = fX.replace([np.inf,-np.inf], np.nan).fillna(fX.median(numeric_only=True))
    wX = wX.replace([np.inf,-np.inf], np.nan).fillna(wX.median(numeric_only=True))
    return fX, y_presence, y_type_raw, wX, w_dates, common

# ========================= Presence & Type Models =========================

def train_presence_model(fX, y_presence, random_state=42):
    Xtr, Xte, ytr, yte = train_test_split(fX.values, y_presence.values, test_size=0.2, random_state=random_state, stratify=y_presence.values)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    if HAS_XGB:
        base = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.07,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=random_state, n_jobs=-1
        )
    else:
        from sklearn.ensemble import RandomForestClassifier
        base = RandomForestClassifier(n_estimators=600, min_samples_leaf=2, random_state=random_state, n_jobs=-1)

    clf = CalibratedClassifierCV(base, method="isotonic", cv=5)
    clf.fit(Xtr_s, ytr)

    # quick print of calibration quality
    proba = clf.predict_proba(Xte_s)[:,1]
    pred = (proba>=0.5).astype(int)
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss
    print("\n=== Presence Model ===")
    print("Accuracy:", accuracy_score(yte, pred))
    print("Balanced Accuracy:", balanced_accuracy_score(yte, pred))
    print("Brier Score:", brier_score_loss(yte, proba))

    return {"scaler":scaler, "clf":clf, "is_xgb":HAS_XGB}

def train_type_model(fX, y_presence, y_type_raw, random_state=42):
    mask = (y_presence.values == 1)
    if mask.sum() < 20:
        print("Not enough positive samples for type model — skipping")
        return None, None

    classes, y_type = np.unique(y_type_raw[mask], return_inverse=True)
    X_pos = fX.values[mask]

    Xtr, Xte, ytr, yte = train_test_split(X_pos, y_type, test_size=0.2, random_state=random_state, stratify=y_type)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    if HAS_XGB:
        base = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.07,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=random_state, n_jobs=-1
        )
    else:
        from sklearn.ensemble import RandomForestClassifier
        base = RandomForestClassifier(n_estimators=600, min_samples_leaf=2, random_state=random_state, n_jobs=-1)

    clf = CalibratedClassifierCV(base, method="isotonic", cv=5)
    clf.fit(Xtr_s, ytr)

    acc = (clf.predict(Xte_s) == yte).mean()
    print("\n=== Disease Type Model (positives only) ===\nAccuracy:", acc)
    return {"scaler":scaler, "clf":clf, "classes":classes}, classes

# ========================= Explainability =========================

def explain_global_importance(model_bundle, fX, feature_names, out_dir="outputs"):
    clf = model_bundle["clf"]
    scaler = model_bundle["scaler"]
    Xs = scaler.transform(fX.values)

    try:
        if HAS_SHAP:
            try:
                explainer = shap.Explainer(getattr(clf, "estimators_", [clf])[0])
                sv = explainer(Xs)
                mean_abs = np.abs(sv.values).mean(axis=0)
                imp = pd.DataFrame({"feature":feature_names, "mean_abs_shap":mean_abs}).sort_values("mean_abs_shap", ascending=False)
                imp.to_csv(os.path.join(out_dir, "shap_global_importance.csv"), index=False)
                plt.figure(figsize=(8,4))
                scatter_series(imp["feature"], imp["mean_abs_shap"])
                plt.xticks(rotation=30, ha="right")
                plt.title("Global Importance (mean |SHAP|)")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "shap_importance_scatter.png"), dpi=150)
                plt.close()
                return
            except Exception as e:
                print("SHAP failed, continuing with variance proxy.", e)
        # fallback: variance proxy
        var = np.var(Xs, axis=0)
        imp = pd.DataFrame({"feature":feature_names, "importance":var}).sort_values("importance", ascending=False)
        imp.to_csv(os.path.join(out_dir, "feature_variance_importance.csv"), index=False)
        plt.figure(figsize=(8,4))
        scatter_series(imp["feature"], imp["importance"])
        plt.xticks(rotation=30, ha="right")
        plt.title("Feature Variance (proxy importance)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "feature_variance_scatter.png"), dpi=150)
        plt.close()
    except Exception as e:
        print("Explainability step skipped:", e)

# ========================= Historical Scoring =========================

def score_historical_risk(presence_bundle, wX, w_dates):
    clf = presence_bundle["clf"]
    scaler = presence_bundle["scaler"]
    Xs = scaler.transform(wX.values)
    risk = clf.predict_proba(Xs)[:,1]
    hist = pd.DataFrame({"Date": w_dates, "risk_prob": risk}).sort_values("Date").reset_index(drop=True)
    return hist

# ========================= Forecasting Models =========================

def seasonal_naive_forecast(hist_df, horizon_days):
    last_year = hist_df.tail(365)["risk_prob"].to_numpy()
    if len(last_year) < 365:
        last_year = np.resize(last_year, 365)
    reps = int(np.ceil(horizon_days/365))
    vals = np.tile(last_year, reps)[:horizon_days]
    start = hist_df["Date"].max() + pd.Timedelta(days=1)
    fut_dates = pd.date_range(start, periods=horizon_days, freq="D")
    return pd.DataFrame({"Date":fut_dates, "risk_prob":vals})

def sarima_forecast(hist_df, horizon_days):
    if not HAS_SM:
        return None
    try:
        series = hist_df.set_index("Date")["risk_prob"].asfreq("D").interpolate()
        m = 7  # weekly seasonality proxy; yearly would be heavy
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,m), enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        fut = res.get_forecast(steps=horizon_days)
        dates = pd.date_range(series.index.max()+timedelta(days=1), periods=horizon_days, freq="D")
        return pd.DataFrame({"Date":dates, "risk_prob":np.clip(fut.predicted_mean.values, 0, 1)})
    except Exception:
        return None

def lstm_forecast(hist_df, horizon_days=60, seq_len=28, random_state=42):
    if not HAS_TF:
        return None
    series = hist_df["risk_prob"].astype(float).values.reshape(-1,1)
    s_min, s_max = series.min(), series.max()
    denom = (s_max - s_min) if s_max > s_min else 1.0
    s_norm = (series - s_min) / denom

    X, y = [], []
    for i in range(len(s_norm)-seq_len):
        X.append(s_norm[i:i+seq_len,0])
        y.append(s_norm[i+seq_len,0])
    if len(X) < 50:  # not enough
        return None
    X = np.array(X); y = np.array(y)
    split = int(len(X)*0.85)
    Xtr, Xval = X[:split], X[split:]
    ytr, yval = y[:split], y[split:]
    Xtr = Xtr.reshape((Xtr.shape[0], Xtr.shape[1], 1))
    Xval = Xval.reshape((Xval.shape[0], Xval.shape[1], 1))

    tf.random.set_seed(random_state)
    model = Sequential([
        LSTM(64, activation="tanh", input_shape=(seq_len,1)),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=0)
    model.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=200, batch_size=32, callbacks=[es], verbose=0)

    last_seq = s_norm[-seq_len:,0].tolist()
    preds = []
    for _ in range(horizon_days):
        x = np.array(last_seq[-seq_len:]).reshape(1, seq_len, 1)
        yhat = model.predict(x, verbose=0)[0,0]
        preds.append(yhat)
        last_seq.append(yhat)

    preds = np.array(preds).reshape(-1,1)*denom + s_min
    fut_dates = pd.date_range(hist_df["Date"].max()+timedelta(days=1), periods=horizon_days, freq="D")
    return pd.DataFrame({"Date":fut_dates, "risk_prob":preds.ravel()})

def choose_and_forecast(hist_df, horizon_days=5*365+1, seq_len=28, out_dir="outputs"):
    fut = lstm_forecast(hist_df, horizon_days=horizon_days, seq_len=seq_len)
    model_name = "LSTM" if fut is not None else None
    if fut is None:
        fut = sarima_forecast(hist_df, horizon_days)
        model_name = "SARIMA" if fut is not None else None
    if fut is None:
        fut = seasonal_naive_forecast(hist_df, horizon_days)
        model_name = "Seasonal-Naive"
    pd.Series({"chosen_model":model_name}).to_csv(os.path.join(out_dir, "chosen_forecast_model.csv"))
    return fut, model_name

# ========================= Backtesting =========================

def walk_forward_backtest(hist_df, k_folds=6, horizon=30, seq_len=28, out_dir="outputs"):
    """Walk-forward on historical risk to compare models. Produces scatter charts (no connecting lines)."""
    series = hist_df.sort_values("Date").reset_index(drop=True)
    n = len(series)
    fold_size = max(int(n*0.1), horizon+seq_len+5)
    metrics = []

    for fold in range(k_folds):
        end = min(n, (fold+1)*fold_size)
        start = max(0, end - fold_size)
        train = series.iloc[start:end-horizon]
        test  = series.iloc[end-horizon:end]
        if len(train) < (seq_len+60) or len(test) < horizon:
            continue
        fut_lstm  = lstm_forecast(train, horizon_days=horizon, seq_len=seq_len) or pd.DataFrame()
        fut_sar   = sarima_forecast(train, horizon_days=horizon) or pd.DataFrame()
        fut_naive = seasonal_naive_forecast(train, horizon_days=horizon)

        actual = test["risk_prob"].values
        preds = {
            "LSTM": fut_lstm["risk_prob"].values if not fut_lstm.empty else None,
            "SARIMA": fut_sar["risk_prob"].values if not fut_sar.empty else None,
            "Seasonal-Naive": fut_naive["risk_prob"].values,
        }
        for name, p in preds.items():
            if p is None or len(p) != len(actual):
                continue
            rmse = mean_squared_error(actual, p, squared=False)
            mae  = mean_absolute_error(actual, p)
            r2   = r2_score(actual, p)
            metrics.append({"fold":fold+1, "model":name, "RMSE":rmse, "MAE":mae, "R2":r2})

        # SCATTER: actual vs each model
        days = test["Date"].values
        plt.figure(figsize=(12,4))
        scatter_series(days, actual, label="Actual")
        if preds["LSTM"] is not None: scatter_series(days, preds["LSTM"], label="LSTM")
        if preds["SARIMA"] is not None: scatter_series(days, preds["SARIMA"], label="SARIMA")
        scatter_series(days, preds["Seasonal-Naive"], label="Seasonal-Naive")
        plt.title(f"Backtest Fold {fold+1}: Actual vs Forecast")
        plt.xlabel("Date"); plt.ylabel("Risk"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"backtest_fold_{fold+1}_scatter.png"), dpi=150)
        plt.close()

        if fold == k_folds-1:
            plt.figure(figsize=(12,4))
            scatter_series(days, actual, label="Actual")
            if preds["LSTM"] is not None: scatter_series(days, preds["LSTM"], label="LSTM")
            if preds["SARIMA"] is not None: scatter_series(days, preds["SARIMA"], label="SARIMA")
            scatter_series(days, preds["Seasonal-Naive"], label="Seasonal-Naive")
            plt.title("Backtest Final Fold — Actual vs Forecast")
            plt.xlabel("Date"); plt.ylabel("Risk"); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "backtest_final_fold_pred_scatter.png"), dpi=150)
            plt.close()

    if not metrics:
        return pd.DataFrame()
    mdf = pd.DataFrame(metrics)
    mdf.to_csv(os.path.join(out_dir, "backtest_metrics.csv"), index=False)

    # SCATTER: RMSE by fold for each model
    try:
        for name, g in mdf.groupby("model"):
            g = g.sort_values("fold")
            plt.figure(figsize=(10,4))
            scatter_series(g["fold"], g["RMSE"])
            plt.title(f"Backtest RMSE by Fold — {name}")
            plt.xlabel("Fold"); plt.ylabel("RMSE"); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"backtest_rmse_scatter_{name}.png"), dpi=150)
            plt.close()
    except Exception:
        pass
    return mdf

# ========================= Uncertainty via Bootstrap =========================

def bootstrap_uncertainty(hist_df, fut_df, base_model_name, n_boot=200, out_dir="outputs"):
    """Residual bootstrap around chosen model using naive one-step residuals as proxy."""
    history = hist_df.sort_values("Date").reset_index(drop=True)
    if len(history) < 365+1:
        residuals = history["risk_prob"].diff().dropna().values  # fallback
    else:
        shifted = history["risk_prob"].shift(365)
        residuals = (history["risk_prob"] - shifted).dropna().values
    if residuals.size == 0:
        residuals = history["risk_prob"].diff().dropna().values
    if residuals.size == 0:
        residuals = np.array([0.0])

    rng = np.random.default_rng(42)
    sims = []
    base = fut_df["risk_prob"].values
    for _ in range(n_boot):
        noise = rng.choice(residuals, size=len(base), replace=True)
        sim = np.clip(base + noise, 0, 1)
        sims.append(sim)
    sims = np.array(sims)
    q05 = np.quantile(sims, 0.05, axis=0)
    q50 = np.quantile(sims, 0.50, axis=0)
    q95 = np.quantile(sims, 0.95, axis=0)

    u = pd.DataFrame({"Date":fut_df["Date"], "q05":q05, "q50":q50, "q95":q95})
    u.to_csv(os.path.join(out_dir, "forecast_5y_uncertainty.csv"), index=False)

    # SHADED BAND ONLY (no median line)
    plt.figure(figsize=(12,5))
    plt.fill_between(u["Date"], u["q05"], u["q95"], alpha=0.2, label="5–95% band")
    plt.title("5-Year Forecast — Uncertainty Band (No Line)")
    plt.xlabel("Date"); plt.ylabel("Risk"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "forecast_5y_uncertainty_band.png"), dpi=150)
    plt.close()
    return u

# ========================= Scenario Analysis =========================

def scenario_analysis(fut_df, scales=None, out_dir="outputs"):
    if scales is None:
        scales = {"baseline":1.0, "wetter":1.15, "drier":0.9, "warmer":1.1, "cooler":0.95}
    S = pd.DataFrame({"Date":fut_df["Date"]})
    for name, s in scales.items():
        S[name] = np.clip(fut_df["risk_prob"].values * s, 0, 1)
    S.to_csv(os.path.join(out_dir, "forecast_5y_scenarios.csv"), index=False)

    # Monthly means per scenario (SCATTER)
    Sm = S.copy()
    Sm["ym"] = Sm["Date"].dt.to_period("M").dt.to_timestamp("M")
    lines = []
    for name in scales.keys():
        m = Sm.groupby("ym")[name].mean().reset_index().rename(columns={name:"risk"})
        m["scenario"] = name
        lines.append(m)
    M = pd.concat(lines, ignore_index=True)
    for name in scales.keys():
        g = M[M["scenario"]==name]
        plt.figure(figsize=(12,4))
        scatter_series(g["ym"], g["risk"])
        plt.title(f"Monthly Mean Risk — Scenario: {name}")
        plt.xlabel("Month"); plt.ylabel("Mean risk"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"scenario_monthly_scatter_{name}.png"), dpi=150)
        plt.close()

    # Combined SCATTER
    plt.figure(figsize=(12,5))
    for name in scales.keys():
        g = M[M["scenario"]==name]
        scatter_series(g["ym"], g["risk"], label=name)
    plt.title("Monthly Mean Risk — All Scenarios")
    plt.xlabel("Month"); plt.ylabel("Mean risk"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scenario_monthly_scatter_all.png"), dpi=150)
    plt.close()
    return S, M

# ========================= Low-Risk Windows =========================

def find_low_risk_windows(df, date_col="Date", risk_col="risk_prob", threshold=0.40, min_days=21):
    s = df[[date_col, risk_col]].sort_values(date_col).reset_index(drop=True)
    mask = s[risk_col] < threshold
    runs = []
    start_i = None
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

# ========================= Disease-Type Decomposition =========================

def disease_type_decomposition(fut_df, y_type_raw=None, type_classes=None, out_dir="outputs"):
    if type_classes is None or y_type_raw is None:
        return None, None
    hist_types = pd.Series(y_type_raw).value_counts(normalize=True)
    weights = {cls: hist_types.get(cls, 0) for cls in type_classes}
    total = sum(weights.values())
    if total == 0:
        weights = {cls: 1/len(type_classes) for cls in type_classes}
    else:
        weights = {cls: v/total for cls, v in weights.items()}

    probs = fut_df[["Date"]].copy()
    for cls in type_classes:
        probs[cls] = fut_df["risk_prob"] * weights[cls]
    probs.to_csv(os.path.join(out_dir, "forecast_5y_disease_types.csv"), index=False)

    # Monthly summary + SCATTER charts
    probs["ym"] = probs["Date"].dt.to_period("M").dt.to_timestamp("M")
    monthly = probs.groupby("ym")[list(type_classes)].mean().reset_index()
    monthly.to_csv(os.path.join(out_dir, "forecast_5y_disease_types_monthly.csv"), index=False)

    # Combined multi-scatter
    plt.figure(figsize=(12,5))
    for cls in type_classes:
        scatter_series(monthly["ym"], monthly[cls], label=cls)
    plt.title("Disease-Type Monthly Average — 5Y Forecast")
    plt.xlabel("Month"); plt.ylabel("Avg probability"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "disease_types_monthly_all_scatter.png"), dpi=150)
    plt.close()

    # Individual SCATTER per disease
    for cls in type_classes:
        plt.figure(figsize=(12,4))
        scatter_series(monthly["ym"], monthly[cls])
        plt.title(f"{cls} — Monthly Average (5Y Forecast)")
        plt.xlabel("Month"); plt.ylabel("Avg probability"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"disease_type_{cls}_monthly_scatter.png"), dpi=150)
        plt.close()

    return probs, monthly

# ========================= Economic Value =========================

def economic_value_analysis(fut_df, low_windows, window_days=120, disease_loss_rate=0.15, yield_value_per_ha=100000.0, out_dir="outputs"):
    """
    Simple expected loss: sum(risk) * disease_loss_rate * yield_value_per_ha
    Compare baseline (first 120 days from forecast start) vs each low-risk window (first 120 days within window).
    """
    if fut_df.empty or low_windows is None or low_windows.empty:
        return None

    start = fut_df["Date"].min()
    base_seg = fut_df[(fut_df["Date"]>=start) & (fut_df["Date"]<start+pd.Timedelta(days=window_days))]
    base_loss = base_seg["risk_prob"].sum() * disease_loss_rate * yield_value_per_ha

    rows = []
    for _, row in low_windows.iterrows():
        seg = fut_df[(fut_df["Date"]>=row["start_date"]) & (fut_df["Date"]<=row["end_date"])].head(window_days)
        if len(seg) < window_days:
            continue
        loss = seg["risk_prob"].sum() * disease_loss_rate * yield_value_per_ha
        rows.append({
            "start_date": row["start_date"],
            "end_date": seg["Date"].iloc[-1],
            "length_days": len(seg),
            "mean_risk": seg["risk_prob"].mean(),
            "expected_loss": loss,
            "expected_savings_vs_baseline": max(0.0, base_loss - loss)
        })
    ev = pd.DataFrame(rows).sort_values(["expected_savings_vs_baseline","length_days"], ascending=[False, False])
    ev.to_csv(os.path.join(out_dir, "economic_value_by_window.csv"), index=False)

    if not ev.empty:
        ev = ev.reset_index(drop=True)
        # SCATTER: savings vs window rank
        plt.figure(figsize=(10,4))
        scatter_series(np.arange(1, len(ev)+1), ev["expected_savings_vs_baseline"])
        plt.title("Expected Savings vs Window Rank (120-day horizon)")
        plt.xlabel("Window rank"); plt.ylabel("Expected savings"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "economic_savings_scatter.png"), dpi=150)
        plt.close()

        # SCATTER: mean risk across top-N windows ordered by start date
        topN = ev.head(10).sort_values("start_date")
        plt.figure(figsize=(12,4))
        scatter_series(pd.to_datetime(topN["start_date"]), topN["mean_risk"])
        plt.title("Mean Risk across Top 10 Windows (by start date)")
        plt.xlabel("Start date"); plt.ylabel("Mean risk"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "economic_top_windows_mean_risk_scatter.png"), dpi=150)
        plt.close()
    return ev

# ========================= Visualization: Core (All Scatter) =========================

def core_scatter_charts(hist_df, fut_df, out_dir="outputs"):
    # Historical & Forecast (scatter)
    plt.figure(figsize=(12,5))
    scatter_series(hist_df["Date"], hist_df["risk_prob"], label="Historical")
    if not fut_df.empty:
        scatter_series(fut_df["Date"], fut_df["risk_prob"], label="Forecast")
    plt.title("Disease Risk: Historical & 5Y Forecast (Scatter)")
    plt.xlabel("Date"); plt.ylabel("Risk"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scatter_hist_vs_forecast.png"), dpi=150)
    plt.close()

    # Rolling 7d & 30d (scatter of rolling means)
    tmp = pd.concat([hist_df.assign(part="hist"), fut_df.assign(part="fut")], ignore_index=True)
    tmp = tmp.sort_values("Date")
    tmp["roll7"] = tmp["risk_prob"].rolling(7, min_periods=1).mean()
    tmp["roll30"] = tmp["risk_prob"].rolling(30, min_periods=1).mean()
    plt.figure(figsize=(12,5))
    scatter_series(tmp["Date"], tmp["roll7"], label="7-day rolling")
    scatter_series(tmp["Date"], tmp["roll30"], label="30-day rolling")
    plt.title("Rolling Mean Risk (7d & 30d) — Scatter")
    plt.xlabel("Date"); plt.ylabel("Rolling mean risk"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scatter_rolling_7_30.png"), dpi=150)
    plt.close()

    # Monthly mean risk (scatter)
    tmp["ym"] = tmp["Date"].dt.to_period("M").dt.to_timestamp("M")
    m = tmp.groupby("ym")["risk_prob"].mean().reset_index()
    plt.figure(figsize=(12,4))
    scatter_series(m["ym"], m["risk_prob"])
    plt.title("Average Monthly Risk (History + Forecast) — Scatter")
    plt.xlabel("Month"); plt.ylabel("Mean risk"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scatter_monthly_mean_all.png"), dpi=150)
    plt.close()

    # Season-year scatter (Maha vs Yala)
    s = add_season_cols(tmp, "Date").groupby(["season_year","season"], as_index=False)["risk_prob"].mean()
    pv = s.pivot(index="season_year", columns="season", values="risk_prob").sort_index()
    plt.figure(figsize=(10,4))
    for col in pv.columns:
        scatter_series(pv.index, pv[col], label=col)
    plt.title("Average Risk by Season-Year (Maha vs Yala) — Scatter")
    plt.xlabel("Season year"); plt.ylabel("Mean risk"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scatter_season_year.png"), dpi=150)
    plt.close()

# ========================= Orchestration =========================

def run_research_pipeline(out_dir="outputs", seq_len=28, k_backtest_folds=6, backtest_horizon=30):
    out_dir = ensure_output_dir(out_dir)
    # 1) Load & align
    fX, y_presence, y_type_raw, wX, w_dates, feature_names = load_datasets()

    # 2) Train presence + (optional) type
    presence_bundle = train_presence_model(fX, y_presence)
    type_bundle, type_classes = train_type_model(fX, y_presence, y_type_raw)

    # 3) Explainability
    explain_global_importance(presence_bundle, fX, feature_names, out_dir)

    # 4) Historical scoring
    hist_df = score_historical_risk(presence_bundle, wX, w_dates)
    hist_df.to_csv(os.path.join(out_dir, "historical_risk.csv"), index=False)

    # 5) Backtesting (scatter per fold + RMSE scatter)
    backtest_df = walk_forward_backtest(hist_df, k_folds=k_backtest_folds, horizon=backtest_horizon, seq_len=seq_len, out_dir=out_dir)

    # 6) Forecast 5 years (choose best available model)
    horizon_days = max((pd.Timestamp(hist_df["Date"].max()) + pd.DateOffset(years=5) - pd.Timestamp(hist_df["Date"].max())).days, 5*365+1)
    fut_df, chosen_model = choose_and_forecast(hist_df, horizon_days=horizon_days, seq_len=seq_len, out_dir=out_dir)
    fut_df.to_csv(os.path.join(out_dir, "forecast_risk_5y_daily.csv"), index=False)

    # 7) Uncertainty bands (shaded band only)
    unc = bootstrap_uncertainty(hist_df, fut_df, chosen_model, n_boot=200, out_dir=out_dir)

    # 8) Low-risk windows
    low_windows = find_low_risk_windows(fut_df, threshold=0.40, min_days=21)
    low_windows.to_csv(os.path.join(out_dir, "forecast_5y_low_risk_windows.csv"), index=False)

    # 9) Scenario analysis (scatter)
    scen_daily, scen_monthly = scenario_analysis(fut_df, out_dir=out_dir)

    # 10) Disease-type decomposition (scatter)
    dtype_daily, dtype_monthly = disease_type_decomposition(fut_df, y_type_raw=y_type_raw, type_classes=type_classes, out_dir=out_dir)

    # 11) Economic value (scatter)
    econ = economic_value_analysis(fut_df, low_windows, out_dir=out_dir)

    # 12) Core scatter charts
    core_scatter_charts(hist_df, fut_df, out_dir)

    # Save summaries
    summary = {
        "chosen_model": chosen_model,
        "forecast_mean_risk": float(fut_df["risk_prob"].mean()),
        "forecast_max_risk": float(fut_df["risk_prob"].max()),
        "forecast_min_risk": float(fut_df["risk_prob"].min()),
        "high_risk_days_gt_0_6": int((fut_df["risk_prob"]>0.6).sum()),
        "n_low_risk_windows": int(len(low_windows))
    }
    pd.Series(summary).to_csv(os.path.join(out_dir, "final_summary.csv"))

    print("\n=== Final Summary ===")
    for k,v in summary.items():
        print(f"{k}: {v}")
    print("Outputs saved to:", os.path.abspath(out_dir))

    return {
        "presence_bundle": presence_bundle,
        "type_bundle": type_bundle,
        "type_classes": type_classes,
        "hist_df": hist_df,
        "backtest_df": backtest_df,
        "fut_df": fut_df,
        "uncertainty": unc,
        "low_windows": low_windows,
        "scenarios_daily": scen_daily,
        "scenarios_monthly": scen_monthly,
        "disease_daily": dtype_daily,
        "disease_monthly": dtype_monthly,
        "economic_value": econ,
        "summary": summary
    }

def main():
    run_research_pipeline(out_dir="outputs")

if __name__ == "__main__":
    main()
