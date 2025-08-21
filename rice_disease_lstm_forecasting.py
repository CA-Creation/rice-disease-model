# rice_disease_full_pipeline.py
# End-to-end rice disease risk modeling, season-aware analysis, SHAP explainability, and LSTM forecasting.
# Author: <you>
# Usage: python rice_disease_full_pipeline.py

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import timedelta

# --- ML imports ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, accuracy_score, balanced_accuracy_score,
    confusion_matrix, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

# Try XGBoost (preferred); graceful fallback to RF
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Try SHAP; graceful fallback
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# Try TensorFlow/Keras for LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except Exception:
    HAS_TF = False


# =============== Utilities ===============

def ensure_output_dir(path="outputs"):
    os.makedirs(path, exist_ok=True)
    return path

def canonicalize(col: str) -> str:
    """Normalize messy column names to a canonical key."""
    c = (col or "").strip().lower()
    # fix common mojibake: Â°C -> °c
    c = c.replace("Â°", "°")
    # remove brackets/percent signs/spaces
    c = c.replace("(", " ").replace(")", " ").replace("%", "").replace("/", " ").replace("-", " ").replace("_", " ")
    c = " ".join(c.split())  # squeeze spaces
    return c

def add_season_cols(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add Sri Lanka seasons: Maha (Oct–Mar) / Yala (Apr–Sep)."""
    df = df.copy()
    d = pd.to_datetime(df[date_col])
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    # Maha spans Oct-Mar (season year labeled by the Oct year)
    season = []
    season_year = []
    for dt in d:
        m = dt.month
        y = dt.year
        if m >= 10 or m <= 3:
            season.append("Maha")
            # For Jan-Mar, season year is previous year's Oct
            sy = y if m >= 10 else y - 1
            season_year.append(sy)
        else:
            season.append("Yala")
            season_year.append(y)
    df["season"] = season
    df["season_year"] = season_year
    return df

def infer_binary(series):
    """Coerce disease presence to 0/1 robustly."""
    s = series.copy()
    if s.dtype.kind in "biufc":
        return (s > 0).astype(int)
    s = s.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1, "y": 1, "true": 1, "present": 1, "1": 1, "pos": 1, "positive": 1,
        "no": 0, "n": 0, "false": 0, "absent": 0, "0": 0, "neg": 0, "negative": 0
    }
    return s.map(mapping).fillna(0).astype(int)

def get_common_feature_map():
    """
    Map diverse column names to canonical keys used by the classifier & scoring.
    Canonical keys we try to use across both datasets:
    ['temperature', 'humidity', 'rainfall', 'wind_speed', 'ph', 'soil_moisture', 'nitrogen', 'potassium']
    """
    # keys are canonical; values are lists of possible raw tokens
    return {
        "temperature": ["temperature °c", "temperature c", "avg temperature", "average temperature", "maximum temperature", "minimum temperature"],
        "humidity": ["humidity", "relative humidity", "avg relative humidity", "average relative humidity"],
        "rainfall": ["rainfall mm", "total rainfall", "rainfall", "precipitation"],
        "wind_speed": ["wind speed m s", "average wind speed", "wind speed"],
        "ph": ["ph level", "average ph", "ph", "maximam ph", "maximum ph", "minimum ph", "min ph"],
        "soil_moisture": ["soil moisture", "soil moisture ", "avg soil moisture", "soil moisture percent"],
        "nitrogen": ["nitrogen content mg kg", "nitrogen"],
        "potassium": ["potassium content mg kg", "potassium"],
        # Sometimes datasets include "salinity" but not moisture; we won't equate them,
        # but we will expose it as an optional feature if present in BOTH.
        "salinity": ["sanility", "salinity", "average sanility", "maximum sanility", "minimum sanility"]
    }

def find_columns(df: pd.DataFrame, wanted: dict) -> dict:
    """Find best matching columns in df for each canonical key."""
    canon_cols = {canonicalize(c): c for c in df.columns}
    match = {}
    for key, options in wanted.items():
        for opt in options:
            if opt in canon_cols:
                match[key] = canon_cols[opt]
                break
    return match


# =============== 1) Load & Harmonize Data ===============

def load_datasets(
    factors_path="Rice_Deseases_Factors_550.csv",
    weather_path="rice_disease_data.csv"
):
    # Read
    factors_df = pd.read_csv(factors_path)
    weather_df = pd.read_csv(weather_path, encoding='ISO-8859-1')


    # Canonicalize columns to search
    f_cols = {canonicalize(c): c for c in factors_df.columns}
    w_cols = {canonicalize(c): c for c in weather_df.columns}

    # Required targets in factors_df
    # 'Disease Presence' and 'Disease Type' (case/space robust)
    # find original names
    def find_col_like(candidates, colmap):
        for cand in candidates:
            if cand in colmap:
                return colmap[cand]
        return None

    disease_presence_col = find_col_like(["disease presence"], f_cols)
    disease_type_col     = find_col_like(["disease type"], f_cols)

    if disease_presence_col is None:
        raise ValueError("Could not find 'Disease Presence' column in factors dataset.")
    if disease_type_col is None:
        raise ValueError("Could not find 'Disease Type' column in factors dataset.")

    # Build feature maps
    wanted = get_common_feature_map()
    f_map = find_columns(factors_df, wanted)
    w_map = find_columns(weather_df, wanted)

    # Keep only intersection of features available in BOTH datasets (to allow scoring weather_df)
    common_keys = sorted(set(f_map.keys()).intersection(set(w_map.keys())))
    # Remove "salinity" unless it's also present in the factors AND weather (safe) — it is already handled by intersection
    if not common_keys:
        raise ValueError("No common feature columns found between the two datasets. Please check column names.")

    # Create aligned feature tables
    fX = pd.DataFrame()
    for key in common_keys:
        fX[key] = pd.to_numeric(factors_df[f_map[key]], errors="coerce")

    # Targets
    y_presence = infer_binary(factors_df[disease_presence_col])
    y_type_raw = factors_df[disease_type_col].astype(str).str.strip()

    # For weather_df, also ensure Date column
    date_col = find_col_like(["date"], w_cols)
    if date_col is None:
        raise ValueError("Could not find a 'Date' column in rice_disease_data.csv")
    wX = pd.DataFrame()
    for key in common_keys:
        wX[key] = pd.to_numeric(weather_df[w_map[key]], errors="coerce")

    # Attach date
    w_dates = pd.to_datetime(weather_df[date_col], errors="coerce")
    if w_dates.isna().any():
        raise ValueError("Some dates could not be parsed. Please ensure 'Date' is in a standard format.")

    # Basic cleaning
    fX = fX.replace([np.inf, -np.inf], np.nan).fillna(fX.median(numeric_only=True))
    wX = wX.replace([np.inf, -np.inf], np.nan).fillna(wX.median(numeric_only=True))

    return fX, y_presence, y_type_raw, wX, w_dates, common_keys


# =============== 2) Supervised Classification (Presence + optional Type) ===============

def train_presence_model(fX, y_presence, random_state=42):
    """
    Train a calibrated classifier for disease presence.
    Prefers XGB; falls back to RF if unavailable.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        fX.values, y_presence.values, test_size=0.2, random_state=random_state, stratify=y_presence.values
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    if HAS_XGB:
        base = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.07,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        base = RandomForestClassifier(
            n_estimators=600, max_depth=None, min_samples_leaf=2, random_state=random_state, n_jobs=-1
        )

    # Probability calibration improves decision usefulness (Brier/reliability)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=5)
    clf.fit(X_train_s, y_train)

    # Evaluate
    y_proba = clf.predict_proba(X_test_s)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    print("\n=== Presence Model Evaluation ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
    print("Brier Score:", brier_score_loss(y_test, y_proba))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    model_bundle = {
        "scaler": scaler,
        "clf": clf,
        "is_xgb": HAS_XGB
    }
    return model_bundle


def train_type_model(fX, y_presence, y_type_raw, random_state=42):
    """
    Optional disease type model (trained only on presence==1).
    """
    mask = (y_presence.values == 1)
    if mask.sum() < 20:
        print("Not enough positive samples to train a disease type classifier. Skipping.")
        return None, None

    # encode labels
    type_classes, y_type = np.unique(y_type_raw[mask], return_inverse=True)

    X_pos = fX.values[mask]
    X_train, X_test, y_train, y_test = train_test_split(
        X_pos, y_type, test_size=0.2, random_state=random_state, stratify=y_type
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    if HAS_XGB:
        base = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.07,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=random_state, n_jobs=-1
        )
    else:
        base = RandomForestClassifier(
            n_estimators=600, max_depth=None, min_samples_leaf=2, random_state=random_state, n_jobs=-1
        )

    clf = CalibratedClassifierCV(base, method="isotonic", cv=5)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    print("\n=== Disease Type Model Evaluation (positives only) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Report:\n", classification_report(y_test, y_pred, digits=3))

    bundle = {"scaler": scaler, "clf": clf, "classes": type_classes}
    return bundle, type_classes


# =============== 3) SHAP Explainability / Fallback ===============

def explain_global_importance(model_bundle, fX, feature_names, out_dir="outputs"):
    clf = model_bundle["clf"]
    scaler = model_bundle["scaler"]

    Xs = scaler.transform(fX.values)
    if HAS_SHAP:
        try:
            # Use TreeExplainer if tree-based; KernelExplainer otherwise (slower)
            if model_bundle["is_xgb"]:
                explainer = shap.Explainer(clf.estimators_[0], feature_names=feature_names)
            else:
                # Calibrated RF wrapped; get underlying estimator if possible
                base = clf.estimators_[0] if hasattr(clf, "estimators_") else clf
                explainer = shap.TreeExplainer(getattr(base, "base_estimator_", base))
            sv = explainer(Xs)
            # Mean abs SHAP per feature
            mean_abs = np.abs(sv.values).mean(axis=0)
            imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}) \
                    .sort_values("mean_abs_shap", ascending=False)
            imp.to_csv(os.path.join(out_dir, "shap_global_importance.csv"), index=False)
            print("\nTop features by SHAP (mean |value|):")
            print(imp.head(10).to_string(index=False))

            # Plain matplotlib bar (no styling)
            plt.figure(figsize=(8, 5))
            plt.bar(imp["feature"], imp["mean_abs_shap"])
            plt.title("Global Feature Importance (mean |SHAP|)")
            plt.ylabel("Mean |SHAP value|")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "shap_importance_bar.png"), dpi=150)
            plt.close()
            return
        except Exception as e:
            print("SHAP failed, falling back to permutation importance.", e)

    # Fallback: permutation importance on calibrated model
    print("\nUsing permutation importance (fallback).")
    r = permutation_importance(clf, Xs, clf.predict(fX.values if hasattr(clf, "predict") else Xs), n_repeats=10, random_state=42)
    imp = pd.DataFrame({"feature": feature_names, "importance": r.importances_mean}).sort_values("importance", ascending=False)
    imp.to_csv(os.path.join(out_dir, "permutation_importance.csv"), index=False)
    plt.figure(figsize=(8, 5))
    plt.bar(imp["feature"], imp["importance"])
    plt.title("Global Feature Importance (Permutation)")
    plt.ylabel("Mean importance")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "permutation_importance_bar.png"), dpi=150)
    plt.close()


# =============== 4) Score Historical Risk on Weather Data ===============

def score_historical_risk(model_bundle, wX, w_dates):
    clf = model_bundle["clf"]
    scaler = model_bundle["scaler"]
    Xs = scaler.transform(wX.values)
    risk = clf.predict_proba(Xs)[:, 1]
    hist_df = pd.DataFrame({"Date": w_dates, "risk_prob": risk})
    hist_df = hist_df.sort_values("Date").reset_index(drop=True)
    return hist_df


# =============== 5) LSTM Forecast on Risk Series ===============

def lstm_forecast_risk(hist_df, horizon_days=60, seq_len=28, random_state=42):
    if not HAS_TF:
        print("TensorFlow not available; skipping LSTM forecast.")
        return pd.DataFrame(columns=["Date", "risk_prob"])

    series = hist_df["risk_prob"].astype(float).values.reshape(-1, 1)

    # scale to 0-1 for stability (min-max)
    s_min, s_max = series.min(), series.max()
    denom = (s_max - s_min) if s_max > s_min else 1.0
    s_norm = (series - s_min) / denom

    # Build sequences
    X_seq, y_seq = [], []
    for i in range(len(s_norm) - seq_len):
        X_seq.append(s_norm[i:i+seq_len, 0])
        y_seq.append(s_norm[i+seq_len, 0])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Train/val split (last 15% for validation)
    n = len(X_seq)
    split = int(n * 0.85)
    X_tr, X_val = X_seq[:split], X_seq[split:]
    y_tr, y_val = y_seq[:split], y_seq[split:]

    X_tr = X_tr.reshape((X_tr.shape[0], X_tr.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

    tf.random.set_seed(random_state)
    model = Sequential([
        LSTM(64, activation="tanh", input_shape=(seq_len, 1)),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=0)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=200, batch_size=32, callbacks=[es], verbose=0)

    # Forecast iteratively
    last_seq = s_norm[-seq_len:, 0].tolist()
    preds = []
    for _ in range(horizon_days):
        x = np.array(last_seq[-seq_len:]).reshape(1, seq_len, 1)
        yhat = model.predict(x, verbose=0)[0, 0]
        preds.append(yhat)
        last_seq.append(yhat)

    # De-normalize
    preds = np.array(preds).reshape(-1, 1) * denom + s_min
    future_dates = pd.date_range(start=hist_df["Date"].max() + timedelta(days=1), periods=horizon_days, freq="D")
    fut_df = pd.DataFrame({"Date": future_dates, "risk_prob": preds.ravel()})
    return fut_df


# =============== 6) Season/Maha-Yala Analysis & Plots ===============

def analyze_and_plot(hist_df, fut_df, out_dir="outputs"):
    plt.figure(figsize=(12, 5))
    plt.plot(hist_df["Date"], hist_df["risk_prob"], label="Historical risk")
    if len(fut_df):
        plt.plot(fut_df["Date"], fut_df["risk_prob"], label="Forecast risk")
    plt.title("Rice Disease Risk: Historical & Forecast")
    plt.xlabel("Date"); plt.ylabel("Risk probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "risk_timeseries.png"), dpi=150)
    plt.close()

    # Combine and season stats
    all_df = pd.concat([hist_df.assign(part="history"), fut_df.assign(part="forecast")], ignore_index=True)
    all_df = add_season_cols(all_df, "Date")

    # Monthly risk
    monthly = all_df.groupby("month", as_index=False)["risk_prob"].mean()
    plt.figure(figsize=(8, 4))
    plt.bar(monthly["month"].astype(int), monthly["risk_prob"])
    plt.title("Average Monthly Disease Risk (History + Forecast)")
    plt.xlabel("Month"); plt.ylabel("Mean risk")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "monthly_risk_bar.png"), dpi=150)
    plt.close()

    # Season risk (Maha vs Yala)
    season_mean = all_df.groupby("season", as_index=False)["risk_prob"].mean().sort_values("risk_prob")
    plt.figure(figsize=(6, 4))
    plt.bar(season_mean["season"], season_mean["risk_prob"])
    plt.title("Average Season Risk (Maha vs Yala)")
    plt.xlabel("Season"); plt.ylabel("Mean risk")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "season_risk_bar.png"), dpi=150)
    plt.close()

    # Season-year trend (useful for reporting)
    seas_year = all_df.groupby(["season", "season_year"], as_index=False)["risk_prob"].mean()
    pivot = seas_year.pivot(index="season_year", columns="season", values="risk_prob").sort_index()
    pivot.to_csv(os.path.join(out_dir, "season_year_risk.csv"), index=True)

    # Best/worst months and seasons (by mean risk)
    best_month = int(monthly.loc[monthly["risk_prob"].idxmin(), "month"])
    worst_month = int(monthly.loc[monthly["risk_prob"].idxmax(), "month"])
    best_season = season_mean.iloc[0]["season"]
    worst_season = season_mean.iloc[-1]["season"]

    summary = {
        "best_month_lowest_risk": best_month,
        "worst_month_highest_risk": worst_month,
        "best_season_lowest_risk": best_season,
        "worst_season_highest_risk": worst_season
    }
    pd.Series(summary).to_csv(os.path.join(out_dir, "best_worst_summary.csv"))
    print("\n=== Season/Month Summary ===")
    print(summary)
    return all_df, summary


# =============== 7) Main Orchestration ===============

def main():
    out_dir = ensure_output_dir("outputs")
    print("Loading & aligning datasets...")
    fX, y_presence, y_type_raw, wX, w_dates, feature_names = load_datasets()

    print("Training presence classifier (calibrated)...")
    presence_bundle = train_presence_model(fX, y_presence)

    print("Explainability (SHAP or fallback)...")
    explain_global_importance(presence_bundle, fX, feature_names, out_dir=out_dir)

    print("Scoring historical risk on Anuradhapura dataset...")
    hist_df = score_historical_risk(presence_bundle, wX, w_dates)
    hist_df.to_csv(os.path.join(out_dir, "historical_risk.csv"), index=False)

    # (Optional) disease type model
    type_bundle, type_classes = train_type_model(fX, y_presence, y_type_raw)
    if type_bundle is not None:
        # Score likely type only when risk > 0.5 for reporting
        Xs_w = type_bundle["scaler"].transform(wX.values)
        type_pred = type_bundle["clf"].predict(Xs_w)
        type_labels = pd.Series(type_pred).map({i: c for i, c in enumerate(type_classes)})
        typed = hist_df.copy()
        typed["predicted_type"] = type_labels
        typed.loc[typed["risk_prob"] < 0.5, "predicted_type"] = "None/Low"
        typed.to_csv(os.path.join(out_dir, "historical_type_when_risky.csv"), index=False)

    print("LSTM forecasting future disease risk...")
    fut_df = lstm_forecast_risk(hist_df, horizon_days=60, seq_len=28)
    fut_df.to_csv(os.path.join(out_dir, "forecast_risk_60d.csv"), index=False)

    print("Season-wise analysis & plotting...")
    all_df, summary = analyze_and_plot(hist_df, fut_df, out_dir=out_dir)

    # Partial Dependence (optional, simple top-3 features), pure matplotlib
    try:
        clf = presence_bundle["clf"]
        scaler = presence_bundle["scaler"]
        Xs = scaler.transform(fX.values)
        # pick top 3 features by variance as a simple heuristic; or use feature_importances_ if available
        top_feats = feature_names
        if hasattr(clf, "estimator_") and hasattr(clf.estimator_, "feature_importances_"):
            fi = clf.estimator_.feature_importances_
            top_idx = np.argsort(fi)[::-1][:min(3, len(feature_names))]
            top_feats = [feature_names[i] for i in top_idx]
        elif hasattr(clf, "feature_importances_"):
            fi = clf.feature_importances_
            top_idx = np.argsort(fi)[::-1][:min(3, len(feature_names))]
            top_feats = [feature_names[i] for i in top_idx]
        else:
            # fallback: variance
            top_idx = np.argsort(np.var(Xs, axis=0))[::-1][:min(3, len(feature_names))]
            top_feats = [feature_names[i] for i in top_idx]

        for feat in top_feats:
            feat_idx = feature_names.index(feat)
            fig, ax = plt.subplots(figsize=(6, 4))
            try:
                PartialDependenceDisplay.from_estimator(clf, Xs, [feat_idx], ax=ax)
                ax.set_title(f"Partial Dependence: {feat}")
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f"pdp_{feat}.png"), dpi=150)
                plt.close(fig)
            except Exception:
                plt.scatter(fX.values[:, feat_idx], clf.predict_proba(Xs)[:, 1], s=6)
                plt.title(f"Risk vs {feat}")
                plt.xlabel(feat); plt.ylabel("Predicted risk")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"risk_vs_{feat}.png"), dpi=150)
                plt.close()

    except Exception as e:
        print("PDP step skipped:", e)

    print(f"\nAll done. Outputs saved in: {os.path.abspath(out_dir)}")
    print("Key files:")
    print(" - outputs/historical_risk.csv")
    print(" - outputs/forecast_risk_60d.csv")
    print(" - outputs/season_year_risk.csv")
    print(" - outputs/best_worst_summary.csv")
    print(" - outputs/shap_importance_bar.png (or permutation_importance_bar.png)")
    print(" - outputs/risk_timeseries.png, monthly_risk_bar.png, season_risk_bar.png")
    if type_bundle is not None:
        print(" - outputs/historical_type_when_risky.csv")


if __name__ == "__main__":
    main()
