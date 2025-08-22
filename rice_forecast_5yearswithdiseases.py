import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# ========================= Utils =========================

def canonicalize(col: str) -> str:
    c = (col or "").strip().lower().replace("Â°", "°")
    for ch in ["(", ")", "%", "/", "-", "_"]:
        c = c.replace(ch, " ")
    return " ".join(c.split())

def get_common_feature_map():
    return {
        "temperature": ["temperature °c","temperature c","avg temperature","average temperature","maximum temperature","minimum temperature"],
        "humidity": ["humidity","relative humidity","avg relative humidity","average relative humidity"],
        "rainfall": ["rainfall mm","total rainfall","rainfall","precipitation"],
        "wind_speed": ["wind speed m s","average wind speed","wind speed"],
        "ph": ["ph level","average ph","ph","maximum ph","minimum ph"],
        "soil_moisture": ["soil moisture","avg soil moisture","soil moisture percent"],
        "nitrogen": ["nitrogen content mg kg","nitrogen"],
        "potassium": ["potassium content mg kg","potassium"],
        "salinity": ["salinity","average salinity","maximum salinity","minimum salinity"]
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

# ========================= Main Pipeline =========================

def train_and_predict(factors_path="Rice_Deseases_Factors_550.csv",
                      forecast_path="rice_forecast_5years.csv",
                      out_path="rice_forecast_5yearswithdiseases.csv"):

    # --- Load training dataset (with disease presence/type) ---
    fdf = pd.read_csv(factors_path)
    fcols = {canonicalize(c): c for c in fdf.columns}

    # Identify target columns
    presence_col = fcols.get("disease presence")
    dtype_col    = fcols.get("disease type")

    if presence_col is None or dtype_col is None:
        raise ValueError("Training dataset must include 'Disease Presence' and 'Disease Type' columns.")

    y_presence = infer_binary(fdf[presence_col])
    y_type_raw = fdf[dtype_col].astype(str).str.strip()

    # --- Feature alignment between training and forecast ---
    wanted = get_common_feature_map()
    fmap = find_columns(fdf, wanted)

    # Training features
    fX = pd.DataFrame({k: pd.to_numeric(fdf[fmap[k]], errors="coerce") for k in fmap})
    fX = fX.fillna(fX.median(numeric_only=True))

    # --- Load forecast dataset (future weather) ---
    wdf = pd.read_csv(forecast_path, encoding="ISO-8859-1")
    wmap = find_columns(wdf, wanted)
    common = sorted(set(fmap).intersection(set(wmap)))
    if not common:
        raise ValueError("No common feature columns between training and forecast datasets.")

    wX = pd.DataFrame({k: pd.to_numeric(wdf[wmap[k]], errors="coerce") for k in common})
    wX = wX.fillna(wX.median(numeric_only=True))

    # --- Train Presence Model ---
    Xtr, Xte, ytr, yte = train_test_split(fX[common].values, y_presence.values,
                                          test_size=0.2, random_state=42, stratify=y_presence.values)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    base_model = RandomForestClassifier(n_estimators=600, min_samples_leaf=2, random_state=42, n_jobs=-1)
    presence_clf = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
    presence_clf.fit(Xtr_s, ytr)

    # --- Train Type Model (positives only) ---
    mask = (y_presence.values == 1)
    type_classes = None
    type_clf = None
    type_scaler = None
    if mask.sum() > 20:
        classes, y_type = np.unique(y_type_raw[mask], return_inverse=True)
        X_pos = fX[common].values[mask]

        Xtr, Xte, ytr, yte = train_test_split(X_pos, y_type, test_size=0.2, random_state=42, stratify=y_type)
        type_scaler = StandardScaler()
        Xtr_s = type_scaler.fit_transform(Xtr)
        Xte_s = type_scaler.transform(Xte)

        type_model = RandomForestClassifier(n_estimators=600, min_samples_leaf=2, random_state=42, n_jobs=-1)
        type_clf = CalibratedClassifierCV(type_model, method="isotonic", cv=5)
        type_clf.fit(Xtr_s, ytr)
        type_classes = classes

    # --- Predict on 5-year forecast ---
    wX_s = scaler.transform(wX[common].values)
    presence_prob = presence_clf.predict_proba(wX_s)[:,1]
    presence_pred = (presence_prob >= 0.5).astype(int)

    results = wdf.copy()
    results["Disease_Presence_Prob"] = presence_prob
    results["Disease_Presence_Pred"] = presence_pred

    if type_clf is not None:
        # Assign random diseases first
        results["Predicted_Disease_Type"] = [random.choice(type_classes) for _ in range(len(results))]

        # For rows predicted as disease present, overwrite with model prediction
        pos_idx = (presence_pred == 1)
        if pos_idx.any():
            Xpos = wX[common].values[pos_idx]
            Xpos_s = type_scaler.transform(Xpos)
            type_preds = type_clf.predict(Xpos_s)
            results.loc[pos_idx, "Predicted_Disease_Type"] = [type_classes[i] for i in type_preds]
    else:
        # If no type model, assign generic random labels
        results["Predicted_Disease_Type"] = [f"RandomDisease{random.randint(1,5)}" for _ in range(len(results))]

    # --- Save to CSV ---
    results.to_csv(out_path, index=False)
    print(f"✅ Saved predictions to {out_path}")

if __name__ == "__main__":
    train_and_predict()
