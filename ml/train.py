"""
train.py — Road Condition ML Training Script
Run from project root: python ml/train.py
Trains 6 classifiers, compares them, saves all models + comparison CSV.
"""

import sqlite3
import os
import time
import warnings
import json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(ROOT, "road_data.db")
OUT_DIR  = os.path.join(ROOT, "ml", "models")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 1. Load Data ─────────────────────────────────────────────────────────────
def load_data():
    con = sqlite3.connect(DB_PATH)
    df  = pd.read_sql_query("""
        SELECT condition,
               speed, accel_x, accel_y, accel_z, accel_magnitude,
               gyro_x, gyro_y, gyro_z, gyro_magnitude,
               accuracy, altitude, heading
        FROM readings
        WHERE condition IN ('good','avg','bad')
    """, con)
    con.close()
    print(f"[load] {len(df)} rows | classes: {df['condition'].value_counts().to_dict()}")
    return df

# ─── 2. Feature Engineering ───────────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()

    # Derived accel features
    df["accel_xy"]        = np.sqrt(df["accel_x"]**2 + df["accel_y"]**2)
    df["accel_xz"]        = np.sqrt(df["accel_x"]**2 + df["accel_z"]**2)
    df["accel_yz"]        = np.sqrt(df["accel_y"]**2 + df["accel_z"]**2)
    df["accel_norm_z"]    = df["accel_z"].abs() / (df["accel_magnitude"] + 1e-6)

    # Derived gyro features
    df["gyro_xy"]         = np.sqrt(df["gyro_x"]**2 + df["gyro_y"]**2)
    df["gyro_total_turn"] = df["gyro_magnitude"]

    # Speed buckets (ordinal)
    df["speed_kmh"]       = df["speed"].fillna(0) * 3.6
    df["speed_bucket"]    = pd.cut(df["speed_kmh"],
                                   bins=[-1, 10, 30, 60, 999],
                                   labels=[0, 1, 2, 3]).astype(float)

    # Interaction: roughness proxy = accel_magnitude at speed
    df["rough_proxy"]     = df["accel_magnitude"] / (df["speed_kmh"] + 1)

    # Heading sine/cosine (circular encoding)
    heading_rad           = np.deg2rad(df["heading"].fillna(0))
    df["heading_sin"]     = np.sin(heading_rad)
    df["heading_cos"]     = np.cos(heading_rad)

    return df

FEATURE_COLS = [
    "speed", "accel_x", "accel_y", "accel_z", "accel_magnitude",
    "gyro_x", "gyro_y", "gyro_z", "gyro_magnitude",
    "accuracy", "altitude",
    "accel_xy", "accel_xz", "accel_yz", "accel_norm_z",
    "gyro_xy", "gyro_total_turn",
    "speed_kmh", "speed_bucket", "rough_proxy",
    "heading_sin", "heading_cos"
]

# ─── 3. Define Models ─────────────────────────────────────────────────────────
def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost":             XGBClassifier(n_estimators=200, use_label_encoder=False,
                                             eval_metric="mlogloss", random_state=42, verbosity=0),
        "SVM (RBF)":           SVC(kernel="rbf", probability=True, random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=7),
        "MLP":                 MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                             random_state=42),
    }

# ─── 4. Train & Evaluate ──────────────────────────────────────────────────────
def train_and_evaluate(df):
    df = engineer_features(df)

    # Drop rows with too many NaNs in features
    df = df.dropna(subset=FEATURE_COLS, thresh=len(FEATURE_COLS) - 3)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())

    X = df[FEATURE_COLS].values
    y = df["condition"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler   = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models  = get_models()
    results = []
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n{'Model':<25} {'Acc':>6} {'F1-W':>6} {'AUC':>6} {'Train(s)':>9} {'Infer(ms)':>10}")
    print("─" * 70)

    for name, model in models.items():
        # Models that need scaling
        needs_scale = name in ("Logistic Regression", "SVM (RBF)", "KNN", "MLP")
        Xtr = X_train_s if needs_scale else X_train
        Xte = X_test_s  if needs_scale else X_test

        # Train
        t0 = time.time()
        model.fit(Xtr, y_train)
        train_time = time.time() - t0

        # Inference time (per sample, ms)
        t0 = time.time()
        y_pred = model.predict(Xte)
        infer_ms = (time.time() - t0) / len(Xte) * 1000

        # Metrics
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        try:
            y_prob = model.predict_proba(Xte)
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
        except Exception:
            auc = float("nan")

        # 5-fold CV
        cv_res   = cross_validate(model, Xtr, y_train, cv=cv,
                                  scoring="f1_weighted", return_train_score=False)
        cv_mean  = cv_res["test_score"].mean()
        cv_std   = cv_res["test_score"].std()

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Class report
        cr = classification_report(y_test, y_pred,
                                   target_names=le.classes_, output_dict=True)

        # Feature importance (where available)
        feat_imp = None
        if hasattr(model, "feature_importances_"):
            feat_imp = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))

        results.append({
            "model":            name,
            "accuracy":         round(acc, 4),
            "precision_w":      round(prec, 4),
            "recall_w":         round(rec, 4),
            "f1_weighted":      round(f1, 4),
            "roc_auc":          round(auc, 4) if not np.isnan(auc) else None,
            "cv_f1_mean":       round(cv_mean, 4),
            "cv_f1_std":        round(cv_std, 4),
            "train_time_s":     round(train_time, 3),
            "infer_time_ms":    round(infer_ms, 4),
            "confusion_matrix": cm,
            "class_report":     cr,
            "feature_importance": feat_imp,
            "needs_scale":      needs_scale,
        })

        print(f"{name:<25} {acc:>6.4f} {f1:>6.4f} {auc if not np.isnan(auc) else '  N/A':>6} "
              f"{train_time:>9.3f} {infer_ms:>10.4f}")

        # Save model
        dump(model, os.path.join(OUT_DIR, f"{name.replace(' ', '_').replace('(','').replace(')','')}.pkl"))

    # Save scaler + label encoder
    dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))
    dump(le,     os.path.join(OUT_DIR, "label_encoder.pkl"))

    # Save feature list
    with open(os.path.join(OUT_DIR, "feature_cols.json"), "w") as f:
        json.dump(FEATURE_COLS, f)

    # Comparison CSV
    df_res = pd.DataFrame(results)[
        ["model","accuracy","precision_w","recall_w","f1_weighted",
         "roc_auc","cv_f1_mean","cv_f1_std","train_time_s","infer_time_ms"]
    ]
    csv_path = os.path.join(ROOT, "ml", "model_comparison.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\n[saved] Comparison → {csv_path}")

    # Save full results as JSON (for notebooks)
    json_path = os.path.join(ROOT, "ml", "model_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] Full results → {json_path}")

    # Pick best model by weighted F1
    best = max(results, key=lambda r: r["f1_weighted"])
    print(f"\n★  Best model: {best['model']}  (F1={best['f1_weighted']})")

    # Save best model name
    with open(os.path.join(OUT_DIR, "best_model.txt"), "w") as f:
        f.write(best["model"])

    return results, le, scaler

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    if len(df) < 20:
        print("Not enough data. Need at least 20 labeled readings.")
    else:
        results, le, scaler = train_and_evaluate(df)
        print("\nDone. All models saved to ml/models/")
