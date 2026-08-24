"""
predict.py — Load best saved model and predict a single sensor reading.
Used by the FastAPI predict endpoint and for testing.

Usage:
    from ml.predict import predict_condition
    result = predict_condition({
        "speed": 8.3, "accel_x": 0.1, "accel_y": 0.2, "accel_z": 9.8,
        "accel_magnitude": 9.81, "gyro_x": 0.01, "gyro_y": 0.02,
        "gyro_z": 0.0, "gyro_magnitude": 0.02,
        "accuracy": 5.0, "altitude": 920.0, "heading": 45.0
    })
    # → {"condition": "good", "confidence": 0.87, "model": "Random Forest", "probabilities": {...}}
"""

import os
import json
import numpy as np
from joblib import load

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "ml", "models")

_cache = {}

def _load_artifacts():
    if _cache:
        return _cache

    best_name_path = os.path.join(MODEL_DIR, "best_model.txt")
    if not os.path.exists(best_name_path):
        raise FileNotFoundError("No trained model found. Run ml/train.py first.")

    with open(best_name_path) as f:
        best_name = f.read().strip()

    model_file = best_name.replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
    model      = load(os.path.join(MODEL_DIR, model_file))
    scaler     = load(os.path.join(MODEL_DIR, "scaler.pkl"))
    le         = load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

    with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)

    # Check if this model needs scaling
    results_path = os.path.join(ROOT, "ml", "model_results.json")
    needs_scale  = True  # safe default
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        for r in results:
            if r["model"] == best_name:
                needs_scale = r.get("needs_scale", True)
                break

    _cache.update({
        "model":        model,
        "scaler":       scaler,
        "le":           le,
        "feature_cols": feature_cols,
        "best_name":    best_name,
        "needs_scale":  needs_scale,
    })
    return _cache

def _engineer(raw: dict) -> np.ndarray:
    """Apply same feature engineering as train.py from a raw sensor dict."""
    d = {k: (raw.get(k) or 0.0) for k in [
        "speed","accel_x","accel_y","accel_z","accel_magnitude",
        "gyro_x","gyro_y","gyro_z","gyro_magnitude",
        "accuracy","altitude","heading"
    ]}

    speed_kmh = d["speed"] * 3.6

    features = {
        "speed":           d["speed"],
        "accel_x":         d["accel_x"],
        "accel_y":         d["accel_y"],
        "accel_z":         d["accel_z"],
        "accel_magnitude": d["accel_magnitude"],
        "gyro_x":          d["gyro_x"],
        "gyro_y":          d["gyro_y"],
        "gyro_z":          d["gyro_z"],
        "gyro_magnitude":  d["gyro_magnitude"],
        "accuracy":        d["accuracy"],
        "altitude":        d["altitude"],
        "accel_xy":        np.sqrt(d["accel_x"]**2 + d["accel_y"]**2),
        "accel_xz":        np.sqrt(d["accel_x"]**2 + d["accel_z"]**2),
        "accel_yz":        np.sqrt(d["accel_y"]**2 + d["accel_z"]**2),
        "accel_norm_z":    abs(d["accel_z"]) / (d["accel_magnitude"] + 1e-6),
        "gyro_xy":         np.sqrt(d["gyro_x"]**2 + d["gyro_y"]**2),
        "gyro_total_turn": d["gyro_magnitude"],
        "speed_kmh":       speed_kmh,
        "speed_bucket":    float(min(3, int(speed_kmh / 10))),
        "rough_proxy":     d["accel_magnitude"] / (speed_kmh + 1),
        "heading_sin":     np.sin(np.deg2rad(d["heading"])),
        "heading_cos":     np.cos(np.deg2rad(d["heading"])),
    }

    arts         = _load_artifacts()
    feature_cols = arts["feature_cols"]
    return np.array([[features[c] for c in feature_cols]])

def predict_condition(raw: dict) -> dict:
    """
    raw: dict with sensor keys (same as DB columns)
    Returns: {"condition": str, "confidence": float, "model": str, "probabilities": dict}
    """
    arts   = _load_artifacts()
    model  = arts["model"]
    scaler = arts["scaler"]
    le     = arts["le"]

    X = _engineer(raw)
    if arts["needs_scale"]:
        X = scaler.transform(X)

    pred_idx  = model.predict(X)[0]
    condition = le.inverse_transform([pred_idx])[0]

    probabilities = {}
    confidence    = 1.0
    if hasattr(model, "predict_proba"):
        proba         = model.predict_proba(X)[0]
        probabilities = {cls: round(float(p), 4) for cls, p in zip(le.classes_, proba)}
        confidence    = round(float(proba[pred_idx]), 4)

    return {
        "condition":     condition,
        "confidence":    confidence,
        "model":         arts["best_name"],
        "probabilities": probabilities,
    }


if __name__ == "__main__":
    # Quick test
    test = {
        "speed": 8.3, "accel_x": 0.1, "accel_y": 0.2, "accel_z": 9.8,
        "accel_magnitude": 9.81, "gyro_x": 0.01, "gyro_y": 0.02,
        "gyro_z": 0.0, "gyro_magnitude": 0.02,
        "accuracy": 5.0, "altitude": 920.0, "heading": 45.0
    }
    result = predict_condition(test)
    print(result)
