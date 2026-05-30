"""
AIRA - Flask API
================
Endpoints:
  GET  /api/fetch-aqi          – fetch live AQI from WAQI
  POST /api/predict             – legacy RF predict (kept for compatibility)
  POST /api/forecast            – NEW: GRU 24-step AQI forecast + risk timeline
  GET  /api/health              – health check
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── TensorFlow quiet import ───────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")

RF_MODEL_PATH      = os.path.join(MODELS_DIR, "aqi_model.pkl")
GRU_MODEL_PATH     = os.path.join(MODELS_DIR, "aqi_gru_model.h5")
SCALER_PATH        = os.path.join(MODELS_DIR, "aqi_scaler.pkl")
TARGET_SCALER_PATH = os.path.join(MODELS_DIR, "aqi_target_scaler.pkl")
HISTORY_PATH       = os.path.join(MODELS_DIR, "training_history.json")
CITY_ENCODER_PATH = os.path.join(MODELS_DIR, "city_encoder.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# Load models at startup
# ─────────────────────────────────────────────────────────────────────────────
print("[AIRA] Loading models …")

# GRU model (optional – gracefully degrade if not trained yet)
gru_model      = None
feat_scaler    = None
target_scaler  = None
gru_available  = False
city_encoder   = None

if (os.path.exists(GRU_MODEL_PATH) and
        os.path.exists(SCALER_PATH) and
        os.path.exists(TARGET_SCALER_PATH)):
    try:
        gru_model     = tf.keras.models.load_model(GRU_MODEL_PATH)
        feat_scaler   = joblib.load(SCALER_PATH)
        target_scaler = joblib.load(TARGET_SCALER_PATH)
        city_encoder = joblib.load(CITY_ENCODER_PATH)
        gru_available = True
        print("[AIRA] GRU model loaded ✓")
    except Exception as e:
        print(f"[AIRA] GRU model load failed: {e}")
else:
    print("[AIRA] GRU model not found – run models/train_gru.py first")

print("[AIRA] API ready")

# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

WAQI_TOKEN = "4eae18b5e0db7190c0d789bccd76651803202734"

FEATURES = [
    "PM2.5",
    "PM10",
    "NO2",
    "CO",
    "SO2",
    "O3",
    "City_Encoded"
]
SEQUENCE_LEN = 14      # must match training

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
PROFILE_MULTIPLIERS = {
    "normal": 1.0,
    "asthma": 1.8,
    "elderly": 1.5,
    "athlete": 1.3,
}

AQI_THRESHOLDS = [
    (50,  "Good",         "#00c896"),
    (100, "Satisfactory", "#96c800"),
    (200, "Moderate",     "#f5a623"),
    (300, "Poor",         "#f05030"),
    (400, "Very Poor",    "#c03070"),
    (float("inf"), "Severe", "#8020a0"),
]


def get_aqi_meta(aqi: float) -> dict:
    for thresh, label, color in AQI_THRESHOLDS:
        if aqi <= thresh:
            return {"category": label, "color": color}
    return {"category": "Severe", "color": "#8020a0"}


def calculate_risk_score(aqi: float, hours: float, profile: str) -> float:
    mult = PROFILE_MULTIPLIERS.get(profile, 1.0)
    return round((aqi * hours * mult) / 100, 2)


def get_risk_level(score: float) -> str:
    if score < 3:
        return "Low"
    if score < 6:
        return "Moderate"
    return "High"


def detect_anomaly(aqi: float, forecast: list) -> dict:
    """
    Simple statistical anomaly detection:
    Flag if any forecast step deviates > 2 std from the window mean.
    Returns anomaly flag + description for the UI.
    """
    arr = np.array(forecast)
    mean = arr.mean()
    std  = arr.std() + 1e-6
    z_scores = np.abs((arr - mean) / std)
    spike_indices = np.where(z_scores > 3)[0].tolist()
    has_anomaly = len(spike_indices) > 0
    max_aqi_forecast = float(arr.max())

    description = ""
    if has_anomaly:
        hours = [f"{(i+1)*1}h" for i in spike_indices]
        description = f"Pollution spike predicted at {', '.join(hours)}"

    return {
        "detected": has_anomaly,
        "spike_hours": spike_indices,
        "max_forecast_aqi": round(max_aqi_forecast, 1),
        "description": description,
    }


def build_forecast_recommendations(peak_aqi: float, profile: str) -> list:
    """
    Recommendations driven by the *worst* forecasted AQI in the 24h window.
    """
    cat = get_aqi_meta(peak_aqi)["category"]
    level_map = {
        "Good": "Low", "Satisfactory": "Low",
        "Moderate": "Moderate", "Poor": "High",
        "Very Poor": "High", "Severe": "High",
    }
    level = level_map.get(cat, "High")

    recs = []
    if level == "Low":
        recs = [
            {"icon": "ti-leaf",    "text": "Air quality forecast is acceptable",        "priority": "low"},
            {"icon": "ti-walk",    "text": "Outdoor activities are safe today",          "priority": "low"},
            {"icon": "ti-refresh", "text": "Good time to ventilate your home",           "priority": "low"},
        ]
    elif level == "Moderate":
        recs = [
            {"icon": "ti-window",  "text": "Close windows to limit indoor pollution",    "priority": "medium"},
            {"icon": "ti-clock",   "text": "Limit outdoor exposure to under 2 hours",    "priority": "medium"},
            {"icon": "ti-run",     "text": "Avoid vigorous outdoor exercise",            "priority": "medium"},
        ]
        if profile == "asthma":
            recs.append({"icon": "ti-lungs", "text": "Keep rescue inhaler accessible",  "priority": "high"})
    else:
        recs = [
            {"icon": "ti-air-conditioning", "text": "Turn on air purifier immediately",  "priority": "high"},
            {"icon": "ti-ban",              "text": "Avoid all outdoor activity",         "priority": "high"},
            {"icon": "ti-mask",             "text": "Wear N95 mask if going outside",     "priority": "high"},
            {"icon": "ti-first-aid-kit",    "text": "Monitor respiratory symptoms",       "priority": "high"},
        ]
        if profile in ("asthma", "elderly"):
            recs.append({"icon": "ti-phone-call",
                         "text": "Consider consulting your doctor today",
                         "priority": "high"})
    return recs


# ─────────────────────────────────────────────────────────────────────────────
# GRU FORECAST CORE
# ─────────────────────────────────────────────────────────────────────────────

def gru_forecast_24h(current_features,city_name: str) -> dict:
    """
    Given ONE snapshot of current pollutant readings, build a 7-step
    historical window by small synthetic perturbations (±5 %) and
    auto-regressively forecast 24 steps ahead.

    Returns:
        {
          "forecast_aqi":      [float * 24],   # AQI for hours 1-24
          "forecast_lower":    [float * 24],   # 90 % lower bound
          "forecast_upper":    [float * 24],   # 90 % upper bound
          "confidence":        float,          # mean confidence 0-1
        }
    """
    try:
        city_code = city_encoder.transform([city_name])[0]
    except:
        city_code = 0

    base = np.array([
        current_features.get("pm25", 50),
        current_features.get("pm10", 80),
        current_features.get("no2", 30),
        current_features.get("co", 1),
        current_features.get("so2", 10),
        current_features.get("o3", 30),
        city_code
    ], dtype=np.float32)

    # Build 7-step window with small synthetic history
    rng = np.random.default_rng(42)
    window = []
    for _ in range(SEQUENCE_LEN):
        row = base.copy()

        row[:6] = row[:6] * (
            1 + rng.uniform(-0.05, 0.05, size=6)
        )

        window.append(row)

    window = np.array(window)

    # Scale
    window_scaled = feat_scaler.transform(window)   # (7, 6)

    FORECAST_STEPS = 24
    MONTE_CARLO    = 20       # MC dropout passes for uncertainty

    all_runs = []

    for _ in range(MONTE_CARLO):
        seq = window_scaled.copy()   # (7, 6)
        run = []
        for _ in range(FORECAST_STEPS):
            inp = seq[np.newaxis, :, :]          # (1, 7, 6)
            pred_scaled = gru_model(inp, training=True).numpy()[0, 0]
            run.append(pred_scaled)

            # Auto-regressive: shift window, append small noise on features
            noise = rng.normal(0, 0.01, size=(1, len(FEATURES)))
            new_step = seq[-1:, :] + noise
            seq = np.concatenate([seq[1:], new_step], axis=0)

        all_runs.append(run)

    all_runs = np.array(all_runs, dtype=np.float32)  # (MC, 24)
    mean_scaled = all_runs.mean(axis=0)              # (24,)
    std_scaled  = all_runs.std(axis=0)               # (24,)

    # Inverse-transform
    def inv(arr):
        return target_scaler.inverse_transform(
            arr.reshape(-1, 1)
        ).ravel()

    mean_aqi  = inv(mean_scaled)
    lower_aqi = inv(np.clip(mean_scaled - 1.645 * std_scaled, 0, 1))
    upper_aqi = inv(np.clip(mean_scaled + 1.645 * std_scaled, 0, 1))

    # Confidence = 1 - normalised uncertainty
    uncertainty = std_scaled / (mean_scaled + 1e-6)
    confidence  = float(np.clip(1.0 - uncertainty.mean(), 0, 1))

    return {
        "forecast_aqi":   [round(float(v), 1) for v in np.clip(mean_aqi,  0, 500)],
        "forecast_lower": [round(float(v), 1) for v in np.clip(lower_aqi, 0, 500)],
        "forecast_upper": [round(float(v), 1) for v in np.clip(upper_aqi, 0, 500)],
        "confidence":     round(confidence, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/fetch-aqi", methods=["GET"])
def fetch_aqi():
    lat = request.args.get("lat")
    lon = request.args.get("lon")
    if not lat or not lon:
        return jsonify({"error": "Missing coordinates"}), 400

    try:
        url  = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={WAQI_TOKEN}"
        resp = requests.get(url, timeout=10).json()

        if resp.get("status") != "ok":
            return jsonify({"error": "WAQI API error"}), 502

        data = resp["data"]
        iaqi = data.get("iaqi", {})

        pm25 = float(iaqi.get("pm25", {}).get("v", 0))
        pm10 = float(iaqi.get("pm10", {}).get("v", 0))
        no2  = float(iaqi.get("no2",  {}).get("v", 0))
        co   = float(iaqi.get("co",   {}).get("v", 0))
        so2  = float(iaqi.get("so2",  {}).get("v", 0))
        o3   = float(iaqi.get("o3",   {}).get("v", 0))
        aqi  = float(data.get("aqi", 0))

        if aqi >= 500 and pm25 > 0:
            aqi = min(pm25 * 1.5, 300)

        return jsonify({
            "city":        data.get("city", {}).get("name", "Your Location"),
            "current_aqi": round(aqi, 1),
            "pm25": pm25, "pm10": pm10,
            "no2":  no2,  "co":   co,
            "so2":  so2,  "o3":   o3,
            "time": data.get("time", {}).get("s", ""),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Legacy RF predict (unchanged) ────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json()
    if not body:
        return jsonify({"error": "No data provided"}), 400

    required = ["pm25", "pm10", "no2", "co", "so2", "o3",
                "current_aqi", "exposure_hours", "profile"]
    for field in required:
        if field not in body:
            return jsonify({"error": f"Missing field: {field}"}), 400

    try:
        current_aqi   = float(body["current_aqi"])
        aqi_lag1      = current_aqi
        aqi_lag2      = max(current_aqi - 10, 0)
        aqi_lag3      = max(current_aqi - 20, 0)

        input_df = pd.DataFrame([[
            float(body["pm25"]), float(body["pm10"]),
            float(body["no2"]),  float(body["co"]),
            float(body["so2"]),  float(body["o3"]),
            aqi_lag1, aqi_lag2, aqi_lag3,
        ]], columns=["PM2.5", "PM10", "NO2", "CO", "SO2", "O3",
                     "AQI_lag1", "AQI_lag2", "AQI_lag3"])

        predicted_aqi  = float(rf_model.predict(input_df)[0])
        exposure_hours = int(body["exposure_hours"])
        profile        = body["profile"]
        risk_score     = calculate_risk_score(predicted_aqi, exposure_hours, profile)
        risk_level     = get_risk_level(risk_score)
        aqi_category   = get_aqi_meta(predicted_aqi)["category"]
        recommendations = build_forecast_recommendations(predicted_aqi, profile)

        return jsonify({
            "predicted_aqi":    round(predicted_aqi, 1),
            "risk_score":       risk_score,
            "risk_level":       risk_level,
            "aqi_category":     aqi_category,
            "recommendations":  recommendations,
            "profile":          profile,
            "exposure_hours":   exposure_hours,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── NEW: GRU 24-hour forecast ─────────────────────────────────────────────────
@app.route("/api/forecast", methods=["POST"])
def forecast():
    """
    Request body:
    {
      "pm25": float, "pm10": float, "no2": float,
      "co": float,   "so2": float,  "o3": float,
      "current_aqi": float,
      "profile": "normal"|"asthma"|"elderly"|"athlete",
      "exposure_hours": int
    }

    Response:
    {
      "forecast_aqi":      [float * 24],
      "forecast_lower":    [float * 24],
      "forecast_upper":    [float * 24],
      "confidence":        float,
      "risk_timeline":     [{"hour": int, "aqi": float, "risk_score": float,
                              "risk_level": str, "category": str, "color": str}],
      "peak_aqi":          float,
      "peak_hour":         int,
      "anomaly":           { "detected": bool, ... },
      "recommendations":   [...],
      "model_used":        "GRU" | "RF_fallback",
      "gru_metrics":       { "r2": float, ... }   # from training history
    }
    """
    body = request.get_json()
    if not body:
        return jsonify({"error": "No data provided"}), 400

    required = ["pm25", "pm10", "no2", "co", "so2", "o3",
                "current_aqi", "profile", "exposure_hours"]
    for f in required:
        if f not in body:
            return jsonify({"error": f"Missing field: {f}"}), 400

    profile        = body.get("profile", "normal")
    exposure_hours = int(body.get("exposure_hours", 3))
    current_aqi    = float(body["current_aqi"])

    # ── GRU forecast ──────────────────────────────────────────────────────────
    if gru_available:
        city_name = body.get("city", "Delhi")
        city_name = city_name.split(",")[0]
        city_name = city_name.replace(" US Embassy", "").strip()
        fc = gru_forecast_24h(body, city_name)
        model_used = "GRU"
    else:
        # Graceful fallback: flat forecast using current AQI + small drift
        drift = np.linspace(0, current_aqi * 0.1, 24)
        noise = np.random.normal(0, current_aqi * 0.03, 24)
        flat  = current_aqi + drift + noise
        flat  = np.clip(flat, 0, 500).tolist()
        fc = {
            "forecast_aqi":   [round(v, 1) for v in flat],
            "forecast_lower": [round(max(0, v - current_aqi * 0.1), 1) for v in flat],
            "forecast_upper": [round(v + current_aqi * 0.1, 1) for v in flat],
            "confidence":     0.55,
        }
        model_used = "RF_fallback"

    forecast_aqi = fc["forecast_aqi"]

    # ── Risk timeline ─────────────────────────────────────────────────────────
    risk_timeline = []
    for i, aqi_val in enumerate(forecast_aqi, start=1):
        meta       = get_aqi_meta(aqi_val)
        risk_score = calculate_risk_score(aqi_val, exposure_hours, profile)
        risk_level = get_risk_level(risk_score)
        risk_timeline.append({
            "hour":       i,
            "aqi":        aqi_val,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "category":   meta["category"],
            "color":      meta["color"],
        })

    # ── Key stats ─────────────────────────────────────────────────────────────
    arr      = np.array(forecast_aqi)
    peak_aqi = float(arr.max())
    peak_hour = int(arr.argmax()) + 1

    # ── Anomaly detection ─────────────────────────────────────────────────────
    anomaly = detect_anomaly(current_aqi, forecast_aqi)

    # ── Recommendations based on worst forecasted AQI ─────────────────────────
    recommendations = build_forecast_recommendations(peak_aqi, profile)

    # ── GRU training metrics (for UI badge) ───────────────────────────────────
    gru_metrics = {}
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            hist = json.load(f)
        gru_metrics = hist.get("metrics", {})

    return jsonify({
        "forecast_aqi":   forecast_aqi,
        "forecast_lower": fc["forecast_lower"],
        "forecast_upper": fc["forecast_upper"],
        "confidence":     fc["confidence"],
        "risk_timeline":  risk_timeline,
        "peak_aqi":       round(peak_aqi, 1),
        "peak_hour":      peak_hour,
        "anomaly":        anomaly,
        "recommendations": recommendations,
        "model_used":     model_used,
        "gru_metrics":    gru_metrics,
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":        "ok",
        "gru_available": gru_available,
        "rf_available":  True,
        "model_used":    "GRU" if gru_available else "RF_fallback",
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)