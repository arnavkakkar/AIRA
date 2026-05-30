"""
AIRA - GRU Temporal Neural Network Training
============================================
Trains a GRU model on city_day.csv (Delhi data) to forecast AQI
for the next 24 time steps (days used as proxy for hours in inference).

Run from AIRA root:
    python models/train_gru.py

Outputs saved to AIRA/models/:
    - aqi_gru_model.h5         (trained GRU weights)
    - aqi_scaler.pkl           (MinMaxScaler for features)
    - aqi_target_scaler.pkl    (MinMaxScaler for AQI target)
    - training_history.json    (loss curves for reporting)
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── TensorFlow / Keras ──────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"          # suppress TF info logs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(ROOT, "data", "city_day.csv")
MODEL_DIR   = os.path.join(ROOT, "models")
MODEL_PATH  = os.path.join(MODEL_DIR, "aqi_gru_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "aqi_scaler.pkl")
TARGET_SCALER_PATH = os.path.join(MODEL_DIR, "aqi_target_scaler.pkl")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
SEQUENCE_LEN = 14    # look-back window (14 days of history)
FORECAST_STEPS = 1     # predict 1 step ahead (multi-step done in inference)
FEATURES = [
    "PM2.5",
    "PM10",
    "NO2",
    "CO",
    "SO2",
    "O3",
    "City_Encoded"
]
TARGET        = "AQI"
BATCH_SIZE    = 32
MAX_EPOCHS    = 200
GRU_UNITS_1   = 128
GRU_UNITS_2   = 64
DROPOUT_RATE  = 0.2
LEARNING_RATE = 1e-3

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD & PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess(path: str):
    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.dropna(
        subset=["City", "PM2.5", "PM10", "NO2", "CO", "SO2", "O3", "AQI"]
    )

    print(f"[data] Loaded {len(df)} samples")
    print(f"[data] Cities: {df['City'].nunique()}")

    return df

def make_sequences(feature_arr: np.ndarray,
                   target_arr: np.ndarray,
                   seq_len: int):
    """
    Slide a window of seq_len over the time series.
    X shape: (n_samples, seq_len, n_features)
    y shape: (n_samples,)
    """
    X, y = [], []
    for i in range(len(feature_arr) - seq_len):
        X.append(feature_arr[i : i + seq_len])
        y.append(target_arr[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  BUILD GRU MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_gru(seq_len: int, n_features: int) -> tf.keras.Model:
    """
    Two-layer stacked GRU with:
      - BatchNormalization for stable training
      - Dropout for regularisation
      - Dense output (single AQI prediction)

    Architecture rationale:
      Layer 1 (return_sequences=True) → captures local temporal patterns
      Layer 2 (return_sequences=False) → compresses into fixed representation
      Dense(32) + Dense(1) → regression head
    """
    model = Sequential([
        GRU(GRU_UNITS_1,
            return_sequences=True,
            input_shape=(seq_len, n_features),
            name="gru_1"),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),

        GRU(GRU_UNITS_2,
            return_sequences=False,
            name="gru_2"),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),

        Dense(32, activation="relu", name="dense_1"),
        Dense(1, name="output"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="huber",           # robust to outliers (high-AQI spikes)
        metrics=["mae"]
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def train():
    # ── Load data
    df = load_and_preprocess(DATA_PATH)
    city_encoder = LabelEncoder()
    df["City_Encoded"] = city_encoder.fit_transform(df["City"])

    # ── Scale features independently (scaler saved for inference)
    feat_scaler   = MinMaxScaler()
    target_scaler = MinMaxScaler()

    feature_scaled = feat_scaler.fit_transform(df[FEATURES].values)
    target_scaled  = target_scaler.fit_transform(df[[TARGET]].values).ravel()

    # ── Build sequences
    X_all = []
    y_all = []

    for city in df["City"].unique():

        city_mask = df["City"] == city

        city_features = feature_scaled[city_mask]
        city_target = target_scaled[city_mask]

        if len(city_features) <= SEQUENCE_LEN:
            continue

        X_city, y_city = make_sequences(
            city_features,
            city_target,
            SEQUENCE_LEN
        )

        X_all.append(X_city)
        y_all.append(y_city)

    X = np.concatenate(X_all)
    y = np.concatenate(y_all)

    print(f"[seq] X={X.shape}")
    print(f"[seq] y={y.shape}")

    # ── Train / validation split (time-ordered – no shuffle)
    split = int(len(X) * 0.85)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    print(f"[split] train={len(X_train)}, val={len(X_val)}")

    # ── Model
    model = build_gru(SEQUENCE_LEN, len(FEATURES))
    model.summary()

    # ── Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    # ── Fit
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate on validation set
    y_pred_scaled = model.predict(X_val, verbose=0).ravel()
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = target_scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    print("\n── Validation Metrics ──────────────────────")
    print(f"   MAE  : {mae:.2f}")
    print(f"   RMSE : {rmse:.2f}")
    print(f"   R²   : {r2:.4f}")
    print("────────────────────────────────────────────\n")

    # ── Save model & scalers
    model.save(MODEL_PATH)
    joblib.dump(feat_scaler, SCALER_PATH)
    joblib.dump(target_scaler, TARGET_SCALER_PATH)
    CITY_ENCODER_PATH = os.path.join(MODEL_DIR, "city_encoder.pkl")
    joblib.dump(city_encoder, CITY_ENCODER_PATH)

    # ── Save history for reporting
    hist_data = {
        "train_loss": [float(v) for v in history.history["loss"]],
        "val_loss": [float(v) for v in history.history["val_loss"]],
        "train_mae": [float(v) for v in history.history["mae"]],
        "val_mae": [float(v) for v in history.history["val_mae"]],
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2)
        }
    }
    with open(HISTORY_PATH, "w") as f:
        json.dump(hist_data, f, indent=2)

    print(f"[saved] model  → {MODEL_PATH}")
    print(f"[saved] scaler → {SCALER_PATH}")
    print(f"[saved] target → {TARGET_SCALER_PATH}")
    print(f"[saved] history→ {HISTORY_PATH}")
    return model, feat_scaler, target_scaler, hist_data


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    train()