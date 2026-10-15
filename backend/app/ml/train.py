"""
Trains GlucoseLSTM on a patient's feature table and saves weights + scaler
to /app/saved_models. Also trains a scikit-learn GradientBoostingRegressor
baseline (used for factor/feature-importance analysis, since tree models
give interpretable feature importances that LSTMs don't).
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import joblib

from .model import GlucoseLSTM
from ..data_processing import FEATURE_COLUMNS

MODEL_DIR = os.getenv("MODEL_DIR", "/app/saved_models")
SEQ_LEN = 12  # last 60 minutes at 5-min spacing
HORIZONS = [30, 60, 120]  # minutes ahead
STEPS_AHEAD = {h: h // 5 for h in HORIZONS}  # assumes ~5 min sampling
MODEL_VERSION = "lstm_v1"


def _make_sequences(feature_df):
    """Build (X, y) where X is a sliding window of SEQ_LEN feature vectors
    and y is the glucose value at each horizon after the window's last point."""
    values = feature_df[FEATURE_COLUMNS].values.astype(np.float32)
    glucose = feature_df["glucose_mgdl"].values.astype(np.float32)

    max_step = max(STEPS_AHEAD.values())
    X, y = [], []
    for i in range(SEQ_LEN, len(values) - max_step):
        window = values[i - SEQ_LEN:i]
        targets = [glucose[i + STEPS_AHEAD[h] - 1] for h in HORIZONS]
        X.append(window)
        y.append(targets)
    return np.array(X), np.array(y)


def train_lstm_for_patient(feature_df, patient_id: int, epochs: int = 25):
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y = _make_sequences(feature_df)
    if len(X) < 20:
        raise ValueError("Not enough data to train a model for this patient yet.")

    n_features = X.shape[2]
    scaler = StandardScaler()
    X_flat = X.reshape(-1, n_features)
    scaler.fit(X_flat)
    X_scaled = scaler.transform(X_flat).reshape(X.shape)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = GlucoseLSTM(n_features=n_features, horizons=HORIZONS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_ds)

    model.eval()
    with torch.no_grad():
        val_preds = model(torch.tensor(X_val))
        val_loss = loss_fn(val_preds, torch.tensor(y_val)).item()

    # --- Also train a gradient-boosted tree baseline for feature-importance /
    # factor analysis, predicting the 60-min horizon target from the *current*
    # feature row (non-sequential, easy to interpret).
    tabular_X = feature_df[FEATURE_COLUMNS].values[:-STEPS_AHEAD[60]]
    tabular_y = feature_df["glucose_mgdl"].values[STEPS_AHEAD[60]:]
    gbr = GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=42)
    gbr.fit(tabular_X, tabular_y)

    patient_dir = os.path.join(MODEL_DIR, f"patient_{patient_id}")
    os.makedirs(patient_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(patient_dir, "lstm.pt"))
    joblib.dump(scaler, os.path.join(patient_dir, "scaler.pkl"))
    joblib.dump(gbr, os.path.join(patient_dir, "gbr_factor_model.pkl"))
    with open(os.path.join(patient_dir, "meta.json"), "w") as f:
        json.dump({
            "n_features": n_features,
            "horizons": HORIZONS,
            "seq_len": SEQ_LEN,
            "model_version": MODEL_VERSION,
            "val_loss_mse": val_loss,
            "feature_columns": FEATURE_COLUMNS,
        }, f)

    return {"val_loss_mse": val_loss, "n_train_samples": len(X_train)}
