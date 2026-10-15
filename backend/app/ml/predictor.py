import os
import json
import numpy as np
import torch
import joblib

from .model import GlucoseLSTM
from ..data_processing import FEATURE_COLUMNS
from .train import MODEL_DIR, SEQ_LEN, HORIZONS


class PatientModelNotFound(Exception):
    pass


def _patient_dir(patient_id: int) -> str:
    return os.path.join(MODEL_DIR, f"patient_{patient_id}")


def has_trained_model(patient_id: int) -> bool:
    return os.path.exists(os.path.join(_patient_dir(patient_id), "meta.json"))


def load_patient_model(patient_id: int):
    pdir = _patient_dir(patient_id)
    meta_path = os.path.join(pdir, "meta.json")
    if not os.path.exists(meta_path):
        raise PatientModelNotFound(f"No trained model for patient {patient_id}. Train first.")

    with open(meta_path) as f:
        meta = json.load(f)

    model = GlucoseLSTM(n_features=meta["n_features"], horizons=meta["horizons"])
    model.load_state_dict(torch.load(os.path.join(pdir, "lstm.pt"), map_location="cpu"))
    model.eval()

    scaler = joblib.load(os.path.join(pdir, "scaler.pkl"))
    gbr = joblib.load(os.path.join(pdir, "gbr_factor_model.pkl"))

    return model, scaler, gbr, meta


def predict_next(feature_df, patient_id: int, horizons=None):
    """Uses the last SEQ_LEN rows of the feature table to predict glucose
    at each requested horizon."""
    model, scaler, _, meta = load_patient_model(patient_id)
    horizons = horizons or meta["horizons"]

    window = feature_df[FEATURE_COLUMNS].values.astype(np.float32)[-SEQ_LEN:]
    if len(window) < SEQ_LEN:
        raise ValueError(f"Need at least {SEQ_LEN} recent readings to predict; have {len(window)}.")

    window_scaled = scaler.transform(window).reshape(1, SEQ_LEN, -1)

    with torch.no_grad():
        preds = model(torch.tensor(window_scaled)).numpy().flatten()

    all_horizons = meta["horizons"]
    result = {}
    for h in horizons:
        if h in all_horizons:
            result[h] = float(preds[all_horizons.index(h)])
    return result, meta["model_version"]
