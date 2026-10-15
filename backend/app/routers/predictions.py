from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db
from ..data_processing import build_feature_table
from ..ml.train import train_lstm_for_patient
from ..ml.predictor import predict_next, has_trained_model, PatientModelNotFound
from ..ml.event_detection import assess_reading

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("/train/{patient_id}")
def train_model(patient_id: int, db: Session = Depends(get_db)):
    feature_df = build_feature_table(db, patient_id)
    if feature_df.empty:
        raise HTTPException(404, "No glucose data found for this patient. Generate/load data first.")
    try:
        result = train_lstm_for_patient(feature_df, patient_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"status": "trained", "patient_id": patient_id, **result}


@router.post("", response_model=schemas.PredictionResponse)
def get_predictions(req: schemas.PredictionRequest, db: Session = Depends(get_db)):
    if not has_trained_model(req.patient_id):
        raise HTTPException(400, "No trained model for this patient yet. Call /predictions/train/{patient_id} first.")

    feature_df = build_feature_table(db, req.patient_id)
    if feature_df.empty:
        raise HTTPException(404, "No glucose data found for this patient.")

    try:
        preds, model_version = predict_next(feature_df, req.patient_id, horizons=req.horizons)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except PatientModelNotFound as e:
        raise HTTPException(400, str(e))

    last_slope = float(feature_df.iloc[-1].get("slope_15", 0.0))
    outputs = []
    for horizon, value in preds.items():
        assessment = assess_reading(value, last_slope)
        outputs.append(schemas.PredictionOut(
            horizon_minutes=horizon,
            predicted_glucose=round(value, 1),
            event_flag=assessment.event_type,
            model_version=model_version,
        ))
        db.add(models.Prediction(
            patient_id=req.patient_id,
            horizon_minutes=horizon,
            predicted_glucose=round(value, 1),
            model_version=model_version,
            event_flag=assessment.event_type,
        ))
    db.commit()

    return schemas.PredictionResponse(
        patient_id=req.patient_id,
        as_of=datetime.now(timezone.utc),
        predictions=outputs,
    )
