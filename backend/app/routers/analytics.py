from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import schemas
from ..database import get_db
from ..data_processing import build_feature_table
from ..ml.event_detection import scan_feature_table
from ..ml.factor_analysis import analyze_factors
from ..ml.predictor import has_trained_model

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/events/{patient_id}", response_model=List[schemas.GlucoseEventOut])
def detect_events(patient_id: int, db: Session = Depends(get_db)):
    feature_df = build_feature_table(db, patient_id)
    if feature_df.empty:
        raise HTTPException(404, "No glucose data found for this patient.")
    events = scan_feature_table(feature_df)
    return events


@router.get("/factors/{patient_id}", response_model=schemas.FactorAnalysisResponse)
def factor_analysis(patient_id: int, db: Session = Depends(get_db)):
    if not has_trained_model(patient_id):
        raise HTTPException(400, "No trained model for this patient yet. Call /predictions/train/{patient_id} first.")
    feature_df = build_feature_table(db, patient_id)
    if feature_df.empty:
        raise HTTPException(404, "No glucose data found for this patient.")

    feature_importance, correlations, summary = analyze_factors(feature_df, patient_id)
    return schemas.FactorAnalysisResponse(
        patient_id=patient_id,
        feature_importance=feature_importance,
        correlations=correlations,
        summary=summary,
    )


@router.get("/summary/{patient_id}")
def glucose_summary(patient_id: int, db: Session = Depends(get_db)):
    """Standard CGM metrics: time-in-range, average glucose, estimated A1c, variability."""
    feature_df = build_feature_table(db, patient_id)
    if feature_df.empty:
        raise HTTPException(404, "No glucose data found for this patient.")

    g = feature_df["glucose_mgdl"]
    time_in_range = float(((g >= 70) & (g <= 180)).mean() * 100)
    time_below = float((g < 70).mean() * 100)
    time_above = float((g > 180).mean() * 100)
    avg_glucose = float(g.mean())
    estimated_a1c = round((avg_glucose + 46.7) / 28.7, 2)  # standard ADAG formula
    cv = float(g.std() / g.mean() * 100) if g.mean() else 0.0

    return {
        "patient_id": patient_id,
        "average_glucose_mgdl": round(avg_glucose, 1),
        "estimated_a1c_percent": estimated_a1c,
        "time_in_range_pct": round(time_in_range, 1),
        "time_below_range_pct": round(time_below, 1),
        "time_above_range_pct": round(time_above, 1),
        "coefficient_of_variation_pct": round(cv, 1),
        "n_readings": int(len(g)),
    }
