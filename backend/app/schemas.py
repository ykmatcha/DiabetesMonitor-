from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel


class GlucoseReadingIn(BaseModel):
    patient_id: int
    recorded_at: datetime
    glucose_mgdl: float
    source: str = "cgm"


class GlucoseReadingOut(GlucoseReadingIn):
    reading_id: int

    class Config:
        from_attributes = True


class InsulinDoseIn(BaseModel):
    patient_id: int
    administered_at: datetime
    units: float
    insulin_type: str = "bolus"


class MealIn(BaseModel):
    patient_id: int
    consumed_at: datetime
    carbs_grams: float
    meal_type: str = "snack"


class ExerciseIn(BaseModel):
    patient_id: int
    started_at: datetime
    duration_minutes: int
    intensity: str = "moderate"
    activity_type: str = "walking"


class SleepIn(BaseModel):
    patient_id: int
    sleep_start: datetime
    sleep_end: datetime
    quality_score: Optional[float] = None


class StressIn(BaseModel):
    patient_id: int
    recorded_at: datetime
    stress_level: float


class PredictionRequest(BaseModel):
    patient_id: int
    horizons: List[int] = [30, 60, 120]


class PredictionOut(BaseModel):
    horizon_minutes: int
    predicted_glucose: float
    event_flag: str
    model_version: str


class PredictionResponse(BaseModel):
    patient_id: int
    as_of: datetime
    predictions: List[PredictionOut]


class GlucoseEventOut(BaseModel):
    detected_at: datetime
    event_type: str
    glucose_mgdl: float
    severity: str

    class Config:
        from_attributes = True


class FactorAnalysisResponse(BaseModel):
    patient_id: int
    feature_importance: dict
    correlations: dict
    summary: str
