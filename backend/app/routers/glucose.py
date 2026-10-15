from datetime import datetime, timedelta, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select

from .. import models, schemas
from ..database import get_db
from ..mock_data import generate_mock_patient_data

router = APIRouter(prefix="/glucose", tags=["glucose"])


@router.post("/mock-data/{external_id}")
def create_mock_data(external_id: str, days: int = 14, db: Session = Depends(get_db)):
    """Generates synthetic CGM + lifestyle data for a demo patient.
    Replace with real ingestion once you have actual CGM exports."""
    patient_id = generate_mock_patient_data(db, external_id=external_id, days=days)
    return {"patient_id": patient_id, "external_id": external_id, "days_generated": days}


@router.post("/readings", response_model=schemas.GlucoseReadingOut)
def add_reading(reading: schemas.GlucoseReadingIn, db: Session = Depends(get_db)):
    obj = models.GlucoseReading(**reading.dict())
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


@router.get("/readings/{patient_id}", response_model=List[schemas.GlucoseReadingOut])
def get_readings(patient_id: int, hours: int = 24, db: Session = Depends(get_db)):
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    rows = db.execute(
        select(models.GlucoseReading)
        .where(models.GlucoseReading.patient_id == patient_id,
               models.GlucoseReading.recorded_at >= since)
        .order_by(models.GlucoseReading.recorded_at)
    ).scalars().all()
    return rows


@router.post("/meals")
def add_meal(meal: schemas.MealIn, db: Session = Depends(get_db)):
    obj = models.Meal(**meal.dict())
    db.add(obj)
    db.commit()
    return {"status": "created", "meal_id": obj.meal_id}


@router.post("/insulin")
def add_insulin(dose: schemas.InsulinDoseIn, db: Session = Depends(get_db)):
    obj = models.InsulinDose(**dose.dict())
    db.add(obj)
    db.commit()
    return {"status": "created", "dose_id": obj.dose_id}


@router.post("/exercise")
def add_exercise(entry: schemas.ExerciseIn, db: Session = Depends(get_db)):
    obj = models.ExerciseLog(**entry.dict())
    db.add(obj)
    db.commit()
    return {"status": "created", "exercise_id": obj.exercise_id}


@router.post("/sleep")
def add_sleep(entry: schemas.SleepIn, db: Session = Depends(get_db)):
    obj = models.SleepLog(**entry.dict())
    db.add(obj)
    db.commit()
    return {"status": "created", "sleep_id": obj.sleep_id}


@router.post("/stress")
def add_stress(entry: schemas.StressIn, db: Session = Depends(get_db)):
    obj = models.StressLog(**entry.dict())
    db.add(obj)
    db.commit()
    return {"status": "created", "stress_id": obj.stress_id}


@router.get("/patients")
def list_patients(db: Session = Depends(get_db)):
    rows = db.execute(select(models.Patient)).scalars().all()
    return [{"patient_id": p.patient_id, "external_id": p.external_id,
             "diabetes_type": p.diabetes_type} for p in rows]
