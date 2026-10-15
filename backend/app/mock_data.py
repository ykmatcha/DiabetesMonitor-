"""
Generates physiologically-plausible synthetic glucose + lifestyle data so the
whole platform can be exercised end-to-end before real CGM exports are wired in.

Swap this out later: as long as your real data lands in the same tables
(glucose_readings, insulin_doses, meals, exercise_logs, sleep_logs, stress_logs)
nothing else in the service needs to change.
"""
import random
import math
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from . import models


def _simulate_glucose_series(start: datetime, days: int, base: float = 110.0):
    """Simulate a 5-minute-interval CGM trace with meal spikes, insulin dips,
    circadian drift, and noise."""
    n_points = days * 24 * 12  # every 5 minutes
    readings = []
    meals = []
    insulin = []
    exercise = []
    sleep = []
    stress = []

    glucose = base
    t = start

    # schedule meals roughly at 8am, 1pm, 7pm each day, plus random snacks
    for day in range(days):
        day_start = start + timedelta(days=day)
        for hour, mtype in [(8, "breakfast"), (13, "lunch"), (19, "dinner")]:
            meals.append({
                "consumed_at": day_start.replace(hour=hour, minute=random.randint(0, 30)),
                "carbs_grams": random.uniform(40, 90),
                "meal_type": mtype,
            })
            insulin.append({
                "administered_at": day_start.replace(hour=hour, minute=random.randint(0, 30)),
                "units": random.uniform(4, 9),
                "insulin_type": "bolus",
            })
        # basal insulin once a day
        insulin.append({
            "administered_at": day_start.replace(hour=22, minute=0),
            "units": random.uniform(10, 14),
            "insulin_type": "basal",
        })
        # random chance of exercise
        if random.random() < 0.5:
            exercise.append({
                "started_at": day_start.replace(hour=random.choice([7, 17, 18]), minute=0),
                "duration_minutes": random.choice([20, 30, 45, 60]),
                "intensity": random.choice(["low", "moderate", "high"]),
                "activity_type": random.choice(["walking", "running", "cycling", "gym"]),
            })
        sleep.append({
            "sleep_start": day_start.replace(hour=23, minute=0),
            "sleep_end": (day_start + timedelta(days=1)).replace(hour=7, minute=0),
            "quality_score": random.uniform(50, 95),
        })
        for _ in range(3):
            stress.append({
                "recorded_at": day_start.replace(hour=random.randint(6, 22)),
                "stress_level": round(random.uniform(1, 8), 1),
            })

    meal_effects = []  # (time, carbs) pending absorption
    insulin_effects = []  # (time, units) pending action

    for m in meals:
        meal_effects.append([m["consumed_at"], m["carbs_grams"]])
    for i in insulin:
        insulin_effects.append([i["administered_at"], i["units"]])

    for idx in range(n_points):
        current_time = start + timedelta(minutes=5 * idx)

        # circadian baseline drift (slightly higher glucose early morning - dawn phenomenon)
        hour_of_day = current_time.hour + current_time.minute / 60
        circadian = 8 * math.sin((hour_of_day - 5) / 24 * 2 * math.pi)

        # meal contribution: carbs raise glucose over ~2 hours after eating
        meal_contrib = 0.0
        for m_time, carbs in meal_effects:
            minutes_since = (current_time - m_time).total_seconds() / 60
            if 0 <= minutes_since <= 180:
                # peak around 45-60 min, decaying afterward
                effect = carbs * 0.9 * math.exp(-((minutes_since - 50) ** 2) / (2 * 45 ** 2))
                meal_contrib += effect

        # insulin contribution: lowers glucose over ~4 hours (bolus) / slow steady (basal)
        insulin_contrib = 0.0
        for i_time, units in insulin_effects:
            minutes_since = (current_time - i_time).total_seconds() / 60
            if 0 <= minutes_since <= 300:
                effect = units * 6 * math.exp(-((minutes_since - 90) ** 2) / (2 * 70 ** 2))
                insulin_contrib += effect

        noise = random.gauss(0, 4)

        target = base + circadian + meal_contrib - insulin_contrib
        # smooth glucose toward target (first-order lag) to avoid unrealistic jumps
        glucose = glucose + (target - glucose) * 0.15 + noise
        glucose = max(40, min(400, glucose))

        readings.append({
            "recorded_at": current_time,
            "glucose_mgdl": round(glucose, 1),
            "source": "cgm",
        })

    return readings, meals, insulin, exercise, sleep, stress


def generate_mock_patient_data(db: Session, external_id: str = "demo-patient-1", days: int = 14):
    """Creates (or reuses) a demo patient and populates `days` worth of
    synthetic glucose + lifestyle data. Returns the patient_id."""
    patient = db.query(models.Patient).filter_by(external_id=external_id).first()
    if not patient:
        patient = models.Patient(external_id=external_id, diabetes_type="type1")
        db.add(patient)
        db.commit()
        db.refresh(patient)

    start = datetime.now(timezone.utc) - timedelta(days=days)
    readings, meals, insulin, exercise, sleep, stress = _simulate_glucose_series(start, days)

    db.bulk_save_objects([
        models.GlucoseReading(patient_id=patient.patient_id, **r) for r in readings
    ])
    db.bulk_save_objects([
        models.Meal(patient_id=patient.patient_id, **m) for m in meals
    ])
    db.bulk_save_objects([
        models.InsulinDose(patient_id=patient.patient_id, **i) for i in insulin
    ])
    db.bulk_save_objects([
        models.ExerciseLog(patient_id=patient.patient_id, **e) for e in exercise
    ])
    db.bulk_save_objects([
        models.SleepLog(patient_id=patient.patient_id, **s) for s in sleep
    ])
    db.bulk_save_objects([
        models.StressLog(patient_id=patient.patient_id, **s) for s in stress
    ])
    db.commit()

    return patient.patient_id
