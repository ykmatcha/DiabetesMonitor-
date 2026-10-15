"""
Turns raw rows from Postgres into a clean, model-ready feature table.
This is the "Python -> processes glucose data" stage.
"""
from datetime import timedelta
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import select
from . import models


def load_patient_frames(db: Session, patient_id: int):
    """Pull all relevant raw tables for a patient into pandas DataFrames."""
    def _df(model, cols):
        rows = db.execute(select(model).where(model.patient_id == patient_id)).scalars().all()
        if not rows:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame([{c: getattr(r, c) for c in cols} for r in rows])

    glucose = _df(models.GlucoseReading, ["recorded_at", "glucose_mgdl"])
    meals = _df(models.Meal, ["consumed_at", "carbs_grams"])
    insulin = _df(models.InsulinDose, ["administered_at", "units", "insulin_type"])
    exercise = _df(models.ExerciseLog, ["started_at", "duration_minutes", "intensity"])
    sleep = _df(models.SleepLog, ["sleep_start", "sleep_end", "quality_score"])
    stress = _df(models.StressLog, ["recorded_at", "stress_level"])

    for df, col in [(glucose, "recorded_at"), (meals, "consumed_at"),
                     (insulin, "administered_at"), (exercise, "started_at"),
                     (sleep, "sleep_start"), (stress, "recorded_at")]:
        if not df.empty:
            df[col] = pd.to_datetime(df[col], utc=True)

    return glucose, meals, insulin, exercise, sleep, stress


def build_feature_table(db: Session, patient_id: int) -> pd.DataFrame:
    """
    Builds a time-indexed feature table at glucose-reading granularity with:
    - recent glucose stats (rolling mean/std/slope)
    - carbs consumed in last 30/60/120 min
    - insulin units active in last 60/180 min
    - minutes since last exercise, exercise intensity in last 4h
    - hours slept previous night, sleep quality
    - latest reported stress level
    - time-of-day / day-of-week cyclical encodings
    """
    glucose, meals, insulin, exercise, sleep, stress = load_patient_frames(db, patient_id)
    if glucose.empty:
        return pd.DataFrame()

    glucose = glucose.sort_values("recorded_at").reset_index(drop=True)
    glucose["glucose_mgdl"] = glucose["glucose_mgdl"].astype(float)

    # rolling glucose stats (assumes ~5 min spacing; robust to gaps via time-based rolling)
    g = glucose.set_index("recorded_at")
    g["roll_mean_30"] = g["glucose_mgdl"].rolling("30min").mean()
    g["roll_std_30"] = g["glucose_mgdl"].rolling("30min").std().fillna(0)
    g["roll_mean_60"] = g["glucose_mgdl"].rolling("60min").mean()
    g["slope_15"] = g["glucose_mgdl"].diff(3) / 15.0  # approx mg/dL per minute over last ~15min

    def carbs_in_window(ts, minutes):
        if meals.empty:
            return 0.0
        window_start = ts - timedelta(minutes=minutes)
        mask = (meals["consumed_at"] > window_start) & (meals["consumed_at"] <= ts)
        return float(meals.loc[mask, "carbs_grams"].sum())

    def insulin_in_window(ts, minutes):
        if insulin.empty:
            return 0.0
        window_start = ts - timedelta(minutes=minutes)
        mask = (insulin["administered_at"] > window_start) & (insulin["administered_at"] <= ts)
        return float(insulin.loc[mask, "units"].sum())

    def minutes_since_exercise(ts):
        if exercise.empty:
            return 999.0
        past = exercise[exercise["started_at"] <= ts]
        if past.empty:
            return 999.0
        return float((ts - past["started_at"].max()).total_seconds() / 60)

    def hours_slept_prior_night(ts):
        if sleep.empty:
            return np.nan
        past = sleep[sleep["sleep_end"] <= ts]
        if past.empty:
            return np.nan
        last = past.sort_values("sleep_end").iloc[-1]
        return float((last["sleep_end"] - last["sleep_start"]).total_seconds() / 3600)

    def latest_stress(ts):
        if stress.empty:
            return np.nan
        past = stress[stress["recorded_at"] <= ts]
        if past.empty:
            return np.nan
        return float(past.sort_values("recorded_at").iloc[-1]["stress_level"])

    idx = g.index
    g["carbs_30min"] = [carbs_in_window(ts, 30) for ts in idx]
    g["carbs_120min"] = [carbs_in_window(ts, 120) for ts in idx]
    g["insulin_60min"] = [insulin_in_window(ts, 60) for ts in idx]
    g["insulin_180min"] = [insulin_in_window(ts, 180) for ts in idx]
    g["minutes_since_exercise"] = [minutes_since_exercise(ts) for ts in idx]
    g["hours_slept"] = [hours_slept_prior_night(ts) for ts in idx]
    g["stress_level"] = [latest_stress(ts) for ts in idx]

    g["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    g["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    g["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    g["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)

    g = g.reset_index().fillna(method="ffill").fillna(0)
    return g


FEATURE_COLUMNS = [
    "glucose_mgdl", "roll_mean_30", "roll_std_30", "roll_mean_60", "slope_15",
    "carbs_30min", "carbs_120min", "insulin_60min", "insulin_180min",
    "minutes_since_exercise", "hours_slept", "stress_level",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
