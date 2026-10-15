from sqlalchemy import Column, Integer, BigInteger, String, Numeric, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from .database import Base


class Patient(Base):
    __tablename__ = "patients"
    patient_id = Column(Integer, primary_key=True)
    external_id = Column(String(64), unique=True, nullable=False)
    date_of_birth = Column(DateTime)
    diabetes_type = Column(String(16))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class GlucoseReading(Base):
    __tablename__ = "glucose_readings"
    reading_id = Column(BigInteger, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.patient_id"), nullable=False)
    recorded_at = Column(DateTime(timezone=True), nullable=False)
    glucose_mgdl = Column(Numeric(6, 2), nullable=False)
    source = Column(String(32), default="cgm")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class InsulinDose(Base):
    __tablename__ = "insulin_doses"
    dose_id = Column(BigInteger, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.patient_id"), nullable=False)
    administered_at = Column(DateTime(timezone=True), nullable=False)
    units = Column(Numeric(5, 2), nullable=False)
    insulin_type = Column(String(32))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Meal(Base):
    __tablename__ = "meals"
    meal_id = Column(BigInteger, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.patient_id"), nullable=False)
    consumed_at = Column(DateTime(timezone=True), nullable=False)
    carbs_grams = Column(Numeric(6, 2), nullable=False)
    meal_type = Column(String(32))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ExerciseLog(Base):
    __tablename__ = "exercise_logs"
    exercise_id = Column(BigInteger, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.patient_id"), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=False)
    duration_minutes = Column(Integer, nullable=False)
    intensity = Column(String(16))
    activity_type = Column(String(64))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class SleepLog(Base):
    __tablename__ = "sleep_logs"
    sleep_id = Column(BigInteger, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.patient_id"), nullable=False)
    sleep_start = Column(DateTime(timezone=True), nullable=False)
    sleep_end = Column(DateTime(timezone=True), nullable=False)
    quality_score = Column(Numeric(4, 2))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class StressLog(Base):
    __tablename__ = "stress_logs"
    stress_id = Column(BigInteger, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.patient_id"), nullable=False)
    recorded_at = Column(DateTime(timezone=True), nullable=False)
    stress_level = Column(Numeric(4, 2))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Prediction(Base):
    __tablename__ = "predictions"
    prediction_id = Column(BigInteger, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.patient_id"), nullable=False)
    predicted_at = Column(DateTime(timezone=True), server_default=func.now())
    horizon_minutes = Column(Integer, nullable=False)
    predicted_glucose = Column(Numeric(6, 2), nullable=False)
    model_version = Column(String(32), nullable=False)
    actual_glucose = Column(Numeric(6, 2))
    event_flag = Column(String(32))


class GlucoseEvent(Base):
    __tablename__ = "glucose_events"
    event_id = Column(BigInteger, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.patient_id"), nullable=False)
    detected_at = Column(DateTime(timezone=True), nullable=False)
    event_type = Column(String(32), nullable=False)
    glucose_mgdl = Column(Numeric(6, 2), nullable=False)
    severity = Column(String(16))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
