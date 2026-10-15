-- ============================================================
-- AI Diabetes Monitoring & Prediction Platform - Schema
-- ============================================================

CREATE TABLE IF NOT EXISTS patients (
    patient_id      SERIAL PRIMARY KEY,
    external_id     VARCHAR(64) UNIQUE NOT NULL,      -- e.g. anonymized patient code
    date_of_birth   DATE,
    diabetes_type   VARCHAR(16),                       -- 'type1' | 'type2' | 'gestational'
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Core CGM stream (typically every 5 minutes)
CREATE TABLE IF NOT EXISTS glucose_readings (
    reading_id      BIGSERIAL PRIMARY KEY,
    patient_id      INTEGER NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    recorded_at     TIMESTAMPTZ NOT NULL,
    glucose_mgdl    NUMERIC(6,2) NOT NULL,
    source          VARCHAR(32) DEFAULT 'cgm',          -- cgm | fingerstick
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_glucose_patient_time ON glucose_readings (patient_id, recorded_at DESC);

CREATE TABLE IF NOT EXISTS insulin_doses (
    dose_id         BIGSERIAL PRIMARY KEY,
    patient_id      INTEGER NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    administered_at TIMESTAMPTZ NOT NULL,
    units           NUMERIC(5,2) NOT NULL,
    insulin_type    VARCHAR(32),                        -- bolus | basal
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_insulin_patient_time ON insulin_doses (patient_id, administered_at DESC);

CREATE TABLE IF NOT EXISTS meals (
    meal_id         BIGSERIAL PRIMARY KEY,
    patient_id      INTEGER NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    consumed_at     TIMESTAMPTZ NOT NULL,
    carbs_grams     NUMERIC(6,2) NOT NULL,
    meal_type       VARCHAR(32),                        -- breakfast | lunch | dinner | snack
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_meals_patient_time ON meals (patient_id, consumed_at DESC);

CREATE TABLE IF NOT EXISTS exercise_logs (
    exercise_id     BIGSERIAL PRIMARY KEY,
    patient_id      INTEGER NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    started_at      TIMESTAMPTZ NOT NULL,
    duration_minutes INTEGER NOT NULL,
    intensity       VARCHAR(16),                        -- low | moderate | high
    activity_type   VARCHAR(64),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_exercise_patient_time ON exercise_logs (patient_id, started_at DESC);

CREATE TABLE IF NOT EXISTS sleep_logs (
    sleep_id        BIGSERIAL PRIMARY KEY,
    patient_id      INTEGER NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    sleep_start     TIMESTAMPTZ NOT NULL,
    sleep_end       TIMESTAMPTZ NOT NULL,
    quality_score   NUMERIC(4,2),                       -- 0-100
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_sleep_patient_time ON sleep_logs (patient_id, sleep_start DESC);

CREATE TABLE IF NOT EXISTS stress_logs (
    stress_id       BIGSERIAL PRIMARY KEY,
    patient_id      INTEGER NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    recorded_at     TIMESTAMPTZ NOT NULL,
    stress_level    NUMERIC(4,2),                       -- 0-10 self-reported or HRV-derived
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_stress_patient_time ON stress_logs (patient_id, recorded_at DESC);

-- Model predictions, stored for auditing / drift analysis
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id   BIGSERIAL PRIMARY KEY,
    patient_id      INTEGER NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    predicted_at    TIMESTAMPTZ NOT NULL DEFAULT now(),   -- when the prediction was made
    horizon_minutes INTEGER NOT NULL,                     -- 30 | 60 | 120
    predicted_glucose NUMERIC(6,2) NOT NULL,
    model_version   VARCHAR(32) NOT NULL,
    actual_glucose  NUMERIC(6,2),                         -- filled in later for evaluation
    event_flag      VARCHAR(32)                           -- hypo | hyper | rapid_rise | rapid_fall | normal
);
CREATE INDEX IF NOT EXISTS idx_predictions_patient_time ON predictions (patient_id, predicted_at DESC);

-- Detected glucose events (hypo/hyper/rapid change) derived from raw readings
CREATE TABLE IF NOT EXISTS glucose_events (
    event_id        BIGSERIAL PRIMARY KEY,
    patient_id      INTEGER NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    detected_at     TIMESTAMPTZ NOT NULL,
    event_type      VARCHAR(32) NOT NULL,                 -- hypo | hyper | rapid_rise | rapid_fall
    glucose_mgdl    NUMERIC(6,2) NOT NULL,
    severity        VARCHAR(16),                          -- mild | moderate | severe
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_events_patient_time ON glucose_events (patient_id, detected_at DESC);
