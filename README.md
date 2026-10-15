# AI Diabetes Monitoring & Prediction Platform

Turns historical CGM + lifestyle data into glucose forecasts, event alerts
(hypo/hyper/rapid change), and factor explanations — served via API and a
live dashboard.

## Stack
| Layer | Tech |
|---|---|
| Data processing / feature engineering | Python + pandas |
| Storage | PostgreSQL |
| Forecasting | PyTorch (LSTM, multi-horizon) |
| Interpretability / factor analysis | scikit-learn (GradientBoostingRegressor) |
| API | FastAPI |
| Dashboard | Streamlit + Plotly |
| Packaging | Docker Compose |

## Architecture
```
CGM/lifestyle data --> Postgres --> feature engineering (data_processing.py)
                                         |
                          -------------------------------
                          |                              |
                   PyTorch LSTM                  GBR (feature importance)
                   (30/60/120 min forecast)       (factor analysis)
                          |                              |
                          -------------------------------
                                         |
                                    FastAPI routes
                                         |
                                 Streamlit dashboard
```

## Quick start
```bash
docker compose up --build
```
- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501
- Postgres: localhost:5432 (user: diabetes_user / pass: diabetes_pass / db: diabetes_platform)

## Try it end-to-end with mock data
1. Open the dashboard (http://localhost:8501), expand **"Generate demo data"**
   in the sidebar, and click **Generate**. This creates a synthetic patient
   with ~14 days of realistic CGM readings, meals, insulin, exercise, sleep,
   and stress logs.
2. Click **"Train / retrain prediction model"** in the sidebar.
3. Click **"Get predictions"** to see 30/60/120-minute glucose forecasts.
4. Check **"Recent detected events"** for hypo/hyper/rapid-change flags.
5. Click **"Run factor analysis"** to see which lifestyle factors are
   driving that patient's glucose.

Or hit the API directly:
```bash
# generate mock data
curl -X POST "http://localhost:8000/glucose/mock-data/demo-patient-1?days=14"

# train
curl -X POST http://localhost:8000/predictions/train/1

# predict
curl -X POST http://localhost:8000/predictions \
  -H "Content-Type: application/json" \
  -d '{"patient_id": 1, "horizons": [30, 60, 120]}'

# events
curl http://localhost:8000/analytics/events/1

# factor analysis
curl http://localhost:8000/analytics/factors/1

# CGM summary stats (time-in-range, est. A1c, etc.)
curl http://localhost:8000/analytics/summary/1
```

## Plugging in your real mock data
Replace calls to `/glucose/mock-data/{external_id}` with real ingestion —
either:
- Bulk load CSV exports into the same Postgres tables (`glucose_readings`,
  `meals`, `insulin_doses`, `exercise_logs`, `sleep_logs`, `stress_logs`) — see
  `backend/sql/init.sql` for the schema, or
- POST individual records to `/glucose/readings`, `/glucose/meals`,
  `/glucose/insulin`, `/glucose/exercise`, `/glucose/sleep`, `/glucose/stress`.

Nothing else in the pipeline needs to change — feature engineering, training,
prediction, event detection, and factor analysis all key off those same
tables.

## What each model does
- **GlucoseLSTM** (`backend/app/ml/model.py`): a 2-layer LSTM that looks at
  the last 60 minutes of engineered features (rolling glucose stats, active
  carbs, active insulin, time since exercise, sleep, stress, time-of-day) and
  predicts glucose 30/60/120 minutes ahead, jointly.
- **GradientBoostingRegressor** (`backend/app/ml/train.py`): trained
  alongside the LSTM for the same data, used purely for its interpretable
  `feature_importances_` in `/analytics/factors/{patient_id}` — this is what
  answers "what's actually driving my glucose."
- **Event detection** (`backend/app/ml/event_detection.py`): rule-based on
  ADA-style thresholds (hypo <70, severe <54; hyper >180, severe >250; rapid
  change ≥2 mg/dL/min), applied to both raw readings and predictions.

## Notes / next steps
- Models are trained per-patient and saved to `saved_models/patient_{id}/`
  (mounted as a Docker volume so they persist across restarts).
- `SEQ_LEN` (60 min lookback) and horizons (30/60/120 min) assume ~5-minute
  CGM sampling — adjust in `backend/app/ml/train.py` if your real device
  samples differently.
- For production: add auth, real CGM device integrations (Dexcom/Libre
  APIs), model retraining schedules, and alerting (push/SMS) on severe events.
- To swap the dashboard for React later, the FastAPI backend already exposes
  everything the dashboard needs as JSON — `frontend/` can be replaced
  independently without touching `backend/`.
