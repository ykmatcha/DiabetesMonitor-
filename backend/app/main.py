from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import Base, engine
from .routers import glucose, predictions, analytics

# Create tables if they don't already exist (init.sql also runs on first
# Postgres container boot, this is a safety net for non-docker runs).
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Diabetes Monitoring & Prediction Platform",
    description="Predicts future glucose levels, detects hypo/hyperglycemic "
                "events, and explains which lifestyle factors drive a "
                "patient's glucose.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(glucose.router)
app.include_router(predictions.router)
app.include_router(analytics.router)


@app.get("/")
def root():
    return {
        "service": "AI Diabetes Monitoring & Prediction Platform",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok"}
