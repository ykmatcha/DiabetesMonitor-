"""
Detects clinically-relevant glucose events from raw readings and/or predictions:
hypoglycemia, hyperglycemia, rapid rise, rapid fall.

Thresholds follow common clinical conventions (ADA-style):
  - Hypoglycemia: < 70 mg/dL (severe < 54 mg/dL)
  - Hyperglycemia: > 180 mg/dL (severe > 250 mg/dL)
  - Rapid change: > 2 mg/dL per minute sustained over ~15 min
"""
from dataclasses import dataclass

HYPO_THRESHOLD = 70
HYPO_SEVERE = 54
HYPER_THRESHOLD = 180
HYPER_SEVERE = 250
RAPID_CHANGE_RATE = 2.0  # mg/dL per minute


@dataclass
class GlucoseAssessment:
    event_type: str   # normal | hypo | hyper | rapid_rise | rapid_fall
    severity: str      # none | mild | moderate | severe


def assess_reading(glucose_mgdl: float, slope_mgdl_per_min: float = 0.0) -> GlucoseAssessment:
    if glucose_mgdl < HYPO_SEVERE:
        return GlucoseAssessment("hypo", "severe")
    if glucose_mgdl < HYPO_THRESHOLD:
        return GlucoseAssessment("hypo", "moderate")
    if glucose_mgdl > HYPER_SEVERE:
        return GlucoseAssessment("hyper", "severe")
    if glucose_mgdl > HYPER_THRESHOLD:
        return GlucoseAssessment("hyper", "moderate")

    if slope_mgdl_per_min >= RAPID_CHANGE_RATE:
        return GlucoseAssessment("rapid_rise", "mild" if slope_mgdl_per_min < 3.5 else "moderate")
    if slope_mgdl_per_min <= -RAPID_CHANGE_RATE:
        return GlucoseAssessment("rapid_fall", "mild" if slope_mgdl_per_min > -3.5 else "moderate")

    return GlucoseAssessment("normal", "none")


def scan_feature_table(feature_df):
    """Runs assess_reading over an entire feature table (must contain
    'glucose_mgdl' and 'slope_15' columns) and returns a list of event dicts."""
    events = []
    for _, row in feature_df.iterrows():
        result = assess_reading(row["glucose_mgdl"], row.get("slope_15", 0.0))
        if result.event_type != "normal":
            events.append({
                "detected_at": row["recorded_at"],
                "event_type": result.event_type,
                "glucose_mgdl": float(row["glucose_mgdl"]),
                "severity": result.severity,
            })
    return events
