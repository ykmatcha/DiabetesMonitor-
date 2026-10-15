"""
Explains WHAT is driving a patient's glucose: feature importances from the
gradient-boosted tree baseline, plus simple Pearson correlations, so the
platform can answer "which factors matter most for me" in plain terms.
"""
import numpy as np
from ..data_processing import FEATURE_COLUMNS
from .predictor import load_patient_model

READABLE_NAMES = {
    "roll_mean_30": "Recent glucose trend (30 min avg)",
    "roll_std_30": "Recent glucose volatility",
    "roll_mean_60": "Recent glucose trend (60 min avg)",
    "slope_15": "Rate of glucose change",
    "carbs_30min": "Carbs eaten in last 30 min",
    "carbs_120min": "Carbs eaten in last 2 hours",
    "insulin_60min": "Insulin active in last hour",
    "insulin_180min": "Insulin active in last 3 hours",
    "minutes_since_exercise": "Time since last exercise",
    "hours_slept": "Hours slept previous night",
    "stress_level": "Reported stress level",
    "hour_sin": "Time of day",
    "hour_cos": "Time of day",
    "dow_sin": "Day of week",
    "dow_cos": "Day of week",
    "glucose_mgdl": "Current glucose level",
}


def analyze_factors(feature_df, patient_id: int):
    _, _, gbr, meta = load_patient_model(patient_id)
    importances = gbr.feature_importances_
    cols = meta["feature_columns"]

    # merge cyclical hour/day pairs for readability
    merged = {}
    for col, imp in zip(cols, importances):
        label = READABLE_NAMES.get(col, col)
        merged[label] = merged.get(label, 0.0) + float(imp)

    total = sum(merged.values()) or 1.0
    feature_importance = {k: round(v / total, 4) for k, v in
                           sorted(merged.items(), key=lambda kv: -kv[1])}

    corr_cols = [c for c in FEATURE_COLUMNS if c != "glucose_mgdl"]
    correlations = {}
    for c in corr_cols:
        if feature_df[c].std() > 0:
            corr = float(np.corrcoef(feature_df[c], feature_df["glucose_mgdl"])[0, 1])
            correlations[READABLE_NAMES.get(c, c)] = round(corr, 3)

    top_factor = next(iter(feature_importance))
    summary = (
        f"The strongest driver of this patient's glucose right now is '{top_factor}'. "
        f"Together, the top 3 factors "
        f"({', '.join(list(feature_importance.keys())[:3])}) account for "
        f"{round(sum(list(feature_importance.values())[:3]) * 100)}% of the model's "
        f"predictive signal."
    )

    return feature_importance, correlations, summary
