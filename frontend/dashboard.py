import os
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Diabetes Monitoring Platform", layout="wide")
st.title("🩸 AI Diabetes Monitoring & Prediction Platform")

# ---------------- Sidebar: patient selection / setup ----------------
st.sidebar.header("Patient Setup")

with st.sidebar.expander("Generate demo data", expanded=False):
    demo_id = st.text_input("Demo patient external ID", value="demo-patient-1")
    demo_days = st.slider("Days of synthetic history", 3, 60, 14)
    if st.button("Generate mock CGM + lifestyle data"):
        r = requests.post(f"{API_BASE_URL}/glucose/mock-data/{demo_id}", params={"days": demo_days})
        if r.ok:
            st.success(f"Generated data: {r.json()}")
        else:
            st.error(r.text)

patients_resp = requests.get(f"{API_BASE_URL}/glucose/patients")
patients = patients_resp.json() if patients_resp.ok else []

if not patients:
    st.warning("No patients yet. Use the sidebar to generate demo data, or ingest real records via the API.")
    st.stop()

patient_labels = {f"{p['external_id']} (id={p['patient_id']})": p["patient_id"] for p in patients}
selected_label = st.sidebar.selectbox("Select patient", list(patient_labels.keys()))
patient_id = patient_labels[selected_label]

hours_back = st.sidebar.slider("Hours of history to display", 6, 24 * 14, 48)

if st.sidebar.button("Train / retrain prediction model"):
    with st.spinner("Training LSTM + baseline model..."):
        r = requests.post(f"{API_BASE_URL}/predictions/train/{patient_id}")
    if r.ok:
        st.sidebar.success(f"Trained. {r.json()}")
    else:
        st.sidebar.error(r.text)

# ---------------- Main: glucose chart ----------------
readings_resp = requests.get(f"{API_BASE_URL}/glucose/readings/{patient_id}", params={"hours": hours_back})
readings = readings_resp.json() if readings_resp.ok else []

if readings:
    df = pd.DataFrame(readings)
    df["recorded_at"] = pd.to_datetime(df["recorded_at"])

    col1, col2 = st.columns([3, 1])

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["recorded_at"], y=df["glucose_mgdl"],
                                  mode="lines", name="Glucose (mg/dL)",
                                  line=dict(color="#2563eb", width=2)))
        fig.add_hrect(y0=70, y1=180, fillcolor="green", opacity=0.08, line_width=0,
                      annotation_text="Target range", annotation_position="top left")
        fig.add_hline(y=70, line_dash="dot", line_color="orange")
        fig.add_hline(y=180, line_dash="dot", line_color="orange")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                           yaxis_title="mg/dL", xaxis_title="Time")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        summary_resp = requests.get(f"{API_BASE_URL}/analytics/summary/{patient_id}")
        if summary_resp.ok:
            s = summary_resp.json()
            st.metric("Average glucose", f"{s['average_glucose_mgdl']} mg/dL")
            st.metric("Est. A1c", f"{s['estimated_a1c_percent']}%")
            st.metric("Time in range", f"{s['time_in_range_pct']}%")
            st.metric("Time below range", f"{s['time_below_range_pct']}%")
            st.metric("Time above range", f"{s['time_above_range_pct']}%")
else:
    st.info("No readings in this window yet.")

st.divider()

# ---------------- Predictions ----------------
st.subheader("🔮 Glucose Predictions")
pred_col, event_col = st.columns(2)

with pred_col:
    if st.button("Get predictions (30 / 60 / 120 min)"):
        r = requests.post(f"{API_BASE_URL}/predictions", json={"patient_id": patient_id, "horizons": [30, 60, 120]})
        if r.ok:
            data = r.json()
            for p in data["predictions"]:
                flag_emoji = {"normal": "✅", "hypo": "🔻", "hyper": "🔺",
                              "rapid_rise": "⬆️", "rapid_fall": "⬇️"}.get(p["event_flag"], "•")
                st.write(f"{flag_emoji} **+{p['horizon_minutes']} min:** "
                         f"{p['predicted_glucose']} mg/dL — _{p['event_flag']}_")
        else:
            st.error(r.text)

with event_col:
    st.write("**Recent detected events**")
    ev_resp = requests.get(f"{API_BASE_URL}/analytics/events/{patient_id}")
    if ev_resp.ok:
        events = ev_resp.json()
        if events:
            edf = pd.DataFrame(events).sort_values("detected_at", ascending=False).head(10)
            st.dataframe(edf, use_container_width=True, hide_index=True)
        else:
            st.write("No hypo/hyper/rapid-change events detected in current data.")

st.divider()

# ---------------- Factor analysis ----------------
st.subheader("🧬 What's driving this patient's glucose?")
if st.button("Run factor analysis"):
    r = requests.get(f"{API_BASE_URL}/analytics/factors/{patient_id}")
    if r.ok:
        data = r.json()
        st.write(data["summary"])
        fi_df = pd.DataFrame(list(data["feature_importance"].items()),
                              columns=["Factor", "Relative importance"])
        fig2 = go.Figure(go.Bar(x=fi_df["Relative importance"], y=fi_df["Factor"], orientation="h"))
        fig2.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.error(r.text)
