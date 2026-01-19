import streamlit as st

st.set_page_config(page_title="Upload Data", layout="wide")

def render_sidebar():
    st.sidebar.subheader("Core")
    st.sidebar.page_link("Home.py", label="Home")
    st.sidebar.page_link("pages/1_Upload_Data.py", label="Upload Data")
    st.sidebar.page_link("pages/2_Discovery.py", label="Patient Flow")
    st.sidebar.page_link("pages/3_Patient_Flow.py", label="Patient Flow")
    st.sidebar.page_link("pages/4_Queue_Visualization.py", label="Queue Visualization")

    st.sidebar.subheader("Advanced")
    st.sidebar.page_link("pages/5_Prediction_and_Anomalies.py", label="Prediction and Anomalies")
    st.sidebar.page_link("pages/6_Queue_Simulation.py", label="Queue Simulations")
render_sidebar()

st.title("Upload Data")

df_raw = st.session_state.get("df_raw")
if df_raw is None:
    st.info("Go to Home and upload a CSV first.")
    st.stop()

cols = list(df_raw.columns)

case_id = st.selectbox("Case ID column", cols)
activity = st.selectbox("Activity/Event column", cols)
timestamp = st.selectbox("Timestamp column", cols)
resource = st.selectbox("Resource column (optional)", ["(none)"] + cols)
attributes = st.multiselect("Attribute columns (optional)", cols)

mapping = {
    "case_id": case_id,
    "activity": activity,
    "timestamp": timestamp,
    "resource": None if resource == "(none)" else resource,
    "attributes": attributes,
}

st.code(mapping, language="python")

if st.button("Save mapping"):
    st.session_state["mapping"] = mapping
    st.success("Mapping saved.")
