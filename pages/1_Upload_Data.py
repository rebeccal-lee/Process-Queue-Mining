import streamlit as st

st.set_page_config(page_title="Upload Data", layout="wide")
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
