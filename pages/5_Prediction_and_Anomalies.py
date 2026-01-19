import streamlit as st
from Home import render_sidebar

st.set_page_config(page_title="Predictive Statistics and Anomaly Detection", layout="wide")
st.title("Predictive Statistics and Anomaly Detection")

df_raw = st.session_state.get("df_raw")
mapping = st.session_state.get("mapping")

if df_raw is None:
    st.info("Go to Home and upload a CSV first.")
    st.stop()

if mapping is None:
    st.info("Go to Upload Data and save a mapping first.")
    st.stop()

