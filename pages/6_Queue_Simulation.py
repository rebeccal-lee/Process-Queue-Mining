import streamlit as st

st.set_page_config(page_title="Queue Simulation", layout="wide")

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

st.title("Queue Simulation")

df_raw = st.session_state.get("df_raw")
mapping = st.session_state.get("mapping")

if df_raw is None:
    st.info("Go to Home and upload a CSV first.")
    st.stop()

if mapping is None:
    st.info("Go to Upload Data and save a mapping first.")
    st.stop()

