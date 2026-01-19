# Home.py
import streamlit as st
import pandas as pd
from io import StringIO

st.set_page_config(page_title="Home", layout="wide")
st.title("Home")

st.header("TimER: An ED Triage Monitoring Dashboard")
st.write("Upload a CSV to unlock the other pages in the left sidebar.")

@st.cache_data
def read_csv_from_upload(uploaded_file) -> pd.DataFrame:
    # Streamlit UploadedFile -> bytes -> pandas
    content = uploaded_file.getvalue().decode("utf-8")
    return pd.read_csv(StringIO(content))

# Upload on MAIN BODY (not sidebar)
uploaded = st.file_uploader("Upload CSV", type=["csv"])

# Initialize session state keys (reproducible / predictable)
st.session_state.setdefault("df_raw", None)
st.session_state.setdefault("mapping", None)
st.session_state.setdefault("discovery_results", None)

if uploaded is None:
    st.info("No file uploaded yet. Once you upload, use the left sidebar pages.")
    st.stop()

df_raw = read_csv_from_upload(uploaded)
st.session_state["df_raw"] = df_raw

st.success(f"Dataset loaded: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

st.subheader("Preview")
st.dataframe(df_raw.head(25), use_container_width=True)

