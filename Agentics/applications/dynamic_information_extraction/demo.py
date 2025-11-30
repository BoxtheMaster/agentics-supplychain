# main.py
import streamlit as st
from process_agentics_data import data_processing
from pydantic_model_builder import pydantic_model_bulilder_ui

# ---------- Host page with tabs ----------
st.set_page_config(page_title="Dynamic Information Extraction", layout="wide")
st.title("Dynamic Information Extraction")
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "NA"

if "new_model_name" not in st.session_state:
    st.session_state.new_model_name = "NA"

tab_a, tab_b = st.tabs(["Market Analysis", "Define Target Model"])

with tab_a:
    data_processing()  # unique prefix

with tab_b:
    pydantic_model_bulilder_ui()  # unique prefix
