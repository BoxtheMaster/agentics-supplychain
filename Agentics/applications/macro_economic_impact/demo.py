# main.py
import streamlit as st
from macro_economic_analysis import macro_economic_analysis_ui
from pydantic_model_builder import pydantic_model_bulilder_ui

# ---------- Host page with tabs ----------
st.set_page_config(page_title="Multi-App in Tabs", page_icon="🗂️", layout="wide")
st.title("Macro Economic Market Analysis")
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "NA"

if "new_model_name" not in st.session_state:
    st.session_state.new_model_name = "NA"

tab_a, tab_b = st.tabs(["Market Analysis", "Define Target Model"])

with tab_a:
    macro_economic_analysis_ui()  # unique prefix

with tab_b:
    pydantic_model_bulilder_ui()  # unique prefix
