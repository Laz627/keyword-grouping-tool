# app.py - This is your entry point for streamlit run
import streamlit as st

# Set page config as the FIRST Streamlit command
st.set_page_config(
    page_title="Keyword Tagging & Topic Generation Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import your main logic
import main_logic

# Run the app
main_logic.run_app()
