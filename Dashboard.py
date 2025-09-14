import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import dashboard_html, dashboard_css
import os

def main():
    load_dotenv()
    st.set_page_config(page_title="Dashboard", initial_sidebar_state="collapsed")

    # Apply custom CSS and HTML for dashboard
    st.write(dashboard_css, unsafe_allow_html=True)
    st.write(dashboard_html, unsafe_allow_html=True)

    # Gemini API integration status
    st.markdown("### Connected to Gemini API")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        st.success("Gemini API Key detected ✅")
    else:
        st.warning("No Gemini API Key found. Please set GEMINI_API_KEY in .env ⚠️")

if __name__ == "__main__":
    main()
