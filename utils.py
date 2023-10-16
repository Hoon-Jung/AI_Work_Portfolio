import streamlit as st
import os

def set_openai_api_key():
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv(
            "OPENAI_API_KEY") or st.session_state.get("openai_api_key", ""))
    if openai_api_key:
        st.session_state.openai_api_key = openai_api_key
    else:
        st.error("Please add your OpenAI API key to continue.")
        st.info("Obtain your key from this link: https://platform.openai.com/account/api-keys")
        st.stop()