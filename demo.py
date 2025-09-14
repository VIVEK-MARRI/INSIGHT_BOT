import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env file
load_dotenv()

st.title("Gemini API Key Test")

# Check API key
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not found in environment!")
else:
    st.success("GEMINI_API_KEY loaded successfully!")

    user_input = st.text_input("Ask something to the Gemini model:")

    if user_input:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-turbo",  # or "gemini-1.5-pro" if quota allows
                api_key=api_key,
                temperature=0
            )

            # Correct way to call and extract response
            response = llm.generate([{"role": "user", "content": user_input}])
            text_response = response.generations[0][0].text

            st.write("Response from Gemini API:")
            st.write(text_response)

        except Exception as e:
            st.error(f"Error calling Gemini API: {e}")
