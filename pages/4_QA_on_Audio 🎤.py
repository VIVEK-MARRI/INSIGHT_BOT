import streamlit as st
from htmlTemplates import css, bot_template, user_template
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import speech_recognition as sr
from google.api_core.exceptions import ResourceExhausted
import os, time

# ---------- Extract Text from Audio ----------
def get_audio_text(audio_file):
    text = "Text extracted from audio:\n"
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text += recognizer.recognize_google(audio_data)
    except sr.RequestError as e:
        text += f"\n[API Error: {e}]"
    except sr.UnknownValueError:
        text += "\n[Could not understand audio]"
    except Exception as e:
        text += f"\n[Error: {e}]"
    return text

# ---------- Text Splitting ----------
def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=800, chunk_overlap=150, length_function=len
    )
    return splitter.split_text(text)

# ---------- Vector Store ----------
def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
    return Chroma.from_texts(texts=chunks, embedding=embeddings)

# ---------- Conversational Chain ----------
def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # switched to flash to reduce quota issues
        api_key=os.environ.get("GEMINI_API_KEY"),
        temperature=0
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

# ---------- Handle User Input ----------
def handle_user_input(user_question):
    try:
        response = st.session_state.conversation({"question": user_question})
        st.session_state.chat_history = response["chat_history"]

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    except ResourceExhausted:
        st.warning("⚠️ Gemini quota exceeded. Retrying in 10 seconds...")
        time.sleep(10)
        return handle_user_input(user_question)
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# ---------- Streamlit App ----------
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Audio", page_icon=":microphone:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Audio :microphone:")
    user_question = st.text_input("Ask a question about your audio:")
    if user_question and st.session_state.conversation:
        handle_user_input(user_question)
    elif user_question:
        st.warning("Please upload and process audio first.")

    with st.sidebar:
        st.subheader("Upload Your Audio")
        audio_file = st.file_uploader(
            "Upload a WAV audio file and click 'Process'", type=["wav"]
        )

        if st.button("Process"):
            if audio_file:
                with st.spinner("Processing audio..."):
                    try:
                        raw_text = get_audio_text(audio_file)
                        chunks = get_text_chunks(raw_text)
                        vectorstore = get_vectorstore(chunks)
                        st.sidebar.success("✅ Processing completed!")
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.error("Please upload an audio file.")

if __name__ == "__main__":
    main()
