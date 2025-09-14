import streamlit as st
from htmlTemplates import css, bot_template, user_template
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import whisper
from pytube import YouTube
from google.api_core.exceptions import ResourceExhausted
import os, datetime, time

# ---------- Extract Text from YouTube Video ----------
def get_video_text(url):
    model = whisper.load_model("base")
    try:
        yt_video = YouTube(url)
        stream = yt_video.streams.filter(only_audio=True).first()
        audio_file = "video_audio.mp4"
        stream.download(filename=audio_file)
    except Exception as e:
        st.error(f"Error downloading video: {e}")
        return ""

    start_time = datetime.datetime.now()
    try:
        output = model.transcribe(audio_file, fp16=False)
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return ""
    finally:
        if os.path.exists(audio_file):
            os.remove(audio_file)

    end_time = datetime.datetime.now()
    st.info(f"Transcription completed in {end_time - start_time}")
    return output.get('text', "")

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
        model="gemini-1.5-flash",  # Flash model reduces quota issues
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
            if i % 2 == 0:  # user
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:  # bot
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    except ResourceExhausted:
        st.warning("⚠️ Gemini quota exceeded. Retrying in 10 seconds...")
        time.sleep(10)
        return handle_user_input(user_question)
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ---------- Streamlit App ----------
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with YouTube Video", page_icon=":clapper:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with YouTube Video :clapper:")
    user_question = st.text_input("Ask a question about your video:")
    if user_question and st.session_state.conversation:
        handle_user_input(user_question)
    elif user_question:
        st.warning("Please process a video first.")

    with st.sidebar:
        st.subheader("YouTube Video URL")
        url = st.text_input("Enter the video URL:")

        if st.button("Process"):
            if url:
                if not url.startswith("http"):
                    st.error("Invalid URL")
                    return
                with st.spinner("Downloading and transcribing video..."):
                    try:
                        raw_text = get_video_text(url)
                        if raw_text.strip() == "":
                            st.error("No text extracted from video.")
                            return
                        chunks = get_text_chunks(raw_text)
                        vectorstore = get_vectorstore(chunks)
                        st.sidebar.success("✅ Processing completed!")
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.error("Please enter a YouTube video URL.")

if __name__ == "__main__":
    main()
