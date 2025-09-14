import streamlit as st
from htmlTemplates import css, bot_template, user_template
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from PIL import Image
import pytesseract
from google.api_core.exceptions import ResourceExhausted
import os, time

# ---------- Extract Text from Images ----------
def get_image_text(image_docs):
    text = "Text extracted from images:\n"
    for image in image_docs:
        try:
            img = Image.open(image)
            data = pytesseract.image_to_string(img).strip()
            text += "\n" + (data if data else "[No text found in image]")
        except Exception as e:
            text += f"\n[Error reading image: {e}]"
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
        model="gemini-1.5-flash",  # switched to flash for better quota
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
    st.set_page_config(page_title="Chat with Images", page_icon=":camera:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Images :camera:")
    user_question = st.text_input("Ask a question about your images:")
    if user_question and st.session_state.conversation:
        handle_user_input(user_question)
    elif user_question:
        st.warning("Please upload and process images first.")

    with st.sidebar:
        st.subheader("Upload Your Images")
        image_docs = st.file_uploader(
            "Select images and click 'Process'",
            accept_multiple_files=True,
            type=["png", "jpg", "jpeg"]
        )

        if st.button("Process"):
            if image_docs:
                with st.spinner("Processing images..."):
                    try:
                        raw_text = get_image_text(image_docs)
                        chunks = get_text_chunks(raw_text)
                        vectorstore = get_vectorstore(chunks)
                        st.sidebar.success("✅ Processing completed!")
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.error("Please upload at least one image.")

if __name__ == "__main__":
    main()
