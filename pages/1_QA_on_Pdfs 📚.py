import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from google.api_core.exceptions import ResourceExhausted
from htmlTemplates import css, bot_template, user_template
import os, time

# ---------- PDF Text Extraction ----------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# ---------- Text Splitting ----------
def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=800, chunk_overlap=150, length_function=len
    )
    return splitter.split_text(text)

# ---------- Vector Store ----------
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
    return Chroma.from_texts(texts=text_chunks, embedding=embeddings)

# ---------- Conversational Chain ----------
def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # switched from pro → flash
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
def handle_userinput(user_question):
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
        return handle_userinput(user_question)
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# ---------- Streamlit App ----------
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload PDFs and click 'Process'", accept_multiple_files=True, type="pdf"
        )
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        chunks = get_text_chunks(raw_text)
                        vectorstore = get_vectorstore(chunks)
                        st.sidebar.success("✅ Processing completed!")
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.error("Please upload at least one PDF.")

if __name__ == "__main__":
    main()
