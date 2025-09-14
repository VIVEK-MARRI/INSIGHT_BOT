import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from google.api_core.exceptions import ResourceExhausted
from htmlTemplates import css, bot_template, user_template
import os, time

# ---------- Load Website Text ----------
def get_website_text(url):
    loader = WebBaseLoader(url)
    return loader.load()

# ---------- Text Splitting ----------
def get_text_chunks(documents):
    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=800, chunk_overlap=150, length_function=len
    )
    return splitter.split_documents(documents)

# ---------- Vector Store ----------
def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
    return Chroma.from_documents(documents=chunks, embedding=embeddings)

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
    st.set_page_config(page_title="Chat with Website", page_icon=":globe_with_meridians:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Website :globe_with_meridians:")
    user_question = st.text_input("Ask a question about the website:")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Enter Website URL")
        url = st.text_input("URL (must start with http:// or https://)")
        if st.button("Process"):
            if url and url.startswith("http"):
                with st.spinner("Processing website..."):
                    try:
                        docs = get_website_text(url)
                        chunks = get_text_chunks(docs)
                        vectorstore = get_vectorstore(chunks)
                        st.sidebar.success("✅ Processing completed!")
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.error("Please enter a valid URL.")

if __name__ == "__main__":
    main()
