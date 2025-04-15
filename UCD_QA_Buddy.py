
import streamlit as st
import os
from io import BytesIO
from PIL import Image
import tempfile
import base64
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# ---- STYLING AND HTML FOR THE CHAT ----
st.set_page_config(page_title="UCD Q/A buddy", page_icon=":robot_face:")

CSS = """
<style>
.chat-message {
    padding: 1.5rem; 
    border-radius: 0.5rem; 
    margin-bottom: 1rem; 
    display: flex;
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
}
.chat-message .avatar {
  width: 20%;
  text-align: center;
}
.chat-message .avatar img {
  max-width: 60px;
  max-height: 60px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
"""

logo_path = r"C:\Users\Sagar\Tech_Consulting_GenAI\logo.jpg"

with open(logo_path, "rb") as f:
    logo_data = f.read()

# Base64-encode the binary data
encoded_logo = base64.b64encode(logo_data).decode("utf-8")

bot_template = f"""
<div class="chat-message bot">
    <div class="avatar">
        <!-- Embed as data URI -->
        <img src="data:image/jpeg;base64,{encoded_logo}" />
    </div>
    <div class="message">{{{{MSG}}}}</div>
</div>
"""

user_template = f"""
<div class="chat-message user">
    <div class="avatar">
        <img src="data:image/jpeg;base64,{encoded_logo}" />
    </div>
    <div class="message">{{{{MSG}}}}</div>
</div>
"""



st.markdown(CSS, unsafe_allow_html=True)

# -----------------------------
# Helper Functions
# -----------------------------
def add_logo(logo_file, width=80, height=40):
    """Return a resized PIL image of the provided logo file path."""
    img = Image.open(logo_file)
    return img.resize((width, height))

def load_pdfs_as_documents(uploaded_files):
    all_docs = []
    for uploaded_file in uploaded_files:
        # Write in-memory file to a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            file_bytes = uploaded_file.read()
            tmp.write(file_bytes)
            tmp.flush()  # make sure all data is written

            # Now pass the temp file path to PyPDFLoader
            loader = PyPDFLoader(tmp.name)
            docs = loader.load()

        all_docs.extend(docs)
    return all_docs

def split_into_chunks(documents, chunk_size=1000, chunk_overlap=100):
    """Split documents into smaller chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

# -----------------------------
# Main App
# -----------------------------
def main():
    # Title / Header
    st.title("UCD Q/A buddy")

    # Sidebar for PDF Upload
    st.sidebar.header("Upload your PDF(s) below")
    uploaded_pdfs = st.sidebar.file_uploader(
        "Select one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Once user clicks "Process"...
    if st.sidebar.button("Process"):
        if uploaded_pdfs:
            with st.spinner("Loading and splitting PDFs..."):
                # 1) Load the uploaded PDFs as documents
                docs = load_pdfs_as_documents(uploaded_pdfs)

                # 2) Split into chunks
                split_docs = split_into_chunks(docs, chunk_size=1000, chunk_overlap=100)

                # 3) Create in-memory Chroma vectorstore
                embeddings = OpenAIEmbeddings()  # Make sure your OPENAI_API_KEY is set in your environment
                vectorstore = Chroma.from_documents(split_docs, embedding=embeddings)

                # 4) Create a RetrievalQA chain
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
                st.session_state["qa_chain"] = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
                )
                st.success("PDFs processed! You can now ask questions below.")
        else:
            st.warning("Please upload at least one PDF before clicking 'Process'.")

    # User's question input
    user_question = st.text_input("Ask a question about your documents:")
    if user_question and "qa_chain" in st.session_state:
        response = st.session_state["qa_chain"].run(user_question)
        # Update chat history
        st.session_state["chat_history"].append(("user", user_question))
        st.session_state["chat_history"].append(("bot", response))

    # Display Chat History
    for role, msg in st.session_state["chat_history"]:
        if role == "user":
            st.markdown(user_template.replace("{{MSG}}", msg), unsafe_allow_html=True)
        else:  # bot
            st.markdown(bot_template.replace("{{MSG}}", msg), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
