import streamlit as st # type: ignore
from PyPDF2 import PdfReader # type: ignore
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from langchain.chains.question_answering import load_qa_chain # type: ignore
from langchain.prompts import PromptTemplate # type: ignore
from dotenv import load_dotenv # type: ignore

# ---------------------------
# Load API key from .env
# ---------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---------------------------
# Extract text from PDFs
# ---------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf.seek(0)  # reset pointer
            pdf_reader = PdfReader(BytesIO(pdf.read()))
            for page in pdf_reader.pages:
                if page.extract_text():
                    text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
            continue
    return text

# ---------------------------
# Split text into chunks
# ---------------------------
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

# ---------------------------
# Create vector store
# ---------------------------
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# ---------------------------
# Create conversational chain with Gemini 2.0 Flash
# ---------------------------
def get_conversational_chain():
    prompt_template = """
   You answer ONLY using the information in the Context. If the Context does not contain the answer, reply exactly:
"answer is not available in the context".

Rules:
- Be concise and specific. Prefer bullet points for multi-part answers.
- Copy numbers, units, dates, and names exactly as written in the Context.
- If a table is relevant, reproduce only the needed rows/columns in Markdown.
- Do not invent facts or rely on prior knowledge.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",   # ✅ Updated model
        temperature=0.3,
        google_api_key=GEMINI_API_KEY
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# ---------------------------
# Handle user query
# ---------------------------
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("Reply:", response["output_text"])

# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    st.set_page_config(page_title="Chat with Multiple PDFs")
    st.header("📄 Chat with Multiple PDFs using Gemini 2.0 Flash 💁")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📌 Menu")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True,
            type=["pdf"]
        )
        if st.button("Submit & Process") and pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if raw_text.strip():
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ Processing Complete! You can now ask questions.")
                else:
                    st.warning("⚠️ No extractable text found in the uploaded PDFs.")

# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    main()
