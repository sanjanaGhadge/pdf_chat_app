# 📄 Chat with Multiple PDFs using Gemini 2.0 Flash 💁

Interact with your PDF documents using AI! This Streamlit app allows you to **upload multiple PDFs**, process their content into searchable chunks, and **ask questions** that are answered using Google’s Gemini 2.0 AI.

---

## 🌐 Live Demo
Check out the live app here: [**Open App**](https://sanjanaghadge-pdf-chat-app-app-wny1th.streamlit.app/)

---

## 🚀 Features
- Upload **multiple PDFs** at once  
- Extract text from PDFs (including tables)  
- Split text into chunks for efficient searching  
- Build a **FAISS vector store** for semantic search  
- Ask natural language questions and get AI-generated answers  
- Clear, concise, and context-specific responses  
- Secure API key handling using Streamlit Secrets  

---

## 🛠️ Technologies Used
- **Python 3.13**  
- **Streamlit** for interactive web UI  
- **PyPDF2** for PDF parsing  
- **LangChain** for chaining AI models  
- **FAISS** for vector-based search  
- **Google Gemini 2.0 Flash** as the LLM  
- **dotenv** for local development secrets  

---

## ⚡ How to Use
1. Clone the repo:
```bash
git clone https://github.com/sanjanaGhadge/pdf_chat_app.git
cd pdf_chat_app
