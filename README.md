# 🏥 Intel-Unnati-Industrial-Training_PS-4 [EduRag]

Companies have thousands of PDF documents like reports, manuals, and policies. Searching inside these documents is hard because the information is unstructured. Our task is to build a tool that converts PDFs into a structured format so that information can be easily searched and retrieved.

## 🧠 Overview

This project is a document-aware intelligent assistant that answers user queries strictly using information extracted from uploaded PDFs. It implements a Retrieval-Augmented Generation (RAG) pipeline combining vector-based document retrieval with large language models to generate accurate, context-grounded responses. The system supports user authentication and controlled access, ensuring that queries are answered only from authorized documents while preventing hallucinations and external knowledge leakage. The project is designed for secure and scalable document question-answering applications.

---

![Application Flow](./assets/applicationFlow.png)

![RAG Workflow Flow](./assets/applicationFlow.png)

![Core Modules](./assets/coreModules.png)

[📄 View Full Project Report (PDF)](./assets/projectReport.pdf)

[📄 View Full Project  (PPT)](./assets/projectReport.ppt)


---

## ⚙️ Tech Stack

- *Backend:* FastAPI (modular)
- *Database:* MongoDB Atlas (for users)
- *Vector DB:* Pinecone (RAG context)
- *LLM:* Groq API using LLaMA-3
- *Embeddings:* Google Generative AI Embeddings
- *Authentication:* HTTP Basic Auth + bcrypt
- *Frontend (Optional):* Streamlit

---

## 🧩 Core Modules

| Module      | Responsibility                                              |
| ----------- | ----------------------------------------------------------- |
| auth/     | Handles authentication (signup, login), hashing with bcrypt |
| chat/     | Manages chat routes and query answering logic using RAG     |
| vectordb/ | Document loading, chunking, and Pinecone indexing           |
| database/ | MongoDB setup and user operations                           |
| main.py   | Entry point for FastAPI app with route inclusion            |

---

## 📡 API Endpoints

| Method | Route          | Description                         |
| ------ | -------------- | ----------------------------------- |
| POST   | /signup      | Register new users                  |
| GET    | /login       | Login with HTTP Basic Auth          |
| POST   | /upload_docs | Admin-only endpoint to upload files |
| POST   | /chat        | Role-sensitive chatbot Q\&A         |

---

## 🚀 Getting Started

1. Clone the repo:

   bash
   git clone https://github.com/yourusername/rbac-medicalAssistant.git
   cd rbac-medicalAssistant
   

2. Create a .env file:

   env
   MONGO_URI=your_mongo_uri
   DB_NAME=your_db_name
   PINECONE_API_KEY=your_pinecone_key
   GOOGLE_API_KEY=your_google_api_key
   GROQ_API_KEY=your_groq_key
   

3. Create venv:

   bash
   uv venv
   .venv/Scripts/activate
   

4. Install dependencies:

   bash
   uv pip install -r requirements.txt
   

5. Run the app:

   bash
   uvicorn main:app --reload
   

---

## 🌱 Future Enhancements

- Add JWT-based Auth + Refresh Tokens
- Build an interactive Streamlit/React-based frontend
- Document download/preview functionality
- Audit logs for medical compliance
- Many more
- *🧍️‍ Contributions are welcome! Feel free to fork and submit PRs.*

---

## 👥 Contributors and Responsibilities

### Supratim Nag (Project Lead)
- GitHub Link = 
- Overall system design and architecture  
- RAG-based document retrieval pipeline  
- Backend development using FastAPI 

### Ismail Sk (ML / NLP Engineer)
- GitHub Link = 
- Overall system design and architecture  
- RAG-based document retrieval pipeline  
- Backend development using FastAPI  

### Sanchari Biswas (Frontend Developer)
- GitHub Link = 
- Overall system design and architecture  
- RAG-based document retrieval pipeline  
- Backend development using FastAPI  