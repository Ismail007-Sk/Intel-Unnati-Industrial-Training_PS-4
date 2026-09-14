import os
import asyncio
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pathlib import Path

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

embed_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)
# llama-3.3-70b-versatile
llm = ChatGroq(
    temperature=0.3,
    model_name="openai/gpt-oss-120b",
    groq_api_key=GROQ_API_KEY
)





PROMPT_PATH = Path(__file__).parent / "prompt.txt"

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    prompt_text = f.read()

prompt = PromptTemplate.from_template(prompt_text)



rag_chain = prompt | llm


async def answer_query(query: str):

    embedding = await asyncio.to_thread(
        embed_model.embed_query, query
    )

    results = await asyncio.to_thread(
        index.query,
        vector=embedding,
        top_k=3,
        include_metadata=True
    )

    contexts = []
    sources = set()

    # ✅ FIX: use results.matches
    for match in results.matches:
        metadata = match.metadata
        contexts.append(metadata.get("text", ""))
        sources.add(metadata.get("source"))

    if not contexts:
        return {"answer": "No relevant information found."}

    docs_text = "\n".join(contexts)

    final_answer = await asyncio.to_thread(
        rag_chain.invoke,
        {"question": query, "context": docs_text}
    )

    return {
        "answer": final_answer.content,
        "sources": list(sources)
    }
