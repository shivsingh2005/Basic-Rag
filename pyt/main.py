from fastapi import FastAPI
from pydantic import BaseModel

from dotenv import load_dotenv
import os

# ---------------- LangChain Imports (Latest 2025) ----------------

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# ----------------------------------------------------------------

# Load .env file
load_dotenv()

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Embeddings model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# 1️⃣ Load text file
def load_docs():
    with open("data/notes.txt", "r", encoding="utf-8") as f:
        return f.read()

# 2️⃣ Split into chunks
def split_docs(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_text(text)

# 3️⃣ Create FAISS vector DB
def create_db():
    raw_text = load_docs()
    chunks = split_docs(raw_text)
    db = FAISS.from_texts(chunks, embedding=embeddings)
    return db

# Create DB once
db = create_db()

# Create retriever
retriever = db.as_retriever()

# FastAPI app
app = FastAPI()

class Query(BaseModel):
    question: str

# Prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Use the retrieved context to answer.\n"
     "If the answer isn't found in the context, reply: 'I don't know from the context.'"),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# 4️⃣ RAG pipeline
def rag_answer(question):
    # New retriever method (2024+)
    docs = retriever.invoke(question)

    context = "\n".join([d.page_content for d in docs])

    prompt = prompt_template.format(
        question=question,
        context=context
    )

    # LLM response
    response = llm.invoke(prompt)

    return response.content

# 5️⃣ API endpoint
@app.post("/ask")
async def ask_rag(query: Query):
    try:
        answer = rag_answer(query.question)
        return {
            "question": query.question,
            "answer": answer
        }
    except Exception as e:
        return {"error": str(e)}
