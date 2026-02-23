from fastapi import FastAPI, UploadFile, File
from backend.services.pdf_loader import load_pdf
from backend.services.text_splitter import split_documents
from backend.services.embedding_service import create_embeddings
from backend.services.vector_store import create_vector_store
from backend.services.rag_service import create_rag_chain
import shutil
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Technology Document RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # safer than "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store = None
rag_chain = None


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vector_store, rag_chain

    file_path = f"temp_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    documents = load_pdf(file_path)
    chunks = split_documents(documents)

    embeddings = create_embeddings()
    vector_store = create_vector_store(chunks, embeddings)
    rag_chain = create_rag_chain(vector_store)

    os.remove(file_path)

    return {"message": "Document processed successfully"}


@app.post("/ask")
async def ask_question(query: str):
    global rag_chain

    if rag_chain is None:
        return {"error": "Upload a document first"}

    response = rag_chain.invoke(query)

    if isinstance(response, dict) and "result" in response:
        return {"answer": response["result"]}

    return {"answer": response}

