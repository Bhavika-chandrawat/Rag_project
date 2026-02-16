from backend.services.pdf_loader import load_pdf
from backend.services.text_splitter import split_documents
from backend.services.embedding_service import get_embedding_model
from backend.services.vector_store import create_vector_store, save_vector_store

if __name__ == "__main__":
    
    # 1️⃣ Load PDF
    docs = load_pdf("data/docs/sample.pdf")
    print(f"Loaded {len(docs)} pages")
    
    # 2️⃣ Split into chunks
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks")
    
    # 3️⃣ Load embedding model
    embedding_model = get_embedding_model()
    
    # 4️⃣ Create vector store
    vector_store = create_vector_store(chunks, embedding_model)
    
    # 5️⃣ Save vector store
    save_vector_store(vector_store)
    
    print("Vector store created and saved successfully!")
