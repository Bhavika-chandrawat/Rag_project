from backend.services.pdf_loader import load_pdf
from backend.services.text_splitter import split_documents
from backend.services.embedding_service import get_embedding_model
from backend.services.vector_store import create_vector_store, save_vector_store, load_vector_store
from backend.services.rag_service import create_rag_chain


if __name__ == "__main__":

    # 1️⃣ Load PDF
    docs = load_pdf("data/docs/sample.pdf")
    print(f"Loaded {len(docs)} pages")

    # 2️⃣ Chunk
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    # 3️⃣ Embedding model
    embedding_model = get_embedding_model()

    # 4️⃣ Create vector store
    vector_store = create_vector_store(chunks, embedding_model)
    save_vector_store(vector_store)

    print("Vector store ready.")

    # 5️⃣ Load vector store
    vector_store = load_vector_store(embedding_model)

    # 6️⃣ Create RAG chain
    rag_chain = create_rag_chain(vector_store)

    # 7️⃣ Ask question
    query = input("\nAsk a question: ")

    response = rag_chain.invoke(query)


    print("\nAnswer:")
    print(response)





# from backend.services.pdf_loader import load_pdf
# from backend.services.text_splitter import split_documents
# from backend.services.embedding_service import get_embedding_model
# from backend.services.vector_store import create_vector_store, save_vector_store
# from backend.services.vector_store import load_vector_store

# if __name__ == "__main__":
    
#     # 1️⃣ Load PDF
#     docs = load_pdf("data/docs/sample.pdf")
#     print(f"Loaded {len(docs)} pages")
    
#     # 2️⃣ Split into chunks
#     chunks = split_documents(docs)
#     print(f"Created {len(chunks)} chunks")
    
#     # 3️⃣ Load embedding model
#     embedding_model = get_embedding_model()
    
#     # 4️⃣ Create vector store
#     vector_store = create_vector_store(chunks, embedding_model)
    
#     # 5️⃣ Save vector store
#     save_vector_store(vector_store)

#     # Load stored vector DB
#     vector_store = load_vector_store(embedding_model)

#     # Test query
#     query = "What is the role of API Gateway?"
#     results = vector_store.similarity_search(query, k=3)

#     print("\nTop Retrieved Chunks:\n")
#     for doc in results:
#         print(doc.page_content)
#         print("-----")
    
#     print("Vector store created and saved successfully!")
