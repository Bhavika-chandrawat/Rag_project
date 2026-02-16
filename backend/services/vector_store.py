from langchain_community.vectorstores import FAISS

def create_vector_store(chunks, embedding_model):
    vector_store = FAISS.from_documents(chunks, embedding_model)
    return vector_store

def save_vector_store(vector_store, path="data/vector_store"):
    vector_store.save_local(path)

def load_vector_store(embedding_model, path="data/vector_store"):
    return FAISS.load_local(
        path,
        embedding_model,
        allow_dangerous_deserialization=True
    )
