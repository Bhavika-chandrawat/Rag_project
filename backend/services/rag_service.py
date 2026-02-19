from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def create_rag_chain(vector_store):

    retriever = vector_store.as_retriever()

    llm = ChatOllama(
        model="mistral",
        temperature=0
)


    prompt = ChatPromptTemplate.from_template(
        """
        Answer strictly using the context below.
        If answer is not found, say "Not found in document."

        Context:
        {context}

        Question:
        {question}
        """
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
