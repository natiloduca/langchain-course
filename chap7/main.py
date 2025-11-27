import os

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    print("Starting data retrieval with classic RetrievalQA chain...")

    # --- 1. Define the SAME Embeddings Model used for ingestion ---
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- 2. Define the LLM used for generating the final answer ---
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
    )

    query = "what is Pinecone in machine learning?"
    # --- 3. Connect to the Pinecone Vector Store and get the retriever ---
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings_model
    )
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrieval_chain.invoke(input={"input": query})

    print(result)
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {question}
    Helpful Answer:"""

    custom_promt = PromptTemplate.from_template(template=template)

    rag_chain = (
        (
            {
                "context": vectorstore.as_retriever() | format_docs,
                "question": RunnablePassthrough(),
            }
        )
        | custom_promt
        | llm
    )
    res = rag_chain.invoke(query)
    print(res)
