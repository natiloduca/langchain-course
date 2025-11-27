import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()
if __name__ == "__main__":
    print("Starting ingestion...")
    loader = TextLoader("/Users/nloduca/workspaces/langchain-course/mediumblog1.txt")
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=384, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    # embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("ingesting...")
    PineconeVectorStore.from_documents(
        texts, embeddings_model, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")
