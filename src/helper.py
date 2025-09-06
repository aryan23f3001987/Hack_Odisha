from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
import os


# --------------------------
# 1. Extract Data From the PDF File
# --------------------------
def load_pdf_file(data: str) -> List[Document]:
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


# --------------------------
# 2. Keep only minimal metadata
# --------------------------
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs


# --------------------------
# 3. Split into Text Chunks
# --------------------------
def text_split(extracted_data: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# --------------------------
# 4. OpenAI Embeddings (matches store_index.py → 1536-dim)
# --------------------------
def download_openai_embeddings():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # 1536 dimensions
        api_key=os.getenv("OPENAI_API_KEY")
    )
    return embeddings