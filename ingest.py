from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import tiktoken
import os

load_dotenv(override=True)

model_name = os.getenv("ENCODING_MODEL", "gpt-4.1-nano")

def load_document(folder_name = "docs", encoding="utf-8"):
    """
    Load a document from the specified file path.

    Args:
        file_path (str): The path to the document file.
    Returns:
        str: The content of the document.
    """
    # Implementation for loading document
    docs = []
    for path in Path(folder_name).rglob("*.pdf"):
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():  # Check if text is not None and not empty
                docs.append(Document(page_content=text, metadata={"source": str(path.name), "page": i + 1}))

    return docs

def number_of_tokens(text):
    """
    Calculate the number of tokens in a given text using the specified model.

    Args:
        text (str): The input text.
        model_name (str): The name of the model to use for tokenization.
    """

    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)


def split_document(document, chunk_size=800, chunk_overlap=100):
    """
    Split a document into smaller chunks.

    Args:
        document (Document): The document to be split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of overlapping characters between chunks.
    Returns:
        list: A list of Document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(document)

    return chunks


def embed_documents(documents, embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"), db_name = os.getenv("CHROMA_DB_NAME", "chroma_db")):
    """
    Embed a list of documents using the specified embedding model.

    Args:
        documents (list): A list of Document objects to be embedded.
        embedding_model (str): The embedding model to use for generating embeddings.
        db_name (str): The name of the Chroma database to use.
    Returns:
        list: A list of embedded Document objects.
    """
    # embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    embeddings = OpenAIEmbeddings(model=embedding_model)
    if os.path.exists(db_name):
        Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(documents, embedding=embeddings, persist_directory=db_name)

    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    print(f"Embedded {len(documents)} documents and stored in Chroma database '{db_name}'.")

if __name__ == "__main__":
    documents = load_document()
    print(f"Loaded {len(documents)} pages")

    chunks = split_document(documents)
    print(f"Split into {len(chunks)} chunks")

    embed_documents(chunks)