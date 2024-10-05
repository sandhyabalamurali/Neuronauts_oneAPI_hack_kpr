import argparse
import os
import shutil
import logging
from langchain_chroma import Chroma
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
import torch

CHROMA_PATH = "chroma"
DATA_PATH = "datasets"
BATCH_SIZE = 1000  # Define a batch size for processing

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TransformerEmbedding:
    def __init__(self, model_name='medicalai/ClinicalBERT', batch_size=64):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.batch_size = batch_size

    def embed_query(self, query):
        return self.embed_documents([query])[0]

    def embed_documents(self, documents):
        embeddings = []
        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i:i+self.batch_size]
            inputs = self.tokenizer(batch_docs, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).tolist()
            embeddings.extend(batch_embeddings)
            torch.cuda.empty_cache()  # Clear GPU memory after each batch
        return embeddings

def get_embedding_function():
    return TransformerEmbedding()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    if args.reset:
        logger.info("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store
    try:
        documents = load_documents()
        chunks = split_documents(documents)
        add_to_chroma(chunks, BATCH_SIZE)
    except Exception as e:
        logger.error(f"An error occurred in main process: {e}")

def load_documents():
    if not os.path.isdir(DATA_PATH):
        raise FileNotFoundError(f"Data directory '{DATA_PATH}' does not exist.")
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = document_loader.load()
    logger.info(f"Loaded {len(documents)} documents.")
    return documents

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(chunks)} chunks.")
    return chunks

def add_to_chroma(chunks: list[Document], batch_size=1000):
    if not os.path.isdir(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)
    
    try:
        embedding_function = get_embedding_function()
        logger.info("Embedding function created successfully.")
    except Exception as e:
        logger.error(f"Failed to create embedding function: {e}")
        return

    try:
        # Initialize Chroma with persistence enabled
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        logger.info("Chroma database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Chroma database: {e}")
        return

    # Calculate Page IDs
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Get the list of existing document IDs in the database
    existing_ids = set(db.get()["ids"])
    logger.info(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata.get("id") not in existing_ids]

    if new_chunks:
        logger.info(f"👉 Adding new documents: {len(new_chunks)}")

        # Add in batches
        for i in range(0, len(new_chunks), batch_size):
            batch_chunks = new_chunks[i:i+batch_size]
            batch_chunk_ids = [chunk.metadata.get("id") for chunk in batch_chunks]
            try:
                db.add_documents(batch_chunks, ids=batch_chunk_ids)
                logger.info(f"Batch {i // batch_size + 1}: {len(batch_chunks)} documents added.")
            except Exception as e:
                logger.error(f"Failed to add batch documents to Chroma: {e}")
    else:
        logger.info("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0 

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        chunk.metadata["id"] = chunk_id
        last_page_id = current_page_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        logger.info("Database cleared.")
    else:
        logger.info("No database found to clear.")

if __name__ == "__main__":
    main()
