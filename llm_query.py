# import nest_asyncio
# nest_asyncio.apply()
import json
import os
import logging

# # if use huggingface API
# from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
# from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding

# if use local LLM
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
import chromadb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
COLLECTION_NAME = "doc"
DB_PATH = "chroma_db"  # Path to store the ChromaDB database
CHAT_HISTORY_DIR = "chat_histories"
# Global variables to store initialized models
chroma_client = None
vector_store = None
index = None

def initialize_rag(api_token, embedding_model, llm_model, chunk_size, chunk_overlap, role):
    """Initialize and update the RAG database (ChromaDB) with new documents."""
    global chroma_client, vector_store, index

    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(DB_PATH)  # Persistent storage

    # Create/retrieve the collection
    chroma_collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    # Set up the vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Create a StorageContext
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Set up text splitter
    text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # # Set up Hugging Face LLM
    # Settings.llm = HuggingFaceInferenceAPI(
    #     model_name=llm_model,
    #     token=api_token,
    # )
    # define LLM
    Settings.llm = Ollama(model=llm_model, request_timeout=500.0) # Replace with your Ollama model

    # # Set up Hugging Face embedding model
    # Settings.embed_model = HuggingFaceInferenceAPIEmbedding(
    #     model_name=embedding_model,
    #     token=api_token,
    # )
    # define embedding model (you can also use Ollama for embeddings if supported)
    Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)

    # Store settings
    Settings.chunk_size = chunk_size
    Settings.text_splitter = text_splitter

    # Check if the index exists
    index_exists = os.path.lexists(DB_PATH) and os.path.isdir(DB_PATH) and os.listdir(DB_PATH)

    # Load existing index if it exists
    if index_exists:
        try:
            logger.info("Loading existing index...")
            index = load_index_from_storage(storage_context)  # Now safe to call
        except ValueError:
            logger.warning("Index not found in storage, creating a new one...")
            index_exists = False  # Force re-creation
    else:
        logger.info("Creating a new index...")

   
    # Load documents differently based on the role
    if role == 'Teacher':
        # Load documents from the documents directory
        docs_from_files = SimpleDirectoryReader("documents").load_data()
        logger.info("Loaded documents from 'documents' directory.")

        # Load chat histories from the chat_histories directory
        chat_docs = []
        if os.path.exists(CHAT_HISTORY_DIR):
            for file_name in os.listdir(CHAT_HISTORY_DIR):
                if file_name.endswith(".json"):
                    file_path = os.path.join(CHAT_HISTORY_DIR, file_name)
                    try:
                        with open(file_path, "r", encoding="utf-8") as file:
                            chat_data = json.load(file)
                        # Combine chat messages into a single text blob
                        chat_text = "\n".join([f"{entry['sender']}: {entry['message']}" for entry in chat_data])
                        # Create a Document with a unique id (using the file name)
                        doc = Document(text=chat_text, doc_id=f"chat_{file_name}")
                        chat_docs.append(doc)
                    except Exception as e:
                        logger.error(f"Error reading {file_name}: {e}")
        else:
            logger.warning(f"Chat histories directory '{CHAT_HISTORY_DIR}' does not exist.")

        # Combine both sets of documents
        documents = docs_from_files + chat_docs
        logger.info("Combined documents from both 'documents' and 'chat_histories' directories for RAG.")
    else:
        # For roles other than Teacher, use only the documents folder
        documents = SimpleDirectoryReader("documents").load_data()

    stored_docs = set(chroma_collection.get()["ids"])  # Get existing document IDs in ChromaDB
    new_docs = [doc for doc in documents if doc.doc_id not in stored_docs]

    if new_docs:
        logger.info(f"Adding {len(new_docs)} new document(s) to the database...")
        index = VectorStoreIndex.from_documents(new_docs, storage_context=storage_context, transformations=[text_splitter])
        index.storage_context.persist(DB_PATH)  # Persist the updated database
        logger.info("Index updated with new documents.")
    else:
        logger.info("No new documents detected. Skipping update.")

def hugging_face_query(prompt, role):
    """Query the preloaded RAG index instead of rebuilding it."""
    global index
    if index is None:
        return "Error: Index has not been initialized. Call initialize_rag() first."
    role = role
    query_engine = index.as_query_engine()
    response = query_engine.query(prompt)
    return response.response  # Ensure we return only the text response

if __name__ == "__main__":
    # Test prompt
    prompt = "What is product marketing mix?"
    hugging_face_query(prompt)
