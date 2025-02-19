import json
import os
import logging
from collections import defaultdict
import pickle

# if use huggingface API
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding

# if use local LLM
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# remaining llama index functions
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
import chromadb

import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    # Constants
    COLLECTION_NAME = "doc"
    DB_PATH = "chroma_db"  # Path to store the ChromaDB database
    CHAT_HISTORY_DIR = "chat_histories"
    FILE_NODES_PATH = "file_nodes.pkl"


    def __init__(self, api_token, embedding_model, llm_model, chunk_size, chunk_overlap):
        """    
        Configure llama index Settings with the provided parameters

        Args:
            api_token (str): API token for authentication with the LLM and embedding model services.
            embedding_model (str): Name or identifier of the embedding model used for vectorization.
            llm_model (str): Name or identifier of the large language model used for generating responses.
            chunk_size (int): Size of each text chunk (number of characters or tokens).
            chunk_overlap (int): Number of overlapping characters or tokens between consecutive chunks.                      

        Returns:
            None or an initialized RAG system object (depending on the implementation).
        """       
        #----------------------------------------------------------------------     
        # Global variables for the RAGSystem
        #----------------------------------------------------------------------
        # Global variables for database and query/chat engine
        self.chroma_client = None
        self.vector_store = None
        self.index = None
        self.engine = None        


        #----------------------------------------------------------------------
        # Configure llama index Settings
        #----------------------------------------------------------------------
        # Set up text splitter
        text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # if api token for huggingface is provided
        if api_token:
            # Set up Hugging Face LLM
            Settings.llm = HuggingFaceInferenceAPI(
                model_name=llm_model,
                token=api_token,
            )

            # Set up Hugging Face embedding model
            Settings.embed_model = HuggingFaceInferenceAPIEmbedding(
                model_name=embedding_model,
                token=api_token,
            )            
        else:
            # define LLM
            Settings.llm = Ollama(model=llm_model, request_timeout=500.0)  # Replace with your Ollama model

            # define embedding model (you can also use Ollama for embeddings if supported)
            Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)

        # Store settings
        Settings.chunk_size = chunk_size
        Settings.text_splitter = text_splitter    


    def initialize_rag(self, role):
        """
        Initializes a RAG database (ChromaDB) system based on role
        
        Args:
            role (str): The role or persona that the RAG system should assume when generating responses.

        Returns:
            None or an initialized RAG system object (depending on the implementation).            
        """

        #----------------------------------------------------------------------
        # Create/Load Index
        #----------------------------------------------------------------------
        # Check if the index exists (check before init the Chroma database)
        index_exists = os.path.lexists(self.DB_PATH) and os.path.isdir(self.DB_PATH) and os.listdir(self.DB_PATH)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(self.DB_PATH)  # Persistent storage

        # Create/retrieve the collection
        chroma_collection = self.chroma_client.get_or_create_collection(name=self.COLLECTION_NAME)

        # Set up the vector store
        self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # (Remark 1): Create/Load index
        if index_exists:
            # load existing index
            logger.info("Loading existing index...")
                        
            # Create a StorageContext
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store, persist_dir=self.DB_PATH)    

            # load the existing index from storage
            self.index = load_index_from_storage(storage_context)  # Now safe to call

            # load the existing file and nodes pair dictionary
            with open(self.FILE_NODES_PATH, 'rb') as f:
                self.file_nodes = pickle.load(f)            

        else:
            # create a new index
            logger.info("Creating a new index...")

            # Create a StorageContext
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)    

            # Load documents via SimpleDirectoryReader
            documents = SimpleDirectoryReader("documents").load_data()

            # (Remark 2): We avoid initializing index using the high-level function to allow customization
            # self.index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, transformations=[text_splitter])

            # parse nodes
            nodes = Settings.text_splitter.get_nodes_from_documents(documents)

            # get file and nodes
            self.file_nodes = defaultdict(list)
            for node in nodes:
                file_name = node.metadata['file_name']
                node_id = node.node_id      
                ref_doc_id = node.ref_doc_id          
                self.file_nodes[file_name].append(node_id)           

            # load the nodes as Vector Store Index
            self.index = VectorStoreIndex(nodes, storage_context=storage_context)        
        
        
        # Save the database and file_nodes
        self.save()


        #----------------------------------------------------------------------
        # Create the corresponding chat/query engine
        #----------------------------------------------------------------------                
        # set role
        self.role = role    

        # Create engine        
        if self.role.lower() == "student":
            # Create query engine
            self.engine = self.index.as_query_engine()        
        elif self.role.lower() == "teacher":
            # (Remark 5): Create chat engine
            self.engine = self.index.as_chat_engine(chat_mode="condense_question", streaming=False) 


    
    def update(self, filename):
        """Update the index with the new documents"""
        # load the new documents
        new_documents = SimpleDirectoryReader(input_files=[filename]).load_data()

        # split to nodes
        new_nodes = Settings.text_splitter.get_nodes_from_documents(new_documents)

        node_ids = []
        for node in new_nodes:
            # record node ids
            node_ids.append(node.node_id)

        # (Remark 3): update index by inserting nodes
        self.index.insert_nodes(new_nodes)
        
        # update file_nodes
        self.file_nodes[os.path.basename(filename)] = node_ids
        
        # Save the database and file_nodes
        self.save()

        # reset query/chat engine
        self.reset_engine()


    def delete(self, filename):
        """
        Delete all documents in the index that originated from the given filename.
        All nodes corresponding to the document will be deleted based on node_ids.
        """
        if self.index is None:
            logger.error("Index has not been initialized.")
            return

        # get the nodes to be deleted
        delete_nodes = self.file_nodes[os.path.basename(filename)]
        print(self.file_nodes)
        print(delete_nodes)

        # (Remark 4): delete nodes from index based on node_ids
        self.index.delete_nodes(node_ids=delete_nodes, delete_from_docstore=True)

        # delete the nodes corresponding to this file(name)
        del self.file_nodes[os.path.basename(filename)]

        # Save the database and file_nodes
        self.save()

        # reset query/chat engine
        self.reset_engine()        


    def chat(self, prompt):
        """Query the preloaded RAG index instead of rebuilding it."""
        if self.index is None:
            return "Error: Index has not been initialized. Call initialize_rag() first."

        if self.role.lower() == "student":
            # query engine will query based on prompt            
            response = self.engine.query(prompt)
            return response.response  # Ensure we return only the text response

        elif self.role.lower() == "teacher":
            # chat engine will create chat stream
            response = self.engine.stream_chat(prompt)
            return response # different compared to query engine


    def save(self):
        """Save the Chroma database and file_nodes"""
        # Save the updated file_nodes
        with open(self.FILE_NODES_PATH, 'wb') as f:
            pickle.dump(self.file_nodes, f)                          

        # Save (persist) the updated database
        self.index.storage_context.persist(self.DB_PATH)


    def reset_engine(self):
        """Reset the chat/query engine"""
        if self.role.lower() == "student":
            # query engine no need reset, because no chat history
            pass
        else:
            # chat engine reset chat history
            self.engine.reset()    

    
    def generate_chat_title(self, chat_history):
        """
        Generates a chat title based on the first item of the chat history using the LLM specified in Settings.llm.
        
        Args:
            chat_history (list): A list of chat messages. Each element can be a tuple (sender, message)
                                or a single string.
        
        Returns:
            str: A generated chat title.
        """
        if not chat_history:
            return "Untitled Chat"

        # Extract the first item. If chat_history elements are tuples/lists, use the first element (assumed to be the sender/message).
        first_item = chat_history[0][0] if isinstance(chat_history[0], (list, tuple)) else chat_history[0]

        # Create a prompt that instructs the LLM to generate a concise title
        prompt = (
        f"Generate an extremely concise chat title (max 5 words) based on the following message:\n\n"
        f"\"{first_item}\""
        )
        
        try:
            # Use the LLM from Settings to generate a title.
            # Assuming Settings.llm is callable (i.e. it has a __call__ method) that returns a string.
            prompt_template = PromptTemplate(template=prompt)
            title = Settings.llm.predict(prompt_template)
            return title.strip()
        except Exception as e:
            logger.error(f"Error generating chat title: {e}")
            return "Untitled Chat"





#----------------------------------------------------------------------
# Remark 1: Tips regarding storage_context
#----------------------------------------------------------------------
# 
# if index exists, then please include argument persist_dir for when creating StorageContext
# else, dont include the argument persist_dir, since we have just created it
# 
# https://github.com/run-llama/llama_index/issues/9110#issuecomment-1841522080


#----------------------------------------------------------------------
# Remark 2: Use low-level parsing of documents instead of high-level
#----------------------------------------------------------------------
# 
# Use low-level parsing, because we want to manually get the node id
# 
# https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/
# https://docs.llamaindex.ai/en/stable/understanding/loading/loading/#lower-level-transformation-api


#----------------------------------------------------------------------
# Remark 3: Insert nodes to index
#----------------------------------------------------------------------
# 
# We can easily insert nodes into the index
# 
# https://docs.llamaindex.ai/en/stable/api_reference/indices/#llama_index.core.indices.base.BaseIndex.insert_nodes


#----------------------------------------------------------------------
# Remark 4: Delete nodes from index
#----------------------------------------------------------------------
# 
# We can easily delete nodes into the index based on node ids
# 
# https://docs.llamaindex.ai/en/stable/api_reference/indices/#llama_index.core.indices.base.BaseIndex.delete_nodes 



#----------------------------------------------------------------------
# Remark 5: Chat engine
#----------------------------------------------------------------------
# 
# Some of the tutorials for chat engine is either too hard or too simple (lacks info)
# This reference is just perfect
# 
# https://docs.llamaindex.ai/en/v0.10.34/examples/customization/streaming/chat_engine_condense_question_stream_response/


