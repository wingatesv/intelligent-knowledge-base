import os
import logging
import pickle
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy.engine import make_url
from collections import defaultdict

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.prompts.base import PromptTemplate

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(
        self,
        embedding_model: str,
        llm_model: str,
        connection_string: str,
        db_name: str,
        table_name: str,
        pickle_file_path: str,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
    ):
        """
        Configure LlamaIndex (LLM, embeddings, chunking) but defer index build to initialize_rag().
        """
        # Database connection defaults (must include default DB)
        self.connection_string = connection_string
        self.db_name = db_name
        self.table_name = table_name
        self.pickle_file_path = pickle_file_path

        # Domain-specific system prompt
        system_prompt = (
            "You are a specialized Knowledge Base Agent for Product Lifecycle Management (PLM). "
            "You understand PLM concepts including EBOM, MBOM, CBOM, change management, variant management, "
            "manufacturing processes, CAD data, configuration management, and PLM software like PTC Windchill and PEVO. "
            "When responding, provide concise, accurate answers with references to the knowledge base documents."
            "If you are not sure with the answer, just say that you don't know the answer."
        )

        # LlamaIndex configuration
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        Settings.text_splitter = splitter
        Settings.llm = Ollama(model=llm_model, request_timeout=500.0, system_prompt=system_prompt)
        Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)

        # Placeholders
        self.vector_store = None
        self.index = None
        self.engine = None
        self.file_nodes = {}

    def _get_admin_conn_params(self) -> dict:
        url = make_url(self.connection_string)
        return {
            "host": url.host,
            "port": url.port,
            "user": url.username,
            "password": url.password,
            "dbname": url.database or "postgres",
        }

    def create_database(self) -> None:
        """
        Ensure the target DB exists by connecting to the admin DB and creating if missing.
        """
        params = self._get_admin_conn_params()
        with psycopg2.connect(**params) as conn:
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s;", (self.db_name,)
                )
                if cur.fetchone():
                    logger.info("Database '%s' already exists.", self.db_name)
                else:
                    cur.execute(f"CREATE DATABASE {self.db_name}")
                    logger.info("Created database '%s'.", self.db_name)

    def _connect_target_db(self):
        """Connect directly to the target database."""
        url = make_url(self.connection_string)
        return psycopg2.connect(
            host=url.host,
            port=url.port,
            user=url.username,
            password=url.password,
            dbname=self.db_name,
        )

    def _has_vectors(self) -> bool:
        try:
            with self._connect_target_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT EXISTS (SELECT 1 FROM {self.table_name} LIMIT 1);")
                    return cur.fetchone()[0]
        except psycopg2.errors.UndefinedTable:
            logger.warning("Table '%s' does not exist yet.", self.table_name)
            return False

    def initialize_rag(self, knowledge_database_dir):
        """
        Build or load the RAG index:
        1) Ensure DB
        2) Instantiate PGVectorStore (auto‐DDL)
        3) Load existing index if vectors exist; else build from docs
        4) Initialize query engine
        """
        # 1) Ensure database
        self.create_database()

        # 2) Vector store with idempotent schema setup
        url = make_url(self.connection_string)
        self.vector_store = PGVectorStore.from_params(
            database=self.db_name,
            host=url.host,
            port=url.port,
            user=url.username,
            password=url.password,
            table_name=self.table_name,
            embed_dim=384,
            hnsw_kwargs={
                "hnsw_m": 16,
                "hnsw_ef_construction": 64,
                "hnsw_ef_search": 40,
                "hnsw_dist_method": "vector_cosine_ops",
            },
        )

        # 3) Storage context
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # 4) Load or build
        if self._has_vectors():
            logger.info("Loading existing index from Postgres.")
            self.index = load_index_from_storage(storage_context)
            # reload file_nodes mapping
            try:
                with open(self.pickle_file_path, "rb") as f:
                    self.file_nodes = pickle.load(f)
                logger.info("Loaded file_nodes mapping.")
            except FileNotFoundError:
                logger.warning("No persisted file_nodes; mapping is empty.")
        else:
            logger.info("Building new index from documents.")
            docs = SimpleDirectoryReader(knowledge_database_dir).load_data()
            nodes = Settings.text_splitter.get_nodes_from_documents(docs)

            # build and persist file_nodes
            self.file_nodes = defaultdict(list)
            for n in nodes:
                fname = n.metadata.get("file_name", os.path.basename(n.node_id))
                self.file_nodes[fname].append(n.node_id)
            with open(self.pickle_file_path, "wb") as f:
                pickle.dump(self.file_nodes, f)

            # index nodes
            self.index = VectorStoreIndex(nodes, storage_context=storage_context)
            logger.info("Persisted file_nodes mapping and indexed nodes.")

        # 5) Query engine
        self.engine = self.index.as_query_engine()
        return self.engine

    def update(self, filename: str):
        """Ingest a new document into the existing index."""
        if self.index is None:
            raise RuntimeError("Index uninitialized. Call initialize_rag() first.")

        # load & split
        docs = SimpleDirectoryReader(input_files=[filename]).load_data()
        new_nodes = Settings.text_splitter.get_nodes_from_documents(docs)

        # record and insert
        ids = [n.node_id for n in new_nodes]
        self.index.insert_nodes(new_nodes)
        self.file_nodes[os.path.basename(filename)] = ids

        # persist mapping
        with open(self.pickle_file_path, "wb") as f:
            pickle.dump(self.file_nodes, f)
        logger.info("Updated file_nodes for '%s'.", filename)

        # refresh engine
        self.engine = self.index.as_query_engine()

    def delete(self, filename: str):
        """Remove nodes and metadata for the specified file from the index."""
        if self.index is None:
            raise RuntimeError("Index uninitialized. Call initialize_rag() first.")

        key = os.path.basename(filename)
        ids = self.file_nodes.get(key)
        if not ids:
            logger.warning("No nodes found for '%s'.", filename)
            return

        self.index.delete_nodes(node_ids=ids, delete_from_docstore=True)
        del self.file_nodes[key]

        # persist mapping
        with open(self.pickle_file_path, "wb") as f:
            pickle.dump(self.file_nodes, f)
        logger.info("Deleted nodes for '%s' and updated mapping.", filename)

        # refresh engine
        self.engine = self.index.as_query_engine()

    def chat(self, prompt: str) -> str:
        """Query the RAG index."""
        if self.index is None:
            raise RuntimeError("Index uninitialized. Call initialize_rag() first.")
        resp = self.engine.query(prompt)
        return resp.response if hasattr(resp, "response") else str(resp)

    def generate_chat_title(self, chat_history) -> str:
        """
        Generate a concise title (<=5 words) for a chat using the LLM.
        """
        if not chat_history:
            return "Untitled Chat"
        first = chat_history[0]
        text = first[1] if isinstance(first, (list, tuple)) else first
        template = (
            f"You are a PLM Knowledge Agent. Generate a concise (max 5 words) chat title based on:\n""{text}"
        )
        try:
            pt = PromptTemplate(template=template)
            title = Settings.llm.predict(pt)
            return title.strip()
        except Exception as e:
            logger.error("Error generating title: %s", e)
            return "Untitled Chat"
