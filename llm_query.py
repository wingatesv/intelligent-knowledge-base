import logging
import pickle
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy.engine import make_url
from collections import defaultdict

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores import PGVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(
        self,
        embedding_model: str,
        llm_model: str,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
    ):
        """
        Sets up LlamaIndex settings (LLM, embedding, chunking) but does NOT build the index
        until initialize_rag() is called.
        """
        # PostgreSQL connection (must include a default DB, e.g. '/postgres')
        self.connection_string = "postgresql://postgres:password@localhost:5432/postgres"
        self.db_name = "vector_db"
        self.table_name = "knowledge_base"
        self.pickle_file_path = "file_nodes.pkl"

        # ─── LlamaIndex Settings ──────────────────────────────────────────────
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        Settings.text_splitter = splitter
        Settings.llm = Ollama(model=llm_model, request_timeout=500.0)
        Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)

        # Placeholders
        self.vector_store = None
        self.index = None
        self.engine = None
        self.file_nodes = {}

    def _get_admin_conn_params(self) -> dict:
        """Parse connection_string and return params for the 'postgres' admin DB."""
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
        Ensures `self.db_name` exists by connecting to the admin DB (usually 'postgres')
        and running CREATE DATABASE only if needed.
        """
        params = self._get_admin_conn_params()
        with psycopg2.connect(**params) as conn:
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            with conn.cursor() as c:
                c.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s;",
                    (self.db_name,)
                )
                if c.fetchone():
                    logger.info("Database '%s' already exists; skipping create.", self.db_name)
                else:
                    c.execute(f"CREATE DATABASE {self.db_name}")
                    logger.info("Created database '%s'.", self.db_name)

    def _connect_target_db(self):
        """Open a connection directly to self.db_name."""
        url = make_url(self.connection_string)
        return psycopg2.connect(
            host=url.host,
            port=url.port,
            user=url.username,
            password=url.password,
            dbname=self.db_name
        )

    def _has_vectors(self) -> bool:
        """
        Returns True if any row exists in the vector table.
        This is a single, fast check with LIMIT 1.
        """
        with self._connect_target_db() as conn:
            with conn.cursor() as c:
                c.execute(f"SELECT EXISTS (SELECT 1 FROM {self.table_name} LIMIT 1);")
                return c.fetchone()[0]

    def initialize_rag(self):
        """
        1) Ensures the database exists
        2) Instantiates PGVectorStore (it auto-creates table+index if missing)
        3) Checks for existing vectors → load_index or build a new one
        4) Creates a query engine
        """
        # 1) Create DB if needed
        self.create_database()

        # 2) Spin up PGVectorStore (idempotent DDL)
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

        # 3) Wrap in StorageContext
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # 4) Load or build the index
        if self._has_vectors():
            logger.info("Loading existing RAG index from Postgres…")
            self.index = load_index_from_storage(storage_context)

            # — if you need file_nodes, load it from disk or another table —
            try:
                with open(self.pickle_file_path, "rb") as f:
                    self.file_nodes = pickle.load(f)
                logger.info("Loaded file_nodes mapping.")
            except FileNotFoundError:
                logger.warning("file_nodes.pkl not found; file_nodes mapping is empty.")
        else:
            logger.info("Building new RAG index…")
            # Load documents
            documents = SimpleDirectoryReader("documents").load_data()
            # Chunk → nodes
            nodes = Settings.text_splitter.get_nodes_from_documents(documents)
            # Build file_nodes map
            self.file_nodes = defaultdict(list)
            for node in nodes:
                fname = node.metadata.get("file_name", "unknown")
                self.file_nodes[fname].append(node.node_id)
            # Index from nodes
            self.index = VectorStoreIndex(nodes, storage_context=storage_context)
            # Persist the mapping for next time
            with open(self.pickle_file_path, "wb") as f:
                pickle.dump(self.file_nodes, f)
            logger.info("Persisted file_nodes mapping to disk.")

        # 5) Create the query engine
        self.engine = self.index.as_query_engine()
        return self.engine
