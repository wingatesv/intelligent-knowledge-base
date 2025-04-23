import os
import pickle
import logging
import psycopg2
from psycopg2 import sql
from collections import defaultdict
from sqlalchemy.engine import make_url

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

logger = logging.getLogger("rag_system")
logging.basicConfig(level=logging.INFO)

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
        Configure LlamaIndex but defer index build until initialize_rag().
        """
        self.connection_string = connection_string
        self.db_name = db_name
        self.table_name = table_name
        self.pickle_file_path = pickle_file_path
        self.knowledge_database_dir = None  # set in initialize_rag

        # LlamaIndex setup
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        Settings.text_splitter = splitter
        system_prompt = (
            "You are a specialized PLM Knowledge Agent. Answer concisely with references."
        )
        Settings.llm = Ollama(model=llm_model, request_timeout=500.0, system_prompt=system_prompt)
        Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        self.embed_dim = Settings.embed_model._model.get_sentence_embedding_dimension()
    

        # placeholders
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
        """Create the target DB if it doesn't exist."""
        params = self._get_admin_conn_params()
        conn = psycopg2.connect(**params)
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s;",
                    (self.db_name,)
                )
                if cur.fetchone():
                    logger.info("Database '%s' exists.", self.db_name)
                else:
                    cur.execute(
                        sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self.db_name))
                    )
                    logger.info("Created database '%s'.", self.db_name)
        finally:
            conn.close()

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

    def _ensure_table_and_column(self):
        """
        Ensure the Postgres table exists with both:
          - id TEXT PRIMARY KEY
          - embedding VECTOR(embed_dim)
          - file_name TEXT NOT NULL
        """
        conn = self._connect_target_db()
        try:
            with conn:
                with conn.cursor() as cur:
                    # create table if missing (vector type requires pgvector)
                    cur.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS {self.table_name} (
                            id TEXT PRIMARY KEY,
                            embedding VECTOR({self.embed_dim}), 
                            file_name TEXT NOT NULL
                        );
                        """
                    )
                    # ensure file_name column exists (in case table was auto-created earlier)
                    cur.execute(
                        sql.SQL("ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS file_name TEXT NOT NULL DEFAULT ''")
                        .format(tbl=sql.Identifier(self.table_name))
                    )
        finally:
            conn.close()

    def initialize_rag(self, knowledge_database_dir: str):
        """
        Build or load the RAG index, and tag each vector row
        with its source filename.
        """
        # persist for delete_file()
        self.knowledge_database_dir = knowledge_database_dir

        # 1) Ensure DB exists
        self.create_database()

        # 2) Ensure our custom table/schema
        self._ensure_table_and_column()

        # 3) Instantiate vector store
        url = make_url(self.connection_string)
        self.vector_store = PGVectorStore.from_params(
            database=self.db_name,
            host=url.host,
            port=url.port,
            user=url.username,
            password=url.password,
            table_name=self.table_name,
            embed_dim=self.embed_dim,
        )

        # 3) Determine if the vector table already has data
        conn = self._connect_target_db()
        try:
            with conn:
                with conn.cursor() as cur:
                    # a) Check if the table exists in public schema
                    cur.execute(
                        """
                        SELECT EXISTS (
                          SELECT 1 FROM information_schema.tables
                         WHERE table_schema = 'public'
                           AND table_name   = %s
                        );
                        """,
                        (self.table_name,),
                    )
                    table_exists = cur.fetchone()[0]  # TRUE if table is present

                    # b) If it exists, count the rows
                    if table_exists:
                        cur.execute(
                            sql.SQL("SELECT COUNT(*) FROM {tbl}")
                            .format(tbl=sql.Identifier(self.table_name))
                        )
                        count = cur.fetchone()[0]
                    else:
                        count = 0
        finally:
            conn.close()

        # 4) Load or build index based on row count
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        if count > 0:
            logger.info("Loading existing index (%d vectors).", count)
            self.index = load_index_from_storage(storage_context)
            try:
                with open(self.pickle_file_path, "rb") as f:
                    self.file_nodes = pickle.load(f)
            except FileNotFoundError:
                self.file_nodes = {}
        else:
            logger.info("Building new index from '%s'.", knowledge_database_dir)
            docs = SimpleDirectoryReader(knowledge_database_dir).load_data()
            nodes = Settings.text_splitter.get_nodes_from_documents(docs)

            # Map filenames → node IDs
            self.file_nodes = defaultdict(list)
            for n in nodes:
                fname = os.path.basename(n.metadata.get("file_name", n.node_id))
                self.file_nodes[fname].append(n.node_id)

            # Build and persist the index
            self.index = VectorStoreIndex(nodes, storage_context=storage_context)
            with open(self.pickle_file_path, "wb") as f:
                pickle.dump(self.file_nodes, f)

            # Tag each row in Postgres with its file_name
            conn = self._connect_target_db()
            try:
                with conn:
                    with conn.cursor() as cur:
                        for fname, ids in self.file_nodes.items():
                            cur.execute(
                                sql.SQL("UPDATE {tbl} SET file_name = %s WHERE id = ANY(%s)")
                                   .format(tbl=sql.Identifier(self.table_name)),
                                (fname, ids),
                            )
            finally:
                conn.close()

        # 5) Create the query engine
        self.engine = self.index.as_query_engine()
        return self.engine

    def update(self, filename: str):
        """Ingest a new document, tag its vectors, and refresh the engine."""
        if self.index is None:
            raise RuntimeError("Call initialize_rag() first.")

        # split & insert nodes
        docs = SimpleDirectoryReader(input_files=[filename]).load_data()
        new_nodes = Settings.text_splitter.get_nodes_from_documents(docs)
        self.index.insert_nodes(new_nodes)

        # record node_ids
        ids = [n.node_id for n in new_nodes]
        key = os.path.basename(filename)
        self.file_nodes[key] = ids

        # tag file_name in Postgres
        conn = self._connect_target_db()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("UPDATE {tbl} SET file_name = %s WHERE id = ANY(%s)")
                        .format(tbl=sql.Identifier(self.table_name)),
                        (key, ids)
                    )
        finally:
            conn.close()

        # persist mapping & refresh engine
        with open(self.pickle_file_path, "wb") as f:
            pickle.dump(self.file_nodes, f)
        self.engine = self.index.as_query_engine()

    def delete(self, filename: str):
        """
        Delete *all* embeddings for this file in one SQL call,
        purge in-memory index & mapping, then rebuild.
        """
        key = os.path.basename(filename)

        # 1) Remove rows from Postgres
        conn = self._connect_target_db()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("DELETE FROM {tbl} WHERE file_name = %s")
                        .format(tbl=sql.Identifier(self.table_name)),
                        (key,)
                    )
            logger.info("Deleted vectors for '%s' from Postgres.", key)
        finally:
            conn.close()

        # 2) Wipe in-memory index & mapping
        self.file_nodes.pop(key, None)
        with open(self.pickle_file_path, "wb") as f:
            pickle.dump(self.file_nodes, f)
        self.vector_store = None
        self.index = None
        self.engine = None

        # 3) Rebuild from whatever files remain
        return self.initialize_rag(self.knowledge_database_dir)

    def chat(self, prompt: str) -> str:
        """Run a RAG query."""
        if self.engine is None:
            raise RuntimeError("Call initialize_rag() first.")
        resp = self.engine.query(prompt)
        return getattr(resp, "response", str(resp))

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
