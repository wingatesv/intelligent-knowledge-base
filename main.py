#----------------------------------------------------------------------
# IMPORTANT REMARKS on gradio file explorer
#----------------------------------------------------------------------
# File explorer will not refresh automatically: https://github.com/gradio-app/gradio/issues/7788
# Hence, what we did is we purposely load the wrong file explorer first, 
# then only we reload back the correct file explorer.
# That is why you will notice that the we will load the wrong directory
# for file explorer in each function, then only we use .then() to call
# the correct file explorer using the correct directory

import json
import shutil
import logging
from pathlib import Path

import yaml
import gradio as gr

from rag_system import RAGSystem

# Configure module-level logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s")


class KnowledgeBaseAgent:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        # Model settings
        self.embedding_model = self.config.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        self.llm_model = self.config.get("LLM_MODEL", "mistral")
        self.chunk_size = self.config.get("CHUNK_SIZE", 512)
        self.chunk_overlap = self.config.get("CHUNK_OVERLAP", 20)
        self.connection_string = self.config.get("CONNECTION_STRING", "postgresql://postgres:password@localhost:5432/postgres")
        self.db_name = self.config.get("DB_NAME", "vector_db")
        self.table_name = self.config.get("TABLE_NAME", "knowledge_base")
        self.pickle_file_path = self.config.get("PICKLE_FILE_PATH", "file_nodes.pkl")

        # Folders
        self.internal_folder = Path(self.config.get("DOC_DIR", "data")).absolute()
        self.chat_history_folder = Path(self.config.get("CHAT_DIR", "chat_histories")).absolute()
        self.internal_folder.mkdir(parents=True, exist_ok=True)
        self.chat_history_folder.mkdir(parents=True, exist_ok=True)

        # State
        self.current_session = None

        # Initialize RAG system
        self.rag_system = RAGSystem(
            embedding_model=self.embedding_model,
            llm_model=self.llm_model,
            connection_string = self.connection_string,
            db_name = self.db_name,
            table_name = self.table_name,
            pickle_file_path = self.pickle_file_path,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self.rag_system.initialize_rag(self.internal_folder)

    def _load_config(self, path: str) -> dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with p.open("r") as f:
            return yaml.safe_load(f)

    def _file_explorer(self, use_chat_history: bool = False) -> gr.FileExplorer:
        root = self.chat_history_folder if use_chat_history else self.internal_folder
        return gr.FileExplorer(
            root_dir=str(root),
            file_count="multiple",
            interactive=True,
            label="Chats" if use_chat_history else "Files",
            every=1,
            ignore_glob="*.ipynb_checkpoints",
        )

    def update_internal_explorer(self):
        return self._file_explorer(use_chat_history=False)

    def update_chat_explorer(self):
        return self._file_explorer(use_chat_history=True)

    def send_query(self, user_input: str, chat_history: list):
        if not user_input.strip():
            return chat_history, ""
        chat_history.append([user_input, ""])
        try:
            response = self.rag_system.chat(user_input)
        except Exception as e:
            logger.error("RAG chat error", exc_info=True)
            response = f"Error: {e}"
        chat_history[-1][1] = str(response)
        return chat_history, ""

    def upload_files(self, files):
        if not files:
            return self.update_internal_explorer()
        for f in files:
            dest = self.internal_folder / Path(f.name).name
            shutil.copy(f.name, dest)
            logger.info(f"Uploaded {dest.name}")
            self.rag_system.update(dest)
        return self.update_internal_explorer()

    def delete_files(self, selected):
        if not selected:
            return self.update_internal_explorer()
        if isinstance(selected, str):
            selected = [selected]
        for name in selected:
            path = self.internal_folder / name
            try:
                path.unlink()
                logger.info(f"Deleted file: {name}")
                self.rag_system.delete(path)
            except FileNotFoundError:
                logger.warning(f"File not found for deletion: {name}")
        return self.update_internal_explorer()

    def load_chat_session(self, session_name: str):
        session_path = self.chat_history_folder / session_name
        try:
            with session_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return [(entry["sender"], entry["message"]) for entry in data]
        except Exception as e:
            logger.error(f"Error loading chat session {session_name}", exc_info=True)
            return []

    def save_chat_session(self, chat_history: list):
        if not chat_history:
            return
        if self.current_session is None:
            title = self.rag_system.generate_chat_title(chat_history)
            self.current_session = title.replace('"', '') + ".json"
        session_path = self.chat_history_folder / self.current_session
        data = [{"sender": s, "message": m} for s, m in chat_history]
        try:
            with session_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Error saving chat session", exc_info=True)

    def save_and_load_chat_session(self, chat_history: list, selected_session: str):
        if chat_history:
            self.save_chat_session(chat_history)
        if selected_session:
            self.current_session = selected_session
            return self.load_chat_session(selected_session)
        return chat_history

    def new_chat_session(self, chat_history: list):
        if chat_history:
            self.save_chat_session(chat_history)
        self.current_session = None
        return [], self.update_internal_explorer()

    def save_chat_and_update(self, chat_history: list):
        self.save_chat_session(chat_history)
        return chat_history, self.update_internal_explorer()

    def launch(self):
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("## PLM SmartBase")

                    upload_btn = gr.File(file_count="multiple", label="Upload")
                    delete_btn = gr.Button("Delete")
                    file_explorer = self._file_explorer(use_chat_history=False)

                    upload_btn.upload(
                        fn=self.upload_files,
                        inputs=upload_btn,
                        outputs=file_explorer,
                    ).then(self.update_internal_explorer, outputs=file_explorer)

                    delete_btn.click(
                        fn=self.delete_files,
                        inputs=file_explorer,
                        outputs=file_explorer,
                    ).then(self.update_internal_explorer, outputs=file_explorer)

                    gr.Markdown("## Chats")
                    new_chat_btn = gr.Button("New chat")
                    save_chat_btn = gr.Button("Save Chat")
                    chat_list = self._file_explorer(use_chat_history=True)

                with gr.Column(scale=8):
                    gr.Markdown("## Chat Interface")
                    chat_interface = gr.Chatbot(label="Chat", type="tuples")
                    user_input_box = gr.Textbox(placeholder="Ask anything", label="Query")
                    send_button = gr.Button("Send", scale=0.005)

                    send_button.click(
                        fn=self.send_query,
                        inputs=[user_input_box, chat_interface],
                        outputs=[chat_interface, user_input_box],
                    )

                    new_chat_btn.click(
                        fn=self.new_chat_session,
                        inputs=[chat_interface],
                        outputs=[chat_interface, chat_list],
                    ).then(self.update_chat_explorer, outputs=chat_list)

                    chat_list.change(
                        fn=self.save_and_load_chat_session,
                        inputs=[chat_interface, chat_list],
                        outputs=chat_interface,
                    )

                    save_chat_btn.click(
                        fn=self.save_chat_and_update,
                        inputs=[chat_interface],
                        outputs=[chat_interface, chat_list],
                    ).then(self.update_chat_explorer, outputs=chat_list)

        demo.launch(share=True)


if __name__ == "__main__":
    agent = KnowledgeBaseAgent("config.yaml")
    agent.launch()
