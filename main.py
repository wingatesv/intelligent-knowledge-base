import os
import json
import datetime
import shutil
import yaml
import logging
import gradio as gr
from rag_system import RAGSystem

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

#----------------------------------------------------------------------
# IMPORTANT REMARKS on gradio file explorer
#----------------------------------------------------------------------
# File explorer will not refresh automatically: https://github.com/gradio-app/gradio/issues/7788
# Hence, what we did is we purposely load the wrong file explorer first, 
# then only we reload back the correct file explorer.
# That is why you will notice that the we will load the wrong directory
# for file explorer in each function, then only we use .then() to call
# the correct file explorer using the correct directory


class KnowledgeBaseAgent:
    def __init__(self, config_path="config.yaml"):
        self.config = self.load_config(config_path)
        self.embedding_model = self.config.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        self.llm_model = self.config.get("LLM_MODEL", "mistral")
        self.chunk_size = self.config.get("CHUNK_SIZE", 512)
        self.chunk_overlap = self.config.get("CHUNK_OVERLAP", 20)

        # Setup folders
        self.internal_folder = self.config.get("DOC_DIR", "data")
        self.internal_folder = os.path.join(os.getcwd(), self.internal_folder)
        os.makedirs(self.internal_folder, exist_ok=True)

        self.base_chat_history_dir = self.config.get("CHAT_DIR", "chat_histories")
        self.base_chat_history_dir = os.path.join(os.getcwd(), self.base_chat_history_dir)
        os.makedirs(self.base_chat_history_dir, exist_ok=True)

        # State variables
        self.current_session = None

        # Initialize RAG system
        self.rag_system = RAGSystem(
            embedding_model=self.embedding_model,
            llm_model=self.llm_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self.rag_system.initialize_rag(self.internal_folder)

    @staticmethod
    def load_config(config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing configuration file: {config_path}")
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def get_chat_history_dir(self):
        """Returns the directory for chat histories"""
        chat_history_dir = os.path.join(self.base_chat_history_dir)
        os.makedirs(chat_history_dir, exist_ok=True)
        return chat_history_dir

    def generate_session_id(self):
        """Generates a new session id using the current timestamp."""
        return f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"


    def send_query(self, user_input, chat_history):
        """Appends the user's query to the chat history and obtains a response via RAG."""
        if not user_input.strip():
            return chat_history, ""
        chat_history = chat_history + [[user_input, ""]]
        try:
            response = self.rag_system.chat(user_input)
        except Exception as e:
            response = f"Error: {str(e)}"
        chat_history[-1][1] = str(response)
        return chat_history, ""

    def upload_files(self, files):
        """Handles file uploads and updates the RAG system."""
        if not files:
            return gr.FileExplorer(root_dir=self.get_chat_history_dir())
        new_files = []
        for file_obj in files:
            filename = os.path.basename(file_obj.name)
            dest_path = os.path.join(self.internal_folder, filename)
            shutil.copy(file_obj.name, dest_path)
            new_files.append(filename)
            logging.info(f"Uploaded: {filename}")
        for filename in new_files:
            self.rag_system.update(os.path.join(self.internal_folder, filename))
        return gr.FileExplorer(root_dir=self.get_chat_history_dir())

    def delete_files(self, selected_files):
        """Handles deletion of selected files and updates the RAG system."""
        if not selected_files:
            return gr.FileExplorer(root_dir=self.get_chat_history_dir())
        if isinstance(selected_files, str):
            selected_files = [selected_files]
        for file_name in selected_files:
            full_path = os.path.join(self.internal_folder, file_name)
            os.remove(full_path)
            logging.info(f"Deleted file: {file_name}")
            self.rag_system.delete(full_path)
        return gr.FileExplorer(root_dir=self.get_chat_history_dir())

    def update_file_explorer_1(self):
        """Refreshes the file explorer for uploaded documents."""
        return gr.FileExplorer(root_dir=self.internal_folder)

    def update_file_explorer_2(self):
        """Refreshes the file explorer for chat histories for the current role."""
        return gr.FileExplorer(root_dir=self.get_chat_history_dir())

    def load_chat_session(self, session_name):
        """Loads a saved chat session."""
        session_path = os.path.join(self.get_chat_history_dir(), session_name)
        try:
            with open(session_path, "r", encoding="utf-8") as file:
                session_data = json.load(file)
            return [(entry["sender"], entry["message"]) for entry in session_data]
        except Exception as e:
            logging.info(f"Error loading chat session {session_name}: {e}")
            return []

    def save_chat_session(self, chat_history):
        """Saves the current chat session."""
        if not chat_history:
            return
        if self.current_session is None:
            self.current_session = self.rag.generate_chat_title(chat_history)
            self.current_session = self.current_session.replace('"', '')
        if not self.current_session.endswith(".json"):
            self.current_session += ".json"
        print('Chat title:', self.current_session)
        session_path = os.path.join(self.get_chat_history_dir(), self.current_session)
        try:
            session_data = [{"sender": msg[0], "message": msg[1]} for msg in chat_history]
            with open(session_path, "w", encoding="utf-8") as file:
                json.dump(session_data, file, indent=2)
        except Exception as e:
            logging.error(f"Error saving chat session: {e}")

    def save_and_load_chat_session(self, chat_history, selected_session):
        """Saves the current session and then loads a selected session."""
        if chat_history:
            self.save_chat_session(chat_history)
        if not selected_session:
            return chat_history
        self.current_session = selected_session
        return self.load_chat_session(selected_session)

    def new_chat_session(self, chat_history):
        """Starts a new chat session, saving the old one if necessary."""
        if chat_history:
            self.save_chat_session(chat_history)
        self.current_session = None
        new_chat = []
        return new_chat, gr.FileExplorer(root_dir=self.internal_folder)

    def save_chat_and_update(self, chat_history):
        """Saves the current chat and returns the updated file explorer."""
        self.save_chat_session(chat_history)
        return chat_history, gr.FileExplorer(root_dir=self.internal_folder)

    def launch(self):
        """Builds and launches the Gradio UI."""
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("## PLM SmartBase")
                    with gr.Row(equal_height=True):
                        upload_button = gr.File(file_count="multiple", label="Upload file")
                        delete_button = gr.Button("Delete")
                    uploaded_file_list = gr.FileExplorer(
                        root_dir=self.internal_folder,
                        file_count="single",
                        interactive=True,
                        label='Uploaded files',
                        every=1,
                        ignore_glob='*.ipynb_checkpoints'
                    )
                    upload_button.upload(
                        fn=self.upload_files,
                        inputs=upload_button,
                        outputs=uploaded_file_list
                    ).then(fn=self.update_file_explorer_1, outputs=uploaded_file_list)
                    delete_button.click(
                        fn=self.delete_files,
                        inputs=uploaded_file_list,
                        outputs=uploaded_file_list
                    ).then(fn=self.update_file_explorer_1, outputs=uploaded_file_list)

                    gr.Markdown("## Chats")
                    new_chat_button = gr.Button("New chat")
                    save_chat_button = gr.Button("Save Chat")
                    chat_history_list = gr.FileExplorer(
                        root_dir=self.get_chat_history_dir(),
                        file_count="single",
                        interactive=True,
                        label='Chats',
                        every=1,
                        ignore_glob='*.ipynb_checkpoints'
                    )

                with gr.Column(scale=8):
                    gr.Markdown("## Chat Interface")
                    chat_interface = gr.Chatbot(label="Chat")
                    with gr.Row(equal_height=True):
                        user_input_box = gr.Textbox(placeholder="Ask anything", label="Query")
                        send_button = gr.Button("Send", scale=0.05)
                    send_button.click(
                        fn=self.send_query,
                        inputs=[user_input_box, chat_interface],
                        outputs=[chat_interface, user_input_box]
                    )
                    new_chat_button.click(
                        fn=self.new_chat_session,
                        inputs=[chat_interface],
                        outputs=[chat_interface, chat_history_list]
                    ).then(fn=self.update_file_explorer_2, outputs=chat_history_list)
                    chat_history_list.change(
                        fn=self.save_and_load_chat_session,
                        inputs=[chat_interface, chat_history_list],
                        outputs=chat_interface
                    )
                   
                    save_chat_button.click(
                        fn=self.save_chat_and_update,
                        inputs=chat_interface,
                        outputs=[chat_interface, chat_history_list]
                    ).then(fn=self.update_file_explorer_2, outputs=chat_history_list)

        demo.launch(share=True)

if __name__ == "__main__":
    app = KnowledgeBaseAgent("config.yaml")
    app.launch()
