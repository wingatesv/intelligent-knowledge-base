import os
import json
import datetime
import shutil
import yaml
import logging
import gradio as gr
from llm_query import RAGSystem

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class RAGChatApp:
    def __init__(self, config_path="config.yaml"):
        self.config = self.load_config(config_path)
        self.api_token = self.config.get("API_TOKEN", "")
        self.embedding_model = self.config.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        self.llm_model = self.config.get("LLM_MODEL", "google/gemma-2-2b-it")
        self.chunk_size = self.config.get("CHUNK_SIZE", 512)
        self.chunk_overlap = self.config.get("CHUNK_OVERLAP", 10)
        self.interface_mode = self.config.get("INTERFACE_MODE", "DARK").upper()

        # Setup folders
        self.internal_folder = self.config.get("DOC_DIR", "documents")
        self.internal_folder = os.path.join(os.getcwd(), self.internal_folder)
        os.makedirs(self.internal_folder, exist_ok=True)

        self.base_chat_history_dir = self.config.get("CHAT_DIR", "chat_histories")
        self.base_chat_history_dir = os.path.join(os.getcwd(), self.base_chat_history_dir)
        os.makedirs(self.base_chat_history_dir, exist_ok=True)

        # State variables
        self.role = "Student"  # default role
        self.current_session = None

        # Initialize RAG system
        self.rag = RAGSystem(
            api_token=self.api_token,
            embedding_model=self.embedding_model,
            llm_model=self.llm_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self.rag.initialize_rag(role=self.role)

    @staticmethod
    def load_config(config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing configuration file: {config_path}")
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def get_chat_history_dir(self):
        """Returns the directory for chat histories corresponding to the current role."""
        role_dir = os.path.join(self.base_chat_history_dir, self.role)
        os.makedirs(role_dir, exist_ok=True)
        return role_dir

    def generate_session_id(self):
        """Generates a new session id using the current timestamp."""
        return f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def change_role(self, role, chat_history):
        """Changes the current role, saves the chat history, and reinitializes RAG."""
        self.save_chat_session(chat_history)
        # Clear chat history when switching role
        chat_history = []
        self.role = role
        try:
            self.rag.initialize_rag(role=role)
            msg = f"Role switched to: {role}. RAG reinitialized."
            logging.info(msg)
        except Exception as e:
            msg = f"Error reinitializing RAG: {str(e)}"
            logging.error(msg)
        file_update = gr.update(visible=(role != "Teacher"), interactive=(role != "Teacher"))
        new_chat, chat_dropdown_update = self.new_chat_session(chat_history)
        return msg, file_update, new_chat, chat_dropdown_update

    def send_query(self, user_input, chat_history):
        """Appends the user's query to the chat history and obtains a response via RAG."""
        if not user_input.strip():
            return chat_history, ""
        chat_history = chat_history + [[user_input, ""]]
        try:
            response = self.rag.chat(user_input)
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
            self.rag.update(os.path.join(self.internal_folder, filename))
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
            self.rag.delete(full_path)
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
        return [], gr.FileExplorer(root_dir=self.get_chat_history_dir())

    def save_chat_and_update(self, chat_history):
        """Saves the current chat and returns the updated file explorer."""
        self.save_chat_session(chat_history)
        return chat_history, gr.FileExplorer(root_dir=self.get_chat_history_dir())

    def launch(self):
        """Builds and launches the Gradio UI."""
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("## Documents")
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
                    with gr.Row(equal_height=True):
                        role_dropdown = gr.Dropdown(
                            choices=["Teacher", "Student"],
                            value="Student",
                            label="User role",
                            scale=0.2
                        )
                        role_status = gr.Textbox(label="Role status", interactive=False, scale=1)
                    chat_interface = gr.Chatbot(label="Chat")
                    with gr.Row(equal_height=True):
                        user_input_box = gr.Textbox(placeholder="Enter your query", label="Your query")
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
                    role_dropdown.change(
                        fn=self.change_role,
                        inputs=[role_dropdown, chat_interface],
                        outputs=[role_status, upload_button, chat_interface, chat_history_list]
                    )
                    save_chat_button.click(
                        fn=self.save_chat_and_update,
                        inputs=chat_interface,
                        outputs=[chat_interface, chat_history_list]
                    ).then(fn=self.update_file_explorer_2, outputs=chat_history_list)

        demo.launch(share=True)

if __name__ == "__main__":
    app = RAGChatApp("config.yaml")
    app.launch()
