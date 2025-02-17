import gradio as gr
import os
import json
import datetime
import shutil
import yaml
import logging
from llm_query import initialize_rag, hugging_face_query

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Configuration
def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing configuration file: {config_path}")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

config = load_config("config.yaml")

# Global Configurations
api_token = config.get("API_TOKEN", "")
embedding_model = config.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
llm_model = config.get("LLM_MODEL", "google/gemma-2-2b-it")
chunk_size = config.get("CHUNK_SIZE", 512)
chunk_overlap = config.get("CHUNK_OVERLAP", 10)
interface_mode = config.get("INTERFACE_MODE", "DARK").upper()

# Setup folders
internal_folder = config.get("DOC_DIR", "documents")
internal_folder = os.path.join(os.getcwd(), internal_folder)
os.makedirs(internal_folder, exist_ok=True)

chat_history_dir = config.get("CHAT_DIR", "chat_histories")
chat_history_dir = os.path.join(os.getcwd(), chat_history_dir)
os.makedirs(chat_history_dir, exist_ok=True)

role_global = "Student"
current_session = None  # Active chat session

# Initialize RAG
initialize_rag(
    api_token=api_token,
    embedding_model=embedding_model,
    llm_model=llm_model,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    role=role_global
)

### **Helper Functions**
def change_role(role):
    global role_global
    role_global = role
    try:
        initialize_rag(
            api_token=api_token,
            embedding_model=embedding_model,
            llm_model=llm_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            role=role
        )
        msg = f"Role switched to: {role}. RAG reinitialized."
        logging.info(msg)
    except Exception as e:
        msg = f"Error reinitializing RAG: {str(e)}"
        logging.error(msg)
    
    file_update = gr.update(visible=(role != "Teacher"), interactive=(role != "Teacher"))
    return msg, file_update

def send_query(user_input, chat_history):
    if not user_input.strip():
        return chat_history, ""
    chat_history = chat_history + [[user_input, ""]]
    try:
        response = hugging_face_query(user_input, role_global)
    except Exception as e:
        response = f"Error: {str(e)}"
    chat_history[-1][1] = str(response)
    return chat_history, ""

def upload_files(files):
    if not files:
        return "No files uploaded."
    new_files = []
    for file_obj in files:
        file_name = os.path.basename(file_obj.name)
        dest_path = os.path.join(internal_folder, file_name)
        shutil.copy(file_obj.name, dest_path)
        new_files.append(file_name)
    return f"Uploaded: {', '.join(new_files)}"

def load_existing_chat_sessions():
    if not os.path.exists(chat_history_dir):
        return []
    session_files = sorted(os.listdir(chat_history_dir))
    return [os.path.splitext(session)[0] for session in session_files]

def load_chat_session(session_name):
    """Loads and formats chat history correctly for gr.Chatbot"""
    session_path = os.path.join(chat_history_dir, session_name + ".json")
    try:
        with open(session_path, "r", encoding="utf-8") as file:
            session_data = json.load(file)
        return [(entry["sender"], entry["message"]) for entry in session_data]
    except Exception as e:
        logging.error(f"Error loading chat session {session_name}: {e}")
        return []

def save_chat_session(chat_history):
    """Saves chat history in the correct JSON format"""
    global current_session
    if not chat_history:
        return
    if current_session is None:
        current_session = f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_path = os.path.join(chat_history_dir, current_session + ".json")
    try:
        session_data = [{"sender": msg[0], "message": msg[1]} for msg in chat_history]
        with open(session_path, "w", encoding="utf-8") as file:
            json.dump(session_data, file, indent=2)
    except Exception as e:
        logging.error(f"Error saving chat session: {e}")

def new_chat_session(chat_history):
    """Save session before clearing if chat exists"""
    if chat_history:
        save_chat_session(chat_history)

    global current_session
    current_session = None

    updated_choices = load_existing_chat_sessions()
    return [], {"choices": updated_choices}

def save_on_exit(chat_history):
    """Save chat session before UI exits"""
    save_chat_session(chat_history)

def list_files_in_internal_folder():
    try:
        files = os.listdir(internal_folder)
        if not files:
            return "No files uploaded."
        return "\n".join(files)
    except Exception as e:
        return f"Error accessing internal folder: {str(e)}"

def upload_files(files):
    """Handle file uploads and reinitialize the RAG system."""
    if not files:
        return "No files uploaded."
    new_files = []
    for file_obj in files:
        file_name = os.path.basename(file_obj.name)
        dest_path = os.path.join(internal_folder, file_name)
        shutil.copy(file_obj.name, dest_path)
        new_files.append(file_name)
    try:
        # Reinitialize the RAG system after uploading new files
        initialize_rag(
              api_token=api_token,
              embedding_model=embedding_model,
              llm_model=llm_model,
              chunk_size=chunk_size,
              chunk_overlap=chunk_overlap,
              role=role_global
          )
        logging.info(f"Uploaded files: {', '.join(new_files)}. RAG system reinitialized.")
    except Exception as e:
        logging.error(f"Error reinitializing RAG system: {str(e)}")
        return f"Error reinitializing RAG system: {str(e)}"
    return list_files_in_internal_folder()


### **Gradio UI Implementation**
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## Documents")
            upload_button = gr.File(file_count="multiple", label="Upload & View Documents")
            upload_status = gr.Textbox(label="Uploaded Files", interactive=False, value=list_files_in_internal_folder())
            upload_button.upload(fn=upload_files, inputs=upload_button, outputs=upload_status)

            gr.Markdown("## Chats")
            new_chat_button = gr.Button("New Chat")
            chat_history_list = gr.Dropdown(choices=load_existing_chat_sessions(), label="Chat History", interactive=True)

        with gr.Column(scale=8):
            with gr.Row(equal_height=True):
                role_dropdown = gr.Dropdown(choices=["Teacher", "Student"], value="Student", label="Role", scale=0.2)
                role_status = gr.Textbox(label="Role Status", interactive=False, scale=1)
                role_dropdown.change(fn=change_role, inputs=role_dropdown, outputs=role_status)
                settings_button = gr.Button("⚙", scale=0.05)

            chat_interface = gr.Chatbot(label="Chat")
            with gr.Row(equal_height=True):
                user_input_box = gr.Textbox(placeholder="Enter your query", label="Your Query")
                send_button = gr.Button("Send", scale=0.05)

            send_button.click(fn=send_query, inputs=[user_input_box, chat_interface], outputs=[chat_interface, user_input_box])
            new_chat_button.click(fn=new_chat_session, inputs=[chat_interface], outputs=[chat_interface, chat_history_list])
            chat_history_list.change(fn=load_chat_session, inputs=chat_history_list, outputs=chat_interface)


demo.launch(share=True)
