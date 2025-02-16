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

# ------------------ Configuration and Setup ------------------

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing configuration file: {config_path}")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

config = load_config("config.yaml")

# Get configuration settings
api_token = config.get("API_TOKEN", "")
embedding_model = config.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
llm_model = config.get("LLM_MODEL", "google/gemma-2-2b-it")
chunk_size = config.get("CHUNK_SIZE", 512)
chunk_overlap = config.get("CHUNK_OVERLAP", 10)
interface_mode = config.get("INTERFACE_MODE", "DARK").upper()

# Setup folders for documents and chat histories
internal_folder = config.get("DOC_DIR", "documents")
internal_folder = os.path.join(os.getcwd(), internal_folder)
os.makedirs(internal_folder, exist_ok=True)

chat_history_dir = config.get("CHAT_DIR", "chat_histories")
chat_history_dir = os.path.join(os.getcwd(), chat_history_dir)
os.makedirs(chat_history_dir, exist_ok=True)

# Default role
role_global = "Student"

# Initialize RAG with default settings
initialize_rag(
    api_token=api_token,
    embedding_model=embedding_model,
    llm_model=llm_model,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    role=role_global
)

# ------------------ Function Definitions ------------------

def change_role(role):
    """
    Change the role for the LLM. If the role is Teacher, disable file uploads.
    Reinitialize RAG with the new role.
    """
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
    # If role is Teacher, disable file upload; otherwise, enable it.
    file_update = gr.File.update(visible=True, interactive=True) if role != "Teacher" else gr.File.update(visible=False, interactive=False)
    return msg, file_update

def send_query(user_input, chat_history):
    """
    Append the user query to the chat history, call the LLM and then update the chat with the response.
    """
    if not user_input.strip():
        return chat_history, ""
    # Add user message with empty model reply
    chat_history = chat_history + [[user_input, ""]]
    try:
        response = hugging_face_query(user_input, role_global)
    except Exception as e:
        response = f"Error: {str(e)}"
    # Update the last message with the response
    chat_history[-1][1] = str(response)
    return chat_history, ""

def upload_files(files):
    """
    Copy uploaded files to the internal folder and reinitialize RAG.
    """
    if not files:
        return "No files uploaded."
    new_files = []
    for file_obj in files:
        # file_obj.name is the local temp path provided by Gradio
        file_name = os.path.basename(file_obj.name)
        dest_path = os.path.join(internal_folder, file_name)
        shutil.copy(file_obj.name, dest_path)
        new_files.append(file_name)
    try:
        initialize_rag(
            api_token=api_token,
            embedding_model=embedding_model,
            llm_model=llm_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            role=role_global
        )
        msg = f"Uploaded files: {', '.join(new_files)}. RAG updated."
        logging.info(msg)
    except Exception as e:
        msg = f"Error updating RAG: {str(e)}"
        logging.error(msg)
    return msg

def new_chat_session():
    """Clear the chat history to start a new chat session."""
    return []

# ------------------ Gradio Interface ------------------

with gr.Blocks() as demo:
    gr.Markdown("# RAG Chat Interface")
    
    with gr.Row():
        role_dropdown = gr.Dropdown(choices=["Student", "Teacher"], value="Student", label="Role")
        role_status = gr.Textbox(label="Role Status", interactive=False)
        # Changing role updates both the status message and file uploader state.
        role_dropdown.change(fn=change_role, inputs=role_dropdown, outputs=[role_status, None])
    
    with gr.Row():
        file_upload = gr.File(file_count="multiple", label="Upload Documents")
        file_status = gr.Textbox(label="File Upload Status", interactive=False)
        file_upload.upload(fn=upload_files, inputs=file_upload, outputs=file_status)
    
    chatbot = gr.Chatbot(label="Chat")
    user_input_box = gr.Textbox(placeholder="Enter your query", label="Your Query")
    
    with gr.Row():
        send_button = gr.Button("Send")
        clear_button = gr.Button("New Chat Session")
    
    send_button.click(fn=send_query, inputs=[user_input_box, chatbot], outputs=[chatbot, user_input_box])
    clear_button.click(fn=new_chat_session, inputs=None, outputs=chatbot)
    
demo.launch()

