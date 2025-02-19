import gradio as gr
import os
import json
import datetime
import shutil
import yaml
import logging
from llm_query import RAGSystem

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

# Global session variables
role_global = "Student"

def generate_session_id():
    """Generates a new session id using the current timestamp."""
    return f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

# On startup, generate a new session id
# current_session = generate_session_id()
current_session = None

# Initialize RAG
rag = RAGSystem()
rag.initialize_rag(
    api_token=api_token,
    embedding_model=embedding_model,
    llm_model=llm_model,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    role=role_global
)

### **Helper Functions**
def change_role(role, chat_history):
    global role_global
    role_global = role
    try:
        rag.initialize_rag(
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
    
    # Update file uploader visibility/interactivity based on role
    file_update = gr.update(visible=(role != "Teacher"), interactive=(role != "Teacher"))
    
    # Create a new chat session (clears the chat interface and updates the dropdown)
    new_chat, chat_dropdown_update = new_chat_session(chat_history)
    
    # Return four outputs: role status message, file update, cleared chat interface, and updated chat history dropdown
    return msg, file_update, new_chat, chat_dropdown_update

def send_query(user_input, chat_history):
    if not user_input.strip():
        return chat_history, ""
    chat_history = chat_history + [[user_input, ""]]
    try:
        response = rag.chat(user_input, role_global)
    except Exception as e:
        response = f"Error: {str(e)}"
    chat_history[-1][1] = str(response)
    return chat_history, ""


def upload_files(files):
    """Handle file uploads and reinitialize the RAG system."""
    if not files:
        return gr.FileExplorer(root_dir=chat_history_dir)
    new_files = []
    for file_obj in files:
        filename = os.path.basename(file_obj.name)
        dest_path = os.path.join(internal_folder, filename)
        shutil.copy(file_obj.name, dest_path)
        new_files.append(filename)
        logging.info(f"Uploaded: {filename}")
    for filename in new_files:
        rag.update(os.path.join(internal_folder, filename))


    return gr.FileExplorer(root_dir=chat_history_dir)

def delete_files(selected_files):
    """
    Handle deletion of selected files, update the RAG system and refresh the file explorer.
    Assumes that RAGSystem has a 'delete' method to remove file content from its index.
    """
    if not selected_files:
        return gr.FileExplorer(root_dir=chat_history_dir)
    # Ensure a list if a single file is selected
    if isinstance(selected_files, str):
        selected_files = [selected_files]
    for file_name in selected_files:
        full_path = os.path.join(internal_folder, file_name)
        os.remove(full_path)
        logging.info(f"Deleted file: {file_name}")
        # Update RAG to remove the file's content from its index
        rag.delete(full_path)

    return gr.FileExplorer(root_dir=chat_history_dir)

def update_file_explorer_1():
    """
    File explorer will not refresh automatically: https://github.com/gradio-app/gradio/issues/7788
    """
    return gr.FileExplorer(root_dir=internal_folder)

def update_file_explorer_2():
    """
    File explorer will not refresh automatically: https://github.com/gradio-app/gradio/issues/7788
    """
    return gr.FileExplorer(root_dir=chat_history_dir)


def load_chat_session(session_name):
    """Loads and formats chat history correctly for gr.Chatbot."""
    session_path = os.path.join(chat_history_dir, session_name)
    try:
        with open(session_path, "r", encoding="utf-8") as file:
            session_data = json.load(file)
        return [(entry["sender"], entry["message"]) for entry in session_data]
    except Exception as e:
        logging.info(f"Error loading chat session {session_name}: {e}")
        return []

def save_chat_session(chat_history):
    """Saves chat history in the correct JSON format."""
    global current_session
    if not chat_history:
        return

    if current_session is None:
      current_session = rag.generate_chat_title(chat_history)
      current_session = current_session.replace('"', '')
    # Check if current_session already ends with ".json"; if not, append it.
    current_session = current_session if current_session.endswith(".json") else current_session + ".json"
    # current_session = title if title.endswith(".json") else title + ".json"
    print('Chat title: ', current_session)

    session_path = os.path.join(chat_history_dir, current_session)

    
    try:
        session_data = [{"sender": msg[0], "message": msg[1]} for msg in chat_history]
        with open(session_path, "w", encoding="utf-8") as file:
            json.dump(session_data, file, indent=2)
    except Exception as e:
        logging.error(f"Error saving chat session: {e}")

def save_and_load_chat_session(chat_history, selected_session):
    """
    Saves the current chat session and then loads the chat history
    from the selected session. If the selected session is None, then
    the current chat_history is returned unchanged.
    """
    global current_session
    if chat_history:
        save_chat_session(chat_history)
    if not selected_session:
        # If no valid session is selected, do nothing.
        return chat_history
    current_session = selected_session
    return load_chat_session(selected_session)

def new_chat_session(chat_history):
    """
    Saves the current chat session (if nonempty), generates a new session id,
    and updates the dropdown choices to include the new session id.
    """
    global current_session

    # If there's an existing chat, save it
    if chat_history:
        save_chat_session(chat_history)
    
    # Generate a new session id
    # current_session = generate_session_id()
    current_session = None
    
    # Clear the chat history for the new session
    return [], gr.FileExplorer(root_dir=internal_folder)

def save_chat_and_update(chat_history):
    """
    Saves the current chat session with the current session id,
    updates the chat history list, but keeps the chat interface unchanged.
    """
    save_chat_session(chat_history)
    return chat_history, gr.FileExplorer(root_dir=internal_folder)


### **Gradio UI Implementation**
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## Documents")
            with gr.Row(equal_height=True):
                upload_button = gr.File(file_count="multiple", label="Upload file")
                delete_button = gr.Button("Delete")

            uploaded_file_list = gr.FileExplorer(root_dir=internal_folder, file_count="single", interactive=True, label='Uploaded files', every=1, ignore_glob='*.ipynb_checkpoints')
           
            upload_button.upload(fn=upload_files, inputs=upload_button, outputs=uploaded_file_list).then(fn=update_file_explorer_1, outputs=uploaded_file_list)
            delete_button.click(fn=delete_files, inputs=uploaded_file_list, outputs=uploaded_file_list).then(fn=update_file_explorer_1,  outputs=uploaded_file_list)

            gr.Markdown("## Chats")
            new_chat_button = gr.Button("New chat")
            save_chat_button = gr.Button("Save Chat")
          
            chat_history_list = gr.FileExplorer(root_dir=chat_history_dir, file_count="single", interactive=True, label='Chats', every=1, ignore_glob='*.ipynb_checkpoints')
            
            

        with gr.Column(scale=8):
            with gr.Row(equal_height=True):
                role_dropdown = gr.Dropdown(choices=["Teacher", "Student"], value="Student", label="User role", scale=0.2)
                role_status = gr.Textbox(label="Role status", interactive=False, scale=1)
                
                # settings_button = gr.Button("⚙", scale=0.05)

            chat_interface = gr.Chatbot(label="Chat")
            with gr.Row(equal_height=True):
                user_input_box = gr.Textbox(placeholder="Enter your query", label="Your query")
                send_button = gr.Button("Send", scale=0.05)

            send_button.click(fn=send_query, inputs=[user_input_box, chat_interface], outputs=[chat_interface, user_input_box])
            new_chat_button.click(fn=new_chat_session, inputs=[chat_interface], outputs=[chat_interface, chat_history_list]).then(fn=update_file_explorer_2, outputs=chat_history_list)
            chat_history_list.change(fn=save_and_load_chat_session, inputs=[chat_interface, chat_history_list], outputs=chat_interface)
            role_dropdown.change(fn=change_role, inputs=[role_dropdown, chat_interface], outputs=[role_status, upload_button, chat_interface, chat_history_list] )
            save_chat_button.click(fn=save_chat_and_update, inputs=chat_interface, outputs=[chat_interface, chat_history_list]).then(fn=update_file_explorer_2, outputs=chat_history_list)

demo.launch(share=True)
