import os
import json
import uuid
from glob import glob
from datetime import datetime
from typing import List, Dict, Optional, Any

# Directory to store chat history JSON files
BASE_HISTORY_DIR = os.path.join("data", "chat_history")

def _get_user_dir(user_id: str) -> str:
    """Helper to ensure user directory exists and return path."""
    user_dir = os.path.join(BASE_HISTORY_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def list_conversations(user_id: str) -> List[Dict[str, Any]]:
    """
    Returns a list of conversation summaries (id, title, date) 
    for the sidebar.
    """
    user_dir = _get_user_dir(user_id)
    files = glob(os.path.join(user_dir, "*.json"))
    
    results = []
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                # We only need metadata for the list view
                results.append({
                    "id": data.get("id"),
                    "title": data.get("title", "Untitled"),
                    "updated_at": data.get("updated_at")
                })
        except Exception:
            continue
            
    # Sort by date descending (newest first)
    results.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return results

def get_conversation(user_id: str, conv_id: str) -> Optional[Dict[str, Any]]:
    """
    Returns the full conversation object including messages.
    """
    user_dir = _get_user_dir(user_id)
    fpath = os.path.join(user_dir, f"{conv_id}.json")
    
    if not os.path.exists(fpath):
        return None
        
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def save_conversation(user_id: str, payload: Any) -> str:
    """
    Saves a new conversation or updates an existing one.
    'payload' is expected to be the Pydantic model from main.py
    """
    user_dir = _get_user_dir(user_id)
    
    # Generate a new ID for this conversation
    # (In a real app, you might check if payload has an ID to update instead)
    conv_id = uuid.uuid4().hex
    
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    conversation_data = {
        "id": conv_id,
        "user_id": user_id,
        "title": payload.title,
        "messages": payload.messages,  # list of message objects
        "created_at": timestamp,
        "updated_at": timestamp
    }
    
    fpath = os.path.join(user_dir, f"{conv_id}.json")
    
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(conversation_data, f, ensure_ascii=False, indent=2)
        
    return conv_id

def delete_conversation(user_id: str, conv_id: str) -> bool:
    """
    Deletes a specific conversation file.
    """
    user_dir = _get_user_dir(user_id)
    fpath = os.path.join(user_dir, f"{conv_id}.json")
    
    if os.path.exists(fpath):
        os.remove(fpath)
        return True
    return False

def clear_all_conversations(user_id: str):
    """
    Deletes all conversation files for this user.
    """
    user_dir = _get_user_dir(user_id)
    files = glob(os.path.join(user_dir, "*.json"))
    for f in files:
        os.remove(f)