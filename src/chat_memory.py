import os
import json

MEMORY_DIR = "sessions"

def load_memory(username):
    if not os.path.exists(MEMORY_DIR):
        os.makedirs(MEMORY_DIR)
    filepath = os.path.join(MEMORY_DIR, f"{username}.json")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return []

def save_memory(username, memory):
    if not os.path.exists(MEMORY_DIR):
        os.makedirs(MEMORY_DIR)
    filepath = os.path.join(MEMORY_DIR, f"{username}.json")
    with open(filepath, "w") as f:
        json.dump(memory, f)