
from collections import defaultdict
from threading import Lock

# Store conversations separately by session ID
conversations = defaultdict(list)

# Protect shared memory during concurrent requests
memory_lock = Lock()

# Maximum number of previous message pairs
MAX_TURNS = 10


def get_history(session_id: str) -> list:
    """
    Return a copy of a session's conversation history.
    """
    with memory_lock:
        return [
            message.copy()
            for message in conversations.get(session_id, [])
        ]


def save_message(session_id: str, role: str, content: str):
    """
    Save a message and limit conversation history.
    """
    if role not in ("user", "assistant"):
        raise ValueError("Invalid message role")

    with memory_lock:
        conversations[session_id].append({
            "role": role,
            "content": content
        })

        # Each turn contains a user and assistant message
        conversations[session_id] = (
            conversations[session_id][-MAX_TURNS * 2:]
        )


def clear_history(session_id: str):
    """
    Delete a session's conversation history.
    """
    with memory_lock:
        conversations.pop(session_id, None)
