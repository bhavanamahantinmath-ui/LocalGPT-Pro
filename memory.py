"""
Per-User Conversation Memory & Context Management for LocalGPT Pro.
Ensures every turn and chat thread is strictly bound to the active user's UID.
"""

from typing import List, Dict, Any, Optional
from database import Database

DEFAULT_SYSTEM_PROMPT = (
    "You are LocalGPT Pro, an intelligent, helpful, and concise AI assistant running 100% locally on this device. "
    "You provide clear, well-structured, and accurate responses with formatted markdown and code blocks when appropriate."
)

class ConversationMemory:
    def __init__(self, uid: str, db: Optional[Database] = None):
        self.uid = uid
        self.db = db or Database()
        self.active_chat_id: Optional[str] = None
        self.active_title: str = "New Chat"
        self.system_prompt: str = DEFAULT_SYSTEM_PROMPT
        self.messages: List[Dict[str, Any]] = []

        # Load user's default system prompt if set
        user = self.db.get_user(self.uid)
        if user and user.get("default_system_prompt"):
            self.system_prompt = user["default_system_prompt"]

    def initialize_or_resume(self, chat_id: Optional[str] = None) -> str:
        """
        Loads an existing chat for this user or creates a fresh one.
        """
        if chat_id:
            chat = self.db.get_chat(self.uid, chat_id)
            if chat:
                self.active_chat_id = chat_id
                self.active_title = chat["title"]
                self.messages = self.db.get_chat_messages(self.uid, chat_id)
                return self.active_chat_id

        # Look for the user's latest chat
        chats = self.db.get_user_chats(self.uid)
        if chats:
            latest = chats[0]
            self.active_chat_id = latest["chat_id"]
            self.active_title = latest["title"]
            self.messages = self.db.get_chat_messages(self.uid, latest["chat_id"])
            return self.active_chat_id

        return self.create_new_chat("New Chat")

    def create_new_chat(self, title: str = "New Chat") -> str:
        chat_id = self.db.create_chat(self.uid, title=title)
        self.active_chat_id = chat_id
        self.active_title = title
        self.messages = []
        return chat_id

    def load_chat(self, chat_id: str):
        chat = self.db.get_chat(self.uid, chat_id)
        if chat:
            self.active_chat_id = chat_id
            self.active_title = chat["title"]
            self.messages = self.db.get_chat_messages(self.uid, chat_id)

    def rename_chat(self, chat_id: str, new_title: str):
        self.db.update_chat_title(self.uid, chat_id, new_title)
        if self.active_chat_id == chat_id:
            self.active_title = new_title

    def toggle_pin(self, chat_id: str) -> bool:
        return self.db.toggle_chat_pin(self.uid, chat_id)

    def delete_chat(self, chat_id: str):
        self.db.delete_chat(self.uid, chat_id)
        if self.active_chat_id == chat_id:
            self.active_chat_id = None
            self.messages = []
            self.active_title = "New Chat"

    # -------------------------------------------------------------------------
    # Message Operations
    # -------------------------------------------------------------------------
    def add_user_message(self, content: str) -> str:
        if not self.active_chat_id:
            self.create_new_chat()

        # Auto-update title on first message
        if len(self.messages) == 0 and self.active_title in ["New Chat", "Untitled Chat"]:
            first_title = content.strip().split("\n")[0][:32]
            if first_title:
                self.rename_chat(self.active_chat_id, first_title)

        msg_id = self.db.save_message(
            uid=self.uid,
            chat_id=self.active_chat_id,
            role="user",
            content=content
        )
        self.messages.append({
            "id": msg_id,
            "chat_id": self.active_chat_id,
            "uid": self.uid,
            "role": "user",
            "content": content,
            "sources": None
        })
        return msg_id

    def add_assistant_message(
        self,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        if not self.active_chat_id:
            self.create_new_chat()

        msg_id = self.db.save_message(
            uid=self.uid,
            chat_id=self.active_chat_id,
            role="assistant",
            content=content,
            sources=sources
        )
        self.messages.append({
            "id": msg_id,
            "chat_id": self.active_chat_id,
            "uid": self.uid,
            "role": "assistant",
            "content": content,
            "sources": sources
        })
        return msg_id

    def edit_user_message_and_truncate(self, msg_id: str, new_content: str):
        target_idx = None
        for i, m in enumerate(self.messages):
            if m["id"] == msg_id:
                target_idx = i
                break

        if target_idx is None:
            return

        self.db.delete_messages_after(self.uid, self.active_chat_id, msg_id, inclusive=False)
        self.db.update_message(self.uid, msg_id, new_content)

        self.messages[target_idx]["content"] = new_content
        self.messages = self.messages[:target_idx + 1]

    def remove_last_turn(self):
        if self.messages and self.messages[-1]["role"] == "assistant":
            last_msg = self.messages.pop()
            self.db.delete_messages_after(self.uid, self.active_chat_id, last_msg["id"], inclusive=True)

    def get_messages_for_llm(self, max_turns: int = 10) -> List[Dict[str, str]]:
        windowed = self.messages[-max_turns:] if len(self.messages) > max_turns else self.messages
        llm_messages = []
        for m in windowed:
            llm_messages.append({
                "role": m["role"],
                "content": m["content"]
            })
        return llm_messages

