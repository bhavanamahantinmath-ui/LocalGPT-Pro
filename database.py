"""
Database layer for LocalGPT Pro with Multi-User Data Isolation.
Manages users, per-user chats, messages, document metadata, and preferences using SQLite & SQLAlchemy.
"""

import os
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Text, Boolean, DateTime, ForeignKey, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "localgpt_pro.db")

Base = declarative_base()

class UserModel(Base):
    __tablename__ = "users"
    uid = Column(String(128), primary_key=True)
    email = Column(String(255), nullable=True)
    display_name = Column(String(255), nullable=True)
    photo_url = Column(Text, nullable=True)
    provider = Column(String(64), default="firebase")
    created_at = Column(String(64), nullable=False)
    last_login = Column(String(64), nullable=False)
    theme = Column(String(32), default="dark")
    default_system_prompt = Column(Text, nullable=True)

class ChatModel(Base):
    __tablename__ = "chats"
    chat_id = Column(String(64), primary_key=True)
    uid = Column(String(128), ForeignKey("users.uid"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    pinned = Column(Boolean, default=False, nullable=False, index=True)
    created_at = Column(String(64), nullable=False)
    updated_at = Column(String(64), nullable=False, index=True)

class MessageModel(Base):
    __tablename__ = "messages"
    message_id = Column(String(64), primary_key=True)
    chat_id = Column(String(64), ForeignKey("chats.chat_id"), nullable=False, index=True)
    uid = Column(String(128), ForeignKey("users.uid"), nullable=False, index=True)
    role = Column(String(32), nullable=False)
    content = Column(Text, nullable=False)
    sources_json = Column(Text, nullable=True)
    created_at = Column(String(64), nullable=False, index=True)

class DocumentModel(Base):
    __tablename__ = "documents"
    doc_id = Column(String(64), primary_key=True)
    uid = Column(String(128), ForeignKey("users.uid"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(32), nullable=False)
    file_size = Column(Integer, nullable=False)
    pages = Column(Integer, default=1)
    chunks = Column(Integer, nullable=False)
    uploaded_at = Column(String(64), nullable=False)

class SettingsModel(Base):
    __tablename__ = "settings"
    uid = Column(String(128), primary_key=True)
    temperature = Column(Float, default=0.7)
    top_k = Column(Integer, default=50)
    top_p = Column(Float, default=0.9)
    max_tokens = Column(Integer, default=512)

class Database:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            connect_args={"check_same_thread": False},
            echo=False
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(self.session_factory)

    # -------------------------------------------------------------------------
    # User Management
    # -------------------------------------------------------------------------
    def upsert_user(
        self,
        uid: str,
        email: Optional[str] = None,
        display_name: Optional[str] = None,
        photo_url: Optional[str] = None,
        provider: str = "firebase"
    ) -> Dict[str, Any]:
        session = self.Session()
        now = datetime.now().isoformat()
        try:
            user = session.query(UserModel).filter_by(uid=uid).first()
            if not user:
                user = UserModel(
                    uid=uid,
                    email=email or "",
                    display_name=display_name or (email.split("@")[0] if email else "User"),
                    photo_url=photo_url or "",
                    provider=provider,
                    created_at=now,
                    last_login=now,
                    theme="dark"
                )
                session.add(user)
                # Also create default settings
                settings = SettingsModel(uid=uid)
                session.add(settings)
            else:
                user.last_login = now
                if email:
                    user.email = email
                if display_name:
                    user.display_name = display_name
                if photo_url:
                    user.photo_url = photo_url
            session.commit()
            return self.get_user(uid)
        finally:
            session.close()

    def get_user(self, uid: str) -> Optional[Dict[str, Any]]:
        session = self.Session()
        try:
            user = session.query(UserModel).filter_by(uid=uid).first()
            if not user:
                return None
            return {
                "uid": user.uid,
                "email": user.email,
                "display_name": user.display_name,
                "photo_url": user.photo_url,
                "provider": user.provider,
                "created_at": user.created_at,
                "last_login": user.last_login,
                "theme": user.theme,
                "default_system_prompt": user.default_system_prompt
            }
        finally:
            session.close()

    def update_user_preferences(
        self,
        uid: str,
        display_name: Optional[str] = None,
        theme: Optional[str] = None,
        default_system_prompt: Optional[str] = None
    ):
        session = self.Session()
        try:
            user = session.query(UserModel).filter_by(uid=uid).first()
            if user:
                if display_name is not None:
                    user.display_name = display_name
                if theme is not None:
                    user.theme = theme
                if default_system_prompt is not None:
                    user.default_system_prompt = default_system_prompt
                session.commit()
        finally:
            session.close()

    # -------------------------------------------------------------------------
    # User Settings
    # -------------------------------------------------------------------------
    def get_settings(self, uid: str) -> Dict[str, Any]:
        session = self.Session()
        try:
            st = session.query(SettingsModel).filter_by(uid=uid).first()
            if not st:
                return {"temperature": 0.7, "top_k": 50, "top_p": 0.9, "max_tokens": 512}
            return {
                "temperature": st.temperature,
                "top_k": st.top_k,
                "top_p": st.top_p,
                "max_tokens": st.max_tokens
            }
        finally:
            session.close()

    def update_settings(self, uid: str, temperature: float, top_k: int, top_p: float, max_tokens: int):
        session = self.Session()
        try:
            st = session.query(SettingsModel).filter_by(uid=uid).first()
            if not st:
                st = SettingsModel(uid=uid, temperature=temperature, top_k=top_k, top_p=top_p, max_tokens=max_tokens)
                session.add(st)
            else:
                st.temperature = temperature
                st.top_k = top_k
                st.top_p = top_p
                st.max_tokens = max_tokens
            session.commit()
        finally:
            session.close()

    # -------------------------------------------------------------------------
    # Chats (Filtered strictly by uid)
    # -------------------------------------------------------------------------
    def create_chat(self, uid: str, title: str = "New Chat") -> str:
        session = self.Session()
        chat_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        try:
            chat = ChatModel(
                chat_id=chat_id,
                uid=uid,
                title=title,
                pinned=False,
                created_at=now,
                updated_at=now
            )
            session.add(chat)
            session.commit()
            return chat_id
        finally:
            session.close()

    def get_user_chats(self, uid: str, search_query: Optional[str] = None) -> List[Dict[str, Any]]:
        session = self.Session()
        try:
            q = session.query(ChatModel).filter(ChatModel.uid == uid)
            if search_query and search_query.strip():
                q = q.filter(ChatModel.title.ilike(f"%{search_query.strip()}%"))
            chats = q.order_by(ChatModel.pinned.desc(), ChatModel.updated_at.desc()).all()
            return [
                {
                    "chat_id": c.chat_id,
                    "uid": c.uid,
                    "title": c.title,
                    "pinned": c.pinned,
                    "created_at": c.created_at,
                    "updated_at": c.updated_at
                }
                for c in chats
            ]
        finally:
            session.close()

    def get_user_chats_grouped(self, uid: str, search_query: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Groups chats into: Pinned, Today, Yesterday, Previous 7 Days, Older
        """
        all_chats = self.get_user_chats(uid, search_query=search_query)
        groups = {
            "Pinned": [],
            "Today": [],
            "Yesterday": [],
            "Previous 7 Days": [],
            "Older": []
        }

        now = datetime.now()
        today_date = now.date()
        yesterday_date = today_date - timedelta(days=1)
        seven_days_ago = today_date - timedelta(days=7)

        for c in all_chats:
            if c["pinned"]:
                groups["Pinned"].append(c)
                continue

            try:
                chat_dt = datetime.fromisoformat(c["updated_at"]).date()
            except Exception:
                chat_dt = today_date

            if chat_dt == today_date:
                groups["Today"].append(c)
            elif chat_dt == yesterday_date:
                groups["Yesterday"].append(c)
            elif chat_dt >= seven_days_ago:
                groups["Previous 7 Days"].append(c)
            else:
                groups["Older"].append(c)

        return groups

    def get_chat(self, uid: str, chat_id: str) -> Optional[Dict[str, Any]]:
        session = self.Session()
        try:
            c = session.query(ChatModel).filter_by(uid=uid, chat_id=chat_id).first()
            if not c:
                return None
            return {
                "chat_id": c.chat_id,
                "uid": c.uid,
                "title": c.title,
                "pinned": c.pinned,
                "created_at": c.created_at,
                "updated_at": c.updated_at
            }
        finally:
            session.close()

    def update_chat_title(self, uid: str, chat_id: str, new_title: str):
        session = self.Session()
        now = datetime.now().isoformat()
        try:
            c = session.query(ChatModel).filter_by(uid=uid, chat_id=chat_id).first()
            if c:
                c.title = new_title
                c.updated_at = now
                session.commit()
        finally:
            session.close()

    def toggle_chat_pin(self, uid: str, chat_id: str) -> bool:
        session = self.Session()
        try:
            c = session.query(ChatModel).filter_by(uid=uid, chat_id=chat_id).first()
            if c:
                c.pinned = not c.pinned
                session.commit()
                return c.pinned
            return False
        finally:
            session.close()

    def delete_chat(self, uid: str, chat_id: str):
        session = self.Session()
        try:
            # Delete messages first
            session.query(MessageModel).filter_by(uid=uid, chat_id=chat_id).delete()
            session.query(ChatModel).filter_by(uid=uid, chat_id=chat_id).delete()
            session.commit()
        finally:
            session.close()

    # -------------------------------------------------------------------------
    # Messages (Filtered strictly by uid)
    # -------------------------------------------------------------------------
    def save_message(
        self,
        uid: str,
        chat_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        msg_id: Optional[str] = None
    ) -> str:
        session = self.Session()
        now = datetime.now().isoformat()
        msg_id = msg_id or str(uuid.uuid4())
        sources_str = json.dumps(sources) if sources else None

        try:
            msg = MessageModel(
                message_id=msg_id,
                chat_id=chat_id,
                uid=uid,
                role=role,
                content=content,
                sources_json=sources_str,
                created_at=now
            )
            session.add(msg)
            # Update chat's updated_at
            chat = session.query(ChatModel).filter_by(uid=uid, chat_id=chat_id).first()
            if chat:
                chat.updated_at = now
            session.commit()
            return msg_id
        finally:
            session.close()

    def get_chat_messages(self, uid: str, chat_id: str) -> List[Dict[str, Any]]:
        session = self.Session()
        try:
            msgs = (
                session.query(MessageModel)
                .filter_by(uid=uid, chat_id=chat_id)
                .order_by(MessageModel.created_at.asc())
                .all()
            )
            result = []
            for m in msgs:
                item = {
                    "id": m.message_id,
                    "chat_id": m.chat_id,
                    "uid": m.uid,
                    "role": m.role,
                    "content": m.content,
                    "created_at": m.created_at,
                    "sources": None
                }
                if m.sources_json:
                    try:
                        item["sources"] = json.loads(m.sources_json)
                    except Exception:
                        pass
                result.append(item)
            return result
        finally:
            session.close()

    def update_message(self, uid: str, msg_id: str, new_content: str):
        session = self.Session()
        try:
            msg = session.query(MessageModel).filter_by(uid=uid, message_id=msg_id).first()
            if msg:
                msg.content = new_content
                session.commit()
        finally:
            session.close()

    def delete_messages_after(self, uid: str, chat_id: str, msg_id: str, inclusive: bool = False):
        session = self.Session()
        try:
            target = session.query(MessageModel).filter_by(uid=uid, message_id=msg_id).first()
            if not target:
                return
            ts = target.created_at
            q = session.query(MessageModel).filter(
                MessageModel.uid == uid,
                MessageModel.chat_id == chat_id
            )
            if inclusive:
                q = q.filter(MessageModel.created_at >= ts)
            else:
                q = q.filter(MessageModel.created_at > ts)
            q.delete(synchronize_session=False)
            session.commit()
        finally:
            session.close()

    # -------------------------------------------------------------------------
    # Documents (Filtered strictly by uid)
    # -------------------------------------------------------------------------
    def register_document(
        self,
        uid: str,
        filename: str,
        file_type: str,
        file_size: int,
        num_chunks: int,
        num_pages: int = 1
    ) -> str:
        session = self.Session()
        now = datetime.now().isoformat()
        try:
            # Check existing for this user
            doc = session.query(DocumentModel).filter_by(uid=uid, filename=filename).first()
            if doc:
                doc.file_type = file_type
                doc.file_size = file_size
                doc.chunks = num_chunks
                doc.pages = num_pages
                doc.uploaded_at = now
                doc_id = doc.doc_id
            else:
                doc_id = str(uuid.uuid4())
                doc = DocumentModel(
                    doc_id=doc_id,
                    uid=uid,
                    filename=filename,
                    file_type=file_type,
                    file_size=file_size,
                    pages=num_pages,
                    chunks=num_chunks,
                    uploaded_at=now
                )
                session.add(doc)
            session.commit()
            return doc_id
        finally:
            session.close()

    def get_user_documents(self, uid: str) -> List[Dict[str, Any]]:
        session = self.Session()
        try:
            docs = (
                session.query(DocumentModel)
                .filter_by(uid=uid)
                .order_by(DocumentModel.uploaded_at.desc())
                .all()
            )
            return [
                {
                    "doc_id": d.doc_id,
                    "filename": d.filename,
                    "file_type": d.file_type,
                    "file_size": d.file_size,
                    "pages": d.pages,
                    "chunks": d.chunks,
                    "uploaded_at": d.uploaded_at
                }
                for d in docs
            ]
        finally:
            session.close()

    def delete_document(self, uid: str, filename: str):
        session = self.Session()
        try:
            session.query(DocumentModel).filter_by(uid=uid, filename=filename).delete()
            session.commit()
        finally:
            session.close()

    def get_stats(self, uid: str) -> Dict[str, int]:
        session = self.Session()
        try:
            chat_count = session.query(ChatModel).filter_by(uid=uid).count()
            msg_count = session.query(MessageModel).filter_by(uid=uid).count()
            doc_count = session.query(DocumentModel).filter_by(uid=uid).count()
            return {
                "chats": chat_count,
                "messages": msg_count,
                "documents": doc_count
            }
        finally:
            session.close()

