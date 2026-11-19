"""
Per-User Chat Coordination and Streaming Generation for LocalGPT Pro.
"""

from typing import List, Dict, Any, Optional, Tuple, Generator
from memory import ConversationMemory
from rag import RAGPipeline
from model import LocalLLMEngine

class ChatCoordinator:
    def __init__(
        self,
        engine: LocalLLMEngine,
        memory: ConversationMemory,
        rag: RAGPipeline
    ):
        self.engine = engine
        self.memory = memory
        self.rag = rag

    def generate_response_stream(
        self,
        user_input: str,
        enable_rag: bool = True,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        max_new_tokens: int = 512
    ) -> Tuple[Generator[str, None, None], List[Dict[str, Any]], str]:
        user_msg_id = self.memory.add_user_message(user_input)

        sources = []
        context_str = ""

        if enable_rag and self.rag.vector_store.total_chunks > 0:
            context_str, sources = self.rag.retrieve_context(user_input, top_k=4)

        history_messages = self.memory.get_messages_for_llm(max_turns=8)
        
        if context_str:
            augmented_query = self.rag.build_rag_prompt(user_input, context_str)
            if history_messages and history_messages[-1]["role"] == "user":
                history_messages[-1]["content"] = augmented_query
        else:
            augmented_query = user_input

        stream_gen = self.engine.stream_generate(
            messages=history_messages,
            system_prompt=self.memory.system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        return stream_gen, sources, augmented_query

    def save_assistant_response(
        self,
        response_text: str,
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        return self.memory.add_assistant_message(
            content=response_text,
            sources=sources
        )

    def regenerate_last_response(
        self,
        enable_rag: bool = True,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        max_new_tokens: int = 512
    ) -> Optional[Tuple[Generator[str, None, None], List[Dict[str, Any]], str]]:
        self.memory.remove_last_turn()
        
        last_user_query = None
        for m in reversed(self.memory.messages):
            if m["role"] == "user":
                last_user_query = m["content"]
                break

        if not last_user_query:
            return None

        sources = []
        context_str = ""
        if enable_rag and self.rag.vector_store.total_chunks > 0:
            context_str, sources = self.rag.retrieve_context(last_user_query, top_k=4)

        history_messages = self.memory.get_messages_for_llm(max_turns=8)
        if context_str and history_messages and history_messages[-1]["role"] == "user":
            augmented_query = self.rag.build_rag_prompt(last_user_query, context_str)
            history_messages[-1]["content"] = augmented_query
        else:
            augmented_query = last_user_query

        stream_gen = self.engine.stream_generate(
            messages=history_messages,
            system_prompt=self.memory.system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        return stream_gen, sources, augmented_query

    def edit_user_message_and_respond(
        self,
        msg_id: str,
        new_content: str,
        enable_rag: bool = True,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        max_new_tokens: int = 512
    ) -> Tuple[Generator[str, None, None], List[Dict[str, Any]], str]:
        self.memory.edit_user_message_and_truncate(msg_id, new_content)

        sources = []
        context_str = ""
        if enable_rag and self.rag.vector_store.total_chunks > 0:
            context_str, sources = self.rag.retrieve_context(new_content, top_k=4)

        history_messages = self.memory.get_messages_for_llm(max_turns=8)
        if context_str and history_messages and history_messages[-1]["role"] == "user":
            augmented_query = self.rag.build_rag_prompt(new_content, context_str)
            history_messages[-1]["content"] = augmented_query
        else:
            augmented_query = new_content

        stream_gen = self.engine.stream_generate(
            messages=history_messages,
            system_prompt=self.memory.system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        return stream_gen, sources, augmented_query

