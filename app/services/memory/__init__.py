"""
Conversation Memory Service.
Manages conversation history and context persistence.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

from app.config import get_settings
from app.core.session import Session, ConversationTurn, ConversationContext

logger = logging.getLogger(__name__)
settings = get_settings()


class ConversationMemoryService:
    """
    Service for managing conversation memory and context.
    
    Features:
    - In-memory storage for active sessions
    - Persistence for long-term context
    - Context summarization for LLM injection
    """
    
    def __init__(self):
        self._conversations: Dict[str, List[ConversationTurn]] = {}
        self._contexts: Dict[str, ConversationContext] = {}
    
    async def get_history(
        self,
        session_id: str,
        n: Optional[int] = None
    ) -> List[ConversationTurn]:
        """Get conversation history for a session."""
        history = self._conversations.get(session_id, [])
        
        if n is not None:
            return history[-n:]
        
        return history
    
    async def add_turn(
        self,
        session_id: str,
        turn: ConversationTurn
    ):
        """Add a turn to conversation history."""
        if session_id not in self._conversations:
            self._conversations[session_id] = []
        
        self._conversations[session_id].append(turn)
        
        # Trim if exceeding max
        max_turns = settings.MAX_CONVERSATION_TURNS
        if len(self._conversations[session_id]) > max_turns:
            self._conversations[session_id] = self._conversations[session_id][-max_turns:]
    
    async def get_context(
        self,
        session_id: str
    ) -> ConversationContext:
        """Get context for a session."""
        if session_id not in self._contexts:
            self._contexts[session_id] = ConversationContext()
        
        return self._contexts[session_id]
    
    async def update_context(
        self,
        session_id: str,
        **kwargs
    ):
        """Update context for a session."""
        if session_id not in self._contexts:
            self._contexts[session_id] = ConversationContext()
        
        self._contexts[session_id].update(**kwargs)
    
    async def clear_session(
        self,
        session_id: str
    ):
        """Clear all data for a session."""
        if session_id in self._conversations:
            del self._conversations[session_id]
        
        if session_id in self._contexts:
            del self._contexts[session_id]
    
    async def build_context_prompt(
        self,
        session_id: str,
        n_turns: Optional[int] = None
    ) -> str:
        """
        Build a context string for LLM prompt injection.
        Includes recent history and current context state.
        """
        n_turns = n_turns or settings.CONTEXT_WINDOW_SIZE
        
        history = await self.get_history(session_id, n_turns)
        context = await self.get_context(session_id)
        
        parts = []
        
        # Add context summary
        context_summary = context.get_context_summary()
        if context_summary != "No prior context":
            parts.append(f"CURRENT CONTEXT:\n{context_summary}")
        
        # Add conversation history
        if history:
            parts.append("\nCONVERSATION HISTORY:")
            for turn in history[-n_turns:]:
                role = turn.role.upper()
                parts.append(f"[{role}]: {turn.content}")
        
        return "\n".join(parts) if parts else ""
    
    async def extract_entities(
        self,
        text: str,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Extract entities from text and update context.
        Simple rule-based extraction for hackathon.
        """
        entities = {}
        context = await self.get_context(session_id)
        
        # Extract order ID patterns (e.g., ORD-12345, #12345)
        import re
        
        order_patterns = [
            r'ORD[-_]?\d{4,}',
            r'#\d{4,}',
            r'order\s*(?:number|no|id)?\s*[:#]?\s*(\d{4,})'
        ]
        
        for pattern in order_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                order_id = match.group(0)
                entities["order_id"] = order_id
                await self.update_context(session_id, last_order_id=order_id)
                break
        
        # Extract phone numbers
        phone_pattern = r'\b[6-9]\d{9}\b'
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            entities["phone"] = phone_match.group(0)
            await self.update_context(session_id, user_phone=phone_match.group(0))
        
        # Extract categories
        categories = ["electronics", "clothing", "home", "beauty", "sports", "books"]
        text_lower = text.lower()
        for cat in categories:
            if cat in text_lower:
                entities["category"] = cat
                await self.update_context(session_id, last_category=cat)
                break
        
        return entities
    
    async def get_session_stats(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """Get statistics for a session."""
        history = await self.get_history(session_id)
        context = await self.get_context(session_id)
        
        return {
            "session_id": session_id,
            "turn_count": len(history),
            "languages_used": list(set(t.language for t in history if t.language)),
            "context": context.to_dict()
        }
