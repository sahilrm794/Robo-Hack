"""
Session Management for Voice Agent.
Handles user sessions, conversation state, and context tracking.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from uuid import uuid4
import logging

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ConversationContext:
    """
    Mutable context that evolves during conversation.
    Tracks user intent, entities, and conversation state.
    """
    # Language context
    detected_language: str = "en"
    preferred_language: Optional[str] = None
    
    # Intent tracking
    current_intent: Optional[str] = None
    intent_confidence: float = 0.0
    
    # Entity memory - what the user is currently interested in
    last_product_id: Optional[str] = None
    last_product_name: Optional[str] = None
    last_order_id: Optional[str] = None
    last_category: Optional[str] = None
    last_search_query: Optional[str] = None
    
    # Conversation state
    awaiting_confirmation: bool = False
    pending_action: Optional[str] = None
    pending_data: Optional[Dict[str, Any]] = None
    
    # User info (if identified)
    user_phone: Optional[str] = None
    user_name: Optional[str] = None
    
    def update(self, **kwargs):
        """Update context with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            "detected_language": self.detected_language,
            "preferred_language": self.preferred_language,
            "current_intent": self.current_intent,
            "intent_confidence": self.intent_confidence,
            "last_product_id": self.last_product_id,
            "last_product_name": self.last_product_name,
            "last_order_id": self.last_order_id,
            "last_category": self.last_category,
            "last_search_query": self.last_search_query,
            "awaiting_confirmation": self.awaiting_confirmation,
            "pending_action": self.pending_action,
            "user_phone": self.user_phone,
            "user_name": self.user_name
        }
    
    def get_context_summary(self) -> str:
        """Generate a summary string for LLM context injection."""
        parts = []
        
        if self.detected_language and self.detected_language != "en":
            parts.append(f"User language: {self.detected_language}")
        
        if self.last_product_name:
            parts.append(f"Last discussed product: {self.last_product_name}")
        
        if self.last_order_id:
            parts.append(f"Last discussed order: {self.last_order_id}")
        
        if self.current_intent:
            parts.append(f"Current intent: {self.current_intent}")
        
        if self.awaiting_confirmation and self.pending_action:
            parts.append(f"Awaiting confirmation for: {self.pending_action}")
        
        if self.user_phone:
            parts.append(f"User phone: {self.user_phone}")
        
        return "\n".join(parts) if parts else "No prior context"


@dataclass
class ConversationTurn:
    """Single turn in conversation."""
    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    language: Optional[str] = None
    
    # STT metadata
    stt_confidence: Optional[float] = None
    audio_duration_ms: Optional[int] = None
    
    # Tool call metadata
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[Dict[str, Any]] = None
    
    # Timing metadata
    processing_time_ms: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert turn to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "language": self.language,
            "stt_confidence": self.stt_confidence,
            "audio_duration_ms": self.audio_duration_ms,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "tool_output": self.tool_output,
            "processing_time_ms": self.processing_time_ms
        }
    
    def to_llm_message(self) -> Dict[str, str]:
        """Convert to LLM message format."""
        return {
            "role": self.role if self.role != "tool" else "assistant",
            "content": self.content
        }


@dataclass
class Session:
    """
    User session containing conversation history and context.
    """
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    # Context and memory
    context: ConversationContext = field(default_factory=ConversationContext)
    turns: List[ConversationTurn] = field(default_factory=list)
    
    # Session metadata
    is_active: bool = True
    turn_count: int = 0
    total_audio_duration_ms: int = 0
    
    def add_turn(self, turn: ConversationTurn):
        """Add a conversation turn."""
        self.turns.append(turn)
        self.turn_count += 1
        self.last_activity = datetime.now()
        
        if turn.audio_duration_ms:
            self.total_audio_duration_ms += turn.audio_duration_ms
        
        # Trim old turns if exceeding max
        max_turns = settings.MAX_CONVERSATION_TURNS
        if len(self.turns) > max_turns:
            self.turns = self.turns[-max_turns:]
    
    def get_recent_turns(self, n: Optional[int] = None) -> List[ConversationTurn]:
        """Get the n most recent turns."""
        n = n or settings.CONTEXT_WINDOW_SIZE
        return self.turns[-n:] if self.turns else []
    
    def get_llm_messages(self, n: Optional[int] = None) -> List[Dict[str, str]]:
        """Get turns formatted for LLM input."""
        recent = self.get_recent_turns(n)
        return [turn.to_llm_message() for turn in recent]
    
    def is_expired(self) -> bool:
        """Check if session has expired due to inactivity."""
        timeout = timedelta(minutes=settings.SESSION_TIMEOUT_MINUTES)
        return datetime.now() - self.last_activity > timeout
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "context": self.context.to_dict(),
            "turn_count": self.turn_count,
            "total_audio_duration_ms": self.total_audio_duration_ms,
            "is_active": self.is_active
        }


class SessionManager:
    """
    Manages user sessions with automatic cleanup.
    Thread-safe session storage and retrieval.
    """
    
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the session manager and cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Session manager started")
    
    async def stop(self):
        """Stop the session manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Session manager stopped")
    
    async def create_session(self, session_id: Optional[str] = None) -> Session:
        """Create a new session."""
        async with self._lock:
            # Check session limit
            if len(self._sessions) >= settings.MAX_SESSIONS:
                # Remove oldest inactive session
                await self._evict_oldest()
            
            session_id = session_id or str(uuid4())
            session = Session(session_id=session_id)
            self._sessions[session_id] = session
            
            logger.info(f"Created new session: {session_id}")
            return session
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get an existing session."""
        async with self._lock:
            session = self._sessions.get(session_id)
            
            if session and session.is_expired():
                await self._remove_session(session_id)
                return None
            
            return session
    
    async def get_or_create_session(self, session_id: Optional[str] = None) -> Session:
        """Get existing session or create new one."""
        if session_id:
            session = await self.get_session(session_id)
            if session:
                return session
        
        return await self.create_session(session_id)
    
    async def update_session(self, session: Session):
        """Update session in storage."""
        async with self._lock:
            session.last_activity = datetime.now()
            self._sessions[session.session_id] = session
    
    async def delete_session(self, session_id: str):
        """Delete a session."""
        async with self._lock:
            await self._remove_session(session_id)
    
    async def get_active_session_count(self) -> int:
        """Get count of active sessions."""
        async with self._lock:
            return sum(1 for s in self._sessions.values() if s.is_active and not s.is_expired())
    
    async def _remove_session(self, session_id: str):
        """Remove session (must be called with lock held)."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Removed session: {session_id}")
    
    async def _evict_oldest(self):
        """Evict oldest inactive session (must be called with lock held)."""
        if not self._sessions:
            return
        
        # Find oldest inactive session
        oldest_session = min(
            self._sessions.values(),
            key=lambda s: s.last_activity
        )
        
        await self._remove_session(oldest_session.session_id)
    
    async def _cleanup_loop(self):
        """Periodically clean up expired sessions."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                async with self._lock:
                    expired = [
                        sid for sid, session in self._sessions.items()
                        if session.is_expired()
                    ]
                    
                    for sid in expired:
                        await self._remove_session(sid)
                    
                    if expired:
                        logger.info(f"Cleaned up {len(expired)} expired sessions")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
