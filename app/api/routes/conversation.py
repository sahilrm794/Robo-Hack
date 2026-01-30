"""
Conversation REST Endpoints.
Manage conversations and access conversation data.
"""

import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


class ConversationMessage(BaseModel):
    """Request model for sending a message."""
    text: str
    language: Optional[str] = "en"
    session_id: Optional[str] = None


class ConversationResponse(BaseModel):
    """Response model for conversation."""
    session_id: str
    response: str
    language: str
    latency_ms: float
    tool_calls: list = []


@router.post("/message", response_model=ConversationResponse)
async def send_message(
    request: Request,
    message: ConversationMessage
):
    """
    Send a text message and get a response.
    This is the main text-based conversation endpoint.
    """
    import time
    from app.core.session import SessionManager, ConversationTurn
    from app.core.pipeline import PipelineOrchestrator
    
    start_time = time.time()
    
    # Get services
    app = request.app
    llm_service = app.state.llm_service
    tool_registry = app.state.tool_registry
    agent_logger = app.state.agent_logger
    memory_service = app.state.memory_service
    
    # Get or create session
    session_manager = SessionManager()
    session = await session_manager.get_or_create_session(message.session_id)
    session.context.detected_language = message.language
    
    # Build pipeline
    pipeline = PipelineOrchestrator(
        stt_service=app.state.stt_service,
        llm_service=llm_service,
        tts_service=app.state.tts_service,
        tool_registry=tool_registry,
        agent_logger=agent_logger
    )
    
    # Build LLM messages
    messages = pipeline._build_llm_messages(session, message.text)
    
    # Get LLM response
    response = await llm_service.complete(
        messages,
        tools=tool_registry.get_tool_schemas()
    )
    
    # Handle tool calls
    tool_results = []
    if response.tool_calls:
        for tc in response.tool_calls:
            result = await tool_registry.execute(
                tc["name"],
                tc["arguments"],
                session
            )
            tool_results.append({
                "tool": tc["name"],
                "input": tc["arguments"],
                "output": result
            })
        
        # Get final response
        response = await llm_service.complete_with_tool_results(
            messages,
            response.tool_calls,
            [{"output": r["output"]} for r in tool_results]
        )
    
    latency_ms = (time.time() - start_time) * 1000
    
    # Update session
    session.add_turn(ConversationTurn(
        role="user",
        content=message.text,
        language=message.language
    ))
    session.add_turn(ConversationTurn(
        role="assistant",
        content=response.content,
        language=message.language,
        processing_time_ms=int(latency_ms)
    ))
    
    # Log the turn
    await agent_logger.log_turn_complete(
        session.session_id,
        message.text,
        response.content,
        message.language,
        tool_results,
        {"total_latency_ms": latency_ms, "llm_latency_ms": latency_ms}
    )
    
    return ConversationResponse(
        session_id=session.session_id,
        response=response.content,
        language=message.language,
        latency_ms=round(latency_ms, 2),
        tool_calls=tool_results
    )


@router.get("/history/{session_id}")
async def get_history(
    request: Request,
    session_id: str,
    limit: int = 10
):
    """Get conversation history for a session."""
    memory_service = request.app.state.memory_service
    
    history = await memory_service.get_history(session_id, n=limit)
    
    if not history:
        return {
            "session_id": session_id,
            "turns": [],
            "message": "No conversation history found"
        }
    
    return {
        "session_id": session_id,
        "turns": [turn.to_dict() for turn in history],
        "count": len(history)
    }


@router.get("/context/{session_id}")
async def get_context(
    request: Request,
    session_id: str
):
    """Get conversation context for a session."""
    memory_service = request.app.state.memory_service
    
    context = await memory_service.get_context(session_id)
    
    return {
        "session_id": session_id,
        "context": context.to_dict()
    }


@router.post("/context/{session_id}")
async def update_context(
    request: Request,
    session_id: str
):
    """Update conversation context."""
    data = await request.json()
    memory_service = request.app.state.memory_service
    
    await memory_service.update_context(session_id, **data)
    context = await memory_service.get_context(session_id)
    
    return {
        "session_id": session_id,
        "context": context.to_dict(),
        "status": "updated"
    }


@router.delete("/session/{session_id}")
async def clear_session(
    request: Request,
    session_id: str
):
    """Clear all data for a session."""
    memory_service = request.app.state.memory_service
    
    await memory_service.clear_session(session_id)
    
    return {
        "session_id": session_id,
        "status": "cleared"
    }


@router.get("/stats/{session_id}")
async def get_session_stats(
    request: Request,
    session_id: str
):
    """Get statistics for a session."""
    memory_service = request.app.state.memory_service
    
    stats = await memory_service.get_session_stats(session_id)
    
    return stats


@router.get("/active")
async def get_active_sessions(request: Request):
    """Get count of active sessions."""
    memory_service = request.app.state.memory_service
    
    return {
        "active_sessions": len(memory_service._conversations),
        "timestamp": datetime.utcnow().isoformat()
    }
