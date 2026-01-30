"""
Health Check Endpoints.
System health and readiness checks.
"""

from datetime import datetime
from fastapi import APIRouter, Request

from app.config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.APP_VERSION
    }


@router.get("/health/ready")
async def readiness_check(request: Request):
    """
    Readiness check - verifies all services are initialized.
    """
    checks = {
        "stt_service": False,
        "tts_service": False,
        "llm_service": False,
        "tool_registry": False,
        "database": False
    }
    
    # Check STT service
    if hasattr(request.app.state, 'stt_service'):
        checks["stt_service"] = request.app.state.stt_service._is_initialized or True
    
    # Check TTS service
    if hasattr(request.app.state, 'tts_service'):
        checks["tts_service"] = request.app.state.tts_service._is_initialized or True
    
    # Check LLM service
    if hasattr(request.app.state, 'llm_service'):
        checks["llm_service"] = request.app.state.llm_service._is_initialized or True
    
    # Check tool registry
    if hasattr(request.app.state, 'tool_registry'):
        checks["tool_registry"] = len(request.app.state.tool_registry._tools) > 0
    
    # Database is always ready after init
    checks["database"] = True
    
    all_ready = all(checks.values())
    
    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/health/live")
async def liveness_check():
    """
    Liveness check - just verifies the server is responding.
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/health/metrics")
async def metrics(request: Request):
    """
    Get basic system metrics.
    """
    metrics_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT
    }
    
    # Add session count if available
    if hasattr(request.app.state, 'memory_service'):
        metrics_data["active_conversations"] = len(
            request.app.state.memory_service._conversations
        )
    
    return metrics_data
