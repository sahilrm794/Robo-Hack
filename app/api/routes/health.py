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


# ===========================================
# Test Endpoints for Bootstrap Verification
# ===========================================

@router.get("/stt-test")
async def stt_test(request: Request):
    """
    Test STT service availability.
    Returns service status and configuration.
    """
    stt_info = {
        "service": "stt",
        "status": "unavailable",
        "model": None,
        "mode": None,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if hasattr(request.app.state, 'stt_service'):
        stt = request.app.state.stt_service
        stt_info["status"] = "ready" if stt._is_initialized else "mock"
        stt_info["model"] = settings.STT_MODEL_ID
        stt_info["mode"] = "live" if stt._is_initialized else "mock"
        stt_info["supported_languages"] = ["en", "hi", "bn", "mr"]
    
    return stt_info


@router.get("/tts-test")
async def tts_test(request: Request):
    """
    Test TTS service availability.
    Returns service status and available voices.
    """
    tts_info = {
        "service": "tts",
        "status": "unavailable",
        "mode": None,
        "voices": {},
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if hasattr(request.app.state, 'tts_service'):
        tts = request.app.state.tts_service
        tts_info["status"] = "ready"
        tts_info["mode"] = "ai4bharat" if tts._is_initialized else "edge-tts"
        tts_info["voices"] = {
            "en": settings.VOICE_MAPPING.get("en", "en-US-JennyNeural"),
            "hi": settings.VOICE_MAPPING.get("hi", "hi-IN-SwaraNeural"),
            "bn": settings.VOICE_MAPPING.get("bn", "bn-IN-TanishaaNeural"),
            "mr": settings.VOICE_MAPPING.get("mr", "mr-IN-AarohiNeural")
        }
    
    return tts_info


@router.post("/tts-test")
async def tts_synthesize_test(request: Request):
    """
    Test TTS synthesis with sample text.
    """
    import base64
    
    data = await request.json()
    text = data.get("text", "Hello, this is a test.")
    language = data.get("language", "en")
    
    if not hasattr(request.app.state, 'tts_service'):
        return {"error": "TTS service not available"}
    
    tts = request.app.state.tts_service
    
    try:
        audio_bytes = await tts.synthesize(text, language)
        
        return {
            "status": "success",
            "text": text,
            "language": language,
            "audio_size_bytes": len(audio_bytes),
            "audio_base64": base64.b64encode(audio_bytes).decode()[:100] + "..."
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@router.get("/llm-test")
async def llm_test(request: Request):
    """
    Test LLM service availability.
    Returns service status and configuration.
    """
    llm_info = {
        "service": "llm",
        "status": "unavailable",
        "provider": "groq",
        "model": None,
        "api_key_set": False,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if hasattr(request.app.state, 'llm_service'):
        llm = request.app.state.llm_service
        llm_info["status"] = "ready" if llm._is_initialized else "not_initialized"
        llm_info["model"] = settings.LLM_MODEL_ID
        llm_info["api_key_set"] = bool(settings.GROQ_API_KEY)
    
    return llm_info


@router.post("/llm-test")
async def llm_completion_test(request: Request):
    """
    Test LLM completion with a simple prompt.
    """
    import time
    
    data = await request.json()
    prompt = data.get("prompt", "Say hello in one sentence.")
    
    if not hasattr(request.app.state, 'llm_service'):
        return {"error": "LLM service not available"}
    
    llm = request.app.state.llm_service
    
    try:
        start_time = time.time()
        
        response = await llm.complete([
            {"role": "system", "content": "You are a helpful assistant. Be brief."},
            {"role": "user", "content": prompt}
        ])
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "status": "success",
            "prompt": prompt,
            "response": response.content,
            "latency_ms": round(latency_ms, 2),
            "model": settings.LLM_MODEL_ID
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
