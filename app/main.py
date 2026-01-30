"""
FastAPI Application Entry Point
===============================
Main application with lifecycle management, middleware, and route mounting.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.core.exceptions import VoiceAgentException
from app.api.routes import voice, conversation, health
from app.db.database import init_db, close_db
from app.services.stt import STTService
from app.services.tts import TTSService
from app.services.llm import LLMService
from app.services.memory import ConversationMemoryService
from app.logging.agent_logger import AgentLogger
from app.tools.registry import ToolRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    logger.info("=" * 60)
    logger.info("Starting Voice Agent Backend")
    logger.info("=" * 60)
    
    # ==================
    # STARTUP
    # ==================
    
    # Ensure directories exist
    Path("./logs").mkdir(parents=True, exist_ok=True)
    Path("./data").mkdir(parents=True, exist_ok=True)
    Path("./models/stt").mkdir(parents=True, exist_ok=True)
    Path("./models/tts").mkdir(parents=True, exist_ok=True)
    
    # Initialize agent logger
    logger.info("Initializing agent logger...")
    app.state.agent_logger = AgentLogger(str(settings.AGENT_LOG_PATH))
    await app.state.agent_logger.log_system_event("Application starting", {
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT
    })
    
    # Initialize database
    logger.info("Initializing database...")
    await init_db()
    
    # Initialize services
    logger.info("Initializing STT service...")
    app.state.stt_service = STTService()
    await app.state.stt_service.initialize()
    
    logger.info("Initializing TTS service...")
    app.state.tts_service = TTSService()
    await app.state.tts_service.initialize()
    
    logger.info("Initializing LLM service...")
    app.state.llm_service = LLMService()
    await app.state.llm_service.initialize()
    
    logger.info("Initializing conversation memory...")
    app.state.memory_service = ConversationMemoryService()
    
    logger.info("Initializing tool registry...")
    app.state.tool_registry = ToolRegistry()
    await app.state.tool_registry.initialize()
    
    logger.info("=" * 60)
    logger.info("Voice Agent Backend Ready!")
    logger.info(f"Server: http://{settings.HOST}:{settings.PORT}")
    logger.info(f"Docs: http://{settings.HOST}:{settings.PORT}/docs")
    logger.info("=" * 60)
    
    await app.state.agent_logger.log_system_event("Application started successfully", {
        "host": settings.HOST,
        "port": settings.PORT
    })
    
    yield  # Application runs here
    
    # ==================
    # SHUTDOWN
    # ==================
    
    logger.info("Shutting down Voice Agent Backend...")
    
    await app.state.agent_logger.log_system_event("Application shutting down", {})
    
    # Cleanup services
    if hasattr(app.state, 'stt_service'):
        await app.state.stt_service.cleanup()
    if hasattr(app.state, 'tts_service'):
        await app.state.tts_service.cleanup()
    if hasattr(app.state, 'llm_service'):
        await app.state.llm_service.cleanup()
    if hasattr(app.state, 'agent_logger'):
        await app.state.agent_logger.close()
    
    # Close database
    await close_db()
    
    logger.info("Shutdown complete.")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    ## Voice-to-Voice AI Customer Support Agent
    
    A low-latency, multilingual voice AI backend for e-commerce customer support.
    
    ### Features:
    - 🎤 Real-time voice conversation via WebSocket
    - 🔄 Multi-turn context awareness
    - 🛍️ Product discovery, FAQ lookup, order tracking
    - 🌐 Supports Hindi, Bengali, Marathi, English
    - ⚡ Sub-second response latency
    
    ### Pipeline:
    ```
    Audio → STT (IndicConformer) → LLM (Groq) → TTS (Indic TTS) → Audio
    ```
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ==================
# MIDDLEWARE
# ==================

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add request timing information to response headers."""
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds() * 1000
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests."""
    logger.debug(f"{request.method} {request.url.path}")
    response = await call_next(request)
    return response


# ==================
# EXCEPTION HANDLERS
# ==================

@app.exception_handler(VoiceAgentException)
async def voice_agent_exception_handler(request: Request, exc: VoiceAgentException):
    """Handle custom Voice Agent exceptions."""
    logger.error(f"VoiceAgentException: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.exception(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "details": str(exc) if settings.DEBUG else None
        }
    )


# ==================
# ROUTES
# ==================

# Mount route modules
app.include_router(health.router, tags=["Health"])
app.include_router(voice.router, prefix="/api/v1/voice", tags=["Voice"])
app.include_router(conversation.router, prefix="/api/v1/conversation", tags=["Conversation"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


# ==================
# DEBUG ENDPOINTS
# ==================

if settings.DEBUG:
    @app.get("/debug/config")
    async def debug_config():
        """Debug endpoint to view configuration (DEBUG mode only)."""
        return {
            "environment": settings.ENVIRONMENT,
            "supported_languages": settings.SUPPORTED_LANGUAGES,
            "stt_model": settings.STT_MODEL_ID,
            "tts_model": settings.TTS_MODEL_ID,
            "llm_model": settings.LLM_MODEL_ID
        }
