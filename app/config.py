"""
Configuration management for the Voice Agent.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # =========================
    # Application Settings
    # =========================
    APP_NAME: str = "Voice Agent"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    ENVIRONMENT: str = Field(default="development", description="Environment name")
    
    # =========================
    # API Keys
    # =========================
    GROQ_API_KEY: str = Field(..., description="Groq API key for LLM inference")
    HF_TOKEN: Optional[str] = Field(default=None, description="HuggingFace token for model access")
    
    # =========================
    # Server Settings
    # =========================
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    WORKERS: int = Field(default=1, description="Number of workers")
    
    # =========================
    # Database Settings
    # =========================
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./data/voice_agent.db",
        description="Database connection URL"
    )
    DATABASE_ECHO: bool = Field(default=False, description="Echo SQL queries")
    
    # =========================
    # Model Settings
    # =========================
    STT_MODEL_ID: str = Field(
        default="ai4bharat/indicconformer_stt-hi-hybrid_ctc_rnnt-13M",
        description="STT model identifier"
    )
    TTS_MODEL_ID: str = Field(
        default="ai4bharat/indic-tts-coqui-indo_aryan-gpu",
        description="TTS model identifier"
    )
    LLM_MODEL_ID: str = Field(
        default="llama-3.1-70b-versatile",
        description="Groq LLM model to use"
    )
    
    # Model paths
    MODELS_DIR: Path = Field(default=Path("./models"), description="Directory for downloaded models")
    STT_MODEL_PATH: Optional[Path] = Field(default=None, description="Local STT model path")
    TTS_MODEL_PATH: Optional[Path] = Field(default=None, description="Local TTS model path")
    
    # =========================
    # Audio Settings
    # =========================
    AUDIO_SAMPLE_RATE: int = Field(default=16000, description="Audio sample rate in Hz")
    AUDIO_CHANNELS: int = Field(default=1, description="Number of audio channels")
    AUDIO_CHUNK_SIZE: int = Field(default=1600, description="Audio chunk size (100ms at 16kHz)")
    
    # =========================
    # Pipeline Settings
    # =========================
    STT_CONFIDENCE_THRESHOLD: float = Field(
        default=0.6, 
        description="Minimum STT confidence to accept transcription"
    )
    MAX_CONVERSATION_TURNS: int = Field(
        default=10, 
        description="Maximum conversation turns to keep in memory"
    )
    CONTEXT_WINDOW_SIZE: int = Field(
        default=5, 
        description="Number of recent turns to include in LLM context"
    )
    
    # =========================
    # Latency Settings
    # =========================
    LLM_TIMEOUT_SECONDS: float = Field(default=5.0, description="LLM API timeout")
    STT_TIMEOUT_SECONDS: float = Field(default=10.0, description="STT processing timeout")
    TTS_TIMEOUT_SECONDS: float = Field(default=10.0, description="TTS processing timeout")
    
    # =========================
    # Session Settings
    # =========================
    SESSION_TIMEOUT_MINUTES: int = Field(default=30, description="Session idle timeout")
    MAX_SESSIONS: int = Field(default=100, description="Maximum concurrent sessions")
    
    # =========================
    # Logging Settings
    # =========================
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    AGENT_LOG_PATH: Path = Field(
        default=Path("./logs/agent_log.md"), 
        description="Path to agent markdown log"
    )
    APP_LOG_PATH: Path = Field(
        default=Path("./logs/app.log"), 
        description="Path to application log"
    )
    
    # =========================
    # Supported Languages
    # =========================
    SUPPORTED_LANGUAGES: List[str] = Field(
        default=["en", "hi", "bn", "mr"],
        description="Supported language codes"
    )
    DEFAULT_LANGUAGE: str = Field(default="en", description="Default language")
    
    # =========================
    # Caching Settings
    # =========================
    ENABLE_CACHE: bool = Field(default=True, description="Enable result caching")
    CACHE_TTL_SECONDS: int = Field(default=300, description="Cache TTL in seconds")
    CACHE_MAX_SIZE: int = Field(default=1000, description="Maximum cache entries")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Language mapping for display names
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi (हिन्दी)",
    "bn": "Bengali (বাংলা)",
    "mr": "Marathi (मराठी)"
}

# Language to voice mapping for TTS
LANGUAGE_VOICES = {
    "en": "english_female",
    "hi": "hindi_female",
    "bn": "bengali_female",
    "mr": "marathi_female"
}

# Category definitions
PRODUCT_CATEGORIES = [
    "electronics",
    "clothing",
    "home",
    "beauty",
    "sports",
    "books",
    "toys",
    "grocery"
]

# Order status definitions
ORDER_STATUSES = [
    "pending",
    "confirmed",
    "processing",
    "shipped",
    "out_for_delivery",
    "delivered",
    "cancelled",
    "returned"
]

# Return reasons
RETURN_REASONS = [
    "defective",
    "wrong_item",
    "not_as_described",
    "changed_mind",
    "size_issue",
    "quality_issue",
    "late_delivery"
]

# Policy types
POLICY_TYPES = [
    "returns",
    "refunds",
    "shipping",
    "privacy",
    "warranty",
    "cancellation"
]
