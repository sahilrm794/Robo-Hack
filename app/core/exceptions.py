"""
Core exceptions for the Voice Agent.
Custom exception classes for structured error handling.
"""

from typing import Optional, Dict, Any


class VoiceAgentException(Exception):
    """Base exception for Voice Agent errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "VOICE_AGENT_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


# =========================
# STT Exceptions
# =========================

class STTException(VoiceAgentException):
    """Base exception for STT errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="STT_ERROR",
            status_code=500,
            details=details
        )


class STTModelNotLoadedException(STTException):
    """Raised when STT model is not loaded."""
    
    def __init__(self):
        super().__init__(
            message="STT model is not loaded. Please wait for initialization.",
            details={"error_type": "model_not_loaded"}
        )


class STTLowConfidenceException(STTException):
    """Raised when STT confidence is too low."""
    
    def __init__(self, confidence: float, threshold: float):
        super().__init__(
            message=f"STT confidence ({confidence:.2f}) below threshold ({threshold:.2f})",
            details={"confidence": confidence, "threshold": threshold}
        )


class STTTimeoutException(STTException):
    """Raised when STT processing times out."""
    
    def __init__(self, timeout_seconds: float):
        super().__init__(
            message=f"STT processing timed out after {timeout_seconds} seconds",
            details={"timeout_seconds": timeout_seconds}
        )


class STTNoAudioException(STTException):
    """Raised when no audio is detected."""
    
    def __init__(self):
        super().__init__(
            message="No audio detected in input",
            details={"error_type": "no_audio"}
        )


# =========================
# TTS Exceptions
# =========================

class TTSException(VoiceAgentException):
    """Base exception for TTS errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="TTS_ERROR",
            status_code=500,
            details=details
        )


class TTSModelNotLoadedException(TTSException):
    """Raised when TTS model is not loaded."""
    
    def __init__(self):
        super().__init__(
            message="TTS model is not loaded. Please wait for initialization.",
            details={"error_type": "model_not_loaded"}
        )


class TTSUnsupportedLanguageException(TTSException):
    """Raised when TTS language is not supported."""
    
    def __init__(self, language: str, supported: list):
        super().__init__(
            message=f"Language '{language}' is not supported for TTS",
            details={"language": language, "supported_languages": supported}
        )


# =========================
# LLM Exceptions
# =========================

class LLMException(VoiceAgentException):
    """Base exception for LLM errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="LLM_ERROR",
            status_code=500,
            details=details
        )


class LLMAPIException(LLMException):
    """Raised when Groq API returns an error."""
    
    def __init__(self, api_error: str, status_code: int = 500):
        super().__init__(
            message=f"LLM API error: {api_error}",
            details={"api_error": api_error, "api_status_code": status_code}
        )


class LLMTimeoutException(LLMException):
    """Raised when LLM processing times out."""
    
    def __init__(self, timeout_seconds: float):
        super().__init__(
            message=f"LLM processing timed out after {timeout_seconds} seconds",
            details={"timeout_seconds": timeout_seconds}
        )


class LLMRateLimitException(LLMException):
    """Raised when LLM API rate limit is exceeded."""
    
    def __init__(self, retry_after: Optional[float] = None):
        super().__init__(
            message="LLM API rate limit exceeded",
            details={"retry_after_seconds": retry_after}
        )


# =========================
# Tool Exceptions
# =========================

class ToolException(VoiceAgentException):
    """Base exception for tool execution errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="TOOL_ERROR",
            status_code=500,
            details=details
        )


class ToolNotFoundException(ToolException):
    """Raised when requested tool is not found."""
    
    def __init__(self, tool_name: str):
        super().__init__(
            message=f"Tool '{tool_name}' not found in registry",
            details={"tool_name": tool_name}
        )


class ToolExecutionException(ToolException):
    """Raised when tool execution fails."""
    
    def __init__(self, tool_name: str, error: str):
        super().__init__(
            message=f"Tool '{tool_name}' execution failed: {error}",
            details={"tool_name": tool_name, "error": error}
        )


class ToolValidationException(ToolException):
    """Raised when tool input validation fails."""
    
    def __init__(self, tool_name: str, validation_errors: list):
        super().__init__(
            message=f"Tool '{tool_name}' input validation failed",
            details={"tool_name": tool_name, "validation_errors": validation_errors}
        )


# =========================
# Session Exceptions
# =========================

class SessionException(VoiceAgentException):
    """Base exception for session errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="SESSION_ERROR",
            status_code=400,
            details=details
        )


class SessionNotFoundException(SessionException):
    """Raised when session is not found."""
    
    def __init__(self, session_id: str):
        super().__init__(
            message=f"Session '{session_id}' not found",
            details={"session_id": session_id}
        )


class SessionExpiredException(SessionException):
    """Raised when session has expired."""
    
    def __init__(self, session_id: str):
        super().__init__(
            message=f"Session '{session_id}' has expired",
            details={"session_id": session_id}
        )


class SessionLimitException(SessionException):
    """Raised when maximum session limit is reached."""
    
    def __init__(self, max_sessions: int):
        super().__init__(
            message=f"Maximum number of sessions ({max_sessions}) reached",
            details={"max_sessions": max_sessions}
        )


# =========================
# Database Exceptions
# =========================

class DatabaseException(VoiceAgentException):
    """Base exception for database errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            status_code=500,
            details=details
        )


class RecordNotFoundException(DatabaseException):
    """Raised when a database record is not found."""
    
    def __init__(self, entity: str, identifier: str):
        super().__init__(
            message=f"{entity} with identifier '{identifier}' not found",
            details={"entity": entity, "identifier": identifier}
        )


# =========================
# Pipeline Exceptions
# =========================

class PipelineException(VoiceAgentException):
    """Base exception for pipeline errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="PIPELINE_ERROR",
            status_code=500,
            details=details
        )


class PipelineStageException(PipelineException):
    """Raised when a pipeline stage fails."""
    
    def __init__(self, stage: str, error: str):
        super().__init__(
            message=f"Pipeline stage '{stage}' failed: {error}",
            details={"stage": stage, "error": error}
        )
