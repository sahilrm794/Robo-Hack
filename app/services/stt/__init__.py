"""
Speech-to-Text Service using AI4Bharat IndicConformer.
Supports streaming transcription for low-latency processing.
"""

import asyncio
import io
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional, List
import numpy as np

from app.config import get_settings
from app.core.exceptions import (
    STTException,
    STTModelNotLoadedException,
    STTNoAudioException,
    STTTimeoutException
)

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class STTResult:
    """Result from speech-to-text transcription."""
    text: str
    language: str
    confidence: float
    audio_duration_ms: int
    is_final: bool = True
    alternatives: Optional[List[str]] = None
    processing_time_ms: Optional[float] = None


class STTService:
    """
    Speech-to-Text service using AI4Bharat IndicConformer.
    
    Supports:
    - Streaming transcription for low latency
    - 22 Indian languages including Hindi, Bengali, Marathi, Tamil, Telugu, etc.
    - CTC and RNNT decoding strategies
    - Language detection
    """
    
    def __init__(self):
        self._model = None
        self._is_initialized = False
        self._device = "cuda"  # Will fallback to CPU if needed
        
        # All 22 supported Indian languages
        self._supported_languages = [
            "as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks",
            "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat",
            "sd", "ta", "te", "ur", "en"
        ]
        
        # Default decoding strategy
        self._decoder = "ctc"  # Options: "ctc" or "rnnt"
    
    async def initialize(self):
        """Initialize STT models. Load in background to not block startup."""
        try:
            logger.info("Initializing STT service...")
            
            # Run model loading in thread pool to not block
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_models)
            
            self._is_initialized = True
            logger.info("STT service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize STT service: {e}")
            # Don't raise - allow app to start, will use fallback
            self._is_initialized = False
    
    def _load_models(self):
        """Load AI4Bharat IndicConformer model (runs in thread pool)."""
        try:
            import torch
            import torchaudio
            from transformers import AutoModel
            
            # Check device availability
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, using CPU for STT")
                self._device = "cpu"
            
            model_id = settings.STT_MODEL_ID
            logger.info(f"Loading STT model: {model_id}")
            
            # Load AI4Bharat IndicConformer with trust_remote_code
            self._model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                token=settings.HF_TOKEN
            )
            
            # Note: IndicConformer handles device placement internally
            logger.info(f"STT model loaded on device: {self._device}")
            
            # Warm up with dummy inference
            self._warmup()
            
            logger.info("STT models loaded successfully")
            
        except ImportError as e:
            logger.warning(f"STT dependencies not installed: {e}")
            logger.info("Using mock STT service for development")
        except Exception as e:
            logger.error(f"Failed to load STT models: {e}")
            raise
    
    def _warmup(self):
        """Warm up model with dummy inference."""
        try:
            import torch
            
            # Generate dummy audio (1 second of silence)
            dummy_audio = torch.zeros(1, 16000, dtype=torch.float32)
            
            # Run warmup transcription with Hindi
            _ = self._model(dummy_audio, "hi", self._decoder)
            
            logger.info("STT model warmed up")
            
        except Exception as e:
            logger.warning(f"STT warmup failed: {e}")
    
    async def transcribe(
        self,
        audio_data: bytes,
        language_hint: Optional[str] = None
    ) -> STTResult:
        """
        Transcribe complete audio data.
        
        Args:
            audio_data: Raw audio bytes (16kHz, 16-bit, mono PCM)
            language_hint: Optional language hint for better accuracy
        
        Returns:
            STTResult with transcription and metadata
        """
        if not self._is_initialized:
            # Fallback for development/testing
            return await self._mock_transcribe(audio_data, language_hint)
        
        start_time = time.time()
        
        try:
            # Convert bytes to numpy array
            audio_array = self._bytes_to_array(audio_data)
            
            if len(audio_array) < 1600:  # Less than 100ms
                raise STTNoAudioException()
            
            # Run inference in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._transcribe_sync,
                audio_array,
                language_hint
            )
            
            result.processing_time_ms = (time.time() - start_time) * 1000
            return result
            
        except STTException:
            raise
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise STTException(f"Transcription failed: {e}")
    
    async def transcribe_streaming(
        self,
        audio_chunks: AsyncIterator[bytes],
        language_hint: Optional[str] = None
    ) -> STTResult:
        """
        Transcribe streaming audio chunks.
        Collects audio until silence detected, then transcribes.
        
        Args:
            audio_chunks: Async iterator of audio chunk bytes
            language_hint: Optional language hint
        
        Returns:
            STTResult with transcription
        """
        # Collect audio chunks
        audio_buffer = bytearray()
        chunk_count = 0
        
        try:
            async for chunk in audio_chunks:
                audio_buffer.extend(chunk)
                chunk_count += 1
                
                # Timeout protection
                if len(audio_buffer) > 16000 * 30 * 2:  # 30 seconds max
                    break
            
            if len(audio_buffer) < 3200:  # Less than 200ms
                raise STTNoAudioException()
            
            # Transcribe collected audio
            return await self.transcribe(bytes(audio_buffer), language_hint)
            
        except asyncio.TimeoutError:
            raise STTTimeoutException(settings.STT_TIMEOUT_SECONDS)
    
    def _transcribe_sync(
        self,
        audio_array: np.ndarray,
        language_hint: Optional[str]
    ) -> STTResult:
        """Synchronous transcription using AI4Bharat IndicConformer (runs in thread pool)."""
        import torch
        import torchaudio
        
        # Normalize audio to [-1, 1] range
        if audio_array.max() > 1.0:
            audio_array = audio_array / 32768.0
        
        # Convert to torch tensor with proper shape (1, samples)
        wav = torch.from_numpy(audio_array).float().unsqueeze(0)
        
        # Ensure mono audio
        if wav.dim() > 2:
            wav = torch.mean(wav, dim=0, keepdim=True)
        
        # Resample if needed (model expects 16kHz)
        target_sample_rate = 16000
        current_sr = settings.AUDIO_SAMPLE_RATE
        if current_sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=current_sr, new_freq=target_sample_rate)
            wav = resampler(wav)
        
        # Determine language (use hint or default to Hindi)
        language = language_hint if language_hint in self._supported_languages else "hi"
        
        # Run inference with IndicConformer
        # model(audio, language_code, decoder_type)
        transcription = self._model(wav, language, self._decoder)
        
        # Calculate audio duration
        audio_duration_ms = int(len(audio_array) / settings.AUDIO_SAMPLE_RATE * 1000)
        
        # Detect actual language from transcription text if no hint provided
        detected_language = self._detect_language(transcription, language_hint)
        
        return STTResult(
            text=transcription.strip() if transcription else "",
            language=detected_language,
            confidence=0.9,  # IndicConformer doesn't return confidence directly
            audio_duration_ms=audio_duration_ms,
            is_final=True
        )
    
    def _detect_language(
        self,
        text: str,
        hint: Optional[str] = None
    ) -> str:
        """Simple language detection based on script."""
        if not text:
            return hint or "en"
        
        # Count character scripts
        devanagari = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        bengali = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
        latin = sum(1 for c in text if 'a' <= c.lower() <= 'z')
        
        total = len(text.replace(" ", ""))
        if total == 0:
            return hint or "en"
        
        # Determine primary script
        if devanagari / total > 0.5:
            # Could be Hindi or Marathi - use hint
            return hint if hint in ["hi", "mr"] else "hi"
        elif bengali / total > 0.5:
            return "bn"
        elif latin / total > 0.5:
            return "en"
        
        return hint or "en"
    
    def _bytes_to_array(self, audio_bytes: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array."""
        # Assume 16-bit PCM
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        return audio_array.astype(np.float32)
    
    async def _mock_transcribe(
        self,
        audio_data: bytes,
        language_hint: Optional[str]
    ) -> STTResult:
        """Mock transcription for development/testing."""
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Return a mock result
        audio_duration_ms = int(len(audio_data) / 32)  # Rough estimate
        
        return STTResult(
            text="This is a mock transcription for testing.",
            language=language_hint or "en",
            confidence=0.95,
            audio_duration_ms=audio_duration_ms,
            is_final=True,
            processing_time_ms=100.0
        )
    
    async def cleanup(self):
        """Cleanup resources."""
        if self._model is not None:
            del self._model
            self._model = None
        
        self._is_initialized = False
        logger.info("STT service cleaned up")
