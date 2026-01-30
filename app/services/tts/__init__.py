"""
Text-to-Speech Service using AI4Bharat Indic TTS.
Supports streaming audio synthesis for low-latency output.
"""

import asyncio
import io
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional, List
import numpy as np

from app.config import get_settings, LANGUAGE_VOICES
from app.core.exceptions import (
    TTSException,
    TTSModelNotLoadedException,
    TTSUnsupportedLanguageException
)

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class TTSResult:
    """Result from text-to-speech synthesis."""
    audio_data: bytes
    sample_rate: int
    duration_ms: int
    language: str
    processing_time_ms: Optional[float] = None


class TTSService:
    """
    Text-to-Speech service using AI4Bharat Indic TTS.
    
    Supports:
    - Streaming synthesis for low latency
    - Multiple Indian languages (Hindi, Bengali, Marathi, English)
    - Natural native speech output
    - Sentence-by-sentence generation
    """
    
    def __init__(self):
        self._model = None
        self._is_initialized = False
        self._device = "cuda"
        self._supported_languages = ["hi", "bn", "mr", "en"]
        
        # Voice configurations per language
        self._voices = {
            "hi": {"speaker": "hindi_female", "model": "ai4bharat/indic-tts-coqui-hi"},
            "bn": {"speaker": "bengali_female", "model": "ai4bharat/indic-tts-coqui-bn"},
            "mr": {"speaker": "marathi_female", "model": "ai4bharat/indic-tts-coqui-mr"},
            "en": {"speaker": "english_female", "model": "ai4bharat/indic-tts-coqui-en"}
        }
    
    async def initialize(self):
        """Initialize TTS models."""
        try:
            logger.info("Initializing TTS service...")
            
            # Run model loading in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_models)
            
            self._is_initialized = True
            logger.info("TTS service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS service: {e}")
            self._is_initialized = False
    
    def _load_models(self):
        """Load TTS models (runs in thread pool)."""
        try:
            import torch
            
            # Check device availability
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, using CPU for TTS")
                self._device = "cpu"
            
            # For hackathon, use a lightweight TTS approach
            # Option 1: AI4Bharat Indic TTS
            # Option 2: Coqui TTS
            # Option 3: Edge TTS (for demo)
            
            try:
                from TTS.api import TTS
                
                # Load multi-lingual model
                self._model = TTS(
                    model_name="tts_models/multilingual/multi-dataset/your_tts",
                    progress_bar=False
                ).to(self._device)
                
                logger.info("Coqui TTS loaded successfully")
                
            except ImportError:
                logger.warning("Coqui TTS not installed, using edge-tts fallback")
                self._model = "edge-tts"  # Will use edge_tts library
            
            logger.info("TTS models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load TTS models: {e}")
            logger.info("Using mock TTS for development")
    
    async def synthesize(
        self,
        text: str,
        language: str = "en",
        voice: Optional[str] = None
    ) -> bytes:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            language: Language code (hi, bn, mr, en)
            voice: Optional voice identifier
        
        Returns:
            Audio bytes (16kHz, 16-bit PCM)
        """
        if language not in self._supported_languages:
            raise TTSUnsupportedLanguageException(language, self._supported_languages)
        
        if not text.strip():
            return b''
        
        if not self._is_initialized:
            return await self._mock_synthesize(text, language)
        
        try:
            loop = asyncio.get_event_loop()
            audio_data = await loop.run_in_executor(
                None,
                self._synthesize_sync,
                text,
                language,
                voice
            )
            return audio_data
            
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            # Fallback to edge-tts
            return await self._edge_tts_synthesize(text, language)
    
    async def synthesize_streaming(
        self,
        text: str,
        language: str = "en",
        voice: Optional[str] = None
    ) -> AsyncIterator[bytes]:
        """
        Stream synthesized audio in chunks.
        Enables playback to start before full synthesis complete.
        
        Args:
            text: Text to synthesize
            language: Language code
            voice: Optional voice identifier
        
        Yields:
            Audio chunk bytes
        """
        if not text.strip():
            return
        
        if not self._is_initialized:
            async for chunk in self._mock_synthesize_streaming(text, language):
                yield chunk
            return
        
        try:
            # For streaming, we'll synthesize in chunks based on punctuation
            sentences = self._split_into_sentences(text)
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                
                # Synthesize each sentence
                audio_data = await self.synthesize(sentence, language, voice)
                
                # Yield in smaller chunks for smooth streaming
                chunk_size = 3200  # 100ms at 16kHz
                for i in range(0, len(audio_data), chunk_size):
                    yield audio_data[i:i + chunk_size]
                    await asyncio.sleep(0.01)  # Small delay for smooth streaming
                    
        except Exception as e:
            logger.error(f"TTS streaming error: {e}")
            raise TTSException(f"TTS streaming failed: {e}")
    
    def _synthesize_sync(
        self,
        text: str,
        language: str,
        voice: Optional[str]
    ) -> bytes:
        """Synchronous synthesis (runs in thread pool)."""
        import torch
        import numpy as np
        
        if isinstance(self._model, str) and self._model == "edge-tts":
            # Use edge-tts - but this is async, so we handle differently
            raise Exception("Use edge_tts_synthesize instead")
        
        # Use Coqui TTS
        voice_config = self._voices.get(language, self._voices["en"])
        
        # Generate audio
        wav = self._model.tts(
            text=text,
            language=language,
            speaker=voice or voice_config["speaker"]
        )
        
        # Convert to 16-bit PCM
        if isinstance(wav, np.ndarray):
            # Normalize and convert
            wav = wav / np.max(np.abs(wav))
            wav = (wav * 32767).astype(np.int16)
            return wav.tobytes()
        elif isinstance(wav, list):
            wav = np.array(wav)
            wav = wav / np.max(np.abs(wav))
            wav = (wav * 32767).astype(np.int16)
            return wav.tobytes()
        
        return wav
    
    async def _edge_tts_synthesize(
        self,
        text: str,
        language: str
    ) -> bytes:
        """Fallback synthesis using edge-tts."""
        try:
            import edge_tts
            
            # Voice mapping for edge-tts
            voices = {
                "hi": "hi-IN-SwaraNeural",
                "bn": "bn-IN-TanishaaNeural",
                "mr": "mr-IN-AarohiNeural",
                "en": "en-IN-NeerjaNeural"
            }
            
            voice = voices.get(language, voices["en"])
            
            # Generate audio
            communicate = edge_tts.Communicate(text, voice)
            audio_data = b''
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            return audio_data
            
        except ImportError:
            logger.warning("edge-tts not installed")
            return await self._mock_synthesize(text, language)
        except Exception as e:
            logger.error(f"edge-tts error: {e}")
            return await self._mock_synthesize(text, language)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for streaming."""
        import re
        
        # Split on sentence-ending punctuation
        # Include Hindi danda (।)
        sentences = re.split(r'(?<=[.!?।])\s+', text)
        
        return [s.strip() for s in sentences if s.strip()]
    
    async def _mock_synthesize(
        self,
        text: str,
        language: str
    ) -> bytes:
        """Mock synthesis for development/testing."""
        await asyncio.sleep(0.1)
        
        # Generate silent audio of appropriate length
        # Rough estimate: 100ms per word
        words = len(text.split())
        duration_ms = words * 100
        samples = int(settings.AUDIO_SAMPLE_RATE * duration_ms / 1000)
        
        # Generate silence (or low-frequency tone for debugging)
        audio = np.zeros(samples, dtype=np.int16)
        
        return audio.tobytes()
    
    async def _mock_synthesize_streaming(
        self,
        text: str,
        language: str
    ) -> AsyncIterator[bytes]:
        """Mock streaming synthesis."""
        audio_data = await self._mock_synthesize(text, language)
        
        chunk_size = 3200  # 100ms
        for i in range(0, len(audio_data), chunk_size):
            yield audio_data[i:i + chunk_size]
            await asyncio.sleep(0.05)
    
    async def cleanup(self):
        """Cleanup resources."""
        if self._model is not None and not isinstance(self._model, str):
            del self._model
            self._model = None
        
        self._is_initialized = False
        logger.info("TTS service cleaned up")
