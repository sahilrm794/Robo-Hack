"""
Voice WebSocket Endpoints.
Handles real-time audio streaming for voice conversations.
"""

import asyncio
import json
import logging
import time
import io
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.core.session import Session, SessionManager, ConversationTurn
from app.core.pipeline import PipelineOrchestrator

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()

# Session manager for WebSocket connections
session_manager = SessionManager()


def convert_webm_to_pcm(webm_data: bytes) -> bytes:
    """Convert WebM audio to raw PCM (16kHz, mono, 16-bit)."""
    import numpy as np
    import io
    import tempfile
    import os
    
    # First, try using pydub with ffmpeg (most reliable for webm)
    try:
        from pydub import AudioSegment
        
        # Write to temp file (pydub works better with files for webm)
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
            f.write(webm_data)
            temp_path = f.name
        
        try:
            audio = AudioSegment.from_file(temp_path, format="webm")
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            pcm_data = audio.raw_data
            logger.info(f"Converted WebM to PCM: {len(pcm_data)} bytes")
            return pcm_data
        finally:
            os.unlink(temp_path)  # Clean up temp file
            
    except Exception as e:
        logger.warning(f"pydub conversion failed: {e}")
    
    # Fallback: Try soundfile
    try:
        import soundfile as sf
        
        audio_io = io.BytesIO(webm_data)
        data, samplerate = sf.read(audio_io)
        
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # Resample to 16kHz if needed
        if samplerate != 16000:
            import torchaudio
            import torch
            tensor = torch.from_numpy(data).float().unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=samplerate, new_freq=16000)
            tensor = resampler(tensor)
            data = tensor.squeeze().numpy()
        
        # Convert to 16-bit PCM
        data = (data * 32767).astype(np.int16)
        return data.tobytes()
        
    except Exception as e:
        logger.warning(f"soundfile conversion failed: {e}")
    
    # Last resort: return raw data and hope STT can handle it
    logger.warning("All audio conversion methods failed, returning raw data")
    return webm_data

@router.websocket("/stream")
async def voice_stream(
    websocket: WebSocket,
    session_id: Optional[str] = None
):
    """
    WebSocket endpoint for real-time voice streaming.
    
    Protocol:
    1. Client connects with optional session_id
    2. Client sends audio chunks (binary)
    3. Client sends JSON message {"type": "end"} when done speaking
    4. Server processes audio and streams back audio response
    5. Server sends JSON message {"type": "end"} when response complete
    
    Message Types:
    - Binary: Audio data (16kHz, 16-bit PCM, mono)
    - JSON: Control messages
        - {"type": "end"} - End of audio input
        - {"type": "config", "language": "hi"} - Set language
        - {"type": "cancel"} - Cancel current processing
    """
    await websocket.accept()
    
    # Get or create session
    session = await session_manager.get_or_create_session(session_id)
    session_id = session.session_id
    
    # Get services from app state
    app = websocket.app
    stt_service = app.state.stt_service
    tts_service = app.state.tts_service
    llm_service = app.state.llm_service
    tool_registry = app.state.tool_registry
    agent_logger = app.state.agent_logger
    
    # Create pipeline orchestrator
    pipeline = PipelineOrchestrator(
        stt_service=stt_service,
        llm_service=llm_service,
        tts_service=tts_service,
        tool_registry=tool_registry,
        agent_logger=agent_logger
    )
    
    # Log session start
    await agent_logger.log_session_start(session_id, session.context.detected_language)
    
    # Send session info to client
    await websocket.send_json({
        "type": "session",
        "session_id": session_id,
        "language": session.context.detected_language
    })
    
    try:
        while True:
            # Collect audio chunks until end signal
            audio_buffer = bytearray()
            
            while True:
                message = await websocket.receive()
                
                if message["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect()
                
                # Handle binary audio data
                if "bytes" in message:
                    audio_buffer.extend(message["bytes"])
                    continue
                
                # Handle text/JSON messages
                if "text" in message:
                    try:
                        data = json.loads(message["text"])
                        msg_type = data.get("type")
                        
                        if msg_type == "end":
                            # End of audio input, process it
                            break
                        
                        elif msg_type == "config":
                            # Update configuration
                            if "language" in data:
                                session.context.preferred_language = data["language"]
                                session.context.detected_language = data["language"]
                            continue
                        
                        elif msg_type == "cancel":
                            # Cancel current processing
                            audio_buffer.clear()
                            await websocket.send_json({"type": "cancelled"})
                            continue
                        
                        elif msg_type == "ping":
                            await websocket.send_json({"type": "pong"})
                            continue
                        
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON message: {message['text']}")
                        continue
            
            # Process audio if we have any
            if len(audio_buffer) > 0:
                start_time = time.time()
                
                # Send processing indicator
                await websocket.send_json({"type": "processing"})
                
                try:
                    # Convert audio if needed (WebM -> PCM)
                    pcm_audio = convert_webm_to_pcm(bytes(audio_buffer))
                    
                    # Create async iterator from audio buffer
                    async def audio_chunks():
                        chunk_size = 3200  # 100ms at 16kHz
                        for i in range(0, len(pcm_audio), chunk_size):
                            yield pcm_audio[i:i + chunk_size]
                    
                    # Collect audio chunks to send back
                    audio_response = bytearray()
                    
                    # Process through pipeline and stream response
                    async for audio_chunk in pipeline.process_audio_streaming(
                        audio_chunks(),
                        session
                    ):
                        audio_response.extend(audio_chunk)
                        # Send audio chunk to client
                        await websocket.send_bytes(audio_chunk)
                    
                    # Get transcript from session
                    user_text = ""
                    agent_text = ""
                    if session.turns:
                        # Get last user turn
                        for turn in reversed(session.turns):
                            if turn.role == "user" and not user_text:
                                user_text = turn.content
                            elif turn.role == "assistant" and not agent_text:
                                agent_text = turn.content
                            if user_text and agent_text:
                                break
                    
                    # Send end of response with transcript
                    total_time = (time.time() - start_time) * 1000
                    await websocket.send_json({
                        "type": "response",
                        "user_text": user_text,
                        "agent_text": agent_text,
                        "latency_ms": round(total_time, 2)
                    })
                    
                except Exception as e:
                    logger.error(f"Pipeline error: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Failed to process audio",
                        "details": str(e) if settings.DEBUG else None
                    })
            
            # Update session
            await session_manager.update_session(session)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
    finally:
        # Cleanup
        await session_manager.update_session(session)


@router.post("/process")
async def process_audio(
    request: Request,
    session_id: Optional[str] = None
):
    """
    REST endpoint for non-streaming audio processing.
    
    Accepts:
    - Audio file upload (multipart/form-data)
    - Raw audio body (application/octet-stream)
    
    Returns:
    - Response audio (audio/wav)
    - Response text
    - Latency metrics
    """
    from fastapi import UploadFile, File
    from fastapi.responses import Response
    
    # Get session
    session = await session_manager.get_or_create_session(session_id)
    
    # Get audio data
    content_type = request.headers.get("content-type", "")
    
    if "multipart/form-data" in content_type:
        form = await request.form()
        audio_file = form.get("audio")
        if audio_file:
            audio_data = await audio_file.read()
        else:
            raise HTTPException(status_code=400, detail="No audio file provided")
    else:
        audio_data = await request.body()
    
    if len(audio_data) < 100:
        raise HTTPException(status_code=400, detail="Audio data too short")
    
    # Get services
    app = request.app
    stt_service = app.state.stt_service
    tts_service = app.state.tts_service
    llm_service = app.state.llm_service
    tool_registry = app.state.tool_registry
    agent_logger = app.state.agent_logger
    
    # Create pipeline
    pipeline = PipelineOrchestrator(
        stt_service=stt_service,
        llm_service=llm_service,
        tts_service=tts_service,
        tool_registry=tool_registry,
        agent_logger=agent_logger
    )
    
    # Process audio
    try:
        text, audio_response, metrics = await pipeline.process_audio_batch(
            audio_data,
            session
        )
        
        # Update session
        await session_manager.update_session(session)
        
        # Return based on Accept header
        accept = request.headers.get("accept", "application/json")
        
        if "audio" in accept:
            return Response(
                content=audio_response,
                media_type="audio/wav",
                headers={
                    "X-Response-Text": text[:100],
                    "X-Session-Id": session.session_id,
                    "X-Latency-Ms": str(round(metrics.total_latency_ms, 2))
                }
            )
        else:
            return {
                "session_id": session.session_id,
                "response_text": text,
                "language": session.context.detected_language,
                "metrics": metrics.to_dict(),
                "audio_base64": None  # Could add base64 encoded audio
            }
            
    except Exception as e:
        logger.exception(f"Audio processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/text")
async def process_text(
    request: Request,
    session_id: Optional[str] = None
):
    """
    Process text input (for testing without audio).
    
    Body:
    {
        "text": "User message",
        "language": "en"  // optional
    }
    """
    data = await request.json()
    text = data.get("text", "")
    language = data.get("language", "en")
    
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    # Get session
    session = await session_manager.get_or_create_session(session_id)
    session.context.detected_language = language
    
    # Get services
    app = request.app
    llm_service = app.state.llm_service
    tool_registry = app.state.tool_registry
    agent_logger = app.state.agent_logger
    
    # Build messages
    from app.core.pipeline import PipelineOrchestrator
    
    pipeline = PipelineOrchestrator(
        stt_service=app.state.stt_service,
        llm_service=llm_service,
        tts_service=app.state.tts_service,
        tool_registry=tool_registry,
        agent_logger=agent_logger
    )
    
    messages = pipeline._build_llm_messages(session, text)
    
    # Get LLM response
    start_time = time.time()
    
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
                "result": result
            })
        
        # Get final response with tool results
        response = await llm_service.complete_with_tool_results(
            messages,
            response.tool_calls,
            [{"output": r["result"]} for r in tool_results]
        )
    
    latency_ms = (time.time() - start_time) * 1000
    
    # Update session
    session.add_turn(ConversationTurn(role="user", content=text, language=language))
    session.add_turn(ConversationTurn(role="assistant", content=response.content, language=language))
    await session_manager.update_session(session)
    
    return {
        "session_id": session.session_id,
        "response": response.content,
        "language": language,
        "tool_calls": tool_results,
        "latency_ms": round(latency_ms, 2)
    }


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session information."""
    session = await session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session.to_dict()


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    await session_manager.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}
