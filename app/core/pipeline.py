"""
Pipeline Orchestrator for Voice Agent.
Coordinates the STT → LLM → TTS pipeline with streaming support.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator, Optional, Dict, Any, List, Tuple
import logging

from app.config import get_settings
from app.core.session import Session, ConversationTurn
from app.core.exceptions import (
    PipelineException,
    PipelineStageException,
    STTLowConfidenceException,
    LLMTimeoutException
)
from app.services.stt import STTService, STTResult
from app.services.tts import TTSService
from app.services.llm import LLMService, LLMResponse
from app.services.memory import ConversationMemoryService
from app.tools.registry import ToolRegistry
from app.logging.agent_logger import AgentLogger

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class PipelineMetrics:
    """Metrics for a single pipeline execution."""
    start_time: float = field(default_factory=time.time)
    stt_start: Optional[float] = None
    stt_end: Optional[float] = None
    llm_start: Optional[float] = None
    llm_first_token: Optional[float] = None
    llm_end: Optional[float] = None
    tool_start: Optional[float] = None
    tool_end: Optional[float] = None
    tts_start: Optional[float] = None
    tts_first_audio: Optional[float] = None
    tts_end: Optional[float] = None
    
    @property
    def stt_latency_ms(self) -> Optional[float]:
        if self.stt_start and self.stt_end:
            return (self.stt_end - self.stt_start) * 1000
        return None
    
    @property
    def llm_latency_ms(self) -> Optional[float]:
        if self.llm_start and self.llm_end:
            return (self.llm_end - self.llm_start) * 1000
        return None
    
    @property
    def llm_ttft_ms(self) -> Optional[float]:
        """Time to first token."""
        if self.llm_start and self.llm_first_token:
            return (self.llm_first_token - self.llm_start) * 1000
        return None
    
    @property
    def tool_latency_ms(self) -> Optional[float]:
        if self.tool_start and self.tool_end:
            return (self.tool_end - self.tool_start) * 1000
        return None
    
    @property
    def tts_latency_ms(self) -> Optional[float]:
        if self.tts_start and self.tts_end:
            return (self.tts_end - self.tts_start) * 1000
        return None
    
    @property
    def tts_ttfa_ms(self) -> Optional[float]:
        """Time to first audio."""
        if self.tts_start and self.tts_first_audio:
            return (self.tts_first_audio - self.tts_start) * 1000
        return None
    
    @property
    def total_latency_ms(self) -> float:
        return (time.time() - self.start_time) * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "stt_latency_ms": self.stt_latency_ms,
            "llm_latency_ms": self.llm_latency_ms,
            "llm_ttft_ms": self.llm_ttft_ms,
            "tool_latency_ms": self.tool_latency_ms,
            "tts_latency_ms": self.tts_latency_ms,
            "tts_ttfa_ms": self.tts_ttfa_ms,
            "total_latency_ms": self.total_latency_ms
        }


@dataclass
class PipelineResult:
    """Result of a complete pipeline execution."""
    session_id: str
    user_text: str
    agent_text: str
    language: str
    metrics: PipelineMetrics
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    stt_confidence: float = 1.0
    error: Optional[str] = None


class PipelineOrchestrator:
    """
    Orchestrates the voice pipeline: STT → LLM → TTS
    Supports streaming for low-latency responses.
    """
    
    def __init__(
        self,
        stt_service: STTService,
        llm_service: LLMService,
        tts_service: TTSService,
        tool_registry: ToolRegistry,
        agent_logger: AgentLogger
    ):
        self.stt = stt_service
        self.llm = llm_service
        self.tts = tts_service
        self.tools = tool_registry
        self.logger = agent_logger
    
    async def process_audio_streaming(
        self,
        audio_chunks: AsyncIterator[bytes],
        session: Session
    ) -> AsyncIterator[bytes]:
        """
        Process audio input and stream audio output.
        This is the main streaming pipeline.
        
        Flow:
        1. Collect audio and run STT
        2. Build prompt with context
        3. Stream LLM response
        4. Execute tools if needed
        5. Stream TTS output as sentences complete
        """
        metrics = PipelineMetrics()
        tool_calls = []
        
        try:
            # ==================
            # Stage 1: STT
            # ==================
            metrics.stt_start = time.time()
            
            stt_result = await self.stt.transcribe_streaming(
                audio_chunks,
                language_hint=session.context.detected_language
            )
            
            metrics.stt_end = time.time()
            
            # Validate STT result
            if stt_result.confidence < settings.STT_CONFIDENCE_THRESHOLD:
                # Return "please repeat" message
                error_response = self._get_low_confidence_response(
                    session.context.detected_language
                )
                async for chunk in self.tts.synthesize_streaming(
                    error_response,
                    language=session.context.detected_language
                ):
                    yield chunk
                return
            
            # Update session with user input
            user_turn = ConversationTurn(
                role="user",
                content=stt_result.text,
                language=stt_result.language,
                stt_confidence=stt_result.confidence,
                audio_duration_ms=stt_result.audio_duration_ms
            )
            session.add_turn(user_turn)
            
            # Update context with detected language
            session.context.detected_language = stt_result.language
            
            # Log STT result
            await self.logger.log_stt_result(
                session.session_id,
                stt_result.text,
                stt_result.language,
                stt_result.confidence,
                metrics.stt_latency_ms
            )
            
            # ==================
            # Stage 2: LLM
            # ==================
            metrics.llm_start = time.time()
            
            # Build prompt with context
            messages = self._build_llm_messages(session, stt_result.text)
            
            # Stream LLM response
            full_response = ""
            current_sentence = ""
            first_token_received = False
            
            async for chunk in self.llm.stream_completion(
                messages,
                tools=self.tools.get_tool_schemas()
            ):
                if not first_token_received:
                    metrics.llm_first_token = time.time()
                    first_token_received = True
                
                # Check for tool calls
                if chunk.tool_calls:
                    metrics.llm_end = time.time()
                    
                    # Execute tools
                    metrics.tool_start = time.time()
                    tool_results = await self._execute_tools(
                        chunk.tool_calls,
                        session
                    )
                    metrics.tool_end = time.time()
                    tool_calls.extend(tool_results)
                    
                    # Log tool calls
                    for result in tool_results:
                        await self.logger.log_tool_call(
                            session.session_id,
                            result["name"],
                            result["input"],
                            result["output"],
                            result.get("latency_ms")
                        )
                    
                    # Continue LLM with tool results
                    async for final_chunk in self.llm.continue_with_tool_results(
                        messages,
                        chunk.tool_calls,
                        tool_results
                    ):
                        full_response += final_chunk.content
                        current_sentence += final_chunk.content
                        
                        # Check for sentence completion
                        if self._is_sentence_complete(current_sentence):
                            # Start TTS for completed sentence
                            if metrics.tts_start is None:
                                metrics.tts_start = time.time()
                            
                            async for audio_chunk in self.tts.synthesize_streaming(
                                current_sentence.strip(),
                                language=stt_result.language
                            ):
                                if metrics.tts_first_audio is None:
                                    metrics.tts_first_audio = time.time()
                                yield audio_chunk
                            
                            current_sentence = ""
                else:
                    # Regular text chunk
                    full_response += chunk.content
                    current_sentence += chunk.content
                    
                    # Check for sentence completion
                    if self._is_sentence_complete(current_sentence):
                        metrics.llm_end = time.time()
                        
                        # Start TTS for completed sentence
                        if metrics.tts_start is None:
                            metrics.tts_start = time.time()
                        
                        async for audio_chunk in self.tts.synthesize_streaming(
                            current_sentence.strip(),
                            language=stt_result.language
                        ):
                            if metrics.tts_first_audio is None:
                                metrics.tts_first_audio = time.time()
                            yield audio_chunk
                        
                        current_sentence = ""
            
            # Handle remaining text
            if current_sentence.strip():
                if metrics.tts_start is None:
                    metrics.tts_start = time.time()
                
                async for audio_chunk in self.tts.synthesize_streaming(
                    current_sentence.strip(),
                    language=stt_result.language
                ):
                    if metrics.tts_first_audio is None:
                        metrics.tts_first_audio = time.time()
                    yield audio_chunk
            
            metrics.tts_end = time.time()
            metrics.llm_end = metrics.llm_end or time.time()
            
            # Update session with agent response
            agent_turn = ConversationTurn(
                role="assistant",
                content=full_response,
                language=stt_result.language,
                processing_time_ms=int(metrics.total_latency_ms)
            )
            session.add_turn(agent_turn)
            
            # Update context from response
            self._update_context_from_response(session, full_response, tool_calls)
            
            # Log completion
            await self.logger.log_turn_complete(
                session.session_id,
                stt_result.text,
                full_response,
                stt_result.language,
                tool_calls,
                metrics.to_dict()
            )
            
        except Exception as e:
            logger.exception(f"Pipeline error: {e}")
            
            # Return error message in user's language
            error_response = self._get_error_response(
                session.context.detected_language,
                str(e)
            )
            
            async for chunk in self.tts.synthesize_streaming(
                error_response,
                language=session.context.detected_language
            ):
                yield chunk
            
            await self.logger.log_error(
                session.session_id,
                "pipeline_error",
                str(e)
            )
    
    async def process_audio_batch(
        self,
        audio_data: bytes,
        session: Session
    ) -> Tuple[str, bytes, PipelineMetrics]:
        """
        Process complete audio input and return complete audio output.
        Non-streaming version for simpler use cases.
        """
        metrics = PipelineMetrics()
        
        # STT
        metrics.stt_start = time.time()
        stt_result = await self.stt.transcribe(audio_data)
        metrics.stt_end = time.time()
        
        # Update session
        session.context.detected_language = stt_result.language
        user_turn = ConversationTurn(
            role="user",
            content=stt_result.text,
            language=stt_result.language,
            stt_confidence=stt_result.confidence
        )
        session.add_turn(user_turn)
        
        # LLM
        metrics.llm_start = time.time()
        messages = self._build_llm_messages(session, stt_result.text)
        response = await self.llm.complete(messages, tools=self.tools.get_tool_schemas())
        
        # Handle tool calls
        if response.tool_calls:
            metrics.tool_start = time.time()
            tool_results = await self._execute_tools(response.tool_calls, session)
            metrics.tool_end = time.time()
            
            response = await self.llm.complete_with_tool_results(
                messages,
                response.tool_calls,
                tool_results
            )
        
        metrics.llm_end = time.time()
        
        # TTS
        metrics.tts_start = time.time()
        audio_output = await self.tts.synthesize(
            response.content,
            language=stt_result.language
        )
        metrics.tts_end = time.time()
        
        # Update session
        agent_turn = ConversationTurn(
            role="assistant",
            content=response.content,
            language=stt_result.language
        )
        session.add_turn(agent_turn)
        
        return response.content, audio_output, metrics
    
    def _build_llm_messages(
        self,
        session: Session,
        user_message: str
    ) -> List[Dict[str, str]]:
        """Build LLM messages with context and history."""
        
        system_prompt = f"""You are a helpful e-commerce customer support agent for an Indian marketplace.

CURRENT CONTEXT:
{session.context.get_context_summary()}

CRITICAL INSTRUCTIONS:
1. ALWAYS respond in the SAME language as the user's message
2. Be concise - this is a voice conversation, keep responses under 3 sentences
3. Use the provided tools to fetch accurate product/order information
4. If you need to look up a product or order, use the appropriate tool
5. Never make up product prices, availability, or order status - always use tools
6. If the user asks about a product they mentioned earlier, use the context
7. For confirmations, be explicit about what you're confirming
8. If unsure, ask a clarifying question

AVAILABLE ACTIONS:
- Search for products by name, category, or description
- Look up frequently asked questions about products
- Track orders by order ID
- Help with returns and cancellations
- Explain company policies (returns, shipping, refunds)

RESPONSE STYLE:
- Natural, conversational tone
- Avoid technical jargon
- Be helpful and empathetic
- Use appropriate cultural context for Indian customers
"""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for turn in session.get_recent_turns(settings.CONTEXT_WINDOW_SIZE):
            messages.append(turn.to_llm_message())
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    async def _execute_tools(
        self,
        tool_calls: List[Dict[str, Any]],
        session: Session
    ) -> List[Dict[str, Any]]:
        """Execute tool calls and return results."""
        results = []
        
        for call in tool_calls:
            start = time.time()
            
            try:
                result = await self.tools.execute(
                    call["name"],
                    call["arguments"],
                    session
                )
                
                results.append({
                    "name": call["name"],
                    "input": call["arguments"],
                    "output": result,
                    "success": True,
                    "latency_ms": (time.time() - start) * 1000
                })
                
            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                results.append({
                    "name": call["name"],
                    "input": call["arguments"],
                    "output": {"error": str(e)},
                    "success": False,
                    "latency_ms": (time.time() - start) * 1000
                })
        
        return results
    
    def _is_sentence_complete(self, text: str) -> bool:
        """Check if text contains a complete sentence."""
        text = text.strip()
        if not text:
            return False
        
        # Check for sentence-ending punctuation
        sentence_enders = ['.', '!', '?', '।', '?', '!']  # Include Hindi danda
        
        for ender in sentence_enders:
            if text.endswith(ender):
                return True
        
        # Check for minimum length with comma (for natural pauses)
        if len(text) > 50 and text.endswith(','):
            return True
        
        return False
    
    def _update_context_from_response(
        self,
        session: Session,
        response: str,
        tool_calls: List[Dict[str, Any]]
    ):
        """Update session context based on agent response and tool calls."""
        for call in tool_calls:
            if call["name"] == "search_products" and call.get("success"):
                output = call.get("output", {})
                products = output.get("products", [])
                if products:
                    session.context.last_product_id = products[0].get("id")
                    session.context.last_product_name = products[0].get("name")
                    session.context.last_search_query = call["input"].get("query")
            
            elif call["name"] == "track_order" and call.get("success"):
                session.context.last_order_id = call["input"].get("order_id")
            
            elif call["name"] == "get_product_faq":
                session.context.last_product_id = call["input"].get("product_id")
    
    def _get_low_confidence_response(self, language: str) -> str:
        """Get response for low STT confidence."""
        responses = {
            "en": "I didn't catch that clearly. Could you please repeat?",
            "hi": "मैं स्पष्ट रूप से नहीं सुन पाया। क्या आप दोहरा सकते हैं?",
            "bn": "আমি স্পষ্টভাবে শুনতে পাইনি। আপনি কি আবার বলবেন?",
            "mr": "मला नीट ऐकू आलं नाही. तुम्ही पुन्हा सांगाल का?"
        }
        return responses.get(language, responses["en"])
    
    def _get_error_response(self, language: str, error: str) -> str:
        """Get generic error response."""
        responses = {
            "en": "I'm having trouble processing that. Let me try again in a moment.",
            "hi": "मुझे इसे समझने में समस्या हो रही है। कृपया थोड़ी देर बाद पुनः प्रयास करें।",
            "bn": "এটি প্রক্রিয়া করতে আমার সমস্যা হচ্ছে। একটু পরে আবার চেষ্টা করুন।",
            "mr": "मला हे समजून घेण्यात अडचण येत आहे. कृपया थोड्या वेळाने पुन्हा प्रयत्न करा."
        }
        return responses.get(language, responses["en"])
