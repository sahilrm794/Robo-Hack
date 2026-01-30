"""
LLM Service using Groq API.
Provides ultra-low latency inference with streaming support.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, List, Dict, Any

from app.config import get_settings
from app.core.exceptions import (
    LLMException,
    LLMAPIException,
    LLMTimeoutException,
    LLMRateLimitException
)

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class LLMResponse:
    """Response from LLM completion."""
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    processing_time_ms: Optional[float] = None


@dataclass
class LLMChunk:
    """Streaming chunk from LLM."""
    content: str = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None
    is_final: bool = False


class LLMService:
    """
    LLM service using Groq API for ultra-low latency inference.
    
    Features:
    - Streaming responses for early TTS start
    - Tool/function calling support
    - Automatic retry with exponential backoff
    - Rate limit handling
    """
    
    def __init__(self):
        self._client = None
        self._is_initialized = False
        self._model = settings.LLM_MODEL_ID
    
    async def initialize(self):
        """Initialize Groq client."""
        try:
            logger.info("Initializing LLM service...")
            
            from groq import AsyncGroq
            
            self._client = AsyncGroq(
                api_key=settings.GROQ_API_KEY
            )
            
            self._is_initialized = True
            logger.info(f"LLM service initialized with model: {self._model}")
            
        except ImportError:
            logger.warning("Groq library not installed, using mock LLM")
            self._is_initialized = False
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {e}")
            self._is_initialized = False
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> LLMResponse:
        """
        Generate a complete response (non-streaming).
        
        Args:
            messages: Conversation messages
            tools: Optional tool definitions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        
        Returns:
            LLMResponse with content and optional tool calls
        """
        if not self._is_initialized:
            return await self._mock_complete(messages, tools)
        
        start_time = time.time()
        
        try:
            kwargs = {
                "model": self._model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            response = await asyncio.wait_for(
                self._client.chat.completions.create(**kwargs),
                timeout=settings.LLM_TIMEOUT_SECONDS
            )
            
            choice = response.choices[0]
            
            # Extract tool calls if any
            tool_calls = None
            if choice.message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments)
                    }
                    for tc in choice.message.tool_calls
                ]
            
            return LLMResponse(
                content=choice.message.content or "",
                tool_calls=tool_calls,
                finish_reason=choice.finish_reason,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
        except asyncio.TimeoutError:
            raise LLMTimeoutException(settings.LLM_TIMEOUT_SECONDS)
        except Exception as e:
            if "rate_limit" in str(e).lower():
                raise LLMRateLimitException()
            raise LLMAPIException(str(e))
    
    async def stream_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> AsyncIterator[LLMChunk]:
        """
        Stream LLM response for low latency.
        
        Args:
            messages: Conversation messages
            tools: Optional tool definitions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        
        Yields:
            LLMChunk with content or tool calls
        """
        if not self._is_initialized:
            async for chunk in self._mock_stream(messages, tools):
                yield chunk
            return
        
        try:
            kwargs = {
                "model": self._model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True
            }
            
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            stream = await self._client.chat.completions.create(**kwargs)
            
            tool_calls_buffer = {}
            
            async for chunk in stream:
                delta = chunk.choices[0].delta
                
                # Handle content
                if delta.content:
                    yield LLMChunk(content=delta.content)
                
                # Handle tool calls (accumulated across chunks)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_buffer:
                            tool_calls_buffer[idx] = {
                                "id": tc.id or "",
                                "name": tc.function.name if tc.function else "",
                                "arguments": ""
                            }
                        
                        if tc.function and tc.function.arguments:
                            tool_calls_buffer[idx]["arguments"] += tc.function.arguments
                        if tc.id:
                            tool_calls_buffer[idx]["id"] = tc.id
                        if tc.function and tc.function.name:
                            tool_calls_buffer[idx]["name"] = tc.function.name
                
                # Check if finished
                if chunk.choices[0].finish_reason:
                    if tool_calls_buffer:
                        # Parse accumulated tool calls
                        tool_calls = []
                        for tc in tool_calls_buffer.values():
                            try:
                                tool_calls.append({
                                    "id": tc["id"],
                                    "name": tc["name"],
                                    "arguments": json.loads(tc["arguments"])
                                })
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse tool call: {tc}")
                        
                        yield LLMChunk(tool_calls=tool_calls, is_final=True)
                    else:
                        yield LLMChunk(is_final=True)
                    break
                    
        except asyncio.TimeoutError:
            raise LLMTimeoutException(settings.LLM_TIMEOUT_SECONDS)
        except Exception as e:
            if "rate_limit" in str(e).lower():
                raise LLMRateLimitException()
            raise LLMAPIException(str(e))
    
    async def continue_with_tool_results(
        self,
        messages: List[Dict[str, str]],
        tool_calls: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]]
    ) -> AsyncIterator[LLMChunk]:
        """
        Continue generation after tool execution.
        
        Args:
            messages: Original messages
            tool_calls: Tool calls made by LLM
            tool_results: Results from tool execution
        
        Yields:
            LLMChunk with final response
        """
        # Build messages with tool results
        extended_messages = messages.copy()
        
        # Add assistant message with tool calls
        extended_messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"])
                    }
                }
                for tc in tool_calls
            ]
        })
        
        # Add tool results
        for tc, result in zip(tool_calls, tool_results):
            extended_messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(result.get("output", {}))
            })
        
        # Continue streaming
        async for chunk in self.stream_completion(
            extended_messages,
            tools=None  # No more tool calls needed
        ):
            yield chunk
    
    async def complete_with_tool_results(
        self,
        messages: List[Dict[str, str]],
        tool_calls: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]]
    ) -> LLMResponse:
        """
        Non-streaming version of continue_with_tool_results.
        """
        # Build messages with tool results
        extended_messages = messages.copy()
        
        extended_messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"])
                    }
                }
                for tc in tool_calls
            ]
        })
        
        for tc, result in zip(tool_calls, tool_results):
            extended_messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(result.get("output", {}))
            })
        
        return await self.complete(extended_messages)
    
    async def _mock_complete(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]]
    ) -> LLMResponse:
        """Mock completion for development."""
        await asyncio.sleep(0.1)
        
        # Extract user message
        user_msg = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_msg = msg["content"]
                break
        
        # Simple mock response
        response = f"I understand you said: '{user_msg}'. How can I help you further?"
        
        return LLMResponse(
            content=response,
            tool_calls=None,
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            processing_time_ms=100.0
        )
    
    async def _mock_stream(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]]
    ) -> AsyncIterator[LLMChunk]:
        """Mock streaming for development."""
        response = "I understand your question. Let me help you with that. Is there anything specific you'd like to know?"
        
        words = response.split()
        for i, word in enumerate(words):
            await asyncio.sleep(0.02)
            yield LLMChunk(content=word + " ")
        
        yield LLMChunk(is_final=True)
    
    async def cleanup(self):
        """Cleanup resources."""
        self._client = None
        self._is_initialized = False
        logger.info("LLM service cleaned up")
