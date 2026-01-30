"""
Agent Logger for Markdown Execution Logs.
Creates human-readable logs for hackathon evaluation and debugging.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

logger = logging.getLogger(__name__)


class AgentLogger:
    """
    Markdown logger for agent execution.
    
    Creates structured, human-readable logs that document:
    - Architectural decisions
    - Pipeline steps
    - Tool calls and reasoning
    - Error handling
    - Performance metrics
    """
    
    def __init__(self, log_path: str = "logs/agent_log.md"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._queue: asyncio.Queue = asyncio.Queue()
        self._writer_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Start the async writer
        self._start_writer()
    
    def _start_writer(self):
        """Start the background log writer."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._writer_task = asyncio.create_task(self._write_loop())
            self._running = True
        except RuntimeError:
            # No event loop, will write synchronously
            pass
    
    async def _write_loop(self):
        """Background loop to write logs asynchronously."""
        while self._running:
            try:
                # Wait for log entries
                entry = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0
                )
                
                # Write to file
                await self._write_entry(entry)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Log writer error: {e}")
    
    async def _write_entry(self, entry: str):
        """Write a log entry to the file."""
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(entry)
                f.write("\n")
        except Exception as e:
            logger.error(f"Failed to write log: {e}")
    
    def _sync_write(self, entry: str):
        """Synchronous write fallback."""
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(entry)
                f.write("\n")
        except Exception as e:
            logger.error(f"Failed to write log: {e}")
    
    async def _log(self, entry: str):
        """Add a log entry to the queue."""
        if self._running and self._writer_task:
            await self._queue.put(entry)
        else:
            self._sync_write(entry)
    
    # =========================
    # Public Logging Methods
    # =========================
    
    async def log_session_start(
        self,
        session_id: str,
        language: str = "en"
    ):
        """Log the start of a new session."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        entry = f"""
---

## 🆕 Session Started: `{session_id}`

**Timestamp:** {timestamp}  
**Initial Language:** {language}

---
"""
        await self._log(entry)
    
    async def log_stt_result(
        self,
        session_id: str,
        text: str,
        language: str,
        confidence: float,
        latency_ms: Optional[float] = None
    ):
        """Log STT transcription result."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Confidence indicator
        if confidence >= 0.9:
            conf_indicator = "🟢"
        elif confidence >= 0.7:
            conf_indicator = "🟡"
        else:
            conf_indicator = "🔴"
        
        entry = f"""### 🎤 User Input | {timestamp}

**Session:** `{session_id}`  
**Transcription:** "{text}"  
**Language:** {language}  
**Confidence:** {conf_indicator} {confidence:.2%}  
{f'**STT Latency:** {latency_ms:.0f}ms' if latency_ms else ''}
"""
        await self._log(entry)
    
    async def log_tool_call(
        self,
        session_id: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Dict[str, Any],
        latency_ms: Optional[float] = None
    ):
        """Log a tool call and its result."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Truncate long outputs
        output_str = json.dumps(tool_output, indent=2, ensure_ascii=False)
        if len(output_str) > 500:
            output_str = output_str[:500] + "\n  ... (truncated)"
        
        input_str = json.dumps(tool_input, indent=2, ensure_ascii=False)
        
        entry = f"""#### 🔧 Tool Call: `{tool_name}` | {timestamp}

**Input:**
```json
{input_str}
```

**Output:**
```json
{output_str}
```
{f'**Execution Time:** {latency_ms:.0f}ms' if latency_ms else ''}
"""
        await self._log(entry)
    
    async def log_llm_response(
        self,
        session_id: str,
        response: str,
        tokens_used: Optional[int] = None,
        latency_ms: Optional[float] = None
    ):
        """Log LLM response."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Truncate long responses for log
        display_response = response
        if len(response) > 500:
            display_response = response[:500] + "..."
        
        entry = f"""### 🤖 Agent Response | {timestamp}

**Session:** `{session_id}`

> {display_response}

{f'**Tokens Used:** {tokens_used}' if tokens_used else ''}
{f'**LLM Latency:** {latency_ms:.0f}ms' if latency_ms else ''}
"""
        await self._log(entry)
    
    async def log_turn_complete(
        self,
        session_id: str,
        user_text: str,
        agent_text: str,
        language: str,
        tool_calls: List[Dict[str, Any]],
        metrics: Dict[str, Any]
    ):
        """Log a complete conversation turn with metrics."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        total_latency = metrics.get("total_latency_ms", 0)
        
        # Latency status
        if total_latency < 800:
            latency_status = "🟢 Excellent"
        elif total_latency < 1000:
            latency_status = "🟡 Good"
        else:
            latency_status = "🔴 Slow"
        
        tools_used = ", ".join([tc.get("name", "unknown") for tc in tool_calls]) if tool_calls else "None"
        
        entry = f"""### ✅ Turn Complete | {timestamp}

**Session:** `{session_id}`  
**Language:** {language}

| Metric | Value |
|--------|-------|
| Total Latency | {latency_status} ({total_latency:.0f}ms) |
| STT | {metrics.get('stt_latency_ms', 'N/A'):.0f}ms |
| LLM | {metrics.get('llm_latency_ms', 'N/A'):.0f}ms |
| LLM TTFT | {metrics.get('llm_ttft_ms', 'N/A'):.0f}ms |
| Tools | {metrics.get('tool_latency_ms', 'N/A'):.0f}ms |
| TTS | {metrics.get('tts_latency_ms', 'N/A'):.0f}ms |

**Tools Used:** {tools_used}

---
"""
        await self._log(entry)
    
    async def log_error(
        self,
        session_id: str,
        error_type: str,
        error_message: str,
        stack_trace: Optional[str] = None
    ):
        """Log an error."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        entry = f"""### ❌ Error | {timestamp}

**Session:** `{session_id}`  
**Type:** `{error_type}`  
**Message:** {error_message}
"""
        
        if stack_trace:
            entry += f"""
<details>
<summary>Stack Trace</summary>

```
{stack_trace}
```

</details>
"""
        
        await self._log(entry)
    
    async def log_system_event(
        self,
        event: str,
        details: Dict[str, Any]
    ):
        """Log a system event."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        details_str = ""
        for key, value in details.items():
            details_str += f"- **{key}:** {value}\n"
        
        entry = f"""### ⚙️ System Event | {timestamp}

**Event:** {event}

{details_str}
---
"""
        await self._log(entry)
    
    async def log_architectural_decision(
        self,
        decision: str,
        rationale: str,
        alternatives: Optional[List[str]] = None
    ):
        """Log an architectural decision."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        alts_str = ""
        if alternatives:
            alts_str = "\n**Alternatives Considered:**\n"
            for alt in alternatives:
                alts_str += f"- {alt}\n"
        
        entry = f"""### 📐 Architectural Decision | {timestamp}

**Decision:** {decision}

**Rationale:** {rationale}
{alts_str}
---
"""
        await self._log(entry)
    
    async def initialize_log(self):
        """Initialize the log file with header."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        header = f"""# 🎙️ Voice Agent Execution Log

**Generated:** {timestamp}  
**Version:** 1.0.0  
**Target Latency:** < 1 second

---

## System Overview

This log documents the execution of the Voice-to-Voice AI Customer Support Agent.

**Pipeline:** Audio → STT → LLM → TTS → Audio

**Supported Languages:**
- English (en)
- Hindi (hi)
- Bengali (bn)
- Marathi (mr)

---

## Execution Log

"""
        
        # Overwrite file with header
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(header)
        
        logger.info(f"Agent log initialized: {self.log_path}")
    
    async def close(self):
        """Close the logger and flush pending entries."""
        self._running = False
        
        if self._writer_task:
            self._writer_task.cancel()
            try:
                await self._writer_task
            except asyncio.CancelledError:
                pass
        
        # Write any remaining entries
        while not self._queue.empty():
            try:
                entry = self._queue.get_nowait()
                self._sync_write(entry)
            except asyncio.QueueEmpty:
                break
        
        logger.info("Agent logger closed")
