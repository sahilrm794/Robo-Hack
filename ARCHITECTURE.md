# Voice-to-Voice AI Customer Support Agent
## System Architecture Documentation

**Version:** 1.0.0  
**Date:** January 30, 2026  
**Target Latency:** < 1 second end-to-end

---

## 1. High-Level System Architecture

### 1.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                          │
│  │   Web App   │    │ Mobile App  │    │   IoT/Kiosk │                          │
│  │  (Browser)  │    │  (Flutter)  │    │   Device    │                          │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                          │
│         │                  │                  │                                  │
│         └──────────────────┼──────────────────┘                                  │
│                            │ WebSocket (Audio Streaming)                         │
│                            ▼                                                     │
└────────────────────────────┬────────────────────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND                                              │
│                            ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    WebSocket Gateway                                     │    │
│  │              (Audio Stream Management)                                   │    │
│  └─────────────────────────┬───────────────────────────────────────────────┘    │
│                            │                                                     │
│  ┌─────────────────────────▼───────────────────────────────────────────────┐    │
│  │                     PIPELINE ORCHESTRATOR                                │    │
│  │         (Manages STT → LLM → TTS flow with streaming)                   │    │
│  └─────┬───────────────────┬───────────────────────────────┬───────────────┘    │
│        │                   │                               │                     │
│        ▼                   ▼                               ▼                     │
│  ┌───────────┐      ┌─────────────┐               ┌─────────────┐               │
│  │    STT    │      │     LLM     │               │     TTS     │               │
│  │  Service  │─────▶│   Service   │──────────────▶│   Service   │               │
│  │(Streaming)│      │  (Groq API) │               │ (Streaming) │               │
│  └───────────┘      └──────┬──────┘               └─────────────┘               │
│                            │                                                     │
│                            ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                      TOOL EXECUTOR                                       │    │
│  │   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │    │
│  │   │ Product  │ │   FAQ    │ │  Order   │ │ Returns  │ │  Policy  │      │    │
│  │   │  Search  │ │  Lookup  │ │ Tracking │ │  Cancel  │ │  Lookup  │      │    │
│  │   └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘      │    │
│  └────────┼────────────┼────────────┼────────────┼────────────┼────────────┘    │
│           │            │            │            │            │                  │
│           └────────────┴────────────┼────────────┴────────────┘                  │
│                                     ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                      DATA ACCESS LAYER                                   │    │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │    │
│  │   │   Products   │  │    Orders    │  │     FAQs     │                  │    │
│  │   │   (SQLite)   │  │   (SQLite)   │  │   (SQLite)   │                  │    │
│  │   └──────────────┘  └──────────────┘  └──────────────┘                  │    │
│  │   ┌──────────────┐  ┌──────────────────────────────────┐                │    │
│  │   │   Policies   │  │  Vector Store (Product Embeddings)│               │    │
│  │   │   (SQLite)   │  │         (ChromaDB / FAISS)        │               │    │
│  │   └──────────────┘  └──────────────────────────────────┘                │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    CONVERSATION MEMORY                                   │    │
│  │   ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────┐     │    │
│  │   │  Short-term      │  │  Context Store   │  │  Session Manager  │     │    │
│  │   │  (Redis/Memory)  │  │  (Last Product,  │  │  (User Sessions)  │     │    │
│  │   │                  │  │   Order, Intent) │  │                   │     │    │
│  │   └──────────────────┘  └──────────────────┘  └───────────────────┘     │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    LOGGING & MONITORING                                  │    │
│  │   ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────┐     │    │
│  │   │  Agent Log       │  │  Metrics Logger  │  │  Error Tracker    │     │    │
│  │   │  (Markdown)      │  │  (Prometheus)    │  │  (Sentry-style)   │     │    │
│  │   └──────────────────┘  └──────────────────┘  └───────────────────┘     │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Data Flow: Microphone → Speaker

```
User speaks into microphone
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│ 1. AUDIO CAPTURE (Client-side)                                 │
│    • Format: 16kHz, 16-bit, mono PCM                          │
│    • Chunk size: 100ms buffers (streaming)                    │
│    • VAD (Voice Activity Detection) on client                 │
└────────────────────────────────────────────────────────────────┘
         │ WebSocket (binary audio chunks)
         ▼
┌────────────────────────────────────────────────────────────────┐
│ 2. STT SERVICE (AI4Bharat IndicConformer)         ~200-300ms  │
│    • Streaming transcription with partial results             │
│    • Language detection (Hindi/Bengali/Marathi/English)       │
│    • Confidence scoring per utterance                         │
│    • Output: Transcribed text + language + confidence         │
└────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│ 3. CONTEXT INJECTION                               ~10ms      │
│    • Fetch conversation history (last 5 turns)                │
│    • Inject user context (last product, order, intent)        │
│    • Build LLM prompt with system instructions               │
└────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│ 4. LLM REASONING (Groq API - Llama 3 70B)         ~150-300ms  │
│    • Streaming response generation                            │
│    • Tool/function calling decisions                          │
│    • Response in same language as input                       │
│    • Token streaming for early TTS start                     │
└────────────────────────────────────────────────────────────────┘
         │
         ├──────────────────────────────────────┐
         ▼                                      ▼
┌─────────────────────────┐     ┌─────────────────────────────────┐
│ 5a. TOOL EXECUTION      │     │ 5b. DIRECT RESPONSE             │
│     (If tools needed)   │     │     (No tools needed)           │
│     ~50-100ms           │     │                                  │
│     • Product search    │     │                                  │
│     • FAQ lookup        │     │                                  │
│     • Order tracking    │     │                                  │
│     • Policy check      │     │                                  │
└──────────┬──────────────┘     └─────────────────────────────────┘
           │                                      │
           └──────────────────┬───────────────────┘
                              ▼
┌────────────────────────────────────────────────────────────────┐
│ 6. TTS SERVICE (AI4Bharat Indic TTS)              ~200-300ms  │
│    • Streaming audio synthesis                                 │
│    • Language-matched voice selection                         │
│    • Chunk-by-chunk audio generation                          │
│    • Output: Audio chunks (16kHz, 16-bit PCM)                 │
└────────────────────────────────────────────────────────────────┘
         │ WebSocket (binary audio chunks)
         ▼
┌────────────────────────────────────────────────────────────────┐
│ 7. AUDIO PLAYBACK (Client-side)                               │
│    • Buffer and play audio chunks                             │
│    • Handle interruptions (barge-in)                          │
└────────────────────────────────────────────────────────────────┘
         │
         ▼
    User hears response
```

---

## 2. FastAPI Backend Design

### 2.1 Folder Structure

```
voice-agent/
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI app initialization
│   ├── config.py                   # Configuration management
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── voice.py            # WebSocket voice endpoints
│   │   │   ├── conversation.py     # REST conversation endpoints
│   │   │   └── health.py           # Health check endpoints
│   │   └── middleware/
│   │       ├── __init__.py
│   │       ├── auth.py             # Session authentication
│   │       └── logging.py          # Request logging middleware
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── pipeline.py             # Main STT→LLM→TTS orchestrator
│   │   ├── session.py              # Session management
│   │   └── exceptions.py           # Custom exceptions
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── stt/
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # STT abstract interface
│   │   │   ├── indic_conformer.py  # AI4Bharat IndicConformer
│   │   │   └── streaming.py        # Streaming STT handler
│   │   ├── tts/
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # TTS abstract interface
│   │   │   ├── indic_tts.py        # AI4Bharat Indic TTS
│   │   │   └── streaming.py        # Streaming TTS handler
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # LLM abstract interface
│   │   │   ├── groq_client.py      # Groq API client
│   │   │   ├── prompts.py          # System prompts
│   │   │   └── streaming.py        # Streaming LLM handler
│   │   └── memory/
│   │       ├── __init__.py
│   │       ├── conversation.py     # Conversation history
│   │       └── context.py          # Context state management
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── registry.py             # Tool registry & executor
│   │   ├── product_search.py       # Product search tool
│   │   ├── faq_lookup.py           # FAQ lookup tool
│   │   ├── order_tracking.py       # Order tracking tool
│   │   ├── returns.py              # Returns & cancellation tool
│   │   └── policy_lookup.py        # Policy lookup tool
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── database.py             # Database connection
│   │   ├── models.py               # SQLAlchemy models
│   │   ├── repositories/
│   │   │   ├── __init__.py
│   │   │   ├── products.py         # Product repository
│   │   │   ├── orders.py           # Order repository
│   │   │   ├── faqs.py             # FAQ repository
│   │   │   └── policies.py         # Policy repository
│   │   └── vector_store.py         # Vector store for semantic search
│   │
│   ├── logging/
│   │   ├── __init__.py
│   │   ├── agent_logger.py         # Markdown agent logger
│   │   └── metrics.py              # Performance metrics
│   │
│   └── schemas/
│       ├── __init__.py
│       ├── audio.py                # Audio data schemas
│       ├── conversation.py         # Conversation schemas
│       ├── tools.py                # Tool I/O schemas
│       └── responses.py            # API response schemas
│
├── data/
│   ├── products.json               # Product catalog (TO BE PROVIDED)
│   ├── faqs.json                   # FAQ database (TO BE PROVIDED)
│   ├── orders.json                 # Sample orders (TO BE PROVIDED)
│   └── policies.json               # Company policies (TO BE PROVIDED)
│
├── models/
│   ├── stt/                        # Downloaded STT models
│   └── tts/                        # Downloaded TTS models
│
├── logs/
│   ├── agent_log.md                # Agent execution log
│   └── app.log                     # Application log
│
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py
│   ├── test_tools.py
│   └── test_services.py
│
├── scripts/
│   ├── download_models.py          # Download AI4Bharat models
│   ├── init_db.py                  # Initialize database
│   └── seed_data.py                # Seed sample data
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

### 2.2 Key Services & Responsibilities

| Service | Responsibility | Async/Sync |
|---------|---------------|------------|
| `PipelineOrchestrator` | Coordinates STT→LLM→TTS flow | Async |
| `STTService` | Audio → Text transcription | Async (streaming) |
| `LLMService` | Text reasoning & tool calling | Async (streaming) |
| `TTSService` | Text → Audio synthesis | Async (streaming) |
| `ToolExecutor` | Execute database queries | Async |
| `ConversationMemory` | Store/retrieve conversation history | Async |
| `ContextManager` | Manage user context state | Sync (in-memory) |
| `AgentLogger` | Log to markdown file | Async (non-blocking) |

### 2.3 Async vs Sync Decisions

```python
# ASYNC: All I/O bound operations
- WebSocket handling
- STT streaming inference
- Groq API calls
- TTS streaming generation
- Database queries
- File I/O for logging

# SYNC: CPU-bound or fast operations
- Context state management (in-memory dict)
- Tool schema validation
- Prompt construction
- Audio chunk encoding
```

---

## 3. STT → LLM → TTS Pipeline

### 3.1 Exact Pipeline Flow

```python
async def process_voice_input(audio_chunks: AsyncIterator[bytes], session: Session):
    """Main pipeline orchestrator"""
    
    # Phase 1: STT (Streaming)
    transcription = await stt_service.transcribe_streaming(
        audio_chunks,
        language_hint=session.context.detected_language
    )
    
    # Phase 2: Context Injection
    prompt = build_llm_prompt(
        user_message=transcription.text,
        conversation_history=session.memory.get_recent(n=5),
        context=session.context,
        tools=tool_registry.get_schemas()
    )
    
    # Phase 3: LLM Reasoning (Streaming)
    async for chunk in llm_service.stream_completion(prompt):
        if chunk.is_tool_call:
            # Execute tool and continue
            tool_result = await tool_executor.execute(chunk.tool_call)
            # Feed result back to LLM for final response
            async for final_chunk in llm_service.continue_with_tool_result(tool_result):
                yield final_chunk
        else:
            # Phase 4: TTS (Streaming, starts as soon as first sentence complete)
            if chunk.is_sentence_complete:
                async for audio_chunk in tts_service.synthesize_streaming(
                    text=chunk.sentence,
                    language=transcription.language
                ):
                    yield audio_chunk
```

### 3.2 Streaming vs Non-Streaming Choices

| Component | Mode | Rationale |
|-----------|------|-----------|
| **STT** | Streaming | Start processing before user finishes speaking |
| **LLM** | Streaming | Send to TTS as tokens arrive, reduce TTFB |
| **TTS** | Streaming | Play audio while still generating |
| **Tool Calls** | Non-streaming | Tools are fast (<100ms), wait for complete result |

### 3.3 Latency Optimization Strategies

```
┌────────────────────────────────────────────────────────────────────────┐
│                    LATENCY BREAKDOWN TARGET                             │
├────────────────────────────────────────────────────────────────────────┤
│ Component          │ Target    │ Optimization Strategy                 │
├────────────────────┼───────────┼───────────────────────────────────────┤
│ Audio to Server    │ ~50ms     │ WebSocket, chunked streaming          │
│ STT Processing     │ ~200ms    │ Streaming, VAD, language detection    │
│ Context Injection  │ ~10ms     │ In-memory context, pre-built prompts  │
│ LLM (Groq)         │ ~150ms    │ Streaming, Llama-3-70B on Groq       │
│ Tool Execution     │ ~50ms     │ SQLite in-memory, indexed queries     │
│ TTS Processing     │ ~200ms    │ Streaming, sentence-level synthesis   │
│ Audio to Client    │ ~50ms     │ WebSocket, chunked streaming          │
├────────────────────┼───────────┼───────────────────────────────────────┤
│ TOTAL (best case)  │ ~500ms    │ With parallelization                  │
│ TOTAL (with tools) │ ~750ms    │ Including tool execution              │
└────────────────────┴───────────┴───────────────────────────────────────┘
```

**Key Optimizations:**

1. **Pipeline Parallelization**: Start TTS as soon as first sentence from LLM
2. **Model Preloading**: Load STT/TTS models at startup
3. **Connection Pooling**: Reuse Groq API connections
4. **Sentence-level TTS**: Don't wait for full response
5. **SQLite WAL Mode**: Fast concurrent reads
6. **Context Caching**: Pre-compute embeddings for products

---

## 4. Context & Memory Handling

### 4.1 Conversation State Storage

```python
@dataclass
class ConversationContext:
    """Mutable context that evolves during conversation"""
    session_id: str
    user_id: Optional[str] = None
    
    # Language context
    detected_language: str = "en"
    preferred_language: Optional[str] = None
    
    # Intent tracking
    current_intent: Optional[str] = None  # "product_search", "order_tracking", etc.
    intent_confidence: float = 0.0
    
    # Entity memory
    last_product_id: Optional[str] = None
    last_product_name: Optional[str] = None
    last_order_id: Optional[str] = None
    last_category: Optional[str] = None
    
    # Conversation state
    awaiting_confirmation: bool = False
    pending_action: Optional[str] = None
    
    # Timestamps
    created_at: datetime
    last_activity: datetime

@dataclass
class ConversationTurn:
    """Single turn in conversation"""
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    timestamp: datetime
    language: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_results: Optional[List[ToolResult]] = None
```

### 4.2 Context Injection into LLM

```python
def build_llm_prompt(
    user_message: str,
    history: List[ConversationTurn],
    context: ConversationContext,
    tools: List[ToolSchema]
) -> List[Message]:
    
    system_prompt = f"""You are a helpful e-commerce customer support agent.
    
CURRENT CONTEXT:
- Language: {context.detected_language}
- User's last viewed product: {context.last_product_name or 'None'}
- User's last order: {context.last_order_id or 'None'}
- Current intent: {context.current_intent or 'General inquiry'}
{f'- Awaiting confirmation for: {context.pending_action}' if context.awaiting_confirmation else ''}

INSTRUCTIONS:
1. Respond in the SAME language as the user's message
2. Be concise - this is a voice conversation
3. Use the provided tools to fetch accurate information
4. If unsure, ask clarifying questions
5. For products, always confirm before proceeding with orders
"""
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history (last 5 turns)
    for turn in history[-5:]:
        messages.append({"role": turn.role, "content": turn.content})
    
    # Add current user message
    messages.append({"role": "user", "content": user_message})
    
    return messages
```

### 4.3 Short-term vs Long-term Memory

```
┌─────────────────────────────────────────────────────────────────────┐
│                      MEMORY ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  SHORT-TERM MEMORY (Session-scoped, In-Memory)                      │
│  ├── Conversation history (last 10 turns)                           │
│  ├── Current context state                                          │
│  ├── Active entities (product, order being discussed)               │
│  └── TTL: Session duration (30 min idle timeout)                    │
│                                                                      │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                      │
│  LONG-TERM MEMORY (Persistent, SQLite/Redis)                        │
│  ├── User preferences (language, communication style)               │
│  ├── Order history (for returning users)                            │
│  ├── Previous support tickets                                       │
│  └── TTL: Indefinite (with user_id)                                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Tool / Function Calling Design

### 5.1 Tool Registry

```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search for products by name, category, or description",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (product name, category, or keywords)"
                    },
                    "category": {
                        "type": "string",
                        "description": "Product category filter",
                        "enum": ["electronics", "clothing", "home", "beauty", "sports"]
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 5
                    },
                    "price_range": {
                        "type": "object",
                        "properties": {
                            "min": {"type": "number"},
                            "max": {"type": "number"}
                        }
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_faq",
            "description": "Get frequently asked questions about a specific product",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Product ID to get FAQs for"
                    },
                    "question_topic": {
                        "type": "string",
                        "description": "Specific topic to filter FAQs",
                        "enum": ["warranty", "shipping", "returns", "usage", "specifications"]
                    }
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "track_order",
            "description": "Track order status and delivery information",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Order ID to track"
                    },
                    "phone_number": {
                        "type": "string",
                        "description": "Phone number associated with the order (for verification)"
                    }
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "initiate_return",
            "description": "Start a return or exchange process for an order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Order ID for return"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for return",
                        "enum": ["defective", "wrong_item", "not_as_described", "changed_mind", "size_issue"]
                    },
                    "action": {
                        "type": "string",
                        "description": "Requested action",
                        "enum": ["refund", "exchange", "store_credit"]
                    }
                },
                "required": ["order_id", "reason", "action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Cancel an order that hasn't shipped yet",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Order ID to cancel"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for cancellation"
                    }
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_policy",
            "description": "Get company policy information",
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_type": {
                        "type": "string",
                        "description": "Type of policy to retrieve",
                        "enum": ["returns", "refunds", "shipping", "privacy", "warranty"]
                    }
                },
                "required": ["policy_type"]
            }
        }
    }
]
```

### 5.2 Tool Input/Output Schemas

```python
# Input Schemas
class ProductSearchInput(BaseModel):
    query: str
    category: Optional[str] = None
    max_results: int = 5
    price_range: Optional[Dict[str, float]] = None

class OrderTrackingInput(BaseModel):
    order_id: str
    phone_number: Optional[str] = None

class ReturnInput(BaseModel):
    order_id: str
    reason: Literal["defective", "wrong_item", "not_as_described", "changed_mind", "size_issue"]
    action: Literal["refund", "exchange", "store_credit"]

# Output Schemas
class ProductSearchResult(BaseModel):
    products: List[Product]
    total_count: int
    search_query: str

class OrderTrackingResult(BaseModel):
    order_id: str
    status: str  # "pending", "shipped", "delivered", "cancelled"
    items: List[OrderItem]
    estimated_delivery: Optional[datetime]
    tracking_number: Optional[str]
    shipping_carrier: Optional[str]

class ReturnResult(BaseModel):
    return_id: str
    status: str
    instructions: str
    refund_amount: Optional[float]
    refund_method: Optional[str]
```

---

## 6. Database Schema (Logical)

### 6.1 Entity Relationship Diagram

```
┌─────────────────────┐       ┌─────────────────────┐
│      PRODUCTS       │       │       ORDERS        │
├─────────────────────┤       ├─────────────────────┤
│ id (PK)             │       │ id (PK)             │
│ name                │       │ user_phone          │
│ name_hi             │◄──────│ status              │
│ name_bn             │       │ total_amount        │
│ name_mr             │       │ shipping_address    │
│ description         │       │ created_at          │
│ description_hi      │       │ updated_at          │
│ description_bn      │       │ estimated_delivery  │
│ description_mr      │       │ tracking_number     │
│ category            │       │ shipping_carrier    │
│ price               │       └─────────┬───────────┘
│ stock_quantity      │                 │
│ image_url           │                 │
│ specifications      │       ┌─────────▼───────────┐
│ created_at          │       │    ORDER_ITEMS      │
└─────────┬───────────┘       ├─────────────────────┤
          │                   │ id (PK)             │
          │                   │ order_id (FK)       │
          │                   │ product_id (FK)     │
          │                   │ quantity            │
          │                   │ unit_price          │
          └───────────────────┴─────────────────────┘

┌─────────────────────┐       ┌─────────────────────┐
│        FAQS         │       │      POLICIES       │
├─────────────────────┤       ├─────────────────────┤
│ id (PK)             │       │ id (PK)             │
│ product_id (FK)     │       │ type                │
│ question            │       │ title               │
│ question_hi         │       │ title_hi            │
│ question_bn         │       │ title_bn            │
│ question_mr         │       │ title_mr            │
│ answer              │       │ content             │
│ answer_hi           │       │ content_hi          │
│ answer_bn           │       │ content_bn          │
│ answer_mr           │       │ content_mr          │
│ topic               │       │ effective_date      │
│ created_at          │       │ updated_at          │
└─────────────────────┘       └─────────────────────┘

┌─────────────────────┐       ┌─────────────────────┐
│      RETURNS        │       │    CONVERSATIONS    │
├─────────────────────┤       ├─────────────────────┤
│ id (PK)             │       │ id (PK)             │
│ order_id (FK)       │       │ session_id          │
│ reason              │       │ user_phone          │
│ action_requested    │       │ turns (JSON)        │
│ status              │       │ context (JSON)      │
│ refund_amount       │       │ created_at          │
│ created_at          │       │ updated_at          │
│ processed_at        │       │ language            │
└─────────────────────┘       └─────────────────────┘
```

### 6.2 SQLAlchemy Models

```python
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship

class Product(Base):
    __tablename__ = "products"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, index=True)
    name_hi = Column(String)  # Hindi
    name_bn = Column(String)  # Bengali
    name_mr = Column(String)  # Marathi
    description = Column(Text)
    description_hi = Column(Text)
    description_bn = Column(Text)
    description_mr = Column(Text)
    category = Column(String, index=True)
    price = Column(Float, nullable=False)
    stock_quantity = Column(Integer, default=0)
    image_url = Column(String)
    specifications = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    faqs = relationship("FAQ", back_populates="product")
    order_items = relationship("OrderItem", back_populates="product")

class Order(Base):
    __tablename__ = "orders"
    
    id = Column(String, primary_key=True)
    user_phone = Column(String, index=True)
    status = Column(String, default="pending")  # pending, confirmed, shipped, delivered, cancelled
    total_amount = Column(Float)
    shipping_address = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    estimated_delivery = Column(DateTime)
    tracking_number = Column(String)
    shipping_carrier = Column(String)
    
    items = relationship("OrderItem", back_populates="order")
    returns = relationship("Return", back_populates="order")

class OrderItem(Base):
    __tablename__ = "order_items"
    
    id = Column(String, primary_key=True)
    order_id = Column(String, ForeignKey("orders.id"))
    product_id = Column(String, ForeignKey("products.id"))
    quantity = Column(Integer)
    unit_price = Column(Float)
    
    order = relationship("Order", back_populates="items")
    product = relationship("Product", back_populates="order_items")

class FAQ(Base):
    __tablename__ = "faqs"
    
    id = Column(String, primary_key=True)
    product_id = Column(String, ForeignKey("products.id"), nullable=True)
    question = Column(Text, nullable=False)
    question_hi = Column(Text)
    question_bn = Column(Text)
    question_mr = Column(Text)
    answer = Column(Text, nullable=False)
    answer_hi = Column(Text)
    answer_bn = Column(Text)
    answer_mr = Column(Text)
    topic = Column(String, index=True)  # warranty, shipping, returns, usage, specifications
    created_at = Column(DateTime, default=datetime.utcnow)
    
    product = relationship("Product", back_populates="faqs")

class Policy(Base):
    __tablename__ = "policies"
    
    id = Column(String, primary_key=True)
    type = Column(String, unique=True, index=True)  # returns, refunds, shipping, privacy, warranty
    title = Column(String)
    title_hi = Column(String)
    title_bn = Column(String)
    title_mr = Column(String)
    content = Column(Text)
    content_hi = Column(Text)
    content_bn = Column(Text)
    content_mr = Column(Text)
    effective_date = Column(DateTime)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
```

---

## 7. Fine-Tuning Strategy

### 7.1 When Fine-Tuning is Necessary

| Scenario | Fine-Tuning Needed? | Reason |
|----------|---------------------|--------|
| Standard Hindi STT | No | IndicConformer handles well |
| **Rural Hindi dialects** | **YES** | Regional vocabulary, accent variations |
| Standard Bengali STT | No | IndicConformer handles well |
| **Rural Bengali dialects** | **YES** | Regional vocabulary, accent variations |
| Standard Marathi STT | No | IndicConformer handles well |
| **Rural Marathi dialects** | **YES** | Regional vocabulary, accent variations |
| TTS voice quality | No | Pre-trained voices are sufficient |
| **TTS for domain terms** | **MAYBE** | E-commerce terminology pronunciation |

### 7.2 Required Datasets (MUST ASK FOR)

```
⚠️ IMPORTANT: The following datasets are REQUIRED for fine-tuning.
   Please provide these if rural dialect support is needed.

┌────────────────────────────────────────────────────────────────────────┐
│ DATASET REQUEST #1: Rural Hindi Speech Corpus                          │
├────────────────────────────────────────────────────────────────────────┤
│ Required for: Fine-tuning IndicConformer for rural Hindi dialects     │
│ Format: Audio files (WAV, 16kHz) + Transcription (JSON/CSV)           │
│ Minimum size: 10-50 hours of transcribed speech                       │
│ Content: Customer support conversations, product inquiries            │
│ Dialects: Bhojpuri, Awadhi, Rajasthani, etc.                         │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ DATASET REQUEST #2: Rural Bengali Speech Corpus                         │
├────────────────────────────────────────────────────────────────────────┤
│ Required for: Fine-tuning IndicConformer for rural Bengali dialects   │
│ Format: Audio files (WAV, 16kHz) + Transcription (JSON/CSV)           │
│ Minimum size: 10-50 hours of transcribed speech                       │
│ Dialects: Rarhi, Bangali, Sylheti, etc.                              │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ DATASET REQUEST #3: Rural Marathi Speech Corpus                         │
├────────────────────────────────────────────────────────────────────────┤
│ Required for: Fine-tuning IndicConformer for rural Marathi dialects   │
│ Format: Audio files (WAV, 16kHz) + Transcription (JSON/CSV)           │
│ Minimum size: 10-50 hours of transcribed speech                       │
│ Dialects: Varhadi, Konkani Marathi, Deccani, etc.                    │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ DATASET REQUEST #4: E-commerce Domain Vocabulary                        │
├────────────────────────────────────────────────────────────────────────┤
│ Required for: Improving TTS pronunciation of domain terms             │
│ Format: Word list with phonetic transcriptions                        │
│ Content: Product names, brand names, e-commerce terminology           │
│ Languages: Hindi, Bengali, Marathi transliterations                   │
└────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Fine-Tuning Process with HuggingFace

```python
from transformers import AutoModelForCTC, AutoProcessor, Trainer, TrainingArguments
from huggingface_hub import login

# Login with HuggingFace token
login(token=os.environ["HF_TOKEN"])

# Load base IndicConformer model
model = AutoModelForCTC.from_pretrained("ai4bharat/indicconformer-hi")
processor = AutoProcessor.from_pretrained("ai4bharat/indicconformer-hi")

# Fine-tuning configuration
training_args = TrainingArguments(
    output_dir="./models/stt/finetuned-rural-hindi",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    save_steps=1000,
    eval_steps=500,
    logging_steps=100,
    push_to_hub=False,  # Keep local for hackathon
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=rural_hindi_dataset,
    eval_dataset=rural_hindi_eval_dataset,
    data_collator=data_collator,
)
trainer.train()
```

---

## 8. Latency Optimization

### 8.1 Optimization Techniques

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     LATENCY OPTIMIZATION MATRIX                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. MODEL PRELOADING                                                     │
│     ├── Load STT model at startup (not per-request)                     │
│     ├── Load TTS model at startup                                       │
│     ├── Warm up models with dummy inference                             │
│     └── Savings: ~2-3 seconds per request                               │
│                                                                          │
│  2. STREAMING EVERYWHERE                                                 │
│     ├── Stream audio FROM client (don't wait for complete audio)       │
│     ├── Stream STT results (partial transcriptions)                     │
│     ├── Stream LLM tokens (Groq streaming)                              │
│     ├── Stream TTS audio TO client (sentence-by-sentence)              │
│     └── Savings: ~500ms reduction in perceived latency                  │
│                                                                          │
│  3. SENTENCE-LEVEL TTS                                                   │
│     ├── Detect sentence boundaries in LLM output                        │
│     ├── Start TTS as soon as first sentence complete                   │
│     ├── Buffer and stream audio chunks                                  │
│     └── Savings: ~200-300ms time-to-first-audio                         │
│                                                                          │
│  4. CONNECTION POOLING                                                   │
│     ├── Reuse Groq API HTTP connections                                 │
│     ├── SQLite connection pooling                                       │
│     └── Savings: ~50-100ms per request                                  │
│                                                                          │
│  5. DATABASE OPTIMIZATION                                                │
│     ├── SQLite in WAL mode (concurrent reads)                           │
│     ├── Indexes on frequently queried columns                           │
│     ├── Pre-computed embeddings for semantic search                     │
│     ├── In-memory caching for hot data (products, policies)            │
│     └── Savings: ~30-50ms per query                                     │
│                                                                          │
│  6. PARALLEL PROCESSING                                                  │
│     ├── Context fetch + prompt building in parallel                     │
│     ├── Multiple tool calls in parallel (when independent)             │
│     └── Savings: ~50-100ms for multi-tool scenarios                     │
│                                                                          │
│  7. CACHING                                                              │
│     ├── Cache product search results (LRU, 5 min TTL)                  │
│     ├── Cache policy lookups (longer TTL)                               │
│     ├── Cache TTS audio for common phrases                              │
│     └── Savings: ~100-200ms for cache hits                              │
│                                                                          │
│  8. BATCHING (for non-realtime)                                          │
│     ├── Batch embedding generation for products                         │
│     ├── Batch log writes (async queue)                                  │
│     └── Impact: Reduces background load                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Model Selection for Speed

| Component | Model Choice | Why |
|-----------|-------------|-----|
| STT | IndicConformer (small/medium) | Good accuracy, reasonable speed |
| LLM | Llama-3-70B-8192 on Groq | Ultra-low latency (~150ms) |
| TTS | Indic TTS (VITS-based) | Real-time capable |

---

## 9. Failure Handling

### 9.1 Error Handling Matrix

```python
class ErrorHandler:
    """Centralized error handling with graceful degradation"""
    
    ERROR_RESPONSES = {
        "stt_low_confidence": {
            "en": "I didn't catch that clearly. Could you please repeat?",
            "hi": "मैं स्पष्ट रूप से नहीं सुन पाया। क्या आप दोहरा सकते हैं?",
            "bn": "আমি স্পষ্টভাবে শুনতে পাইনি। আপনি কি আবার বলবেন?",
            "mr": "मला नीट ऐकू आलं नाही. तुम्ही पुन्हा सांगाल का?"
        },
        "stt_timeout": {
            "en": "I didn't hear anything. Are you still there?",
            "hi": "मुझे कुछ सुनाई नहीं दिया। क्या आप अभी भी वहाँ हैं?",
            "bn": "আমি কিছু শুনতে পাইনি। আপনি কি এখনও আছেন?",
            "mr": "मला काहीच ऐकू आलं नाही. तुम्ही अजून आहात का?"
        },
        "llm_error": {
            "en": "I'm having trouble processing that. Let me try again.",
            "hi": "मुझे इसे समझने में समस्या हो रही है। मैं फिर से कोशिश करता हूँ।",
            "bn": "এটি প্রক্রিয়া করতে আমার সমস্যা হচ্ছে। আমাকে আবার চেষ্টা করতে দিন।",
            "mr": "मला हे समजून घेण्यात अडचण येत आहे. मी पुन्हा प्रयत्न करतो."
        },
        "tool_not_found": {
            "en": "I couldn't find that information. Could you provide more details?",
            "hi": "मुझे वह जानकारी नहीं मिली। क्या आप अधिक विवरण दे सकते हैं?",
            "bn": "আমি সেই তথ্য খুঁজে পাইনি। আপনি কি আরও বিস্তারিত বলবেন?",
            "mr": "मला ती माहिती सापडली नाही. तुम्ही अधिक तपशील देऊ शकता का?"
        },
        "ambiguous_query": {
            "en": "I want to make sure I understand. Are you asking about {option1} or {option2}?",
            "hi": "मैं सुनिश्चित करना चाहता हूँ कि मैं समझ गया। क्या आप {option1} या {option2} के बारे में पूछ रहे हैं?",
            "bn": "আমি নিশ্চিত হতে চাই। আপনি কি {option1} না {option2} সম্পর্কে জিজ্ঞাসা করছেন?",
            "mr": "मला खात्री करायची आहे. तुम्ही {option1} किंवा {option2} बद्दल विचारत आहात का?"
        }
    }
```

### 9.2 Specific Failure Scenarios

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     FAILURE HANDLING SCENARIOS                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  SCENARIO 1: STT Confidence < 0.6                                        │
│  ├── Action: Request user to repeat                                     │
│  ├── Log: Low confidence transcription                                  │
│  ├── Context: Keep previous context intact                              │
│  └── Recovery: Retry with adjusted audio gain suggestion                │
│                                                                          │
│  SCENARIO 2: Partial/Truncated Audio                                     │
│  ├── Detection: Audio duration < 0.5s or abrupt end                    │
│  ├── Action: Prompt user to complete their sentence                     │
│  ├── Log: Partial audio received                                        │
│  └── Recovery: Append next audio chunk before processing                │
│                                                                          │
│  SCENARIO 3: Language Detection Mismatch                                 │
│  ├── Detection: Detected language differs from session language        │
│  ├── Action: Switch to detected language, confirm with user            │
│  ├── Log: Language switch detected                                      │
│  └── Recovery: Update session language preference                       │
│                                                                          │
│  SCENARIO 4: LLM Timeout (> 5 seconds)                                   │
│  ├── Action: Return canned "processing" response                        │
│  ├── Log: LLM timeout                                                   │
│  ├── Retry: Automatic retry with exponential backoff                    │
│  └── Fallback: Simplified prompt if retry fails                         │
│                                                                          │
│  SCENARIO 5: Tool Returns Empty Results                                  │
│  ├── Detection: Empty product/order search                              │
│  ├── Action: Suggest alternatives or ask for clarification             │
│  ├── Log: Empty tool result                                             │
│  └── Recovery: Broaden search or offer related options                  │
│                                                                          │
│  SCENARIO 6: Order Not Found                                             │
│  ├── Detection: Order ID not in database                                │
│  ├── Action: Ask for verification, suggest checking order ID           │
│  ├── Log: Order lookup failed                                           │
│  └── Recovery: Offer to search by phone number instead                  │
│                                                                          │
│  SCENARIO 7: Ambiguous User Intent                                       │
│  ├── Detection: Multiple possible intents with similar confidence      │
│  ├── Action: Ask clarifying question                                    │
│  ├── Log: Ambiguous intent                                              │
│  └── Recovery: Present options to user                                  │
│                                                                          │
│  SCENARIO 8: TTS Generation Failure                                      │
│  ├── Detection: TTS model error                                         │
│  ├── Action: Return text response, notify of audio issue               │
│  ├── Log: TTS failure                                                   │
│  └── Recovery: Fallback to simpler TTS or text-only                    │
│                                                                          │
│  SCENARIO 9: WebSocket Disconnection                                     │
│  ├── Detection: Connection lost mid-conversation                        │
│  ├── Action: Persist conversation state for reconnection               │
│  ├── Log: Session interrupted                                           │
│  └── Recovery: Restore context on reconnection                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Hackathon-Ready Deployment Plan

### 10.1 Local Demo Setup

```bash
# Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU inference)
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM (recommended)

# Quick Start
git clone <repo>
cd voice-agent
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Download models
python scripts/download_models.py

# Initialize database
python scripts/init_db.py
python scripts/seed_data.py

# Set environment variables
copy .env.example .env
# Edit .env with GROQ_API_KEY, HF_TOKEN

# Run
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 10.2 Hardware Requirements

| Configuration | CPU | GPU | RAM | Latency |
|--------------|-----|-----|-----|---------|
| **Minimum (CPU-only)** | 8 cores | None | 16GB | ~2-3s |
| **Recommended (GPU)** | 8 cores | RTX 3060 | 16GB | ~800ms |
| **Optimal (High-end GPU)** | 12 cores | RTX 4080 | 32GB | ~500ms |

### 10.3 Minimal Infrastructure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   HACKATHON DEMO INFRASTRUCTURE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  OPTION A: Single Machine (Recommended for Demo)                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Laptop/Desktop with GPU                                         │    │
│  │  ├── FastAPI Server (Port 8000)                                 │    │
│  │  ├── STT Model (GPU)                                            │    │
│  │  ├── TTS Model (GPU)                                            │    │
│  │  ├── SQLite Database                                            │    │
│  │  └── Web Client (Port 3000)                                     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  OPTION B: Cloud GPU (Google Colab Pro / AWS)                            │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Cloud Instance (T4/A10 GPU)                                     │    │
│  │  ├── FastAPI Server                                             │    │
│  │  ├── Models                                                      │    │
│  │  └── Database                                                    │    │
│  └───────────────────────────┬─────────────────────────────────────┘    │
│                              │ ngrok/cloudflare tunnel                   │
│  ┌───────────────────────────▼─────────────────────────────────────┐    │
│  │  Local Laptop (Demo Presentation)                                │    │
│  │  └── Web Browser Client                                         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.4 Docker Deployment

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

WORKDIR /app

# Install Python
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Download models at build time
RUN python scripts/download_models.py

# Run
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  voice-agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - HF_TOKEN=${HF_TOKEN}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
```

---

## 11. Logging Specification

### 11.1 What Gets Logged

| Event | Level | File | Content |
|-------|-------|------|---------|
| Session Start | INFO | agent_log.md | Session ID, timestamp, language |
| STT Result | DEBUG | agent_log.md | Transcription, confidence, language |
| LLM Prompt | DEBUG | agent_log.md | Full prompt (sanitized) |
| LLM Response | INFO | agent_log.md | Response text, tokens used |
| Tool Call | INFO | agent_log.md | Tool name, input, output |
| TTS Generated | DEBUG | agent_log.md | Text length, audio duration |
| Error | ERROR | agent_log.md | Error type, message, stack trace |
| Latency Metrics | INFO | agent_log.md | Pipeline timings |

### 11.2 Log Format (agent_log.md)

```markdown
---
# Voice Agent Execution Log
Generated: 2026-01-30T10:30:00Z
---

## Session: abc123-def456

### Turn 1 | 10:30:15
**User Input (Hindi)**: "मुझे लाल रंग की साड़ी दिखाओ"
- STT Confidence: 0.92
- Language Detected: hi
- Processing Time: 245ms

**Tool Calls**:
1. `search_products`
   - Input: `{"query": "red saree", "category": "clothing"}`
   - Output: 5 products found
   - Time: 45ms

**Agent Response**:
> मुझे 5 लाल साड़ियाँ मिलीं। सबसे लोकप्रिय है "बनारसी सिल्क साड़ी" जो ₹2,999 में उपलब्ध है।

- TTS Duration: 3.2s audio
- Total Latency: 687ms

---

### Turn 2 | 10:30:25
...
```

### 11.3 Logging Component Responsibility

```python
# Each component logs to the centralized AgentLogger

class AgentLogger:
    """Centralized markdown logger for hackathon evaluation"""
    
    def __init__(self, log_path: str = "logs/agent_log.md"):
        self.log_path = log_path
        self.queue = asyncio.Queue()
        self._start_writer()
    
    async def log_turn(self, session_id: str, turn: TurnLog):
        await self.queue.put(("turn", session_id, turn))
    
    async def log_tool_call(self, session_id: str, tool: ToolCallLog):
        await self.queue.put(("tool", session_id, tool))
    
    async def log_error(self, session_id: str, error: ErrorLog):
        await self.queue.put(("error", session_id, error))
```

---

## 12. Data Requirements Summary

### 12.1 Required Datasets (PLEASE PROVIDE)

| Dataset | Format | Purpose | Status |
|---------|--------|---------|--------|
| Product Catalog | JSON | Product search, display | ❓ **NEEDED** |
| Product FAQs | JSON | FAQ lookup | ❓ **NEEDED** |
| Order Database | JSON | Order tracking | ❓ **NEEDED** |
| Company Policies | JSON | Policy queries | ❓ **NEEDED** |

### 12.2 Optional Datasets (For Fine-tuning)

| Dataset | Format | Purpose | Status |
|---------|--------|---------|--------|
| Rural Hindi Speech | WAV + Transcripts | STT Fine-tuning | ❓ Ask if needed |
| Rural Bengali Speech | WAV + Transcripts | STT Fine-tuning | ❓ Ask if needed |
| Rural Marathi Speech | WAV + Transcripts | STT Fine-tuning | ❓ Ask if needed |
| Domain Vocabulary | CSV | TTS Pronunciation | ❓ Ask if needed |

---

## Next Steps

1. **Receive datasets** for products, FAQs, orders, policies
2. **Implement core pipeline** with streaming
3. **Test latency** on target hardware
4. **Iterate** on error handling and edge cases
5. **Prepare demo** with sample conversations

---

*This document serves as the single source of truth for the voice agent architecture.*
