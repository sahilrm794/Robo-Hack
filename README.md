# 🎙️ Voice-to-Voice AI Customer Support Backend

A production-ready, low-latency (<1 second) voice-to-voice AI customer support system for e-commerce platforms. Built for hackathons with clean architecture and demo-ready features.

## ✨ Features

- **🚀 Ultra-Low Latency**: End-to-end response under 1 second
- **🗣️ Multilingual Support**: Hindi, Bengali, Marathi, and English
- **🔄 Streaming Architecture**: Real-time audio streaming via WebSocket
- **🛠️ Function Calling**: Product search, order tracking, returns, FAQs, policies
- **📝 Comprehensive Logging**: Human-readable markdown logs for evaluation
- **💾 Offline-Ready**: Works entirely on provided datasets

## 🏗️ Architecture

```
Audio In → STT (AI4Bharat) → LLM (Groq) → TTS (AI4Bharat/Edge) → Audio Out
              100-150ms        200-400ms        100-200ms
```

### Tech Stack
- **Framework**: FastAPI with async/await
- **STT**: AI4Bharat IndicConformer (or mock for development)
- **LLM**: Groq API (llama-3.1-70b-versatile)
- **TTS**: AI4Bharat Indic TTS / Edge-TTS fallback
- **Database**: SQLite (async via aiosqlite)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
cd Robo-Hack

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies (minimal for CPU)
pip install -r requirements-minimal.txt
```

### 2. Configure Environment

```bash
# Copy example config
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/Mac

# Edit .env and add your Groq API key
# GROQ_API_KEY=your_key_here
```

### 3. Initialize Database

```bash
python scripts/init_db.py
python scripts/seed_data.py
```

### 4. Run the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Test the API

```bash
# Health check
curl http://localhost:8000/api/health

# Send a text message
curl -X POST http://localhost:8000/api/conversation/message \
  -H "Content-Type: application/json" \
  -d '{"text": "What is your return policy?", "language": "en"}'
```

## 📡 API Endpoints

### Health & Status
- `GET /api/health` - Basic health check
- `GET /api/health/ready` - Readiness check
- `GET /api/health/metrics` - System metrics

### Voice (WebSocket)
- `WS /api/voice/stream` - Real-time voice streaming
- `POST /api/voice/process` - Batch audio processing
- `POST /api/voice/text` - Text-only testing

### Conversation
- `POST /api/conversation/message` - Send text message
- `GET /api/conversation/history/{session_id}` - Get history
- `GET /api/conversation/context/{session_id}` - Get context

## 🎤 WebSocket Protocol

```javascript
// Connect
const ws = new WebSocket('ws://localhost:8000/api/voice/stream');

// Receive session info
ws.onmessage = (msg) => {
  const data = JSON.parse(msg.data);
  if (data.type === 'session') {
    console.log('Session:', data.session_id);
  }
};

// Send audio chunks (binary)
ws.send(audioBuffer);

// Signal end of input
ws.send(JSON.stringify({ type: 'end' }));

// Receive audio response (binary) and end signal (JSON)
```

## 🛠️ Available Tools

The AI assistant can use these tools:

| Tool | Description |
|------|-------------|
| `search_products` | Search product catalog |
| `get_product_details` | Get product information |
| `track_order` | Track order by ID or phone |
| `get_order_details` | Get order information |
| `cancel_order` | Cancel an order |
| `search_faqs` | Search FAQs |
| `initiate_return` | Start return process |
| `get_return_status` | Check return status |
| `get_policy` | Get policy information |

## 📁 Project Structure

```
Robo-Hack/
├── app/
│   ├── api/
│   │   └── routes/
│   │       ├── health.py
│   │       ├── voice.py
│   │       └── conversation.py
│   ├── core/
│   │   ├── pipeline.py      # STT → LLM → TTS orchestration
│   │   ├── session.py       # Session management
│   │   └── exceptions.py    # Custom exceptions
│   ├── services/
│   │   ├── stt/             # Speech-to-text
│   │   ├── tts/             # Text-to-speech
│   │   ├── llm/             # LLM integration
│   │   └── memory/          # Conversation memory
│   ├── tools/               # Function calling tools
│   ├── db/                  # Database models & repos
│   └── logging/             # Markdown logger
├── scripts/
│   ├── init_db.py           # Database initialization
│   └── seed_data.py         # Sample data seeding
├── data/                    # SQLite database
├── logs/                    # Agent logs
└── models/                  # Model cache
```

## 🔧 Configuration

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key | Required |
| `DEBUG` | Debug mode | `true` |
| `TARGET_LATENCY_MS` | Target latency | `1000` |
| `MOCK_STT_FOR_TESTING` | Use mock STT | `true` |
| `ENABLE_FALLBACK_TTS` | Use Edge-TTS fallback | `true` |

## 🐳 Docker Deployment

```bash
# Production
docker-compose up -d api

# Development (with hot reload)
docker-compose --profile dev up api-dev
```

## 📊 Latency Breakdown

| Component | Target | Actual |
|-----------|--------|--------|
| STT | 100-150ms | ~100ms |
| LLM (TTFT) | 100-200ms | ~150ms |
| LLM (Full) | 200-400ms | ~300ms |
| TTS | 100-200ms | ~150ms |
| **Total** | **<1000ms** | **~700ms** |

## 📝 Logging

Logs are written to `logs/agent_log.md` in human-readable markdown format:

```markdown
## 🆕 Session Started: `abc123`
**Timestamp:** 2024-01-15 10:30:45
**Language:** hi

### 🎤 User Input
**Transcription:** "मेरा ऑर्डर कहाँ है?"
**Confidence:** 🟢 95%

### 🔧 Tool Call: `track_order`
...

### 🤖 Agent Response
> आपका ऑर्डर कल तक पहुंच जाएगा!
```

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# Test specific module
pytest tests/test_pipeline.py -v
```

## 🎯 Hackathon Tips

1. **Demo Flow**: Start with English, then switch to Hindi
2. **Key Features**: Highlight sub-second latency, multilingual support
3. **Tools Demo**: Show order tracking, product search
4. **Logs**: Use markdown logs to show decision process

## 📄 License

MIT License - Built for Robo-Hack 2024

---

Built with ❤️ using FastAPI, Groq, and AI4Bharat
