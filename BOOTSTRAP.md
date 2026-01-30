# 🚀 Project Bootstrap Guide

**Voice-to-Voice AI Customer Support Backend**

This guide provides step-by-step instructions to set up, install, and verify the project.

---

## Step 0 – Verify Conda Environment

Before proceeding, check if you're already inside a conda environment.

```bash
# Check current conda environment
echo $CONDA_DEFAULT_ENV
```

**Expected output:**
- If inside an environment: Shows environment name (e.g., `base`, `myenv`)
- If not activated: Shows empty or `base`

```bash
# Alternative: See full conda info
conda info --envs
```

The active environment is marked with `*`.

---

## Step 1 – Project Folder Structure

The project structure is already created. Verify it exists:

```bash
# Navigate to project root
cd ~/Desktop/Robo-Hack

# Verify structure
ls -la
```

**Expected directories:**
```
app/           # Main application code
scripts/       # Utility scripts
data/          # SQLite database (created on first run)
logs/          # Markdown execution logs
models/        # Cached ML models
```

---

## Step 2 – Create Conda Environment

**Environment name:** `voicebot`  
**Python version:** 3.10 (chosen for best compatibility with PyTorch 2.x and NeMo toolkit)

```bash
# Create new conda environment with Python 3.10
conda create --name voicebot python=3.10 --yes
```

**Why Python 3.10?**
- PyTorch 2.x has best support for 3.10
- NeMo toolkit (for AI4Bharat STT) requires 3.9-3.10
- Avoids compatibility issues with newer Python versions

### Activate the Environment

> ⚠️ **Run activation ONLY if the environment is not already active**

```bash
# Check if voicebot is active
echo $CONDA_DEFAULT_ENV
```

If output is NOT `voicebot`, activate it:

```bash
conda activate voicebot
```

### Verify Environment is Active

```bash
# Should show "voicebot"
echo $CONDA_DEFAULT_ENV

# Verify Python version
python --version
# Expected: Python 3.10.x
```

---

## Step 3 – Install Dependencies

### 3.1 Core Backend Dependencies

FastAPI, Uvicorn, and async support.

```bash
# Core web framework and async utilities
pip install fastapi==0.109.0 \
    uvicorn[standard]==0.25.0 \
    python-multipart==0.0.6 \
    websockets==12.0 \
    aiohttp==3.9.1 \
    aiofiles==23.2.1 \
    anyio==4.2.0
```

**Why these versions?** Pinned for stability; tested together.

### 3.2 Database Dependencies

SQLAlchemy with async SQLite.

```bash
# Async database support
pip install sqlalchemy[asyncio]==2.0.25 aiosqlite==0.19.0
```

### 3.3 Configuration Dependencies

Pydantic for settings management.

```bash
# Configuration and environment management
pip install pydantic==2.5.3 pydantic-settings==2.1.0 python-dotenv==1.0.0
```

### 3.4 LLM Client Dependencies

Groq SDK for ultra-low latency LLM inference.

```bash
# Groq Python SDK
pip install groq==0.4.2
```

### 3.5 Audio Processing Dependencies

NumPy, SciPy, and SoundFile for audio handling.

```bash
# Audio processing libraries
pip install numpy==1.26.3 scipy==1.11.4 soundfile==0.12.1
```

### 3.6 TTS Fallback Dependencies

Edge-TTS as fallback (works without GPU).

```bash
# Microsoft Edge TTS (free, no GPU required)
pip install edge-tts==6.1.9
```

### 3.7 STT Dependencies (GPU Required)

> ⚠️ **Skip this section if you don't have a GPU. The app will use mock STT.**

```bash
# PyTorch with CUDA support (for GPU)
# Check your CUDA version first: nvidia-smi
pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# NeMo toolkit for AI4Bharat IndicConformer
pip install nemo-toolkit[asr]==1.22.0

# Indic NLP library for language processing
pip install indic-nlp-library==0.92
```

**For CPU-only systems (slower but works):**

```bash
# PyTorch CPU version
pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
```

### 3.8 Logging Dependencies

Structured logging for markdown output.

```bash
# Structured logging
pip install structlog==24.1.0
```

### 3.9 Testing Dependencies

For running tests and making HTTP requests.

```bash
# Testing utilities
pip install pytest==7.4.4 pytest-asyncio==0.23.3 httpx==0.26.0
```

### Verify All Installations

```bash
# List installed packages
pip list | grep -E "fastapi|uvicorn|groq|torch|edge-tts"
```

**Expected output:**
```
edge-tts          6.1.9
fastapi           0.109.0
groq              0.4.2
torch             2.1.2
uvicorn           0.25.0
```

---

## Step 4 – Model Download / Loading

### 4.1 HuggingFace Login

Some models require HuggingFace authentication.

```bash
# Install HuggingFace CLI
pip install huggingface-hub

# Login with token (get token from https://huggingface.co/settings/tokens)
huggingface-cli login
```

When prompted, paste your HuggingFace token.

### 4.2 Model Cache Location

Models are cached in these locations:

| Model Type | Cache Location |
|------------|----------------|
| HuggingFace | `~/.cache/huggingface/` |
| NeMo | `~/.cache/torch/NeMo/` |
| Local | `./models/` |

### 4.3 Download AI4Bharat STT Model (Optional)

> ⚠️ **Skip if using mock STT for demo**

```bash
# Download IndicConformer model
python -c "
from nemo.collections.asr.models import ASRModel
model = ASRModel.from_pretrained('ai4bharat/indicconformer_stt-hi-hybrid_ctc_rnnt-13M')
print('✅ STT Model downloaded successfully')
"
```

### 4.4 Verify Edge-TTS Voices

```bash
# List available Hindi voices
python -c "
import asyncio
import edge_tts

async def check():
    voices = await edge_tts.list_voices()
    hindi = [v for v in voices if 'hi-IN' in v['Locale']]
    print(f'✅ Found {len(hindi)} Hindi voices')
    for v in hindi[:3]:
        print(f'   - {v[\"ShortName\"]}')

asyncio.run(check())
"
```

**Expected output:**
```
✅ Found 4 Hindi voices
   - hi-IN-MadhurNeural
   - hi-IN-SwaraNeural
   ...
```

### 4.5 Dataset Requirements

> ❓ **Before proceeding, do you need to provide custom datasets?**

The seed script creates sample data. If you have specific datasets for:
- Product catalog (JSON)
- FAQs (JSON)
- Order database (JSON)
- Company policies (Markdown/JSON)

Please provide them now. Otherwise, we'll use the demo data.

---

## Step 5 – Environment Variables

### 5.1 Create .env File

```bash
# Copy example config
cp .env.example .env
```

### 5.2 Set Required Variables

Edit `.env` with your values:

```bash
# Open in editor
nano .env
# Or use: vim .env / code .env
```

**Required variables:**

```env
# Groq API Key (get from https://console.groq.com)
GROQ_API_KEY=gsk_your_key_here

# HuggingFace Token (optional, for model downloads)
HUGGINGFACE_TOKEN=hf_your_token_here

# App Settings
DEBUG=true
MOCK_STT_FOR_TESTING=true
ENABLE_FALLBACK_TTS=true
```

### 5.3 Export Variables (Alternative)

If you prefer not to use .env file:

```bash
# Export directly (add to ~/.bashrc for persistence)
export GROQ_API_KEY="gsk_your_key_here"
export HUGGINGFACE_TOKEN="hf_your_token_here"
```

### 5.4 Verify Environment Variables

```bash
# Check if variables are set
python -c "
import os
groq_key = os.getenv('GROQ_API_KEY', '')
print(f'GROQ_API_KEY: {'✅ Set' if groq_key.startswith('gsk_') else '❌ Missing'}')
"
```

---

## Step 6 – Initialize Database

### 6.1 Create Database Tables

```bash
# Initialize SQLite database
python scripts/init_db.py
```

**Expected output:**
```
🗄️  Initializing database...
✅ Database initialized successfully!
📁 Database location: data/app.db
```

### 6.2 Seed Sample Data

```bash
# Populate with demo data
python scripts/seed_data.py
```

**Expected output:**
```
🌱 Starting database seeding...

📦 Seeding products...
   ✅ Added 10 products
❓ Seeding FAQs...
   ✅ Added 10 FAQs
📜 Seeding policies...
   ✅ Added 6 policies
📋 Seeding orders...
   ✅ Added 5 orders

✅ Database seeding complete!
```

### 6.3 Verify Database

```bash
# Check database file exists
ls -la data/app.db
```

---

## Step 7 – First Runnable Backend

### 7.1 Start the Server

```bash
# Run FastAPI server with hot reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Expected startup output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345]
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Initializing services...
INFO:     ✅ STT Service initialized (mock mode)
INFO:     ✅ TTS Service initialized
INFO:     ✅ LLM Service initialized
INFO:     ✅ Tool Registry initialized with 5 tools
INFO:     Application startup complete.
```

### 7.2 Keep Server Running

Leave this terminal running. Open a **new terminal** for testing.

---

## Step 8 – Testing Process

### 8.1 Test Health Endpoint

```bash
# Basic health check
curl http://localhost:8000/api/health
```

**Expected response:**
```json
{"status":"healthy","timestamp":"2026-01-30T...","version":"1.0.0"}
```

### 8.2 Test Readiness Endpoint

```bash
# Check all services are ready
curl http://localhost:8000/api/health/ready
```

**Expected response:**
```json
{
  "status": "ready",
  "checks": {
    "stt_service": true,
    "tts_service": true,
    "llm_service": true,
    "tool_registry": true,
    "database": true
  }
}
```

### 8.3 Test LLM Endpoint (Text Conversation)

```bash
# Send a text message
curl -X POST http://localhost:8000/api/conversation/message \
  -H "Content-Type: application/json" \
  -d '{"text": "What is your return policy?", "language": "en"}'
```

**Expected response:**
```json
{
  "session_id": "abc123...",
  "response": "Our return policy allows...",
  "language": "en",
  "latency_ms": 450.25,
  "tool_calls": []
}
```

### 8.4 Test with Tool Calling

```bash
# Ask about order tracking (triggers tool call)
curl -X POST http://localhost:8000/api/conversation/message \
  -H "Content-Type: application/json" \
  -d '{"text": "Track my order. My phone is 9876543210", "language": "en"}'
```

**Expected response includes tool_calls:**
```json
{
  "response": "I found your orders...",
  "tool_calls": [
    {
      "tool": "get_orders_by_phone",
      "input": {"phone": "9876543210"},
      "output": {...}
    }
  ]
}
```

### 8.5 Test TTS Endpoint

```bash
# Test TTS synthesis
curl -X POST http://localhost:8000/api/voice/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how can I help you?", "language": "en"}'
```

### 8.6 Verify Logs are Written

```bash
# Check log file exists and has content
cat logs/agent_log.md
```

**Expected log format:**
```markdown
# 🎙️ Voice Agent Execution Log

## Execution Log

### 🆕 Session Started: `abc123`
**Timestamp:** 2026-01-30 10:30:45

### ✅ Turn Complete | 10:30:46
**Session:** `abc123`
| Metric | Value |
|--------|-------|
| Total Latency | 🟢 Excellent (450ms) |
...
```

### 8.7 Run Full Test Suite

```bash
# Run the test client
python scripts/test_client.py
```

**Expected output:**
```
============================================================
🧪 Voice Agent API Test Client
============================================================
Target: http://localhost:8000

🏥 Testing Health Endpoints...
   /api/health: 200
   /api/health/ready: 200

💬 Testing Conversation Endpoint...
   📤 User: Hello, I need help with my order
   🤖 Agent: Hello! I'd be happy to help...
   ⏱️  Latency: 423.5ms

✅ All tests completed!
```

---

## Step 9 – Common Failure Checks

### 9.1 Torch Not Detecting GPU

**Symptom:** `torch.cuda.is_available()` returns `False`

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**Fix:**
1. Verify NVIDIA drivers: `nvidia-smi`
2. Reinstall PyTorch with correct CUDA version:
   ```bash
   pip uninstall torch torchaudio
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

### 9.2 Model Load Taking Too Long

**Symptom:** First request hangs for minutes

**Cause:** Model downloading in background

**Fix:**
1. Pre-download models (Step 4.3)
2. Or use mock mode: `MOCK_STT_FOR_TESTING=true`

### 9.3 Conda Environment Mismatch

**Symptom:** `ModuleNotFoundError` for installed packages

```bash
# Verify you're in the right environment
which python
# Should show: /path/to/conda/envs/voicebot/bin/python
```

**Fix:**
```bash
conda activate voicebot
```

### 9.4 Missing Environment Variables

**Symptom:** `ValidationError: GROQ_API_KEY field required`

```bash
# Check if .env file exists
ls -la .env

# Check if variable is loaded
python -c "from app.config import get_settings; print(get_settings().GROQ_API_KEY[:10])"
```

**Fix:** Create `.env` file (Step 5.1)

### 9.5 Port Already in Use

**Symptom:** `Address already in use`

```bash
# Find process using port 8000
lsof -i :8000

# Kill it
kill -9 <PID>
```

Or use a different port:
```bash
uvicorn app.main:app --reload --port 8001
```

### 9.6 Database Errors

**Symptom:** `sqlite3.OperationalError: no such table`

**Fix:** Re-initialize database:
```bash
rm data/app.db
python scripts/init_db.py
python scripts/seed_data.py
```

---

## Step 10 – Quick Reference Commands

```bash
# Activate environment (if not active)
conda activate voicebot

# Start server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
python scripts/test_client.py

# View logs
tail -f logs/agent_log.md

# Reset database
rm data/app.db && python scripts/init_db.py && python scripts/seed_data.py

# Check all services
curl http://localhost:8000/api/health/ready
```

---

## ✅ Bootstrap Complete!

Your voice agent backend is ready. Next steps:
1. Get a Groq API key from https://console.groq.com
2. Add it to `.env`
3. Start the server
4. Test with `curl` or the test client

For hackathon demo:
- Use mock STT mode (no GPU needed)
- Edge-TTS works great for demos
- Show the markdown logs to judges
