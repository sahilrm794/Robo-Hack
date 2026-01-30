#!/usr/bin/env bash
# =============================================================================
# Voice Agent - Quick Setup Script
# =============================================================================
# This script automates the bootstrap process for the voice agent backend.
# Run with: bash scripts/setup.sh
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "🎙️  Voice Agent - Quick Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# =============================================================================
# Step 1: Check Prerequisites
# =============================================================================
echo ""
echo "📋 Step 1: Checking prerequisites..."

# Check conda
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed. Please install Miniconda or Anaconda first."
    exit 1
fi
print_status "Conda found: $(conda --version)"

# Check current environment
CURRENT_ENV=${CONDA_DEFAULT_ENV:-"none"}
echo "   Current conda environment: $CURRENT_ENV"

# =============================================================================
# Step 2: Create/Activate Environment
# =============================================================================
echo ""
echo "🐍 Step 2: Setting up conda environment..."

ENV_NAME="voicebot"

# Check if environment exists
if conda env list | grep -q "^${ENV_NAME} "; then
    print_warning "Environment '$ENV_NAME' already exists."
    read -p "   Recreate it? (y/N): " recreate
    if [[ $recreate =~ ^[Yy]$ ]]; then
        echo "   Removing existing environment..."
        conda env remove --name $ENV_NAME --yes
        echo "   Creating new environment..."
        conda create --name $ENV_NAME python=3.10 --yes
    fi
else
    echo "   Creating environment '$ENV_NAME' with Python 3.10..."
    conda create --name $ENV_NAME python=3.10 --yes
fi

# Activate environment
echo "   Activating environment..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

print_status "Environment active: $CONDA_DEFAULT_ENV"
print_status "Python version: $(python --version)"

# =============================================================================
# Step 3: Install Dependencies
# =============================================================================
echo ""
echo "📦 Step 3: Installing dependencies..."

# Upgrade pip
pip install --upgrade pip

# Install from requirements
echo "   Installing core dependencies..."
pip install -r requirements-minimal.txt

print_status "Dependencies installed"

# =============================================================================
# Step 4: Setup Environment Variables
# =============================================================================
echo ""
echo "🔐 Step 4: Setting up environment variables..."

if [ ! -f .env ]; then
    cp .env.example .env
    print_warning ".env file created from template"
    echo ""
    echo "   ⚠️  IMPORTANT: Edit .env and add your GROQ_API_KEY"
    echo "   Get your API key from: https://console.groq.com"
    echo ""
    read -p "   Press Enter when you've added your API key..."
else
    print_status ".env file already exists"
fi

# =============================================================================
# Step 5: Initialize Database
# =============================================================================
echo ""
echo "🗄️  Step 5: Initializing database..."

python scripts/init_db.py
python scripts/seed_data.py

print_status "Database initialized with sample data"

# =============================================================================
# Step 6: Verify Installation
# =============================================================================
echo ""
echo "🔍 Step 6: Verifying installation..."

# Check imports
python -c "
import fastapi
import uvicorn
import groq
import edge_tts
import sqlalchemy
print('All core imports successful')
" && print_status "Core imports OK" || print_error "Import check failed"

# Check database
if [ -f "data/app.db" ]; then
    print_status "Database file exists"
else
    print_error "Database file not found"
fi

# =============================================================================
# Step 7: Summary
# =============================================================================
echo ""
echo "=============================================="
echo "🎉 Setup Complete!"
echo "=============================================="
echo ""
echo "To start the server, run:"
echo ""
echo "   conda activate voicebot"
echo "   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "Then test with:"
echo ""
echo "   curl http://localhost:8000/api/health"
echo ""
echo "Or run the test client:"
echo ""
echo "   python scripts/test_client.py"
echo ""
