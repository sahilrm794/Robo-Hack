"""
Download AI4Bharat models for STT and TTS.
Run this script to pre-download models before deployment.
"""

import subprocess
import sys
from pathlib import Path


def check_gpu():
    """Check if GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠️  No GPU detected. Models will run on CPU (slower).")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed. Install with: pip install torch")
        return False


def download_stt_model():
    """Download AI4Bharat STT model."""
    print("\n📥 Downloading STT Model (AI4Bharat IndicConformer)...")
    
    try:
        # Using NeMo to download
        from nemo.collections.asr.models import ASRModel
        
        model_id = "ai4bharat/indicconformer_stt-hi-hybrid_ctc_rnnt-13M"
        model = ASRModel.from_pretrained(model_id)
        
        # Save to local cache
        model_path = Path("models/stt")
        model_path.mkdir(parents=True, exist_ok=True)
        model.save_to(str(model_path / "indicconformer.nemo"))
        
        print(f"✅ STT Model saved to: {model_path}")
        
    except ImportError:
        print("⚠️  NeMo not installed. Install with:")
        print("   pip install nemo-toolkit[asr]")
    except Exception as e:
        print(f"❌ Error downloading STT model: {e}")


def download_tts_models():
    """Download TTS models/voices."""
    print("\n📥 Setting up TTS...")
    
    # For Edge-TTS, voices are fetched at runtime
    # Just verify edge-tts is installed
    try:
        import edge_tts
        print("✅ Edge-TTS available (fallback TTS)")
        
        # List available Hindi voices
        import asyncio
        async def list_voices():
            voices = await edge_tts.list_voices()
            hindi_voices = [v for v in voices if "hi-IN" in v["Locale"]]
            print(f"   Hindi voices: {len(hindi_voices)}")
            for v in hindi_voices[:3]:
                print(f"   - {v['ShortName']}")
        
        asyncio.run(list_voices())
        
    except ImportError:
        print("⚠️  Edge-TTS not installed. Install with:")
        print("   pip install edge-tts")


def setup_ai4bharat_tts():
    """Setup AI4Bharat TTS."""
    print("\n📥 Setting up AI4Bharat TTS...")
    
    try:
        # Clone and setup AI4Bharat TTS
        # Note: This requires manual setup
        print("ℹ️  AI4Bharat TTS requires manual setup:")
        print("   1. Clone: git clone https://github.com/AI4Bharat/Indic-TTS")
        print("   2. Follow installation instructions")
        print("   3. Download voice models for required languages")
        print()
        print("   Using Edge-TTS as fallback is recommended for hackathons.")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main function."""
    print("=" * 60)
    print("🔧 AI4Bharat Model Setup")
    print("=" * 60)
    
    # Check GPU
    has_gpu = check_gpu()
    
    if not has_gpu:
        print("\n⚠️  Without GPU, using mock STT and Edge-TTS fallback is recommended.")
        print("   Set in .env:")
        print("   MOCK_STT_FOR_TESTING=true")
        print("   ENABLE_FALLBACK_TTS=true")
    
    # Download models
    print("\n📦 Model Options:")
    print("1. Download STT model (requires NeMo + GPU)")
    print("2. Setup TTS (Edge-TTS fallback)")
    print("3. Skip (use mock services)")
    
    choice = input("\nSelect option (1/2/3): ").strip()
    
    if choice == "1":
        download_stt_model()
    elif choice == "2":
        download_tts_models()
    else:
        print("\n⏭️  Skipping model download.")
        print("   The app will use mock STT and Edge-TTS fallback.")
    
    print("\n✅ Setup complete!")


if __name__ == "__main__":
    main()
