"""
Configuration file for Event Extraction System.

You can set these values:
1. Directly in this file (simple)
2. Via environment variables (recommended for production)
3. Via .env file with python-dotenv (if installed)

Priority: Environment Variables > This File
"""

import os
from pathlib import Path

# ============================================
# LLM FALLBACK CONFIGURATION
# ============================================

# Enable/disable LLM fallback for new event type detection
USE_LLM_FALLBACK = os.getenv("USE_LLM_FALLBACK", "false").lower() == "true"

# OpenAI API Key (required if USE_LLM_FALLBACK=true)
# Get your key from: https://platform.openai.com/api-keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

# Confidence threshold: if retriever score < this, trigger LLM
# Default: 0.5 (values between 0.0 and 1.0)
LLM_CONFIDENCE_THRESHOLD = float(os.getenv("LLM_CONFIDENCE_THRESHOLD", "0.5"))

# OpenAI model to use (default: gpt-4o-mini for cost efficiency)
# Options: gpt-4o-mini, gpt-4o, gpt-3.5-turbo
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ============================================
# SERVER CONFIGURATION
# ============================================

# Device to use for inference (auto-detected if not set)
DEVICE = os.getenv("DEVICE", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu")

# Maximum input length for models
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))

# Maximum output length for BART
OUTPUT_MAX_LENGTH = int(os.getenv("OUTPUT_MAX_LENGTH", "64"))

# Default top-k events to retrieve per sentence
TOP_K = int(os.getenv("TOP_K", "1"))


# ============================================
# FILE PATHS
# ============================================

# Base directory
BASE_DIR = Path(__file__).parent

# Path to trained model checkpoints
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", BASE_DIR / "checkpoints"))
RETRIEVER_CKPT_DIR = CHECKPOINT_DIR / "retrieve_best_model"
BEST_CHECKPOINT_TXT = CHECKPOINT_DIR / "best_checkpoint.txt"

# Base model names (fallback)
BART_BASE = "facebook/bart-base"
RETRIEVER_BASE = "roberta-base"

# Path to ontology file (event types + questions)
ONTOLOGY_PATH = os.getenv("ONTOLOGY_PATH", "ontoloy/event_role_WIKI_q.json")

# Path to training data and caches
TRAIN_LABELS_PATH = os.getenv("TRAIN_LABELS_PATH", "processing_data/train.json")
EVENT_TYPES_CACHE = os.getenv("EVENT_TYPES_CACHE", "processing_data/event_types.json")


# ============================================
# FRONTEND CORS
# ============================================

# Allowed origins for CORS
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS", 
    "http://localhost:5173,http://localhost:3000"
).split(",")


# ============================================
# LOGGING
# ============================================

# Log level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Enable performance logging (prints BART call counts)
ENABLE_PERFORMANCE_LOGGING = os.getenv("ENABLE_PERFORMANCE_LOGGING", "true").lower() == "true"


# ============================================
# CONFIGURATION SUMMARY
# ============================================

def print_config_summary():
    """Print configuration summary at startup"""
    print("\n" + "=" * 70)
    print("⚙️  Configuration Summary")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Max Length: {MAX_LENGTH}")
    print(f"Top K: {TOP_K}")
    print(f"LLM Fallback: {'✅ Enabled' if USE_LLM_FALLBACK else '❌ Disabled'}")
    if USE_LLM_FALLBACK:
        print(f"  - API Key: {'✅ Set' if OPENAI_API_KEY else '❌ Not set'}")
        print(f"  - Model: {OPENAI_MODEL}")
        print(f"  - Threshold: {LLM_CONFIDENCE_THRESHOLD}")
    print(f"Ontology: {ONTOLOGY_PATH}")
    print(f"Checkpoint: {CHECKPOINT_DIR}")
    print(f"CORS Origins: {', '.join(CORS_ORIGINS)}")
    print("=" * 70 + "\n")


# ============================================
# VALIDATION
# ============================================

def validate_config():
    """Validate configuration and print warnings"""
    warnings = []
    
    if USE_LLM_FALLBACK and not OPENAI_API_KEY:
        warnings.append("⚠️  USE_LLM_FALLBACK is true but OPENAI_API_KEY is not set!")
    
    if not Path(ONTOLOGY_PATH).exists():
        warnings.append(f"⚠️  Ontology file not found: {ONTOLOGY_PATH}")
    
    if not CHECKPOINT_DIR.exists():
        warnings.append(f"⚠️  Checkpoint directory not found: {CHECKPOINT_DIR}")
    
    if warnings:
        print("\n⚠️  Configuration Warnings:")
        for w in warnings:
            print(f"   {w}")
        print()
    
    return len(warnings) == 0


# ============================================
# QUICK SETUP GUIDE
# ============================================

QUICK_SETUP_GUIDE = """
Quick Setup Guide:

To enable LLM fallback:
  1. Edit config.py and set:
       USE_LLM_FALLBACK = True (line 20)
       OPENAI_API_KEY = "sk-your-key-here" (line 24)
  
  OR set environment variables:
       export USE_LLM_FALLBACK=true
       export OPENAI_API_KEY=sk-your-key-here
  
  2. Restart backend

To change device (CPU/GPU):
  export DEVICE=cuda  # or cpu, mps (Mac M1/M2)

To adjust confidence threshold:
  export LLM_CONFIDENCE_THRESHOLD=0.3  # Lower = more LLM calls
"""

# Print setup guide if LLM is disabled
if __name__ == "__main__":
    print_config_summary()
    validate_config()
    
    if not USE_LLM_FALLBACK:
        print(QUICK_SETUP_GUIDE)