#!/usr/bin/env python3
"""
🤖 Self-Learning Multi-User Chatbot Launcher
Run this script to start the chatbot system
"""

import os
import sys
import subprocess
import logging
import signal
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed"""
    logger.info("🔍 Checking dependencies...")
    
    try:
        import uvicorn
        import fastapi
        import gpt4all
        import sentence_transformers
        import chromadb
        import torch
        import sklearn
        import aiosqlite
        logger.info("✅ All core dependencies found!")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.info("💡 Please run: pip install -r requirements.txt")
        return False

def setup_directories():
    """Create necessary directories"""
    logger.info("📁 Setting up directories...")
    
    directories = [
        "models",
        "chroma_db", 
        "logs",
        "data",
        "backups"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"✅ Directory created/verified: {directory}")

def check_gpt4all_model():
    """Check if GPT4All model is available"""
    logger.info("🤖 Checking GPT4All model...")
    
    try:
        from gpt4all import GPT4All
        
        # Try to load a small model first
        model_names = [
            "orca-mini-3b-gguf2-q4_0.gguf",
            "mistral-7b-instruct-v0.1.Q4_0.gguf",
            "gpt4all-j-v1.3-groovy.gguf"
        ]
        
        for model_name in model_names:
            try:
                logger.info(f"📥 Attempting to load model: {model_name}")
                model = GPT4All(model_name, model_path="models/", allow_download=True)
                logger.info(f"✅ Model loaded successfully: {model_name}")
                return True
            except Exception as e:
                logger.warning(f"⚠️ Could not load {model_name}: {str(e)}")
                continue
        
        logger.error("❌ Could not load any GPT4All model")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error checking GPT4All: {str(e)}")
        return False

def start_chatbot(port=8000, workers=1, reload=False):
    """Start the chatbot server"""
    logger.info("🚀 Starting Self-Learning Chatbot...")
    
    try:
        # Import and run the main application
        cmd = [
            sys.executable, "-m", "uvicorn",
            "main:app",
            "--host", "0.0.0.0",
            "--port", str(port),
            "--workers", str(workers)
        ]
        
        if reload:
            cmd.append("--reload")
            
        logger.info(f"🌐 Server will be available at: http://localhost:{port}")
        logger.info("🎯 Features enabled:")
        logger.info("   • Multi-user support (up to 100 concurrent users)")
        logger.info("   • Real-time learning from feedback")
        logger.info("   • GPT4All integration (free & unlimited)")
        logger.info("   • Structured responses with bold formatting")
        logger.info("   • WebSocket support for real-time chat")
        logger.info("   • Confidence scoring and feedback system")
        
        # Start the server
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down chatbot...")
    except Exception as e:
        logger.error(f"❌ Error starting chatbot: {str(e)}")

def main():
    """Main launcher function"""
    print("🤖 Self-Learning Multi-User Chatbot")
    print("=====================================")
    print("Features:")
    print("• Reinforcement Learning from user feedback")
    print("• Multi-user support with session management") 
    print("• GPT4All integration (free & unlimited)")
    print("• RAG pipeline with vector search")
    print("• Real-time learning and model updates")
    print("• Beautiful web interface with WebSocket support")
    print("• Structured responses with bold formatting")
    print("=====================================\n")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Check GPT4All model
    logger.info("⏳ This may take a while on first run as models are downloaded...")
    if not check_gpt4all_model():
        logger.warning("⚠️ GPT4All model check failed, but continuing anyway...")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Self-Learning Chatbot Launcher")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--check-only", action="store_true", help="Only check dependencies and exit")
    
    args = parser.parse_args()
    
    if args.check_only:
        logger.info("✅ Dependency check completed successfully!")
        return
    
    # Start the chatbot
    start_chatbot(port=args.port, workers=args.workers, reload=args.reload)

if __name__ == "__main__":
    main()