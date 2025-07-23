"""
🤖 Self-Learning Multi-User Chatbot with Reinforcement Learning
Features:
- Multi-user support with session management
- Self-learning through reinforcement learning
- RAG pipeline with GPT4All
- Real-time learning from user interactions
- Structured response formatting
"""

import asyncio
import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import uuid

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from src.rag_pipeline import RAGPipeline
from src.reinforcement_learning import RLTrainer
from src.user_manager import UserManager
from src.response_formatter import ResponseFormatter
from src.learning_system import LearningSystem
from src.database import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Self-Learning Multi-User Chatbot",
    description="Advanced chatbot with reinforcement learning and multi-user support",
    version="1.0.0"
)

# CORS middleware for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    confidence: float
    learning_feedback: Dict
    session_id: str
    timestamp: str

class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str
    rating: int  # 1-5 scale
    feedback: Optional[str] = None

# Global components
db_manager = DatabaseManager()
rag_pipeline = RAGPipeline()
rl_trainer = RLTrainer()
user_manager = UserManager()
response_formatter = ResponseFormatter()
learning_system = LearningSystem(rag_pipeline, rl_trainer, db_manager)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, str] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected for session: {session_id}")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.user_sessions:
            del self.user_sessions[session_id]
        logger.info(f"WebSocket disconnected for session: {session_id}")

    async def send_personal_message(self, message: str, session_id: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    logger.info("🚀 Starting Self-Learning Chatbot System...")
    
    # Initialize database
    await db_manager.initialize()
    
    # Load or initialize models
    await rag_pipeline.initialize()
    await rl_trainer.initialize()
    
    # Start background learning process
    asyncio.create_task(learning_system.continuous_learning_loop())
    
    logger.info("✅ System initialized successfully!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down system...")
    await learning_system.save_learned_data()
    await db_manager.close()

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Serve the main chat interface"""
    with open("templates/chat_interface.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage):
    """Main chat endpoint with learning capabilities"""
    try:
        # Generate session ID if not provided
        session_id = message.session_id or str(uuid.uuid4())
        user_id = message.user_id or f"user_{session_id[:8]}"
        
        # Process message through RAG pipeline
        start_time = time.time()
        raw_response = await rag_pipeline.generate_response(
            query=message.message,
            user_id=user_id,
            session_id=session_id
        )
        
        # Apply reinforcement learning for response optimization
        optimized_response = await rl_trainer.optimize_response(
            query=message.message,
            raw_response=raw_response,
            user_feedback_history=await db_manager.get_user_feedback_history(user_id)
        )
        
        # Format response with structure and bold formatting
        formatted_response = response_formatter.format_response(optimized_response)
        
        # Calculate confidence score
        confidence = await rl_trainer.calculate_confidence(
            query=message.message,
            response=formatted_response
        )
        
        # Store interaction for learning
        interaction_data = {
            "user_id": user_id,
            "session_id": session_id,
            "query": message.message,
            "response": formatted_response,
            "confidence": confidence,
            "processing_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat()
        }
        
        await db_manager.store_interaction(interaction_data)
        
        # Trigger online learning
        learning_feedback = await learning_system.process_interaction(interaction_data)
        
        # Update user session
        user_manager.update_session(session_id, user_id, message.message, formatted_response)
        
        return ChatResponse(
            response=formatted_response,
            confidence=confidence,
            learning_feedback=learning_feedback,
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback for reinforcement learning"""
    try:
        # Store feedback
        await db_manager.store_feedback(
            session_id=feedback.session_id,
            message_id=feedback.message_id,
            rating=feedback.rating,
            feedback_text=feedback.feedback
        )
        
        # Trigger immediate learning from feedback
        await learning_system.learn_from_feedback(feedback)
        
        return {"status": "success", "message": "Feedback received and processed"}
        
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process feedback")

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat"""
    await manager.connect(websocket, session_id)
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Create chat message
            message = ChatMessage(
                message=message_data["message"],
                session_id=session_id,
                user_id=message_data.get("user_id")
            )
            
            # Process through chat endpoint logic
            response = await chat_endpoint(message)
            
            # Send response back
            await manager.send_personal_message(
                json.dumps(response.dict()), 
                session_id
            )
            
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(session_id)

@app.get("/api/stats")
async def get_system_stats():
    """Get system learning statistics"""
    try:
        stats = await db_manager.get_system_stats()
        learning_metrics = await learning_system.get_learning_metrics()
        
        return {
            "total_interactions": stats.get("total_interactions", 0),
            "unique_users": stats.get("unique_users", 0),
            "average_confidence": stats.get("average_confidence", 0.0),
            "learning_progress": learning_metrics.get("learning_progress", 0.0),
            "model_accuracy": learning_metrics.get("model_accuracy", 0.0),
            "total_feedback": stats.get("total_feedback", 0),
            "system_uptime": learning_metrics.get("uptime", "0h 0m"),
            "last_learning_update": learning_metrics.get("last_update", "Never")
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get system stats")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "rag_pipeline": await rag_pipeline.health_check(),
            "rl_trainer": await rl_trainer.health_check(),
            "database": await db_manager.health_check(),
            "learning_system": learning_system.is_healthy()
        }
    }

@app.get("/api/model/retrain")
async def trigger_model_retrain():
    """Manually trigger model retraining"""
    try:
        await learning_system.retrain_models()
        return {"status": "success", "message": "Model retraining initiated"}
    except Exception as e:
        logger.error(f"Error triggering retrain: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to trigger retraining")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        reload=False,
        access_log=True
    )