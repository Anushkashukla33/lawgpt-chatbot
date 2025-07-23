"""
🤖 Simplified Self-Learning Multi-User Chatbot
Minimal version for testing and demonstration
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Self-Learning Multi-User Chatbot (Simplified)",
    description="Simplified version for testing and demonstration",
    version="1.0.0"
)

# CORS middleware
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
    session_id: str
    timestamp: str

class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str
    rating: int
    feedback: Optional[str] = None

# Simple in-memory storage (replace with database in production)
conversations = {}
feedback_data = []
user_sessions = {}

# Simple response generator (replace with GPT4All in production)
def generate_simple_response(query: str) -> str:
    """Generate a simple response for testing"""
    query_lower = query.lower()
    
    # Simple keyword-based responses with formatting
    if any(word in query_lower for word in ['hello', 'hi', 'hey']):
        return """**Hello! Welcome to the Self-Learning Chatbot! 👋**

**How I can help you:**
• Answer questions about various topics
• Learn from your feedback to improve
• Support multiple users simultaneously
• Provide **structured responses** with formatting

**What would you like to know?**"""
    
    elif any(word in query_lower for word in ['machine learning', 'ml', 'ai', 'artificial intelligence']):
        return """**Machine Learning Overview:**

**Definition:**
Machine Learning is a subset of **Artificial Intelligence (AI)** that enables computers to learn and improve from experience without being explicitly programmed.

**Key Types:**
• **Supervised Learning** - Learning from labeled data
• **Unsupervised Learning** - Finding patterns in unlabeled data  
• **Reinforcement Learning** - Learning through rewards and penalties

**Applications:**
• Image recognition and computer vision
• Natural language processing
• Recommendation systems
• Autonomous vehicles

**This chatbot uses reinforcement learning to improve from your feedback!**"""
    
    elif any(word in query_lower for word in ['how', 'work', 'function']):
        return """**How This Chatbot Works:**

**🧠 Self-Learning Process:**
1. **User Input** - You ask a question
2. **Response Generation** - I provide an answer with confidence score
3. **Feedback Collection** - You rate my response (1-5 stars)
4. **Learning** - I improve based on your feedback
5. **Better Responses** - Future answers get more accurate

**💎 Key Features:**
• **Multi-user support** - Multiple people can chat simultaneously
• **Real-time learning** - I get better with each interaction
• **Structured responses** - Clear formatting with **bold text**
• **Confidence scoring** - Transparency about answer quality

**Try rating this response to help me learn!**"""
    
    elif any(word in query_lower for word in ['features', 'capability', 'what can you']):
        return """**My Capabilities:**

**🚀 Core Features:**
• **Multi-user chat** - Handle 100+ concurrent users
• **Self-learning** - Improve from user feedback
• **Structured responses** - Professional formatting
• **Real-time updates** - WebSocket support
• **Session management** - Remember our conversation

**🎯 Learning Features:**
• **Reinforcement Learning** - Get smarter over time
• **Confidence Scoring** - Show how sure I am
• **Feedback Integration** - Learn from your ratings
• **Pattern Recognition** - Understand user preferences

**💻 Technical Features:**
• **RAG Pipeline** - Retrieve relevant information
• **Vector Search** - Find similar content
• **Database Storage** - Remember all interactions
• **API Access** - Integrate with other systems

**Rate my responses to help me improve!**"""
    
    elif any(word in query_lower for word in ['thank', 'thanks', 'good', 'great', 'excellent']):
        return """**Thank you! 🙏**

I'm glad I could help! Your **positive feedback** helps me learn and improve.

**How you're helping me learn:**
• **High ratings** teach me what responses work well
• **Specific feedback** helps me understand user preferences  
• **Continued use** provides more learning opportunities

**Keep chatting and rating my responses - together we can make this chatbot even better!**

**What else would you like to know?**"""
    
    else:
        return f"""**I received your question: "{query}"**

**Current Response:**
This is a **demonstration response** from the simplified version of the chatbot. In the full version, I would use **GPT4All** and **RAG pipeline** to provide more sophisticated answers.

**What I can do:**
• Answer questions with **structured formatting**
• Learn from your **feedback ratings**
• Support **multiple users** simultaneously
• Provide **confidence scores** for transparency

**Rate this response to help me learn! ⭐⭐⭐⭐⭐**

*Try asking about "machine learning" or "how do you work" for better examples.*"""

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected for session: {session_id}")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        logger.info(f"WebSocket disconnected for session: {session_id}")

    async def send_personal_message(self, message: str, session_id: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(message)

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Serve the main chat interface"""
    try:
        with open("templates/chat_interface.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html><head><title>Chatbot</title></head>
        <body>
        <h1>🤖 Self-Learning Chatbot</h1>
        <p>Chat interface file not found. Please ensure templates/chat_interface.html exists.</p>
        <p>Try the API at <a href="/docs">/docs</a></p>
        </body></html>
        """)

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage):
    """Main chat endpoint"""
    try:
        # Generate session ID if not provided
        session_id = message.session_id or str(uuid.uuid4())
        user_id = message.user_id or f"user_{session_id[:8]}"
        
        # Generate response
        response_text = generate_simple_response(message.message)
        
        # Simple confidence calculation
        confidence = 0.75 + (len(message.message) % 25) / 100  # Fake confidence 0.75-0.99
        
        # Store conversation
        if session_id not in conversations:
            conversations[session_id] = []
        
        conversations[session_id].append({
            "user": message.message,
            "bot": response_text,
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence
        })
        
        # Track user session
        user_sessions[session_id] = {
            "user_id": user_id,
            "last_activity": datetime.now().isoformat(),
            "message_count": len(conversations[session_id])
        }
        
        return ChatResponse(
            response=response_text,
            confidence=confidence,
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback"""
    try:
        # Store feedback
        feedback_data.append({
            "session_id": feedback.session_id,
            "message_id": feedback.message_id,
            "rating": feedback.rating,
            "feedback": feedback.feedback,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Received feedback: {feedback.rating} stars for session {feedback.session_id}")
        
        return {"status": "success", "message": "Thank you for your feedback! 🙏"}
        
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process feedback")

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat"""
    await manager.connect(websocket, session_id)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Create chat message
            message = ChatMessage(
                message=message_data["message"],
                session_id=session_id,
                user_id=message_data.get("user_id")
            )
            
            # Process message
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
    """Get system statistics"""
    try:
        total_interactions = sum(len(conv) for conv in conversations.values())
        unique_users = len(user_sessions)
        total_feedback = len(feedback_data)
        
        if feedback_data:
            avg_rating = sum(f["rating"] for f in feedback_data) / len(feedback_data)
        else:
            avg_rating = 3.0
        
        return {
            "total_interactions": total_interactions,
            "unique_users": unique_users,
            "total_feedback": total_feedback,
            "average_rating": round(avg_rating, 2),
            "learning_progress": min(100, total_feedback * 2),  # Fake progress
            "model_accuracy": min(0.95, 0.5 + (total_feedback * 0.01)),
            "system_status": "Running (Simplified Mode)"
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return {"error": "Failed to get stats"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mode": "simplified",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(user_sessions),
        "total_conversations": len(conversations)
    }

if __name__ == "__main__":
    print("🤖 Self-Learning Multi-User Chatbot (Simplified)")
    print("=" * 50)
    print("🚀 Starting server...")
    print("🌐 Access at: http://localhost:8000")
    print("📚 API docs: http://localhost:8000/docs")
    print("💡 This is a simplified version for testing")
    print("=" * 50)
    
    uvicorn.run(
        "main_simple:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )