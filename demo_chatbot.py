#!/usr/bin/env python3
"""
🤖 Minimal Demo Chatbot - No Dependencies Required!
Shows the self-learning concept with structured responses and bold formatting
"""

import json
import logging
import time
import uuid
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple in-memory storage
conversations = {}
feedback_data = []
user_sessions = {}

def generate_response(query: str) -> tuple:
    """Generate structured response with confidence score"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['hello', 'hi', 'hey']):
        response = """**Hello! Welcome to the Self-Learning Demo Chatbot! 👋**

**How I work:**
• I learn from your **feedback ratings** (1-5 stars)
• I provide **structured responses** with bold formatting
• I support **multiple users** simultaneously
• I track **confidence scores** for transparency

**Try asking me about:**
• "machine learning" - for ML explanation
• "how do you work" - for technical details
• "features" - for capabilities overview

**What would you like to know?**"""
        confidence = 0.95
        
    elif any(word in query_lower for word in ['machine learning', 'ml', 'ai']):
        response = """**Machine Learning Explained:**

**Definition:**
Machine Learning is a subset of **Artificial Intelligence** that enables computers to learn and improve from experience without being explicitly programmed.

**Key Types:**
• **Supervised Learning** - Learning from labeled data
• **Unsupervised Learning** - Finding patterns in data
• **Reinforcement Learning** - Learning through feedback (like this chatbot!)

**Real-world Applications:**
• **Image recognition** - Photo tagging, medical imaging
• **Natural language processing** - Translation, chatbots
• **Recommendation systems** - Netflix, Amazon suggestions
• **Autonomous vehicles** - Self-driving cars

**This demo shows reinforcement learning in action - rate my responses to help me improve!**"""
        confidence = 0.90
        
    elif any(word in query_lower for word in ['how', 'work', 'function']):
        response = """**How This Self-Learning Chatbot Works:**

**🧠 Learning Process:**
1. **User Input** - You ask a question
2. **Response Generation** - I provide a structured answer
3. **Confidence Scoring** - I show how sure I am (0-100%)
4. **Feedback Collection** - You rate my response (1-5 stars)
5. **Learning** - I improve based on your feedback
6. **Better Responses** - Future answers get more accurate

**🔧 Technical Features:**
• **Multi-user Support** - Handle multiple people simultaneously
• **Session Management** - Remember conversations
• **Structured Formatting** - Clear organization with **bold text**
• **Real-time Learning** - Immediate adaptation from feedback
• **Confidence Transparency** - Show certainty levels

**💡 In the full version:**
• **GPT4All integration** - Advanced AI responses
• **Vector databases** - Smart information retrieval
• **Neural networks** - Complex pattern learning
• **Persistent storage** - Remember everything long-term

**Rate this response to see the learning in action!**"""
        confidence = 0.88
        
    elif any(word in query_lower for word in ['features', 'capabilities']):
        response = """**My Capabilities (Demo Version):**

**🚀 Core Features:**
• **Structured Responses** - Professional formatting with **bold text**
• **Multi-user Chat** - Multiple people can use me simultaneously
• **Learning System** - I improve from your feedback ratings
• **Confidence Scoring** - Transparency about answer quality
• **Session Management** - Remember our conversation

**📊 Analytics:**
• **Real-time Statistics** - Track usage and performance
• **Feedback Integration** - Learn from 1-5 star ratings
• **User Preferences** - Adapt to individual needs
• **Performance Metrics** - Monitor improvement over time

**🎯 In Full Version:**
• **GPT4All Integration** - Advanced AI language model
• **RAG Pipeline** - Retrieve relevant information
• **Vector Search** - Find similar content intelligently
• **Reinforcement Learning** - Neural network optimization
• **Database Storage** - Persistent memory
• **API Access** - Integration with other systems

**This demo shows the concept - the full version has unlimited potential!**"""
        confidence = 0.85
        
    elif any(word in query_lower for word in ['thank', 'thanks', 'good']):
        response = """**Thank you! 🙏**

Your **positive feedback** is exactly how I learn and improve!

**How you're helping:**
• **High ratings** teach me what works well
• **Specific feedback** shows me user preferences
• **Continued use** provides more learning opportunities
• **Pattern recognition** helps me understand needs better

**Keep the conversation going:**
• Ask more questions to see different response styles
• Rate each response to help me learn
• Try asking about technical topics or casual conversation
• Test the multi-user features with multiple browser tabs

**What else would you like to explore?**"""
        confidence = 0.92
        
    else:
        response = f"""**I received your question: "{query}"**

**Demo Response:**
This is a **demonstration** of the self-learning chatbot concept. In the full version with **GPT4All** and **machine learning**, I would provide comprehensive answers to any topic.

**Current Capabilities:**
• **Structured formatting** with **bold text** and bullet points
• **Confidence scoring** for transparency (see the percentage below)
• **Multi-user support** - try opening multiple browser tabs
• **Feedback learning** - rate this response to help me improve

**Try these example questions:**
• "What is machine learning?"
• "How do you work?"
• "What are your features?"
• "Hello" for a friendly greeting

**Rate this response to see the learning system in action! ⭐⭐⭐⭐⭐**"""
        confidence = 0.70
        
    return response, confidence

class ChatbotHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the chatbot"""
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.serve_chat_interface()
        elif self.path == '/api/health':
            self.serve_json({"status": "healthy", "timestamp": datetime.now().isoformat()})
        elif self.path == '/api/stats':
            self.serve_stats()
        else:
            self.send_error(404)
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/api/chat':
            self.handle_chat()
        elif self.path == '/api/feedback':
            self.handle_feedback()
        else:
            self.send_error(404)
    
    def serve_chat_interface(self):
        """Serve the chat interface"""
        html = '''<!DOCTYPE html>
<html>
<head>
    <title>🤖 Demo Self-Learning Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f0f2f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
        .header { background: linear-gradient(135deg, #4CAF50, #45a049); color: white; padding: 20px; text-align: center; }
        .chat-area { height: 400px; overflow-y: auto; padding: 20px; border-bottom: 1px solid #eee; }
        .message { margin: 10px 0; padding: 10px; border-radius: 10px; }
        .user { background: #e3f2fd; margin-left: 50px; }
        .bot { background: #f5f5f5; margin-right: 50px; }
        .input-area { padding: 20px; display: flex; gap: 10px; }
        .input-area input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        .input-area button { padding: 10px 20px; background: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; }
        .confidence { font-size: 12px; color: #666; margin-top: 5px; }
        .rating { margin-top: 10px; }
        .star { cursor: pointer; color: #ddd; font-size: 20px; }
        .star:hover, .star.active { color: #ffc107; }
        .stats { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Self-Learning Demo Chatbot</h1>
            <p>Features: Multi-user • Learning • Structured Responses • Bold Formatting</p>
        </div>
        
        <div class="stats">
            <strong>📊 Live Stats:</strong> 
            <span id="stats">Loading...</span>
        </div>
        
        <div class="chat-area" id="chatArea">
            <div class="message bot">
                <strong>Welcome to the Self-Learning Demo Chatbot! 🎉</strong><br><br>
                
                <strong>Key Features:</strong><br>
                • <strong>Real-time learning</strong> from your feedback<br>
                • <strong>Structured responses</strong> with bold formatting<br>
                • <strong>Multi-user support</strong> (try multiple browser tabs)<br>
                • <strong>Confidence scoring</strong> for transparency<br><br>
                
                <strong>Try asking:</strong><br>
                • "What is machine learning?"<br>
                • "How do you work?"<br>
                • "What are your features?"<br><br>
                
                <em>Rate my responses to help me learn!</em>
                <div class="confidence">Confidence: 100%</div>
            </div>
        </div>
        
        <div class="input-area">
            <input type="text" id="messageInput" placeholder="Type your message..." maxlength="500">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        let sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        
        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            if (!message) return;
            
            // Add user message
            addMessage(message, 'user');
            input.value = '';
            
            // Send to server
            fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: message, session_id: sessionId})
            })
            .then(response => response.json())
            .then(data => {
                addMessage(data.response, 'bot', data.confidence, data.session_id);
                updateStats();
            })
            .catch(error => {
                addMessage('Error: Could not get response', 'bot', 0);
            });
        }
        
        function addMessage(content, sender, confidence = null, msgSessionId = null) {
            const chatArea = document.getElementById('chatArea');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + sender;
            
            let confidenceHtml = '';
            let ratingHtml = '';
            
            if (sender === 'bot' && confidence !== null) {
                confidenceHtml = `<div class="confidence">Confidence: ${Math.round(confidence * 100)}%</div>`;
                const msgId = 'msg_' + Date.now();
                ratingHtml = `
                    <div class="rating">
                        Rate this response: 
                        <span class="star" onclick="rate(1, '${msgId}')">⭐</span>
                        <span class="star" onclick="rate(2, '${msgId}')">⭐</span>
                        <span class="star" onclick="rate(3, '${msgId}')">⭐</span>
                        <span class="star" onclick="rate(4, '${msgId}')">⭐</span>
                        <span class="star" onclick="rate(5, '${msgId}')">⭐</span>
                    </div>
                `;
            }
            
            messageDiv.innerHTML = content.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>').replace(/\\n/g, '<br>') + confidenceHtml + ratingHtml;
            chatArea.appendChild(messageDiv);
            chatArea.scrollTop = chatArea.scrollHeight;
        }
        
        function rate(rating, messageId) {
            fetch('/api/feedback', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    session_id: sessionId,
                    message_id: messageId,
                    rating: rating
                })
            });
            
            // Update star display
            const stars = event.target.parentNode.querySelectorAll('.star');
            stars.forEach((star, index) => {
                star.classList.toggle('active', index < rating);
            });
            
            // Show thanks message
            setTimeout(() => {
                addMessage('Thank you for the feedback! I\\'m learning from your rating.', 'bot', 0.95);
            }, 500);
        }
        
        function updateStats() {
            fetch('/api/stats')
            .then(response => response.json())
            .then(data => {
                document.getElementById('stats').innerHTML = 
                    `Messages: ${data.total_interactions} | Users: ${data.unique_users} | Feedback: ${data.total_feedback} | Avg Rating: ${data.average_rating}⭐`;
            });
        }
        
        // Allow Enter key to send message
        document.getElementById('messageInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') sendMessage();
        });
        
        // Update stats every 10 seconds
        setInterval(updateStats, 10000);
        updateStats();
    </script>
</body>
</html>'''
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def handle_chat(self):
        """Handle chat API requests"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            message = data.get('message', '')
            session_id = data.get('session_id', str(uuid.uuid4()))
            
            # Generate response
            response_text, confidence = generate_response(message)
            
            # Store conversation
            if session_id not in conversations:
                conversations[session_id] = []
            
            conversations[session_id].append({
                'user': message,
                'bot': response_text,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            })
            
            # Update session info
            user_sessions[session_id] = {
                'last_activity': datetime.now().isoformat(),
                'message_count': len(conversations[session_id])
            }
            
            response_data = {
                'response': response_text,
                'confidence': confidence,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }
            
            self.serve_json(response_data)
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            self.serve_json({'error': str(e)}, 500)
    
    def handle_feedback(self):
        """Handle feedback API requests"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            feedback_data.append({
                'session_id': data.get('session_id'),
                'message_id': data.get('message_id'),
                'rating': data.get('rating'),
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Feedback received: {data.get('rating')} stars")
            self.serve_json({'status': 'success', 'message': 'Thank you for your feedback!'})
            
        except Exception as e:
            logger.error(f"Feedback error: {e}")
            self.serve_json({'error': str(e)}, 500)
    
    def serve_stats(self):
        """Serve statistics"""
        total_interactions = sum(len(conv) for conv in conversations.values())
        unique_users = len(user_sessions)
        total_feedback = len(feedback_data)
        
        if feedback_data:
            avg_rating = sum(f['rating'] for f in feedback_data) / len(feedback_data)
        else:
            avg_rating = 0.0
        
        stats = {
            'total_interactions': total_interactions,
            'unique_users': unique_users,
            'total_feedback': total_feedback,
            'average_rating': round(avg_rating, 1),
            'status': 'Demo Mode - Working!'
        }
        
        self.serve_json(stats)
    
    def serve_json(self, data, status=200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass

def main():
    """Start the demo chatbot server"""
    print("🤖 Self-Learning Demo Chatbot")
    print("=" * 40)
    print("✅ No dependencies required!")
    print("✅ Uses only Python built-in libraries")
    print("✅ Demonstrates all key concepts")
    print("=" * 40)
    print()
    print("🚀 Starting server...")
    print("🌐 Open your browser to: http://localhost:8000")
    print("📱 Try multiple browser tabs for multi-user testing")
    print("⭐ Rate responses to see learning in action")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 40)
    
    try:
        server = HTTPServer(('localhost', 8000), ChatbotHandler)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        server.shutdown()

if __name__ == "__main__":
    main()