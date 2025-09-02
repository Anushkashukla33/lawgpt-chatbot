
import os
from flask import Flask, render_template, request, jsonify, session
import json
from datetime import datetime
import uuid
from enhanced_chatbot import chatbot_instance

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

@app.route('/')
def home():
    """Render the main chat interface"""
    # Generate session ID if not exists
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    
    # Get user statistics
    user_stats = chatbot_instance.get_user_statistics(session['user_id'])
    
    return render_template('index.html', 
                         user_stats=user_stats,
                         suggested_questions=chatbot_instance.suggested_questions_db["general"][:4])

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages via POST request"""
    try:
        # Ensure user has session ID
        if 'user_id' not in session:
            session['user_id'] = str(uuid.uuid4())
        
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'error': 'Message cannot be empty'
            }), 400
        
        # Process message with enhanced chatbot
        response = chatbot_instance.process_message(user_message, session['user_id'])
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/history', methods=['GET'])
def get_history():
    """Get conversation history"""
    try:
        history = chatbot_instance.get_conversation_history()
        return jsonify({
            'history': history
        })
    except Exception as e:
        return jsonify({
            'error': f'Error fetching history: {str(e)}'
        }), 500

@app.route('/user-stats', methods=['GET'])
def get_user_stats():
    """Get user statistics"""
    try:
        if 'user_id' not in session:
            session['user_id'] = str(uuid.uuid4())
        
        stats = chatbot_instance.get_user_statistics(session['user_id'])
        return jsonify(stats)
    except Exception as e:
        return jsonify({
            'error': f'Error fetching user stats: {str(e)}'
        }), 500

@app.route('/reset', methods=['POST'])
def reset_conversation():
    """Reset conversation history"""
    try:
        if 'user_id' not in session:
            session['user_id'] = str(uuid.uuid4())
        
        response = chatbot_instance.process_message("reset", session['user_id'])
        return jsonify(response)
    except Exception as e:
        return jsonify({
            'error': f'Error resetting conversation: {str(e)}'
        }), 500

@app.route('/suggestions', methods=['GET'])
def get_suggestions():
    """Get suggested questions based on context"""
    try:
        # Get recent conversation context
        history = chatbot_instance.get_conversation_history()
        context = ""
        if history:
            context = history[-1].get('user_message', '')
        
        suggestions = chatbot_instance.generate_suggested_questions(context, "")
        return jsonify({
            'suggestions': suggestions
        })
    except Exception as e:
        return jsonify({
            'error': f'Error fetching suggestions: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Enhanced Legal Chatbot API is running',
        'features': [
            'Conversation Memory',
            'Tone Adaptation',
            'Personalized Responses',
            'Suggested Questions',
            'User Profiles',
            'Legal Facts of India'
        ]
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)