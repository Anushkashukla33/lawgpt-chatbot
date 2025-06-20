
import os
from legal import chatbot
from flask import Flask, render_template, request, jsonify
import json
from datetime import datetime

app = Flask(__name__)

# Sample suggested questions
SUGGESTED_QUESTIONS = [
    "What does Article 1(1) say?",
    "Which articles in the constitution define the judges of the Supreme Court in India?",
    "Who was Ms. Ammu Swaminathan in the Constituent Assembly?",
    "What is the 38th Amendment Act of the Constitution?"
]

@app.route('/')
def home():
    """Render the main chat interface"""
    return render_template('index.html', suggested_questions=SUGGESTED_QUESTIONS)

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages via POST request"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'error': 'Message cannot be empty'
            }), 400
        
        # Generate response using your model
        bot_response = chatbot(user_message)
        
        return jsonify({
            'user_message': user_message,
            'bot_response': bot_response,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/suggestions', methods=['GET'])
def get_suggestions():
    """Get suggested questions"""
    return jsonify({
        'suggestions': SUGGESTED_QUESTIONS
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Constitution Chatbot API is running'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)