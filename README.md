# Enhanced Legal AI Assistant - India

A sophisticated, interactive chatbot system designed specifically for Indian legal knowledge with advanced features including conversation memory, tone adaptation, and personalized responses.

## 🌟 Features

### Core Capabilities
- **Conversation Memory** - Remembers past interactions and user preferences
- **Tone Adaptation** - Automatically switches between Professional, Casual, and Playful tones
- **Personalized Responses** - Adapts responses based on user profile and conversation history
- **Legal Facts of India** - Comprehensive knowledge base of Indian constitution and laws
- **Smart Suggested Questions** - Context-aware follow-up questions
- **Interactive UI/UX** - Modern, responsive web interface

### Advanced Features
- **User Profiles** - Tracks conversation count, favorite topics, and preferences
- **Session Management** - Persistent user sessions with unique IDs
- **Real-time Tone Detection** - Analyzes user messages to determine appropriate tone
- **Error Handling** - Graceful error handling with user-friendly messages
- **Mobile Responsive** - Works seamlessly on all devices

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Flask
- Required dependencies (see requirements.txt)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd enhanced-legal-chatbot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
export TOGETHER_API_KEY="your-together-api-key"
```

4. **Run the application**
```bash
# Web interface
python app.py

# CLI interface
python cli_chatbot.py
```

5. **Access the web interface**
   - Open your browser and go to `http://localhost:5000`

## 📁 Project Structure

```
enhanced-legal-chatbot/
├── app.py                 # Main Flask application
├── enhanced_chatbot.py    # Enhanced chatbot core logic
├── legal.py              # Original legal chatbot (base)
├── cli_chatbot.py        # Command-line interface
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html        # Web interface template
├── ipc_vector_db/        # Vector database for legal knowledge
└── README.md             # This file
```

## 🎯 Usage Examples

### Web Interface
1. Open the web interface in your browser
2. Start chatting with the AI assistant
3. Use suggested questions for quick access
4. Monitor your conversation history and statistics

### CLI Interface
```bash
python cli_chatbot.py

# Available commands:
# - Type your message to chat
# - 'reset' - Clear conversation history
# - 'stats' - Show user statistics
# - 'history' - Show conversation history
# - 'exit' - Quit the application
```

### API Endpoints

- `GET /` - Main chat interface
- `POST /chat` - Send message and get response
- `GET /history` - Get conversation history
- `GET /user-stats` - Get user statistics
- `POST /reset` - Reset conversation
- `GET /suggestions` - Get suggested questions
- `GET /health` - Health check

## 🔧 Configuration

### Environment Variables
- `TOGETHER_API_KEY` - Your Together AI API key
- `FLASK_SECRET_KEY` - Flask session secret key (for production)

### Customization
You can customize the chatbot by modifying:
- Tone detection patterns in `enhanced_chatbot.py`
- Suggested questions database
- Legal facts database
- UI styling in `templates/index.html`

## 🎨 Tone Adaptation

The chatbot automatically detects and adapts to user tone:

### Professional Tone
- Formal language
- Structured responses
- Legal terminology
- Professional greetings

### Casual Tone
- Friendly language
- Relaxed responses
- Simple explanations
- Informal greetings

### Playful Tone
- Emoji usage
- Enthusiastic responses
- Fun expressions
- Engaging language

## 📊 User Features

### Conversation Memory
- Remembers user name and preferences
- Tracks conversation count
- Stores favorite topics
- Maintains conversation context

### Personalized Responses
- Greets returning users by name
- References previous conversations
- Adapts to user's preferred topics
- Provides contextual suggestions

## 🛠️ Development

### Adding New Features
1. Extend the `EnhancedChatbot` class in `enhanced_chatbot.py`
2. Add new API endpoints in `app.py`
3. Update the web interface in `templates/index.html`
4. Test with the CLI interface

### Testing
```bash
# Test the CLI interface
python cli_chatbot.py

# Test the web interface
python app.py
# Then visit http://localhost:5000
```

## 🔒 Security Considerations

- Change the default Flask secret key in production
- Implement proper user authentication
- Secure API endpoints
- Validate user inputs
- Use HTTPS in production

## 📈 Performance Optimization

- Implement caching for frequently accessed data
- Optimize database queries
- Use async processing for heavy operations
- Implement rate limiting
- Monitor API usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Together AI for the language model
- Hugging Face for embeddings
- FAISS for vector search
- Flask for the web framework

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the code comments

---

**Made with ❤️ for Indian Legal Education**