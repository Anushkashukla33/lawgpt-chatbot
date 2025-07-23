# 🚀 VS Code Setup Guide for Self-Learning Chatbot

## ✅ **Quick Status Check**
- ✅ Python 3.13.3 is installed
- ✅ All project files are present
- ✅ Code is ready to run

## 📋 **Step-by-Step Setup in VS Code**

### **Step 1: Open Project in VS Code**
```bash
# Option 1: If you have VS Code command line
code .

# Option 2: Open VS Code manually
# File -> Open Folder -> Select this project folder
```

### **Step 2: Install Dependencies**
Open VS Code terminal (`Ctrl+`` or `View -> Terminal`) and run:

```bash
# Install all required packages
pip3 install -r requirements.txt

# If you get permission errors, use:
pip3 install --user -r requirements.txt
```

### **Step 3: Test the System** 
Run the test script to make sure everything works:

```bash
python3 test_system.py
```

### **Step 4: Start the Chatbot**
```bash
# Simple start
python3 run_chatbot.py

# Or with custom settings
python3 run_chatbot.py --port 8080 --workers 1
```

### **Step 5: Access the Chatbot**
Open your browser and go to:
- **http://localhost:8000** (main chat interface)
- **http://localhost:8000/docs** (API documentation)

---

## 🛠️ **VS Code Recommended Settings**

### **Install VS Code Extensions:**
1. **Python** (Microsoft) - Essential for Python development
2. **Pylance** - Advanced Python language support
3. **autoDocstring** - Generate docstrings automatically
4. **GitLens** - Enhanced Git capabilities

### **VS Code Settings (settings.json):**
```json
{
    "python.defaultInterpreterPath": "/usr/bin/python3",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "files.autoSave": "afterDelay",
    "files.autoSaveDelay": 1000
}
```

---

## 🐛 **Troubleshooting Common Issues**

### **Issue 1: Dependencies Not Installing**
```bash
# Try upgrading pip first
pip3 install --upgrade pip

# Install each requirement individually if bulk install fails
pip3 install fastapi uvicorn
pip3 install gpt4all sentence-transformers
pip3 install chromadb torch scikit-learn
```

### **Issue 2: GPT4All Model Download Issues**
```bash
# Create models directory manually
mkdir -p models

# The first run will download models automatically
# This may take 5-10 minutes depending on your internet speed
```

### **Issue 3: Port Already in Use**
```bash
# Use a different port
python3 run_chatbot.py --port 8080

# Or find and kill the process using port 8000
lsof -ti:8000 | xargs kill -9
```

### **Issue 4: Python Command Not Found**
```bash
# Use python3 instead of python
python3 run_chatbot.py

# Or create an alias
echo "alias python=python3" >> ~/.bashrc
source ~/.bashrc
```

---

## 🎯 **Quick Test Commands**

### **Test Individual Components:**
```bash
# Test database
python3 -c "from src.database import DatabaseManager; import asyncio; asyncio.run(DatabaseManager().initialize())"

# Test RAG pipeline
python3 -c "from src.rag_pipeline import RAGPipeline; print('RAG OK')"

# Test web interface
python3 -c "with open('templates/chat_interface.html') as f: print('Web interface OK')"
```

### **Check System Health:**
```bash
# After starting the server, test the health endpoint
curl http://localhost:8000/api/health
```

---

## 🔧 **Development Mode**

### **Run in Development Mode with Auto-Reload:**
```bash
python3 run_chatbot.py --reload
```

### **Debug Mode:**
```bash
# Enable debug logging
export CHATBOT_LOG_LEVEL=DEBUG
python3 run_chatbot.py
```

### **VS Code Debug Configuration (.vscode/launch.json):**
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Chatbot",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/run_chatbot.py",
            "args": ["--reload"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        }
    ]
}
```

---

## 📊 **Expected Output When Running**

When you run `python3 run_chatbot.py`, you should see:

```
🤖 Self-Learning Multi-User Chatbot
=====================================
Features:
• Reinforcement Learning from user feedback
• Multi-user support with session management
• GPT4All integration (free & unlimited)
• RAG pipeline with vector search
• Real-time learning and model updates
• Beautiful web interface with WebSocket support
• Structured responses with bold formatting
=====================================

2024-07-21 07:25:00,000 - __main__ - INFO - 🔍 Checking dependencies...
2024-07-21 07:25:01,000 - __main__ - INFO - ✅ All core dependencies found!
2024-07-21 07:25:01,000 - __main__ - INFO - 📁 Setting up directories...
2024-07-21 07:25:01,000 - __main__ - INFO - 🤖 Checking GPT4All model...
2024-07-21 07:25:05,000 - __main__ - INFO - 🚀 Starting Self-Learning Chatbot...
2024-07-21 07:25:05,000 - __main__ - INFO - 🌐 Server will be available at: http://localhost:8000

INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## 🎉 **Success! Your Chatbot is Running**

Once you see "Uvicorn running", your chatbot is live! 

### **What to do next:**
1. ✅ Open http://localhost:8000 in your browser
2. ✅ Start chatting with the AI
3. ✅ Rate responses to help it learn
4. ✅ Watch it improve over time!

### **Key Features to Try:**
- Ask questions and see **bold formatted** responses
- Rate responses with 1-5 stars
- Check the statistics panel (📊 button)
- Try the quick action buttons
- Open multiple browser tabs to test multi-user support

---

## 📞 **Need Help?**

If you encounter any issues:

1. **Check the terminal output** for error messages
2. **Run the test script**: `python3 test_system.py`
3. **Check dependencies**: All packages in requirements.txt must be installed
4. **Verify Python version**: Must be Python 3.8 or higher
5. **Free up memory**: Close other applications if needed

The system is designed to work out of the box, so if you follow these steps, it should run smoothly! 🚀