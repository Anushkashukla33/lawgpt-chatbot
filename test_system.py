#!/usr/bin/env python3
"""
Test script for the Self-Learning Chatbot System
Verifies all components are working correctly
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_database():
    """Test database functionality"""
    logger.info("🗄️ Testing Database...")
    
    try:
        from src.database import DatabaseManager
        
        db = DatabaseManager(db_path="test_chatbot.db")
        await db.initialize()
        
        # Test storing interaction
        interaction_data = {
            "user_id": "test_user",
            "session_id": "test_session",
            "query": "Test question",
            "response": "Test response",
            "confidence": 0.85,
            "processing_time": 1.5
        }
        
        interaction_id = await db.store_interaction(interaction_data)
        logger.info(f"✅ Stored interaction with ID: {interaction_id}")
        
        # Test storing feedback
        await db.store_feedback("test_session", "msg_123", 5, "Great response!")
        logger.info("✅ Stored feedback successfully")
        
        # Test getting stats
        stats = await db.get_system_stats()
        logger.info(f"✅ Retrieved stats: {stats}")
        
        await db.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Database test failed: {str(e)}")
        return False

async def test_rag_pipeline():
    """Test RAG pipeline functionality"""
    logger.info("🤖 Testing RAG Pipeline...")
    
    try:
        from src.rag_pipeline import RAGPipeline
        
        rag = RAGPipeline()
        await rag.initialize()
        
        # Test generating response
        response = await rag.generate_response("What is artificial intelligence?")
        logger.info(f"✅ Generated response: {response[:100]}...")
        
        # Test health check
        health = await rag.health_check()
        logger.info(f"✅ Health check: {health}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ RAG Pipeline test failed: {str(e)}")
        return False

async def test_reinforcement_learning():
    """Test reinforcement learning functionality"""
    logger.info("🧠 Testing Reinforcement Learning...")
    
    try:
        from src.reinforcement_learning import RLTrainer
        
        rl = RLTrainer()
        await rl.initialize()
        
        # Test feature extraction
        features = rl._extract_features("Test query", "Test response")
        logger.info(f"✅ Extracted {len(features)} features")
        
        # Test confidence calculation
        confidence = await rl.calculate_confidence("Test query", "Test response")
        logger.info(f"✅ Calculated confidence: {confidence}")
        
        # Test learning from feedback
        await rl.learn_from_feedback("Test query", "Test response", 4, "Good response")
        logger.info("✅ Learned from feedback")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ RL test failed: {str(e)}")
        return False

def test_user_manager():
    """Test user manager functionality"""
    logger.info("👥 Testing User Manager...")
    
    try:
        from src.user_manager import UserManager
        
        user_mgr = UserManager()
        
        # Test creating sessions
        session_id = user_mgr.create_session("test_user")
        logger.info(f"✅ Created session: {session_id}")
        
        # Test updating session
        user_mgr.update_session(session_id, "test_user", "Hello", "Hi there!")
        logger.info("✅ Updated session with message")
        
        # Test getting session stats
        stats = user_mgr.get_session_stats()
        logger.info(f"✅ Session stats: {stats}")
        
        user_mgr.shutdown()
        return True
        
    except Exception as e:
        logger.error(f"❌ User Manager test failed: {str(e)}")
        return False

def test_response_formatter():
    """Test response formatter functionality"""
    logger.info("💎 Testing Response Formatter...")
    
    try:
        from src.response_formatter import ResponseFormatter
        
        formatter = ResponseFormatter()
        
        # Test formatting response
        raw_response = "This is important information about machine learning. It includes key concepts and main ideas."
        formatted = formatter.format_response(raw_response)
        logger.info(f"✅ Formatted response: {formatted}")
        
        # Test error response
        error_response = formatter.format_error_response("Test error message")
        logger.info("✅ Generated error response")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Response Formatter test failed: {str(e)}")
        return False

async def test_learning_system():
    """Test learning system functionality"""
    logger.info("🎓 Testing Learning System...")
    
    try:
        from src.learning_system import LearningSystem
        from src.rag_pipeline import RAGPipeline
        from src.reinforcement_learning import RLTrainer
        from src.database import DatabaseManager
        
        # Initialize components
        rag = RAGPipeline()
        rl = RLTrainer()
        db = DatabaseManager(db_path="test_learning.db")
        
        await rag.initialize()
        await rl.initialize()
        await db.initialize()
        
        learning_sys = LearningSystem(rag, rl, db)
        
        # Test processing interaction
        interaction_data = {
            "user_id": "test_user",
            "session_id": "test_session",
            "query": "Test question",
            "response": "Test response",
            "confidence": 0.75
        }
        
        feedback = await learning_sys.process_interaction(interaction_data)
        logger.info(f"✅ Processed interaction: {feedback}")
        
        # Test getting metrics
        metrics = await learning_sys.get_learning_metrics()
        logger.info(f"✅ Learning metrics: {metrics}")
        
        await db.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Learning System test failed: {str(e)}")
        return False

async def test_api_integration():
    """Test API integration"""
    logger.info("🔌 Testing API Integration...")
    
    try:
        # Import FastAPI app
        from main import app
        
        # Test that app is properly configured
        assert app.title == "Self-Learning Multi-User Chatbot"
        logger.info("✅ FastAPI app configured correctly")
        
        # Test pydantic models
        from main import ChatMessage, ChatResponse, FeedbackRequest
        
        # Test creating message
        message = ChatMessage(message="Test message")
        logger.info(f"✅ Created chat message: {message.message}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ API Integration test failed: {str(e)}")
        return False

def test_web_interface():
    """Test web interface"""
    logger.info("🌐 Testing Web Interface...")
    
    try:
        # Check if template exists and is readable
        with open("templates/chat_interface.html", "r") as f:
            content = f.read()
        
        # Basic checks
        assert "Self-Learning AI Chatbot" in content
        assert "websocket" in content.lower()
        assert "feedback" in content.lower()
        
        logger.info("✅ Web interface template is valid")
        return True
        
    except Exception as e:
        logger.error(f"❌ Web Interface test failed: {str(e)}")
        return False

async def run_all_tests():
    """Run all tests"""
    print("🧪 Self-Learning Chatbot System Tests")
    print("=====================================\n")
    
    tests = [
        ("Database", test_database),
        ("RAG Pipeline", test_rag_pipeline),
        ("Reinforcement Learning", test_reinforcement_learning),
        ("User Manager", test_user_manager),
        ("Response Formatter", test_response_formatter),
        ("Learning System", test_learning_system),
        ("API Integration", test_api_integration),
        ("Web Interface", test_web_interface),
    ]
    
    results = {}
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {str(e)}")
            results[test_name] = False
        print()
    
    end_time = time.time()
    
    # Print results
    print("📊 Test Results Summary")
    print("=" * 30)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nTests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    
    if passed == total:
        print("\n🎉 All tests passed! The system is ready to use.")
        return True
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please check the errors above.")
        return False

def main():
    """Main test function"""
    try:
        # Run tests
        success = asyncio.run(run_all_tests())
        
        # Cleanup test files
        import os
        test_files = ["test_chatbot.db", "test_learning.db"]
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)
                logger.info(f"🧹 Cleaned up {file}")
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner crashed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()