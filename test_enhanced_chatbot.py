#!/usr/bin/env python3
"""
Test script for the Enhanced Legal Chatbot
"""

def test_imports():
    """Test if all modules can be imported"""
    try:
        from enhanced_chatbot import chatbot_instance
        print("✅ Enhanced chatbot imported successfully!")
        
        from legal import chatbot as legal_chatbot
        print("✅ Legal chatbot imported successfully!")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic chatbot functionality"""
    try:
        from enhanced_chatbot import chatbot_instance
        
        # Test tone detection
        test_message = "Hello! This is awesome! 😊"
        tone = chatbot_instance.detect_tone(test_message)
        print(f"✅ Tone detection works: {tone.value}")
        
        # Test suggested questions generation
        questions = chatbot_instance.generate_suggested_questions("test", test_message)
        print(f"✅ Suggested questions generation works: {len(questions)} questions generated")
        
        # Test user profile
        profile = chatbot_instance.get_user_profile("test_user")
        print(f"✅ User profile creation works: {profile.name}")
        
        return True
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

def test_legal_facts():
    """Test legal facts database"""
    try:
        from enhanced_chatbot import chatbot_instance
        
        facts = chatbot_instance.indian_legal_facts
        print(f"✅ Legal facts database loaded: {len(facts)} categories")
        
        # Test constitution facts
        if 'constitution' in facts:
            print(f"✅ Constitution facts available: {facts['constitution']['articles']} articles")
        
        return True
    except Exception as e:
        print(f"❌ Legal facts test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Enhanced Legal Chatbot...")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Legal Facts Database", test_legal_facts)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The chatbot is ready to use.")
        print("\n🚀 To start the web interface:")
        print("   python3 app.py")
        print("\n💻 To start the CLI interface:")
        print("   python3 cli_chatbot.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()