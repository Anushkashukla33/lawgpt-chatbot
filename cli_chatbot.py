#!/usr/bin/env python3
"""
Enhanced Legal Chatbot CLI Interface
A command-line interface for testing the enhanced chatbot features
"""

import sys
import os
from datetime import datetime
from enhanced_chatbot import chatbot_instance

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("🤖 Enhanced Legal AI Assistant - CLI Version")
    print("=" * 60)
    print("✨ Features:")
    print("   • Conversation Memory")
    print("   • Tone Adaptation (Professional/Casual/Playful)")
    print("   • Personalized Responses")
    print("   • Legal Facts of India")
    print("   • Smart Suggested Questions")
    print("=" * 60)
    print("Commands:")
    print("   • Type your message to chat")
    print("   • 'reset' - Clear conversation history")
    print("   • 'stats' - Show user statistics")
    print("   • 'history' - Show conversation history")
    print("   • 'exit' - Quit the application")
    print("=" * 60)

def print_response(response_data):
    """Print formatted bot response"""
    print(f"\n🤖 Bot ({response_data.get('tone', 'professional')}):")
    print(f"   {response_data['bot_response']}")
    
    if response_data.get('suggested_questions'):
        print(f"\n💡 Suggested Questions:")
        for i, question in enumerate(response_data['suggested_questions'], 1):
            print(f"   {i}. {question}")
    
    print("-" * 60)

def show_user_stats():
    """Display user statistics"""
    stats = chatbot_instance.get_user_statistics()
    print(f"\n📊 User Statistics:")
    print(f"   Total Conversations: {stats['total_conversations']}")
    print(f"   Name: {stats['name'] or 'Not set'}")
    print(f"   Favorite Topics: {', '.join(stats['favorite_topics']) if stats['favorite_topics'] else 'None'}")
    print(f"   Last Interaction: {stats['last_interaction'] or 'Never'}")
    print("-" * 60)

def show_history():
    """Display conversation history"""
    history = chatbot_instance.get_conversation_history()
    if not history:
        print("\n📝 No conversation history found.")
        return
    
    print(f"\n📝 Conversation History ({len(history)} messages):")
    for i, msg in enumerate(history, 1):
        print(f"\n{i}. User: {msg['user_message']}")
        print(f"   Bot ({msg['tone']}): {msg['bot_response']}")
        print(f"   Time: {msg['timestamp']}")
    print("-" * 60)

def main():
    """Main CLI loop"""
    print_banner()
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() == 'exit':
                print("\n👋 Goodbye! Thanks for using the Enhanced Legal AI Assistant!")
                break
            
            elif user_input.lower() == 'stats':
                show_user_stats()
                continue
            
            elif user_input.lower() == 'history':
                show_history()
                continue
            
            elif user_input.lower() == 'reset':
                response = chatbot_instance.process_message("reset")
                print_response(response)
                continue
            
            # Process regular message
            print(f"\n⏳ Processing...")
            response = chatbot_instance.process_message(user_input)
            print_response(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using the Enhanced Legal AI Assistant!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main()