import os
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import random
from dataclasses import dataclass, asdict
from enum import Enum

# Import existing legal chatbot
from legal import chatbot as legal_chatbot

class Tone(Enum):
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    PLAYFUL = "playful"

@dataclass
class UserProfile:
    name: str = ""
    preferences: Dict = None
    conversation_count: int = 0
    last_interaction: str = ""
    favorite_topics: List[str] = None
    
    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}
        if self.favorite_topics is None:
            self.favorite_topics = []

@dataclass
class ChatMessage:
    user_message: str
    bot_response: str
    timestamp: str
    tone: str
    suggested_questions: List[str]

class EnhancedChatbot:
    def __init__(self):
        self.user_profiles: Dict[str, UserProfile] = {}
        self.conversation_history: List[ChatMessage] = []
        self.current_tone = Tone.PROFESSIONAL
        self.session_id = None
        
        # Legal facts of India database
        self.indian_legal_facts = {
            "constitution": {
                "adopted": "November 26, 1949",
                "enacted": "January 26, 1950",
                "articles": "395 articles",
                "amendments": "105 amendments",
                "preamble": "We, the people of India..."
            },
            "fundamental_rights": [
                "Right to Equality (Articles 14-18)",
                "Right to Freedom (Articles 19-22)",
                "Right against Exploitation (Articles 23-24)",
                "Right to Freedom of Religion (Articles 25-28)",
                "Cultural and Educational Rights (Articles 29-30)",
                "Right to Constitutional Remedies (Article 32)"
            ],
            "directive_principles": [
                "Social and Economic Justice",
                "Gandhian Principles",
                "International Peace and Security"
            ]
        }
        
        # Tone-specific response patterns
        self.tone_patterns = {
            Tone.PROFESSIONAL: {
                "greetings": ["Hello", "Good day", "Greetings"],
                "acknowledgments": ["I understand", "That's correct", "Indeed"],
                "transitions": ["Furthermore", "Additionally", "Moreover"],
                "closings": ["Is there anything else I can assist you with?", "Please let me know if you need further clarification."]
            },
            Tone.CASUAL: {
                "greetings": ["Hey there!", "Hi!", "What's up?"],
                "acknowledgments": ["Got it!", "Sure thing!", "Absolutely!"],
                "transitions": ["Also,", "Plus,", "By the way,"],
                "closings": ["Anything else on your mind?", "Need help with something else?"]
            },
            Tone.PLAYFUL: {
                "greetings": ["Hey friend! 👋", "Hello there! ✨", "Hiya! 🎉"],
                "acknowledgments": ["Awesome! 🎯", "Brilliant! ⭐", "Perfect! 🎪"],
                "transitions": ["And guess what? 🎭", "Here's the fun part! 🎨", "Plus this cool thing! 🚀"],
                "closings": ["What else can we explore together? 🌟", "Ready for more adventures? 🎢"]
            }
        }
        
        # Suggested questions database
        self.suggested_questions_db = {
            "constitutional": [
                "What are the fundamental rights in the Indian Constitution?",
                "How many articles are there in the Indian Constitution?",
                "What is the significance of Article 370?",
                "Explain the Directive Principles of State Policy"
            ],
            "legal_procedures": [
                "What is the process of filing a PIL?",
                "How does the Supreme Court work?",
                "What are the different types of writs?",
                "Explain the concept of judicial review"
            ],
            "general": [
                "Tell me about recent legal developments",
                "What are my rights as a citizen?",
                "How can I stay updated with legal changes?",
                "What should I know about consumer rights?"
            ]
        }
    
    def detect_tone(self, user_message: str) -> Tone:
        """Detect user's tone and adapt accordingly"""
        message_lower = user_message.lower()
        
        # Playful indicators
        if any(word in message_lower for word in ['😊', '😄', '😃', '😁', '🎉', '✨', 'awesome', 'cool', 'amazing', 'fantastic']):
            return Tone.PLAYFUL
        
        # Casual indicators
        if any(word in message_lower for word in ['hey', 'hi', 'whatsup', 'sup', 'cool', 'yeah', 'sure', 'okay']):
            return Tone.CASUAL
        
        # Professional indicators
        if any(word in message_lower for word in ['please', 'thank you', 'regards', 'sincerely', 'formal', 'official']):
            return Tone.PROFESSIONAL
        
        return self.current_tone
    
    def generate_suggested_questions(self, context: str, user_message: str) -> List[str]:
        """Generate contextual suggested questions"""
        questions = []
        
        # Analyze user message for context
        if any(word in user_message.lower() for word in ['constitution', 'article', 'fundamental', 'rights']):
            questions.extend(random.sample(self.suggested_questions_db["constitutional"], 2))
        elif any(word in user_message.lower() for word in ['court', 'judge', 'legal', 'procedure', 'pilot']):
            questions.extend(random.sample(self.suggested_questions_db["legal_procedures"], 2))
        else:
            questions.extend(random.sample(self.suggested_questions_db["general"], 2))
        
        # Add one personalized question based on conversation history
        if self.conversation_history:
            last_topic = self._extract_topic(self.conversation_history[-1].user_message)
            if last_topic:
                questions.append(f"Would you like to know more about {last_topic}?")
        
        return questions[:3]  # Return max 3 questions
    
    def _extract_topic(self, message: str) -> Optional[str]:
        """Extract main topic from message"""
        topics = ['constitution', 'rights', 'court', 'law', 'legal', 'judiciary', 'parliament']
        for topic in topics:
            if topic in message.lower():
                return topic
        return None
    
    def format_response(self, response: str, tone: Tone) -> str:
        """Format response according to detected tone"""
        patterns = self.tone_patterns[tone]
        
        # Add tone-appropriate elements
        if tone == Tone.PLAYFUL:
            response = f"{random.choice(patterns['greetings'])} {response}"
            if not response.endswith(('!', '😊', '✨', '🎉')):
                response += " ✨"
        elif tone == Tone.CASUAL:
            if not response.startswith(('Hey', 'Hi', 'What')):
                response = f"{random.choice(patterns['greetings'])} {response}"
        else:  # Professional
            if not response.startswith(('Hello', 'Good', 'Greetings')):
                response = f"{random.choice(patterns['greetings'])} {response}"
        
        return response
    
    def get_user_profile(self, user_id: str = "default") -> UserProfile:
        """Get or create user profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile()
        return self.user_profiles[user_id]
    
    def update_user_profile(self, user_id: str, message: str):
        """Update user profile based on interaction"""
        profile = self.get_user_profile(user_id)
        profile.conversation_count += 1
        profile.last_interaction = datetime.now().isoformat()
        
        # Extract and store favorite topics
        topic = self._extract_topic(message)
        if topic and topic not in profile.favorite_topics:
            profile.favorite_topics.append(topic)
    
    def get_personalized_greeting(self, user_id: str) -> str:
        """Generate personalized greeting based on user profile"""
        profile = self.get_user_profile(user_id)
        
        if profile.name:
            if self.current_tone == Tone.PLAYFUL:
                return f"Hey {profile.name}! Welcome back! ✨"
            elif self.current_tone == Tone.CASUAL:
                return f"Hi {profile.name}! Good to see you again!"
            else:
                return f"Hello {profile.name}. Welcome back to our legal consultation."
        
        if profile.conversation_count == 0:
            if self.current_tone == Tone.PLAYFUL:
                return "Hey there! I'm your friendly legal assistant! 🎉"
            elif self.current_tone == Tone.CASUAL:
                return "Hi! I'm here to help with your legal questions!"
            else:
                return "Greetings. I am your legal assistant, ready to provide information on Indian law and constitution."
        
        if self.current_tone == Tone.PLAYFUL:
            return f"Welcome back! This is our {profile.conversation_count}th chat! 🎪"
        elif self.current_tone == Tone.CASUAL:
            return f"Back again! We've chatted {profile.conversation_count} times now!"
        else:
            return f"Welcome back. This is our {profile.conversation_count}th consultation session."
    
    def process_message(self, user_message: str, user_id: str = "default") -> Dict:
        """Main method to process user message and generate response"""
        try:
            # Detect tone
            detected_tone = self.detect_tone(user_message)
            self.current_tone = detected_tone
            
            # Update user profile
            self.update_user_profile(user_id, user_message)
            
            # Check for special commands
            if user_message.lower().strip() == "reset":
                self.conversation_history.clear()
                return {
                    "bot_response": "Conversation reset successfully! 🔄",
                    "tone": detected_tone.value,
                    "suggested_questions": self.generate_suggested_questions("reset", user_message),
                    "timestamp": datetime.now().isoformat()
                }
            
            # Check for name setting
            name_match = re.search(r"my name is (\w+)", user_message.lower())
            if name_match:
                profile = self.get_user_profile(user_id)
                profile.name = name_match.group(1).title()
                return {
                    "bot_response": f"Nice to meet you, {profile.name}! I'll remember that. 😊",
                    "tone": detected_tone.value,
                    "suggested_questions": self.generate_suggested_questions("introduction", user_message),
                    "timestamp": datetime.now().isoformat()
                }
            
            # Check for greeting
            if any(word in user_message.lower() for word in ['hello', 'hi', 'hey', 'greetings']):
                greeting = self.get_personalized_greeting(user_id)
                return {
                    "bot_response": greeting,
                    "tone": detected_tone.value,
                    "suggested_questions": self.generate_suggested_questions("greeting", user_message),
                    "timestamp": datetime.now().isoformat()
                }
            
            # Get response from legal chatbot
            legal_response = legal_chatbot(user_message)
            
            # Format response according to tone
            formatted_response = self.format_response(legal_response, detected_tone)
            
            # Generate suggested questions
            suggested_questions = self.generate_suggested_questions(legal_response, user_message)
            
            # Create chat message
            chat_message = ChatMessage(
                user_message=user_message,
                bot_response=formatted_response,
                timestamp=datetime.now().isoformat(),
                tone=detected_tone.value,
                suggested_questions=suggested_questions
            )
            
            # Store in conversation history
            self.conversation_history.append(chat_message)
            
            return {
                "bot_response": formatted_response,
                "tone": detected_tone.value,
                "suggested_questions": suggested_questions,
                "timestamp": datetime.now().isoformat(),
                "user_profile": asdict(self.get_user_profile(user_id))
            }
            
        except Exception as e:
            error_response = f"I apologize, but I encountered an error: {str(e)}"
            if detected_tone == Tone.PLAYFUL:
                error_response += " 😅"
            return {
                "bot_response": error_response,
                "tone": detected_tone.value,
                "suggested_questions": ["Let's try a different question", "What would you like to know about?"],
                "timestamp": datetime.now().isoformat(),
                "error": True
            }
    
    def get_conversation_history(self) -> List[Dict]:
        """Get formatted conversation history"""
        return [asdict(msg) for msg in self.conversation_history]
    
    def get_user_statistics(self, user_id: str = "default") -> Dict:
        """Get user interaction statistics"""
        profile = self.get_user_profile(user_id)
        return {
            "total_conversations": profile.conversation_count,
            "favorite_topics": profile.favorite_topics,
            "last_interaction": profile.last_interaction,
            "name": profile.name
        }

# Global chatbot instance
chatbot_instance = EnhancedChatbot()