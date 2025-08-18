"""
Enhanced Legal Chatbot with Mistral AI and Advanced RAG Pipeline
Features: PDF Processing, Conversation Memory, Suggestions, Multi-Modal Support
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# Core imports
import pandas as pd
import numpy as np
from langchain_community.vectorstores import FAISS, Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.schema import Document

# PDF processing
import PyPDF2
import tiktoken

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedLegalChatbot:
    """
    Advanced Legal Chatbot with RAG pipeline, conversation memory,
    and intelligent suggestion system
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the enhanced legal chatbot"""
        self.config = config or self._default_config()
        self.conversation_history = []
        self.user_context = {}
        self.setup_components()
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the chatbot"""
        return {
            "together_api_key": os.getenv('TOGETHER_API_KEY', 'a0ea6b08429661d592dab6beca8f9f95437f7efdbe794e254f4710d04b75e89d'),
            "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
            "embedding_model": "nomic-ai/nomic-embed-text-v1",
            "vector_db_path": "enhanced_vector_db",
            "memory_window": 10,
            "max_tokens": 2048,
            "temperature": 0.7,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "top_k": 6
        }
    
    def setup_components(self):
        """Initialize all chatbot components"""
        try:
            # Set API key
            os.environ['TOGETHER_API_KEY'] = self.config['together_api_key']
            
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config['embedding_model'],
                model_kwargs={
                    "trust_remote_code": True,
                    "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"
                }
            )
            
            # Initialize vector database
            self.setup_vector_database()
            
            # Initialize LLM
            self.llm = Together(
                model=self.config['model_name'],
                temperature=self.config['temperature'],
                max_tokens=self.config['max_tokens'],
                together_api_key=self.config['together_api_key']
            )
            
            # Initialize memory
            self.memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                max_token_limit=1000,
                memory_key="chat_history",
                return_messages=True
            )
            
            # Setup prompt template
            self.setup_prompt_template()
            
            # Initialize QA chain
            self.setup_qa_chain()
            
            logger.info("✅ Enhanced Legal Chatbot initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing chatbot: {str(e)}")
            raise
    
    def setup_vector_database(self):
        """Setup or load vector database"""
        try:
            if os.path.exists(self.config['vector_db_path']):
                # Load existing database
                self.vector_db = FAISS.load_local(
                    self.config['vector_db_path'], 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                logger.info("✅ Loaded existing vector database")
            else:
                # Check for legacy database
                if os.path.exists("ipc_vector_db"):
                    self.vector_db = FAISS.load_local(
                        "ipc_vector_db", 
                        self.embeddings, 
                        allow_dangerous_deserialization=True
                    )
                    logger.info("✅ Loaded legacy vector database")
                else:
                    # Create empty database
                    sample_docs = [Document(page_content="Legal chatbot initialized", metadata={"source": "system"})]
                    self.vector_db = FAISS.from_documents(sample_docs, self.embeddings)
                    logger.info("✅ Created new vector database")
            
            self.retriever = self.vector_db.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": self.config['top_k']}
            )
            
        except Exception as e:
            logger.error(f"❌ Error setting up vector database: {str(e)}")
            raise
    
    def setup_prompt_template(self):
        """Setup enhanced prompt template for legal assistance"""
        self.prompt_template = """[INST]
You are an advanced Legal Assistant AI designed to provide comprehensive legal guidance and support. Your expertise spans multiple areas of law including constitutional law, criminal law, civil law, corporate law, and legal procedures.

🎯 **Your Role & Capabilities:**
- Provide accurate legal information based on the knowledge base
- Explain complex legal concepts in simple, understandable terms
- Offer practical guidance on legal procedures and rights
- Analyze legal documents and cases when provided
- Suggest relevant follow-up questions to deepen understanding
- Maintain conversation context and build upon previous discussions

📚 **Knowledge Base Context:** {context}

💬 **Conversation History:** {chat_history}

❓ **Current Question:** {question}

🔍 **Instructions:**
1. Use the provided context to give accurate, relevant answers
2. If the question is outside the knowledge base, clearly state this limitation
3. Provide examples and analogies to clarify complex legal concepts
4. Suggest 2-3 relevant follow-up questions at the end of your response
5. Maintain a professional yet approachable tone
6. Structure your response with clear headings when appropriate

**ANSWER:**
[/INST]"""
        
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question", "chat_history"]
        )
    
    def setup_qa_chain(self):
        """Setup the conversational retrieval chain"""
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.prompt},
            verbose=True
        )
    
    def process_pdf_documents(self, pdf_paths: List[str]) -> bool:
        """
        Process PDF documents and add them to the vector database
        """
        try:
            documents = []
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config['chunk_size'],
                chunk_overlap=self.config['chunk_overlap']
            )
            
            for pdf_path in pdf_paths:
                if not os.path.exists(pdf_path):
                    logger.warning(f"⚠️ PDF file not found: {pdf_path}")
                    continue
                
                # Load PDF
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                
                # Split into chunks
                chunks = text_splitter.split_documents(pages)
                
                # Add metadata
                for chunk in chunks:
                    chunk.metadata.update({
                        "source": pdf_path,
                        "type": "legal_document",
                        "processed_date": datetime.now().isoformat()
                    })
                
                documents.extend(chunks)
                logger.info(f"✅ Processed {len(chunks)} chunks from {pdf_path}")
            
            if documents:
                # Add to vector database
                self.vector_db.add_documents(documents)
                
                # Save updated database
                self.vector_db.save_local(self.config['vector_db_path'])
                
                logger.info(f"✅ Added {len(documents)} document chunks to vector database")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error processing PDF documents: {str(e)}")
            return False
    
    def generate_suggestions(self, user_message: str, bot_response: str) -> List[str]:
        """
        Generate intelligent follow-up questions based on conversation context
        """
        try:
            # Legal domain specific suggestions
            legal_suggestions = {
                "constitution": [
                    "What are the fundamental rights under the Constitution?",
                    "How does the amendment process work?",
                    "What is the role of the Supreme Court in constitutional matters?"
                ],
                "criminal": [
                    "What are the different types of criminal offenses?",
                    "How does the criminal justice process work?",
                    "What are the rights of an accused person?"
                ],
                "civil": [
                    "What is the difference between civil and criminal law?",
                    "How do I file a civil lawsuit?",
                    "What are the remedies available in civil cases?"
                ],
                "contract": [
                    "What makes a contract legally binding?",
                    "How can a contract be terminated?",
                    "What are the remedies for breach of contract?"
                ],
                "property": [
                    "What are the different types of property rights?",
                    "How is property transferred legally?",
                    "What are the laws regarding property disputes?"
                ]
            }
            
            # Analyze message content for keywords
            message_lower = user_message.lower()
            response_lower = bot_response.lower()
            combined_text = f"{message_lower} {response_lower}"
            
            suggestions = []
            
            # Match legal domains
            for domain, domain_suggestions in legal_suggestions.items():
                if domain in combined_text:
                    suggestions.extend(domain_suggestions[:2])
            
            # Add contextual suggestions based on conversation
            if "article" in combined_text or "section" in combined_text:
                suggestions.append("Can you explain this provision in simpler terms?")
            
            if "case" in combined_text or "judgment" in combined_text:
                suggestions.append("What are similar landmark cases?")
            
            if "procedure" in combined_text or "process" in combined_text:
                suggestions.append("What documents are required for this process?")
            
            # Default suggestions if none found
            if not suggestions:
                suggestions = [
                    "Can you provide an example to illustrate this concept?",
                    "What are the practical implications of this law?",
                    "Are there any recent changes to this legal provision?"
                ]
            
            # Return top 3 unique suggestions
            return list(set(suggestions))[:3]
            
        except Exception as e:
            logger.error(f"❌ Error generating suggestions: {str(e)}")
            return [
                "Can you explain this in more detail?",
                "What are the practical applications?",
                "Are there any related legal concepts?"
            ]
    
    def chat(self, user_message: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main chat function with enhanced features
        """
        try:
            # Handle special commands
            if user_message.lower().strip() == "reset":
                return self.reset_conversation()
            
            if user_message.lower().strip() == "history":
                return self.get_conversation_history()
            
            # Update user context
            if user_context:
                self.user_context.update(user_context)
            
            # Generate response
            result = self.qa_chain.invoke({"question": user_message})
            bot_response = result["answer"]
            
            # Generate suggestions
            suggestions = self.generate_suggestions(user_message, bot_response)
            
            # Store conversation
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_message": user_message,
                "bot_response": bot_response,
                "suggestions": suggestions,
                "user_context": self.user_context.copy()
            }
            
            self.conversation_history.append(conversation_entry)
            
            # Format response
            response = {
                "user_message": user_message,
                "bot_response": bot_response,
                "suggestions": suggestions,
                "timestamp": conversation_entry["timestamp"],
                "conversation_id": len(self.conversation_history),
                "status": "success"
            }
            
            logger.info(f"✅ Generated response for: {user_message[:50]}...")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in chat function: {str(e)}")
            return {
                "user_message": user_message,
                "bot_response": f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question.",
                "suggestions": [
                    "Can you rephrase your question?",
                    "Would you like to try a different legal topic?",
                    "Do you need help with basic legal concepts?"
                ],
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }
    
    def reset_conversation(self) -> Dict[str, Any]:
        """Reset conversation memory and history"""
        self.memory.clear()
        self.conversation_history = []
        self.user_context = {}
        
        return {
            "user_message": "reset",
            "bot_response": "✅ Conversation has been reset. How can I assist you with your legal questions today?",
            "suggestions": [
                "What are my fundamental rights?",
                "How does the legal system work?",
                "Can you explain constitutional law basics?"
            ],
            "timestamp": datetime.now().isoformat(),
            "status": "reset"
        }
    
    def get_conversation_history(self) -> Dict[str, Any]:
        """Get formatted conversation history"""
        return {
            "user_message": "history",
            "bot_response": f"Here's your conversation history ({len(self.conversation_history)} messages):",
            "conversation_history": self.conversation_history,
            "suggestions": [
                "Continue our previous discussion",
                "Ask a new legal question",
                "Reset the conversation"
            ],
            "timestamp": datetime.now().isoformat(),
            "status": "history"
        }
    
    def add_documents_from_directory(self, directory_path: str) -> bool:
        """Add all PDF documents from a directory"""
        try:
            pdf_files = list(Path(directory_path).glob("*.pdf"))
            if pdf_files:
                pdf_paths = [str(pdf) for pdf in pdf_files]
                return self.process_pdf_documents(pdf_paths)
            return False
        except Exception as e:
            logger.error(f"❌ Error processing directory: {str(e)}")
            return False

# Global chatbot instance
enhanced_chatbot = None

def initialize_chatbot(config: Dict[str, Any] = None) -> EnhancedLegalChatbot:
    """Initialize the global chatbot instance"""
    global enhanced_chatbot
    if enhanced_chatbot is None:
        enhanced_chatbot = EnhancedLegalChatbot(config)
    return enhanced_chatbot

def chatbot(user_message: str, user_context: Dict[str, Any] = None) -> str:
    """
    Backward compatible function for existing app.py
    """
    global enhanced_chatbot
    if enhanced_chatbot is None:
        enhanced_chatbot = initialize_chatbot()
    
    response = enhanced_chatbot.chat(user_message, user_context)
    return response["bot_response"]

def enhanced_chat(user_message: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Enhanced chat function returning full response with suggestions
    """
    global enhanced_chatbot
    if enhanced_chatbot is None:
        enhanced_chatbot = initialize_chatbot()
    
    return enhanced_chatbot.chat(user_message, user_context)

# CLI interface for testing
if __name__ == "__main__":
    print("🚀 Enhanced Legal Chatbot with Mistral AI")
    print("=" * 50)
    print("Commands: 'reset' to clear memory, 'history' for conversation history, 'exit' to quit")
    print("=" * 50)
    
    bot = initialize_chatbot()
    
    while True:
        user_input = input("\n💬 You: ").strip()
        
        if user_input.lower() == "exit":
            print("👋 Thank you for using the Enhanced Legal Chatbot!")
            break
        
        response = bot.chat(user_input)
        
        print(f"\n🤖 Bot: {response['bot_response']}")
        
        if response.get('suggestions'):
            print(f"\n💡 Suggested questions:")
            for i, suggestion in enumerate(response['suggestions'], 1):
                print(f"   {i}. {suggestion}")