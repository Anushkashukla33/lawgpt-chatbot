"""
RAG Pipeline with GPT4All Integration
Handles document retrieval, embeddings, and response generation
"""

import asyncio
import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
from gpt4all import GPT4All
from sentence_transformers import SentenceTransformer
import faiss
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Advanced RAG pipeline with GPT4All and vector search"""
    
    def __init__(self):
        self.gpt4all_model = None
        self.embedding_model = None
        self.vector_store = None
        self.chroma_client = None
        self.collection = None
        self.documents = []
        self.embeddings_cache = {}
        self.response_cache = {}
        self.model_name = "mistral-7b-instruct-v0.1.Q4_0.gguf"
        
    async def initialize(self):
        """Initialize all components of the RAG pipeline"""
        logger.info("🔧 Initializing RAG Pipeline...")
        
        try:
            # Initialize GPT4All model
            await self._initialize_gpt4all()
            
            # Initialize embedding model
            await self._initialize_embeddings()
            
            # Initialize vector store
            await self._initialize_vector_store()
            
            # Load or create document corpus
            await self._load_documents()
            
            logger.info("✅ RAG Pipeline initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG Pipeline: {str(e)}")
            raise
    
    async def _initialize_gpt4all(self):
        """Initialize GPT4All model"""
        try:
            # Check if model exists locally, download if not
            model_path = f"models/{self.model_name}"
            os.makedirs("models", exist_ok=True)
            
            logger.info(f"Loading GPT4All model: {self.model_name}")
            self.gpt4all_model = GPT4All(
                model_name=self.model_name,
                model_path="models/",
                allow_download=True,
                device='cpu'  # Use GPU if available
            )
            
            logger.info("🤖 GPT4All model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize GPT4All: {str(e)}")
            # Fallback to a smaller model
            try:
                self.model_name = "orca-mini-3b-gguf2-q4_0.gguf"
                self.gpt4all_model = GPT4All(
                    model_name=self.model_name,
                    model_path="models/",
                    allow_download=True
                )
                logger.info("✅ Fallback model loaded successfully!")
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {str(fallback_error)}")
                raise
    
    async def _initialize_embeddings(self):
        """Initialize sentence transformer for embeddings"""
        try:
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Embedding model loaded!")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    async def _initialize_vector_store(self):
        """Initialize ChromaDB vector store"""
        try:
            # Initialize ChromaDB
            self.chroma_client = chromadb.Client(Settings(
                persist_directory="./chroma_db",
                anonymized_telemetry=False
            ))
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="knowledge_base",
                metadata={"description": "RAG knowledge base"}
            )
            
            logger.info("✅ Vector store initialized!")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise
    
    async def _load_documents(self):
        """Load documents into the vector store"""
        try:
            # Check if we have existing documents
            if self.collection.count() > 0:
                logger.info(f"Found {self.collection.count()} existing documents in vector store")
                return
            
            # Sample knowledge base - you can expand this
            default_documents = [
                {
                    "text": "Artificial Intelligence (AI) is the simulation of human intelligence in machines. It includes machine learning, deep learning, and neural networks.",
                    "metadata": {"category": "AI", "topic": "definition"}
                },
                {
                    "text": "Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.",
                    "metadata": {"category": "ML", "topic": "definition"}
                },
                {
                    "text": "Deep Learning uses neural networks with multiple layers to model and understand complex patterns in data.",
                    "metadata": {"category": "DL", "topic": "definition"}
                },
                {
                    "text": "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language.",
                    "metadata": {"category": "NLP", "topic": "definition"}
                },
                {
                    "text": "Reinforcement Learning is a type of machine learning where agents learn to make decisions by performing actions and receiving rewards or penalties.",
                    "metadata": {"category": "RL", "topic": "definition"}
                }
            ]
            
            await self.add_documents(default_documents)
            logger.info("✅ Default knowledge base loaded!")
            
        except Exception as e:
            logger.error(f"Failed to load documents: {str(e)}")
            raise
    
    async def add_documents(self, documents: List[Dict]):
        """Add documents to the vector store"""
        try:
            texts = [doc["text"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            ids = [f"doc_{i}_{int(time.time())}" for i in range(len(documents))]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts).tolist()
            
            # Add to ChromaDB
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to knowledge base")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise
    
    async def retrieve_relevant_context(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant documents for a query"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            relevant_docs = []
            if results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    relevant_docs.append({
                        "text": doc,
                        "metadata": results["metadatas"][0][i],
                        "relevance_score": 1 - results["distances"][0][i]  # Convert distance to similarity
                    })
            
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Failed to retrieve context: {str(e)}")
            return []
    
    async def generate_response(self, query: str, user_id: str = None, session_id: str = None) -> str:
        """Generate response using RAG pipeline"""
        try:
            # Check cache first
            cache_key = f"{query}_{user_id}"
            if cache_key in self.response_cache:
                logger.info("📋 Returning cached response")
                return self.response_cache[cache_key]
            
            # Retrieve relevant context
            relevant_docs = await self.retrieve_relevant_context(query, k=3)
            
            # Build context string
            context = "\n".join([doc["text"] for doc in relevant_docs])
            
            # Create prompt
            prompt = self._build_prompt(query, context)
            
            # Generate response with GPT4All
            response = await self._generate_with_gpt4all(prompt)
            
            # Cache the response
            self.response_cache[cache_key] = response
            
            # Limit cache size
            if len(self.response_cache) > 1000:
                # Remove oldest entries
                oldest_keys = list(self.response_cache.keys())[:100]
                for key in oldest_keys:
                    del self.response_cache[key]
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            return "I apologize, but I encountered an error while processing your request. Please try again."
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build the prompt for the language model"""
        return f"""**CONTEXT:**
{context}

**INSTRUCTION:**
You are a helpful and knowledgeable AI assistant. Based on the provided context, answer the user's question in a clear, structured, and informative manner. Use **bold text** for important points and organize your response in a logical structure.

**USER QUESTION:**
{query}

**RESPONSE:**
"""
    
    async def _generate_with_gpt4all(self, prompt: str) -> str:
        """Generate response using GPT4All model"""
        try:
            # Use asyncio to run the blocking GPT4All generation in a thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.gpt4all_model.generate(
                    prompt=prompt,
                    max_tokens=512,
                    temp=0.7,
                    top_p=0.9,
                    repeat_penalty=1.1,
                    streaming=False
                )
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"GPT4All generation error: {str(e)}")
            return "I apologize, but I'm unable to generate a response at the moment. Please try again."
    
    async def health_check(self) -> bool:
        """Check if all components are healthy"""
        try:
            # Test GPT4All model
            if self.gpt4all_model is None:
                return False
            
            # Test embedding model
            if self.embedding_model is None:
                return False
            
            # Test vector store
            if self.collection is None:
                return False
            
            # Try a simple generation
            test_response = await self.generate_response("Hello, are you working?")
            
            return len(test_response) > 0
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    async def update_knowledge_base(self, new_documents: List[Dict]):
        """Add new documents to the knowledge base"""
        await self.add_documents(new_documents)
    
    async def get_knowledge_base_stats(self) -> Dict:
        """Get statistics about the knowledge base"""
        return {
            "total_documents": self.collection.count(),
            "cache_size": len(self.response_cache),
            "model_name": self.model_name
        }