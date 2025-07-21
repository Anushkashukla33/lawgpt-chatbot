"""
Reinforcement Learning Trainer for Self-Learning Chatbot
Uses user feedback to improve response quality over time
"""

import asyncio
import logging
import pickle
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import json

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import gymnasium as gym

logger = logging.getLogger(__name__)

class ResponseQualityNetwork(nn.Module):
    """Neural network to predict response quality"""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super(ResponseQualityNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

class RLTrainer:
    """Reinforcement Learning trainer for chatbot optimization"""
    
    def __init__(self):
        self.quality_network = None
        self.optimizer = None
        self.scaler = StandardScaler()
        self.feature_extractor = None
        self.response_history = []
        self.feedback_history = []
        self.learning_rate = 0.001
        self.batch_size = 32
        self.memory_size = 10000
        self.epsilon = 0.1  # Exploration rate
        self.gamma = 0.95   # Discount factor
        self.model_path = "models/rl_quality_model.pth"
        self.is_initialized = False
        
        # Feedback weights
        self.feedback_weights = {
            1: -1.0,  # Very bad
            2: -0.5,  # Bad
            3: 0.0,   # Neutral
            4: 0.5,   # Good
            5: 1.0    # Excellent
        }
        
    async def initialize(self):
        """Initialize the RL trainer"""
        logger.info("🧠 Initializing RL Trainer...")
        
        try:
            # Create models directory
            os.makedirs("models", exist_ok=True)
            
            # Initialize neural network
            self.quality_network = ResponseQualityNetwork()
            self.optimizer = optim.Adam(
                self.quality_network.parameters(), 
                lr=self.learning_rate
            )
            
            # Load existing model if available
            await self._load_model()
            
            # Initialize feature extraction (simple for now)
            await self._initialize_feature_extractor()
            
            self.is_initialized = True
            logger.info("✅ RL Trainer initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RL Trainer: {str(e)}")
            raise
    
    async def _load_model(self):
        """Load existing model if available"""
        try:
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path)
                self.quality_network.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.response_history = checkpoint.get('response_history', [])
                self.feedback_history = checkpoint.get('feedback_history', [])
                logger.info("📦 Loaded existing RL model")
            else:
                logger.info("🆕 No existing model found, starting fresh")
                
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
    
    async def _save_model(self):
        """Save the current model"""
        try:
            torch.save({
                'model_state_dict': self.quality_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'response_history': self.response_history[-1000:],  # Keep last 1000
                'feedback_history': self.feedback_history[-1000:],
                'epoch': len(self.feedback_history)
            }, self.model_path)
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
    
    async def _initialize_feature_extractor(self):
        """Initialize simple feature extraction"""
        try:
            # For now, use simple text features
            # In production, you'd use more sophisticated embeddings
            self.feature_extractor = {
                'length_weight': 0.1,
                'sentiment_weight': 0.3,
                'relevance_weight': 0.6
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize feature extractor: {str(e)}")
            raise
    
    def _extract_features(self, query: str, response: str) -> np.ndarray:
        """Extract features from query and response"""
        try:
            features = []
            
            # Length features
            query_len = len(query.split())
            response_len = len(response.split())
            features.extend([query_len, response_len, response_len / max(query_len, 1)])
            
            # Simple word overlap (relevance proxy)
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            overlap = len(query_words.intersection(response_words))
            features.append(overlap / max(len(query_words), 1))
            
            # Response structure features
            has_bold = "**" in response
            has_structure = any(char in response for char in ["1.", "2.", "-", "•"])
            features.extend([int(has_bold), int(has_structure)])
            
            # Sentiment features (simple)
            positive_words = ["good", "great", "excellent", "helpful", "useful", "clear"]
            negative_words = ["bad", "poor", "unclear", "confusing", "wrong", "error"]
            
            positive_count = sum(1 for word in positive_words if word in response.lower())
            negative_count = sum(1 for word in negative_words if word in response.lower())
            
            features.extend([positive_count, negative_count])
            
            # Pad to fixed size (512 features)
            while len(features) < 512:
                features.append(0.0)
            
            return np.array(features[:512], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            return np.zeros(512, dtype=np.float32)
    
    async def optimize_response(self, query: str, raw_response: str, user_feedback_history: List[Dict]) -> str:
        """Optimize response based on learned preferences"""
        try:
            if not self.is_initialized:
                return raw_response
            
            # Extract features
            features = self._extract_features(query, raw_response)
            
            # Predict quality score
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                quality_score = self.quality_network(features_tensor).item()
            
            # If quality is low, try to improve the response
            if quality_score < 0.6:
                optimized_response = await self._improve_response(query, raw_response, user_feedback_history)
                return optimized_response
            
            return raw_response
            
        except Exception as e:
            logger.error(f"Response optimization error: {str(e)}")
            return raw_response
    
    async def _improve_response(self, query: str, response: str, user_feedback_history: List[Dict]) -> str:
        """Improve response based on learning"""
        try:
            # Analyze user feedback patterns
            feedback_patterns = self._analyze_feedback_patterns(user_feedback_history)
            
            improved_response = response
            
            # Apply improvements based on patterns
            if feedback_patterns.get('prefers_structure', False):
                improved_response = self._add_structure(improved_response)
            
            if feedback_patterns.get('prefers_bold', False):
                improved_response = self._add_bold_formatting(improved_response)
            
            if feedback_patterns.get('prefers_detailed', False):
                improved_response = self._add_details(improved_response)
            
            return improved_response
            
        except Exception as e:
            logger.error(f"Response improvement error: {str(e)}")
            return response
    
    def _analyze_feedback_patterns(self, feedback_history: List[Dict]) -> Dict:
        """Analyze user feedback to identify preferences"""
        patterns = {
            'prefers_structure': False,
            'prefers_bold': False,
            'prefers_detailed': False
        }
        
        if not feedback_history:
            return patterns
        
        try:
            high_rated = [f for f in feedback_history if f.get('rating', 3) >= 4]
            
            if len(high_rated) > 0:
                # Check for structure preference
                structured_responses = sum(1 for f in high_rated 
                                         if any(char in f.get('response', '') for char in ["1.", "2.", "-", "•"]))
                patterns['prefers_structure'] = structured_responses / len(high_rated) > 0.6
                
                # Check for bold preference
                bold_responses = sum(1 for f in high_rated if "**" in f.get('response', ''))
                patterns['prefers_bold'] = bold_responses / len(high_rated) > 0.6
                
                # Check for detail preference
                detailed_responses = sum(1 for f in high_rated if len(f.get('response', '').split()) > 50)
                patterns['prefers_detailed'] = detailed_responses / len(high_rated) > 0.6
            
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern analysis error: {str(e)}")
            return patterns
    
    def _add_structure(self, response: str) -> str:
        """Add structure to response"""
        try:
            lines = response.split('\n')
            if len(lines) <= 2:
                return response
            
            # Add bullet points if not present
            structured_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith(('•', '-', '1.', '2.', '3.')):
                    if len(structured_lines) == 0:
                        structured_lines.append(line)
                    else:
                        structured_lines.append(f"• {line}")
                else:
                    structured_lines.append(line)
            
            return '\n'.join(structured_lines)
            
        except Exception as e:
            logger.error(f"Structure addition error: {str(e)}")
            return response
    
    def _add_bold_formatting(self, response: str) -> str:
        """Add bold formatting to key points"""
        try:
            # Simple bold addition for key terms
            key_terms = ["important", "key", "main", "primary", "essential", "crucial"]
            
            for term in key_terms:
                if term in response.lower() and f"**{term}" not in response.lower():
                    response = response.replace(term, f"**{term}**")
                    response = response.replace(term.capitalize(), f"**{term.capitalize()}**")
            
            return response
            
        except Exception as e:
            logger.error(f"Bold formatting error: {str(e)}")
            return response
    
    def _add_details(self, response: str) -> str:
        """Add more details to response"""
        try:
            if len(response.split()) < 30:
                response += "\n\n**Additional Information:** This topic involves multiple aspects that you might find interesting to explore further."
            
            return response
            
        except Exception as e:
            logger.error(f"Detail addition error: {str(e)}")
            return response
    
    async def calculate_confidence(self, query: str, response: str) -> float:
        """Calculate confidence score for a response"""
        try:
            if not self.is_initialized:
                return 0.5  # Default confidence
            
            features = self._extract_features(query, response)
            
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                confidence = self.quality_network(features_tensor).item()
            
            return max(0.1, min(0.99, confidence))  # Clamp between 0.1 and 0.99
            
        except Exception as e:
            logger.error(f"Confidence calculation error: {str(e)}")
            return 0.5
    
    async def learn_from_feedback(self, query: str, response: str, rating: int, feedback_text: str = None):
        """Learn from user feedback"""
        try:
            if not self.is_initialized:
                return
            
            # Extract features
            features = self._extract_features(query, response)
            
            # Convert rating to target score
            target_score = self.feedback_weights.get(rating, 0.0)
            target_score = (target_score + 1) / 2  # Normalize to [0, 1]
            
            # Store in memory
            self.response_history.append({
                'query': query,
                'response': response,
                'features': features.tolist(),
                'rating': rating,
                'target_score': target_score,
                'timestamp': datetime.now().isoformat()
            })
            
            self.feedback_history.append({
                'rating': rating,
                'feedback_text': feedback_text,
                'timestamp': datetime.now().isoformat()
            })
            
            # Limit memory size
            if len(self.response_history) > self.memory_size:
                self.response_history = self.response_history[-self.memory_size:]
            
            if len(self.feedback_history) > self.memory_size:
                self.feedback_history = self.feedback_history[-self.memory_size:]
            
            # Train if we have enough data
            if len(self.response_history) >= self.batch_size:
                await self._train_step()
            
            # Save model periodically
            if len(self.response_history) % 100 == 0:
                await self._save_model()
            
        except Exception as e:
            logger.error(f"Learning from feedback error: {str(e)}")
    
    async def _train_step(self):
        """Perform a training step"""
        try:
            if len(self.response_history) < self.batch_size:
                return
            
            # Sample batch
            batch_indices = np.random.choice(
                len(self.response_history), 
                size=min(self.batch_size, len(self.response_history)), 
                replace=False
            )
            
            batch_features = []
            batch_targets = []
            
            for idx in batch_indices:
                item = self.response_history[idx]
                batch_features.append(item['features'])
                batch_targets.append(item['target_score'])
            
            # Convert to tensors
            features_tensor = torch.FloatTensor(batch_features)
            targets_tensor = torch.FloatTensor(batch_targets).unsqueeze(1)
            
            # Forward pass
            predictions = self.quality_network(features_tensor)
            
            # Calculate loss
            loss = nn.MSELoss()(predictions, targets_tensor)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            logger.info(f"🎯 Training step completed. Loss: {loss.item():.4f}")
            
        except Exception as e:
            logger.error(f"Training step error: {str(e)}")
    
    async def retrain_model(self):
        """Retrain the entire model"""
        try:
            if len(self.response_history) < 10:
                logger.warning("Not enough data for retraining")
                return
            
            logger.info("🔄 Retraining RL model...")
            
            # Prepare all data
            all_features = []
            all_targets = []
            
            for item in self.response_history:
                all_features.append(item['features'])
                all_targets.append(item['target_score'])
            
            features_tensor = torch.FloatTensor(all_features)
            targets_tensor = torch.FloatTensor(all_targets).unsqueeze(1)
            
            # Training loop
            epochs = 50
            best_loss = float('inf')
            
            for epoch in range(epochs):
                # Shuffle data
                indices = torch.randperm(len(all_features))
                features_shuffled = features_tensor[indices]
                targets_shuffled = targets_tensor[indices]
                
                # Mini-batch training
                total_loss = 0
                num_batches = 0
                
                for i in range(0, len(features_shuffled), self.batch_size):
                    batch_features = features_shuffled[i:i+self.batch_size]
                    batch_targets = targets_shuffled[i:i+self.batch_size]
                    
                    predictions = self.quality_network(batch_features)
                    loss = nn.MSELoss()(predictions, batch_targets)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / num_batches
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    await self._save_model()
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
            
            logger.info(f"✅ Retraining completed. Best loss: {best_loss:.4f}")
            
        except Exception as e:
            logger.error(f"Retraining error: {str(e)}")
    
    async def health_check(self) -> bool:
        """Check if RL trainer is healthy"""
        try:
            if not self.is_initialized:
                return False
            
            # Test feature extraction
            test_features = self._extract_features("test query", "test response")
            
            # Test neural network
            with torch.no_grad():
                features_tensor = torch.FloatTensor(test_features).unsqueeze(0)
                output = self.quality_network(features_tensor)
            
            return True
            
        except Exception as e:
            logger.error(f"RL health check failed: {str(e)}")
            return False
    
    async def get_learning_stats(self) -> Dict:
        """Get learning statistics"""
        return {
            "total_feedback": len(self.feedback_history),
            "total_interactions": len(self.response_history),
            "avg_rating": np.mean([f.get('rating', 3) for f in self.feedback_history]) if self.feedback_history else 3.0,
            "model_trained": self.is_initialized,
            "memory_usage": len(self.response_history)
        }