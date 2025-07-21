"""
Learning System for Continuous Improvement
Coordinates reinforcement learning, feedback processing, and model updates
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

logger = logging.getLogger(__name__)

class LearningSystem:
    """Central system for coordinating learning and improvement"""
    
    def __init__(self, rag_pipeline, rl_trainer, db_manager):
        self.rag_pipeline = rag_pipeline
        self.rl_trainer = rl_trainer
        self.db_manager = db_manager
        
        # Learning configuration
        self.learning_interval = 300  # 5 minutes
        self.batch_learning_size = 50
        self.model_retrain_threshold = 500  # Retrain after 500 feedback items
        self.confidence_threshold = 0.7
        
        # Learning metrics
        self.learning_metrics = {
            'total_interactions_processed': 0,
            'total_feedback_processed': 0,
            'model_updates': 0,
            'last_update': None,
            'learning_rate': 0.001,
            'accuracy_trend': [],
            'user_satisfaction_trend': []
        }
        
        # State tracking
        self.is_learning = False
        self.is_healthy = True
        self.startup_time = datetime.now()
        self.last_retrain = None
        
    async def continuous_learning_loop(self):
        """Main continuous learning loop"""
        logger.info("🎓 Starting continuous learning loop...")
        
        while True:
            try:
                await asyncio.sleep(self.learning_interval)
                
                if not self.is_learning:
                    await self._perform_learning_cycle()
                
            except Exception as e:
                logger.error(f"Error in learning loop: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _perform_learning_cycle(self):
        """Perform one cycle of learning"""
        self.is_learning = True
        
        try:
            logger.info("🔄 Starting learning cycle...")
            
            # Get unprocessed learning data
            learning_data = await self.db_manager.get_learning_data(self.batch_learning_size)
            
            if not learning_data:
                logger.info("📚 No new learning data available")
                return
            
            # Process interactions for pattern learning
            await self._process_interaction_patterns(learning_data)
            
            # Process feedback for reinforcement learning
            await self._process_feedback_batch(learning_data)
            
            # Update knowledge base if needed
            await self._update_knowledge_base(learning_data)
            
            # Check if model retraining is needed
            await self._check_retrain_conditions()
            
            # Update learning metrics
            await self._update_learning_metrics()
            
            logger.info("✅ Learning cycle completed")
            
        except Exception as e:
            logger.error(f"Error in learning cycle: {str(e)}")
        finally:
            self.is_learning = False
    
    async def process_interaction(self, interaction_data: Dict) -> Dict:
        """Process a single interaction for immediate learning"""
        try:
            learning_feedback = {
                'processed': True,
                'learning_applied': False,
                'confidence_updated': False,
                'patterns_detected': []
            }
            
            # Extract patterns from the interaction
            patterns = await self._extract_interaction_patterns(interaction_data)
            learning_feedback['patterns_detected'] = patterns
            
            # Apply immediate learning if confidence is low
            if interaction_data.get('confidence', 0.5) < self.confidence_threshold:
                await self._apply_immediate_learning(interaction_data)
                learning_feedback['learning_applied'] = True
            
            # Update metrics
            self.learning_metrics['total_interactions_processed'] += 1
            
            return learning_feedback
            
        except Exception as e:
            logger.error(f"Error processing interaction: {str(e)}")
            return {'processed': False, 'error': str(e)}
    
    async def learn_from_feedback(self, feedback_request):
        """Process feedback for immediate learning"""
        try:
            # Get the interaction details
            session_id = feedback_request.session_id
            rating = feedback_request.rating
            feedback_text = feedback_request.feedback
            
            # Find the corresponding interaction
            interaction_data = await self._get_interaction_by_session(session_id)
            
            if interaction_data:
                # Apply reinforcement learning
                await self.rl_trainer.learn_from_feedback(
                    query=interaction_data['query'],
                    response=interaction_data['response'],
                    rating=rating,
                    feedback_text=feedback_text
                )
                
                # Update user preferences based on feedback
                await self._update_user_preferences_from_feedback(
                    interaction_data['user_id'],
                    rating,
                    feedback_text
                )
                
                # Store learning metrics
                await self.db_manager.store_learning_metric(
                    'feedback_rating',
                    float(rating),
                    {
                        'session_id': session_id,
                        'feedback_text': feedback_text,
                        'confidence': interaction_data.get('confidence', 0.5)
                    }
                )
                
                self.learning_metrics['total_feedback_processed'] += 1
                
                logger.info(f"📈 Processed feedback: rating={rating}, session={session_id}")
            
        except Exception as e:
            logger.error(f"Error learning from feedback: {str(e)}")
    
    async def _process_interaction_patterns(self, learning_data: List[Dict]):
        """Analyze interaction patterns for insights"""
        try:
            patterns = {
                'common_queries': {},
                'successful_responses': [],
                'low_confidence_areas': [],
                'user_preferences': {}
            }
            
            for item in learning_data:
                query = item.get('query', '')
                confidence = item.get('confidence', 0.5)
                rating = item.get('rating')
                
                # Track common query patterns
                query_words = set(query.lower().split())
                for word in query_words:
                    if len(word) > 3:  # Ignore short words
                        patterns['common_queries'][word] = patterns['common_queries'].get(word, 0) + 1
                
                # Track successful responses
                if rating and rating >= 4:
                    patterns['successful_responses'].append({
                        'query': query,
                        'response': item.get('response', ''),
                        'rating': rating
                    })
                
                # Track low confidence areas
                if confidence < 0.6:
                    patterns['low_confidence_areas'].append({
                        'query': query,
                        'confidence': confidence
                    })
            
            # Store pattern insights
            await self.db_manager.store_learning_metric(
                'interaction_patterns',
                len(learning_data),
                patterns
            )
            
        except Exception as e:
            logger.error(f"Error processing interaction patterns: {str(e)}")
    
    async def _process_feedback_batch(self, learning_data: List[Dict]):
        """Process a batch of feedback for reinforcement learning"""
        try:
            feedback_items = [item for item in learning_data if item.get('rating')]
            
            if not feedback_items:
                return
            
            # Batch process feedback
            for item in feedback_items:
                if item.get('query') and item.get('response') and item.get('rating'):
                    await self.rl_trainer.learn_from_feedback(
                        query=item['query'],
                        response=item['response'],
                        rating=item['rating'],
                        feedback_text=item.get('feedback_text')
                    )
            
            logger.info(f"🎯 Processed {len(feedback_items)} feedback items")
            
        except Exception as e:
            logger.error(f"Error processing feedback batch: {str(e)}")
    
    async def _update_knowledge_base(self, learning_data: List[Dict]):
        """Update knowledge base with new information"""
        try:
            # Extract potential new knowledge from successful interactions
            new_documents = []
            
            for item in learning_data:
                rating = item.get('rating')
                if rating and rating >= 4:  # High-rated responses
                    query = item.get('query', '')
                    response = item.get('response', '')
                    
                    if len(response.split()) > 20:  # Substantial responses
                        new_documents.append({
                            'text': f"Q: {query}\nA: {response}",
                            'metadata': {
                                'type': 'qa_pair',
                                'rating': rating,
                                'timestamp': item.get('timestamp'),
                                'source': 'user_interaction'
                            }
                        })
            
            if new_documents:
                await self.rag_pipeline.add_documents(new_documents)
                logger.info(f"📚 Added {len(new_documents)} new documents to knowledge base")
            
        except Exception as e:
            logger.error(f"Error updating knowledge base: {str(e)}")
    
    async def _check_retrain_conditions(self):
        """Check if model retraining is needed"""
        try:
            # Get feedback count since last retrain
            stats = await self.db_manager.get_system_stats()
            total_feedback = stats.get('total_feedback', 0)
            
            # Check if we should retrain
            should_retrain = False
            
            if self.last_retrain is None and total_feedback >= 100:
                should_retrain = True
                reason = "Initial retraining threshold reached"
            elif self.last_retrain and total_feedback >= self.model_retrain_threshold:
                should_retrain = True
                reason = "Regular retraining threshold reached"
            
            # Check if performance has degraded
            avg_rating = stats.get('average_rating', 3.0)
            if avg_rating < 3.0:
                should_retrain = True
                reason = "Performance degradation detected"
            
            if should_retrain:
                logger.info(f"🔄 Triggering model retraining: {reason}")
                await self.retrain_models()
            
        except Exception as e:
            logger.error(f"Error checking retrain conditions: {str(e)}")
    
    async def retrain_models(self):
        """Retrain the models with accumulated data"""
        try:
            logger.info("🔄 Starting model retraining...")
            
            # Retrain RL model
            await self.rl_trainer.retrain_model()
            
            # Update last retrain time
            self.last_retrain = datetime.now()
            self.learning_metrics['model_updates'] += 1
            
            # Store retraining event
            await self.db_manager.log_system_event(
                'model_retrain',
                {
                    'timestamp': self.last_retrain.isoformat(),
                    'total_feedback': await self._get_total_feedback(),
                    'performance_metrics': await self._get_performance_metrics()
                }
            )
            
            logger.info("✅ Model retraining completed")
            
        except Exception as e:
            logger.error(f"Error retraining models: {str(e)}")
            raise
    
    async def _update_learning_metrics(self):
        """Update learning metrics and trends"""
        try:
            # Calculate current accuracy
            stats = await self.db_manager.get_system_stats()
            avg_rating = stats.get('average_rating', 3.0)
            accuracy = (avg_rating - 1) / 4  # Normalize to 0-1 scale
            
            # Update trends
            self.learning_metrics['accuracy_trend'].append({
                'timestamp': datetime.now().isoformat(),
                'accuracy': accuracy
            })
            
            self.learning_metrics['user_satisfaction_trend'].append({
                'timestamp': datetime.now().isoformat(),
                'satisfaction': avg_rating
            })
            
            # Keep only last 100 data points
            if len(self.learning_metrics['accuracy_trend']) > 100:
                self.learning_metrics['accuracy_trend'] = self.learning_metrics['accuracy_trend'][-100:]
            
            if len(self.learning_metrics['user_satisfaction_trend']) > 100:
                self.learning_metrics['user_satisfaction_trend'] = self.learning_metrics['user_satisfaction_trend'][-100:]
            
            # Update last update time
            self.learning_metrics['last_update'] = datetime.now().isoformat()
            
            # Store metrics in database
            await self.db_manager.store_learning_metric(
                'system_accuracy',
                accuracy,
                {'user_satisfaction': avg_rating}
            )
            
        except Exception as e:
            logger.error(f"Error updating learning metrics: {str(e)}")
    
    async def _extract_interaction_patterns(self, interaction_data: Dict) -> List[str]:
        """Extract patterns from a single interaction"""
        patterns = []
        
        try:
            query = interaction_data.get('query', '').lower()
            confidence = interaction_data.get('confidence', 0.5)
            
            # Detect query types
            if any(word in query for word in ['how', 'what', 'why', 'where', 'when']):
                patterns.append('question_pattern')
            
            if any(word in query for word in ['help', 'assist', 'support']):
                patterns.append('help_request')
            
            if confidence < 0.5:
                patterns.append('low_confidence')
            elif confidence > 0.9:
                patterns.append('high_confidence')
            
            # Detect technical queries
            tech_words = ['code', 'program', 'algorithm', 'api', 'database', 'server']
            if any(word in query for word in tech_words):
                patterns.append('technical_query')
            
        except Exception as e:
            logger.error(f"Error extracting patterns: {str(e)}")
        
        return patterns
    
    async def _apply_immediate_learning(self, interaction_data: Dict):
        """Apply immediate learning for low-confidence interactions"""
        try:
            # This could involve updating response strategies,
            # adjusting confidence calculations, etc.
            pass
            
        except Exception as e:
            logger.error(f"Error applying immediate learning: {str(e)}")
    
    async def _get_interaction_by_session(self, session_id: str) -> Optional[Dict]:
        """Get the most recent interaction for a session"""
        try:
            learning_data = await self.db_manager.get_learning_data(1)
            for item in learning_data:
                # This is a simplified lookup - in practice you'd want a proper query
                pass
            return None
            
        except Exception as e:
            logger.error(f"Error getting interaction by session: {str(e)}")
            return None
    
    async def _update_user_preferences_from_feedback(self, user_id: str, rating: int, feedback_text: str):
        """Update user preferences based on feedback"""
        try:
            preferences = await self.db_manager.get_user_preferences(user_id)
            
            # Update preference trends
            if 'feedback_history' not in preferences:
                preferences['feedback_history'] = []
            
            preferences['feedback_history'].append({
                'rating': rating,
                'feedback': feedback_text,
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep only last 50 feedback items
            preferences['feedback_history'] = preferences['feedback_history'][-50:]
            
            # Calculate preference scores
            recent_ratings = [f['rating'] for f in preferences['feedback_history'][-10:]]
            if recent_ratings:
                preferences['average_satisfaction'] = sum(recent_ratings) / len(recent_ratings)
            
            # Update preferences
            await self.db_manager.update_user_preferences(user_id, preferences)
            
        except Exception as e:
            logger.error(f"Error updating user preferences: {str(e)}")
    
    async def get_learning_metrics(self) -> Dict:
        """Get current learning metrics"""
        uptime = datetime.now() - self.startup_time
        uptime_str = f"{uptime.days}d {uptime.seconds//3600}h {(uptime.seconds//60)%60}m"
        
        return {
            **self.learning_metrics,
            'uptime': uptime_str,
            'is_learning': self.is_learning,
            'last_retrain': self.last_retrain.isoformat() if self.last_retrain else None,
            'learning_progress': min(100, (self.learning_metrics['total_feedback_processed'] / 1000) * 100),
            'model_accuracy': await self._calculate_current_accuracy()
        }
    
    async def _calculate_current_accuracy(self) -> float:
        """Calculate current model accuracy"""
        try:
            stats = await self.db_manager.get_system_stats()
            avg_rating = stats.get('average_rating', 3.0)
            return max(0.0, min(1.0, (avg_rating - 1) / 4))
        except:
            return 0.5
    
    async def _get_total_feedback(self) -> int:
        """Get total feedback count"""
        try:
            stats = await self.db_manager.get_system_stats()
            return stats.get('total_feedback', 0)
        except:
            return 0
    
    async def _get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        try:
            stats = await self.db_manager.get_system_stats()
            return {
                'average_rating': stats.get('average_rating', 3.0),
                'total_interactions': stats.get('total_interactions', 0),
                'average_confidence': stats.get('average_confidence', 0.5)
            }
        except:
            return {}
    
    def is_healthy(self) -> bool:
        """Check if learning system is healthy"""
        return self.is_healthy
    
    async def save_learned_data(self):
        """Save learned data before shutdown"""
        try:
            # Save current learning state
            await self.db_manager.store_learning_metric(
                'shutdown_state',
                1.0,
                self.learning_metrics
            )
            
            logger.info("💾 Learned data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving learned data: {str(e)}")