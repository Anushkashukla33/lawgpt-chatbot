"""
Database Manager for Self-Learning Chatbot
Handles storage of interactions, feedback, and learning data
"""

import asyncio
import logging
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
import aiosqlite

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database manager for chatbot data"""
    
    def __init__(self, db_path: str = "chatbot_data.db"):
        self.db_path = db_path
        self.connection = None
        
    async def initialize(self):
        """Initialize database and create tables"""
        logger.info("🗄️ Initializing Database...")
        
        try:
            # Create database directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
            
            # Create tables
            await self._create_tables()
            
            logger.info("✅ Database initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {str(e)}")
            raise
    
    async def _create_tables(self):
        """Create necessary database tables"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Users table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        session_id TEXT,
                        first_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        total_interactions INTEGER DEFAULT 0,
                        preferences TEXT,
                        feedback_count INTEGER DEFAULT 0
                    )
                """)
                
                # Interactions table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS interactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        session_id TEXT,
                        query TEXT NOT NULL,
                        response TEXT NOT NULL,
                        confidence REAL,
                        processing_time REAL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        context_used TEXT,
                        model_version TEXT,
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    )
                """)
                
                # Feedback table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        interaction_id INTEGER,
                        user_id TEXT,
                        session_id TEXT,
                        rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                        feedback_text TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        processed BOOLEAN DEFAULT FALSE,
                        FOREIGN KEY (interaction_id) REFERENCES interactions (id),
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    )
                """)
                
                # Learning metrics table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS learning_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        additional_data TEXT
                    )
                """)
                
                # Knowledge base table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge_base (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id TEXT UNIQUE,
                        content TEXT NOT NULL,
                        metadata TEXT,
                        embedding_vector TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        usage_count INTEGER DEFAULT 0
                    )
                """)
                
                # System events table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS system_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT NOT NULL,
                        event_data TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        severity TEXT DEFAULT 'INFO'
                    )
                """)
                
                # Create indexes for better performance
                await db.execute("CREATE INDEX IF NOT EXISTS idx_interactions_user_id ON interactions(user_id)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_users_last_interaction ON users(last_interaction)")
                
                await db.commit()
                
        except Exception as e:
            logger.error(f"Failed to create tables: {str(e)}")
            raise
    
    async def store_interaction(self, interaction_data: Dict) -> int:
        """Store a user interaction"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Insert or update user
                await db.execute("""
                    INSERT OR REPLACE INTO users (
                        user_id, session_id, last_interaction, total_interactions
                    ) VALUES (
                        ?, ?, CURRENT_TIMESTAMP, 
                        COALESCE((SELECT total_interactions FROM users WHERE user_id = ?), 0) + 1
                    )
                """, (interaction_data['user_id'], interaction_data['session_id'], interaction_data['user_id']))
                
                # Insert interaction
                cursor = await db.execute("""
                    INSERT INTO interactions (
                        user_id, session_id, query, response, confidence, 
                        processing_time, context_used, model_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    interaction_data['user_id'],
                    interaction_data['session_id'],
                    interaction_data['query'],
                    interaction_data['response'],
                    interaction_data.get('confidence', 0.5),
                    interaction_data.get('processing_time', 0.0),
                    json.dumps(interaction_data.get('context_used', [])),
                    interaction_data.get('model_version', 'v1.0')
                ))
                
                interaction_id = cursor.lastrowid
                await db.commit()
                
                return interaction_id
                
        except Exception as e:
            logger.error(f"Failed to store interaction: {str(e)}")
            raise
    
    async def store_feedback(self, session_id: str, message_id: str, rating: int, feedback_text: str = None):
        """Store user feedback"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Get interaction details
                cursor = await db.execute("""
                    SELECT id, user_id FROM interactions 
                    WHERE session_id = ? 
                    ORDER BY id DESC 
                    LIMIT 1
                """, (session_id,))
                
                interaction = await cursor.fetchone()
                
                if interaction:
                    interaction_id, user_id = interaction
                    
                    # Store feedback
                    await db.execute("""
                        INSERT INTO feedback (
                            interaction_id, user_id, session_id, rating, feedback_text
                        ) VALUES (?, ?, ?, ?, ?)
                    """, (interaction_id, user_id, session_id, rating, feedback_text))
                    
                    # Update user feedback count
                    await db.execute("""
                        UPDATE users 
                        SET feedback_count = feedback_count + 1 
                        WHERE user_id = ?
                    """, (user_id,))
                    
                    await db.commit()
                    
        except Exception as e:
            logger.error(f"Failed to store feedback: {str(e)}")
            raise
    
    async def get_user_feedback_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get user's feedback history"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("""
                    SELECT f.rating, f.feedback_text, f.timestamp, i.query, i.response
                    FROM feedback f
                    JOIN interactions i ON f.interaction_id = i.id
                    WHERE f.user_id = ?
                    ORDER BY f.timestamp DESC
                    LIMIT ?
                """, (user_id, limit))
                
                rows = await cursor.fetchall()
                
                feedback_history = []
                for row in rows:
                    feedback_history.append({
                        'rating': row[0],
                        'feedback_text': row[1],
                        'timestamp': row[2],
                        'query': row[3],
                        'response': row[4]
                    })
                
                return feedback_history
                
        except Exception as e:
            logger.error(f"Failed to get user feedback history: {str(e)}")
            return []
    
    async def get_system_stats(self) -> Dict:
        """Get overall system statistics"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                stats = {}
                
                # Total interactions
                cursor = await db.execute("SELECT COUNT(*) FROM interactions")
                stats['total_interactions'] = (await cursor.fetchone())[0]
                
                # Unique users
                cursor = await db.execute("SELECT COUNT(DISTINCT user_id) FROM users")
                stats['unique_users'] = (await cursor.fetchone())[0]
                
                # Average confidence
                cursor = await db.execute("SELECT AVG(confidence) FROM interactions WHERE confidence IS NOT NULL")
                result = await cursor.fetchone()
                stats['average_confidence'] = result[0] if result[0] else 0.0
                
                # Total feedback
                cursor = await db.execute("SELECT COUNT(*) FROM feedback")
                stats['total_feedback'] = (await cursor.fetchone())[0]
                
                # Average rating
                cursor = await db.execute("SELECT AVG(rating) FROM feedback")
                result = await cursor.fetchone()
                stats['average_rating'] = result[0] if result[0] else 3.0
                
                # Recent activity (last 24 hours)
                cursor = await db.execute("""
                    SELECT COUNT(*) FROM interactions 
                    WHERE timestamp > datetime('now', '-1 day')
                """)
                stats['recent_interactions_24h'] = (await cursor.fetchone())[0]
                
                # Rating distribution
                cursor = await db.execute("""
                    SELECT rating, COUNT(*) 
                    FROM feedback 
                    GROUP BY rating 
                    ORDER BY rating
                """)
                rating_dist = await cursor.fetchall()
                stats['rating_distribution'] = {str(rating): count for rating, count in rating_dist}
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get system stats: {str(e)}")
            return {}
    
    async def get_learning_data(self, limit: int = 1000) -> List[Dict]:
        """Get data for learning algorithms"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("""
                    SELECT i.query, i.response, i.confidence, f.rating, f.feedback_text, i.timestamp
                    FROM interactions i
                    LEFT JOIN feedback f ON i.id = f.interaction_id
                    ORDER BY i.timestamp DESC
                    LIMIT ?
                """, (limit,))
                
                rows = await cursor.fetchall()
                
                learning_data = []
                for row in rows:
                    learning_data.append({
                        'query': row[0],
                        'response': row[1],
                        'confidence': row[2],
                        'rating': row[3],
                        'feedback_text': row[4],
                        'timestamp': row[5]
                    })
                
                return learning_data
                
        except Exception as e:
            logger.error(f"Failed to get learning data: {str(e)}")
            return []
    
    async def store_learning_metric(self, metric_name: str, metric_value: float, additional_data: Dict = None):
        """Store learning metrics"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO learning_metrics (metric_name, metric_value, additional_data)
                    VALUES (?, ?, ?)
                """, (metric_name, metric_value, json.dumps(additional_data) if additional_data else None))
                
                await db.commit()
                
        except Exception as e:
            logger.error(f"Failed to store learning metric: {str(e)}")
    
    async def get_user_preferences(self, user_id: str) -> Dict:
        """Get user preferences"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("""
                    SELECT preferences FROM users WHERE user_id = ?
                """, (user_id,))
                
                result = await cursor.fetchone()
                
                if result and result[0]:
                    return json.loads(result[0])
                else:
                    return {}
                    
        except Exception as e:
            logger.error(f"Failed to get user preferences: {str(e)}")
            return {}
    
    async def update_user_preferences(self, user_id: str, preferences: Dict):
        """Update user preferences"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    UPDATE users 
                    SET preferences = ? 
                    WHERE user_id = ?
                """, (json.dumps(preferences), user_id))
                
                await db.commit()
                
        except Exception as e:
            logger.error(f"Failed to update user preferences: {str(e)}")
    
    async def log_system_event(self, event_type: str, event_data: Dict, severity: str = "INFO"):
        """Log system events"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO system_events (event_type, event_data, severity)
                    VALUES (?, ?, ?)
                """, (event_type, json.dumps(event_data), severity))
                
                await db.commit()
                
        except Exception as e:
            logger.error(f"Failed to log system event: {str(e)}")
    
    async def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data to manage database size"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            async with aiosqlite.connect(self.db_path) as db:
                # Delete old interactions and associated feedback
                await db.execute("""
                    DELETE FROM feedback 
                    WHERE interaction_id IN (
                        SELECT id FROM interactions 
                        WHERE timestamp < ?
                    )
                """, (cutoff_date,))
                
                await db.execute("""
                    DELETE FROM interactions 
                    WHERE timestamp < ?
                """, (cutoff_date,))
                
                # Delete old system events
                await db.execute("""
                    DELETE FROM system_events 
                    WHERE timestamp < ?
                """, (cutoff_date,))
                
                # Delete old learning metrics
                await db.execute("""
                    DELETE FROM learning_metrics 
                    WHERE timestamp < ?
                """, (cutoff_date,))
                
                await db.commit()
                
                logger.info(f"🧹 Cleaned up data older than {days_to_keep} days")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {str(e)}")
    
    async def health_check(self) -> bool:
        """Check database health"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("SELECT 1")
                return True
                
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False
    
    async def close(self):
        """Close database connections"""
        try:
            if self.connection:
                self.connection.close()
            logger.info("📦 Database connections closed")
            
        except Exception as e:
            logger.error(f"Error closing database: {str(e)}")
    
    async def export_data(self, output_file: str):
        """Export data for analysis"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Export interactions
                cursor = await db.execute("""
                    SELECT i.*, f.rating, f.feedback_text
                    FROM interactions i
                    LEFT JOIN feedback f ON i.id = f.interaction_id
                    ORDER BY i.timestamp DESC
                """)
                
                rows = await cursor.fetchall()
                
                data = []
                for row in rows:
                    data.append({
                        'id': row[0],
                        'user_id': row[1],
                        'session_id': row[2],
                        'query': row[3],
                        'response': row[4],
                        'confidence': row[5],
                        'processing_time': row[6],
                        'timestamp': row[7],
                        'context_used': row[8],
                        'model_version': row[9],
                        'rating': row[10],
                        'feedback_text': row[11]
                    })
                
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                
                logger.info(f"📊 Data exported to {output_file}")
                
        except Exception as e:
            logger.error(f"Failed to export data: {str(e)}")
            raise