"""
User Manager for Multi-User Session Management
Handles user sessions, preferences, and concurrent user support
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import threading
import uuid

logger = logging.getLogger(__name__)

class UserSession:
    """Individual user session data"""
    
    def __init__(self, session_id: str, user_id: str):
        self.session_id = session_id
        self.user_id = user_id
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.message_count = 0
        self.conversation_history = []
        self.preferences = {}
        self.is_active = True
        self.feedback_given = 0
        
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
        
    def add_message(self, query: str, response: str):
        """Add message to conversation history"""
        self.conversation_history.append({
            'query': query,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        self.message_count += 1
        self.update_activity()
        
        # Keep only last 50 messages to manage memory
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
    
    def get_session_duration(self) -> float:
        """Get session duration in minutes"""
        return (datetime.now() - self.created_at).total_seconds() / 60
    
    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """Check if session has expired"""
        return (datetime.now() - self.last_activity).total_seconds() > (timeout_minutes * 60)

class UserManager:
    """Manager for handling multiple user sessions"""
    
    def __init__(self):
        self.active_sessions: Dict[str, UserSession] = {}
        self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> set of session_ids
        self.session_lock = threading.RLock()
        self.max_concurrent_users = 100
        self.session_timeout_minutes = 30
        self.cleanup_interval = 300  # 5 minutes
        self.is_running = True
        
        # Start cleanup task
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background task for session cleanup"""
        def cleanup_loop():
            while self.is_running:
                try:
                    self.cleanup_expired_sessions()
                    time.sleep(self.cleanup_interval)
                except Exception as e:
                    logger.error(f"Error in cleanup loop: {str(e)}")
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
    
    def create_session(self, user_id: str = None) -> str:
        """Create a new user session"""
        with self.session_lock:
            # Generate session ID
            session_id = str(uuid.uuid4())
            
            # Generate user ID if not provided
            if not user_id:
                user_id = f"user_{session_id[:8]}"
            
            # Check if we're at capacity
            if len(self.active_sessions) >= self.max_concurrent_users:
                # Remove oldest inactive session
                self._remove_oldest_session()
            
            # Create session
            session = UserSession(session_id, user_id)
            self.active_sessions[session_id] = session
            
            # Track user sessions
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = set()
            self.user_sessions[user_id].add(session_id)
            
            logger.info(f"👤 Created session {session_id} for user {user_id}")
            return session_id
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID"""
        with self.session_lock:
            session = self.active_sessions.get(session_id)
            if session and not session.is_expired(self.session_timeout_minutes):
                session.update_activity()
                return session
            elif session:
                # Session expired, remove it
                self._remove_session(session_id)
            return None
    
    def update_session(self, session_id: str, user_id: str, query: str, response: str):
        """Update session with new message"""
        with self.session_lock:
            session = self.get_session(session_id)
            if session:
                session.add_message(query, response)
            else:
                # Create new session if it doesn't exist
                new_session_id = self.create_session(user_id)
                session = self.get_session(new_session_id)
                if session:
                    session.add_message(query, response)
    
    def get_user_sessions(self, user_id: str) -> List[UserSession]:
        """Get all active sessions for a user"""
        with self.session_lock:
            user_session_ids = self.user_sessions.get(user_id, set())
            sessions = []
            
            for session_id in user_session_ids.copy():
                session = self.active_sessions.get(session_id)
                if session and not session.is_expired(self.session_timeout_minutes):
                    sessions.append(session)
                else:
                    # Remove expired session
                    self._remove_session(session_id)
            
            return sessions
    
    def _remove_session(self, session_id: str):
        """Remove a session"""
        session = self.active_sessions.get(session_id)
        if session:
            user_id = session.user_id
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            # Remove from user sessions
            if user_id in self.user_sessions:
                self.user_sessions[user_id].discard(session_id)
                
                # Remove user entry if no more sessions
                if not self.user_sessions[user_id]:
                    del self.user_sessions[user_id]
            
            logger.info(f"🗑️ Removed session {session_id}")
    
    def _remove_oldest_session(self):
        """Remove the oldest session to make room"""
        if not self.active_sessions:
            return
        
        oldest_session_id = min(
            self.active_sessions.keys(),
            key=lambda sid: self.active_sessions[sid].last_activity
        )
        
        self._remove_session(oldest_session_id)
        logger.info(f"📉 Removed oldest session {oldest_session_id} due to capacity limit")
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        with self.session_lock:
            expired_sessions = []
            
            for session_id, session in self.active_sessions.items():
                if session.is_expired(self.session_timeout_minutes):
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                self._remove_session(session_id)
            
            if expired_sessions:
                logger.info(f"🧹 Cleaned up {len(expired_sessions)} expired sessions")
    
    def get_user_preferences(self, session_id: str) -> Dict:
        """Get user preferences from session"""
        session = self.get_session(session_id)
        if session:
            return session.preferences.copy()
        return {}
    
    def update_user_preferences(self, session_id: str, preferences: Dict):
        """Update user preferences"""
        session = self.get_session(session_id)
        if session:
            session.preferences.update(preferences)
            logger.info(f"📝 Updated preferences for session {session_id}")
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get conversation history for a session"""
        session = self.get_session(session_id)
        if session:
            return session.conversation_history[-limit:]
        return []
    
    def get_session_stats(self) -> Dict:
        """Get statistics about active sessions"""
        with self.session_lock:
            total_sessions = len(self.active_sessions)
            unique_users = len(self.user_sessions)
            
            # Calculate session durations
            durations = [
                session.get_session_duration() 
                for session in self.active_sessions.values()
            ]
            
            avg_duration = sum(durations) / len(durations) if durations else 0
            
            # Message statistics
            total_messages = sum(
                session.message_count 
                for session in self.active_sessions.values()
            )
            
            avg_messages_per_session = total_messages / total_sessions if total_sessions > 0 else 0
            
            return {
                'total_active_sessions': total_sessions,
                'unique_users': unique_users,
                'average_session_duration_minutes': round(avg_duration, 2),
                'total_messages': total_messages,
                'average_messages_per_session': round(avg_messages_per_session, 2),
                'capacity_utilization': round((total_sessions / self.max_concurrent_users) * 100, 2)
            }
    
    def get_user_activity_summary(self, user_id: str) -> Dict:
        """Get activity summary for a specific user"""
        user_sessions = self.get_user_sessions(user_id)
        
        if not user_sessions:
            return {
                'active_sessions': 0,
                'total_messages': 0,
                'session_duration_minutes': 0,
                'feedback_given': 0
            }
        
        total_messages = sum(session.message_count for session in user_sessions)
        total_duration = sum(session.get_session_duration() for session in user_sessions)
        total_feedback = sum(session.feedback_given for session in user_sessions)
        
        return {
            'active_sessions': len(user_sessions),
            'total_messages': total_messages,
            'session_duration_minutes': round(total_duration, 2),
            'feedback_given': total_feedback
        }
    
    def record_feedback(self, session_id: str):
        """Record that feedback was given in a session"""
        session = self.get_session(session_id)
        if session:
            session.feedback_given += 1
    
    def is_user_active(self, user_id: str) -> bool:
        """Check if user has any active sessions"""
        return len(self.get_user_sessions(user_id)) > 0
    
    def get_active_users_count(self) -> int:
        """Get count of unique active users"""
        return len(self.user_sessions)
    
    def force_logout_user(self, user_id: str):
        """Force logout all sessions for a user"""
        with self.session_lock:
            user_session_ids = self.user_sessions.get(user_id, set()).copy()
            
            for session_id in user_session_ids:
                self._remove_session(session_id)
            
            logger.info(f"🚪 Forced logout for user {user_id} ({len(user_session_ids)} sessions)")
    
    def set_session_inactive(self, session_id: str):
        """Mark a session as inactive"""
        session = self.get_session(session_id)
        if session:
            session.is_active = False
    
    def get_load_balancing_info(self) -> Dict:
        """Get information useful for load balancing"""
        with self.session_lock:
            return {
                'current_load': len(self.active_sessions),
                'max_capacity': self.max_concurrent_users,
                'load_percentage': (len(self.active_sessions) / self.max_concurrent_users) * 100,
                'can_accept_new_users': len(self.active_sessions) < self.max_concurrent_users,
                'sessions_by_duration': self._get_sessions_by_duration_bucket()
            }
    
    def _get_sessions_by_duration_bucket(self) -> Dict[str, int]:
        """Get session count by duration buckets"""
        buckets = {
            '0-5min': 0,
            '5-15min': 0,
            '15-30min': 0,
            '30min+': 0
        }
        
        for session in self.active_sessions.values():
            duration = session.get_session_duration()
            
            if duration <= 5:
                buckets['0-5min'] += 1
            elif duration <= 15:
                buckets['5-15min'] += 1
            elif duration <= 30:
                buckets['15-30min'] += 1
            else:
                buckets['30min+'] += 1
        
        return buckets
    
    def shutdown(self):
        """Shutdown the user manager"""
        self.is_running = False
        with self.session_lock:
            self.active_sessions.clear()
            self.user_sessions.clear()
        logger.info("🛑 User manager shutdown complete")