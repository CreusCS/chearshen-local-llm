import json
import os
import sqlite3
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class ChatStorage:
    """Handles persistent storage for chat history and session data"""
    
    def __init__(self, db_path: str = "chat_data.db"):
        """
        Initialize storage with SQLite database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        video_filename TEXT,
                        transcription TEXT,
                        summary TEXT,
                        context_data TEXT
                    )
                """)
                
                # Migrate existing sessions table to add context_data if needed
                cursor.execute("PRAGMA table_info(sessions)")
                columns = [column[1] for column in cursor.fetchall()]
                if 'context_data' not in columns:
                    logger.info("Migrating sessions table to add context_data column")
                    cursor.execute("ALTER TABLE sessions ADD COLUMN context_data TEXT")
                
                # Create chat_messages table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        role TEXT,
                        content TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chat_messages_session 
                    ON chat_messages (session_id)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chat_messages_timestamp 
                    ON chat_messages (timestamp)
                """)
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise
    
    def create_session(self, video_filename: str = None, session_id: str = None) -> str:
        """
        Create a new session
        
        Args:
            video_filename: Optional video filename
            session_id: Optional session ID (if not provided, generates new UUID)
            
        Returns:
            Session ID
        """
        try:
            if session_id is None:
                session_id = str(uuid.uuid4())
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Use INSERT OR IGNORE to handle existing session IDs
                cursor.execute("""
                    INSERT OR IGNORE INTO sessions (session_id, video_filename)
                    VALUES (?, ?)
                """, (session_id, video_filename))
                conn.commit()
            
            logger.info(f"Created/ensured session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Session creation failed: {str(e)}")
            raise
    
    def store_transcription(self, session_id: str, transcription: str):
        """Store transcription for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE sessions 
                    SET transcription = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                """, (transcription, session_id))
                conn.commit()
            
            logger.info(f"Stored transcription for session: {session_id}")
            
        except Exception as e:
            logger.error(f"Transcription storage failed: {str(e)}")
            raise
    
    def update_session_video(self, session_id: str, video_filename: str):
        """Update the video filename for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE sessions
                    SET video_filename = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                """, (video_filename, session_id))
                if cursor.rowcount == 0:
                    cursor.execute(
                        """
                        INSERT INTO sessions (session_id, video_filename)
                        VALUES (?, ?)
                        """,
                        (session_id, video_filename)
                    )
                conn.commit()

            logger.info(f"Updated video for session: {session_id}")

        except Exception as e:
            logger.error(f"Session video update failed: {str(e)}")
            raise

    def store_summary(self, session_id: str, summary: str):
        """Store summary for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE sessions 
                    SET summary = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                """, (summary, session_id))
                conn.commit()
            
            logger.info(f"Stored summary for session: {session_id}")
            
        except Exception as e:
            logger.error(f"Summary storage failed: {str(e)}")
            raise
    
    def store_chat_message(self, session_id: str, role: str, content: str) -> int:
        """
        Store a chat message
        
        Args:
            session_id: Session identifier
            role: Message role (user, assistant, system)
            content: Message content
            
        Returns:
            Message ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO chat_messages (session_id, role, content)
                    VALUES (?, ?, ?)
                """, (session_id, role, content))
                message_id = cursor.lastrowid
                conn.commit()
            
            logger.debug(f"Stored chat message {message_id} for session: {session_id}")
            return message_id
            
        except Exception as e:
            logger.error(f"Chat message storage failed: {str(e)}")
            raise
    
    def get_chat_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve chat history for a session
        
        Args:
            session_id: Session identifier
            limit: Optional limit on number of messages
            
        Returns:
            List of chat messages
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT id, role, content, timestamp
                    FROM chat_messages
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor.execute(query, (session_id,))
                rows = cursor.fetchall()
                
                messages = []
                for row in rows:
                    messages.append({
                        'id': str(row[0]),
                        'role': row[1],
                        'content': row[2],
                        'timestamp': row[3],
                        'session_id': session_id
                    })
                
                return messages
                
        except Exception as e:
            logger.error(f"Chat history retrieval failed: {str(e)}")
            return []
    
    def clear_chat_history(self, session_id: str):
        """Clear chat history for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM chat_messages
                    WHERE session_id = ?
                """, (session_id,))
                conn.commit()
            
            logger.info(f"Cleared chat history for session: {session_id}")
            
        except Exception as e:
            logger.error(f"Chat history clearing failed: {str(e)}")
            raise
    
    def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data including transcription and summary
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT session_id, created_at, updated_at, video_filename, 
                           transcription, summary, context_data
                    FROM sessions
                    WHERE session_id = ?
                """, (session_id,))
                
                row = cursor.fetchone()
                if row:
                    context_data = {}
                    if row[6]:
                        try:
                            context_data = json.loads(row[6])
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in context_data for session {session_id}")
                    
                    return {
                        'session_id': row[0],
                        'created_at': row[1],
                        'updated_at': row[2],
                        'video_filename': row[3],
                        'transcription': row[4],
                        'summary': row[5],
                        'context': context_data
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Session data retrieval failed: {str(e)}")
            return None
    
    def update_session_context(self, session_id: str, context: Dict[str, Any]):
        """
        Update session context data
        
        Args:
            session_id: Session identifier
            context: Context dictionary to store
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                context_json = json.dumps(context)
                cursor.execute("""
                    UPDATE sessions 
                    SET context_data = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                """, (context_json, session_id))
                conn.commit()
            
            logger.debug(f"Updated context for session: {session_id}")
            
        except Exception as e:
            logger.error(f"Session context update failed: {str(e)}")
            raise
    
    def get_all_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get list of all sessions
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session data
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT session_id, created_at, updated_at, video_filename
                    FROM sessions
                    ORDER BY updated_at DESC
                    LIMIT ?
                """, (limit,))
                
                rows = cursor.fetchall()
                sessions = []
                
                for row in rows:
                    sessions.append({
                        'session_id': row[0],
                        'created_at': row[1],
                        'updated_at': row[2],
                        'video_filename': row[3]
                    })
                
                return sessions
                
        except Exception as e:
            logger.error(f"Sessions retrieval failed: {str(e)}")
            return []
    
    def delete_session(self, session_id: str):
        """Delete a session and all associated data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete chat messages first (foreign key constraint)
                cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
                
                # Delete session
                cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                
                conn.commit()
            
            logger.info(f"Deleted session: {session_id}")
            
        except Exception as e:
            logger.error(f"Session deletion failed: {str(e)}")
            raise
    
    def export_session_data(self, session_id: str) -> Dict[str, Any]:
        """
        Export all data for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Complete session data including messages
        """
        try:
            session_data = self.get_session_data(session_id)
            if not session_data:
                return {}
            
            chat_history = self.get_chat_history(session_id)
            
            return {
                'session': session_data,
                'chat_history': chat_history,
                'exported_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Session export failed: {str(e)}")
            return {}
    
    def cleanup_old_sessions(self, days_old: int = 30):
        """
        Clean up sessions older than specified days
        
        Args:
            days_old: Number of days after which sessions are considered old
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old chat messages
                cursor.execute("""
                    DELETE FROM chat_messages 
                    WHERE session_id IN (
                        SELECT session_id FROM sessions 
                        WHERE datetime(updated_at) < datetime('now', '-{} days')
                    )
                """.format(days_old))
                
                # Delete old sessions
                cursor.execute("""
                    DELETE FROM sessions 
                    WHERE datetime(updated_at) < datetime('now', '-{} days')
                """.format(days_old))
                
                deleted_count = cursor.rowcount
                conn.commit()
            
            logger.info(f"Cleaned up {deleted_count} old sessions")
            
        except Exception as e:
            logger.error(f"Session cleanup failed: {str(e)}")
            raise
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count sessions
                cursor.execute("SELECT COUNT(*) FROM sessions")
                session_count = cursor.fetchone()[0]
                
                # Count messages
                cursor.execute("SELECT COUNT(*) FROM chat_messages")
                message_count = cursor.fetchone()[0]
                
                # Get database size
                db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                
                return {
                    'sessions': session_count,
                    'messages': message_count,
                    'db_size_bytes': db_size,
                    'db_size_mb': round(db_size / (1024 * 1024), 2)
                }
                
        except Exception as e:
            logger.error(f"Database stats retrieval failed: {str(e)}")
            return {
                'sessions': 0,
                'messages': 0,
                'db_size_bytes': 0,
                'db_size_mb': 0
            }
    
    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List all sessions with basic info
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session data dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT session_id, video_filename, created_at, updated_at
                    FROM sessions
                    ORDER BY updated_at DESC
                    LIMIT ?
                """, (limit,))
                
                rows = cursor.fetchall()
                
                sessions = []
                for row in rows:
                    sessions.append({
                        'session_id': row[0],
                        'video_filename': row[1],
                        'created_at': row[2],
                        'updated_at': row[3]
                    })
                
                return sessions
                
        except Exception as e:
            logger.error(f"List sessions failed: {str(e)}")
            return []
