import sqlite3
import json
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SQLiteVectorDB:
    def __init__(self, db_path="teammate_finder.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Инициализация SQLite базы"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Таблица пользователей
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT UNIQUE NOT NULL,
                        username TEXT,
                        description TEXT NOT NULL,
                        embedding TEXT,
                        game_type TEXT DEFAULT 'general',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Индекс для user_id
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON users(user_id)")
                
                conn.commit()
            logger.info("SQLite database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def add_user(self, user_data: Dict[str, Any]) -> bool:
        """Добавление пользователя"""
        try:
            # Проверяем и подготавливаем эмбеддинг
            embedding = user_data['embedding']
            logger.info(f"DB: Embedding type: {type(embedding)}, length: {len(embedding)}")

            # Убеждаемся, что это список чисел (а не numpy array)
            if hasattr(embedding, 'tolist'):  # если это numpy array
                embedding = embedding.tolist()

            # Сериализуем в JSON
            embedding_json = json.dumps(embedding)
            logger.info(f"DB: JSON serialization successful, length: {len(embedding_json)}")

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO users 
                    (user_id, username, description, embedding, game_type)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    user_data['user_id'],
                    user_data.get('username'),
                    user_data['description'],
                    embedding_json,  # используем подготовленный JSON
                    user_data.get('game_type', 'general')
                ))

                conn.commit()
                logger.info(f"User {user_data['user_id']} added successfully")
                return True

        except Exception as e:
            logger.error(f"Error adding user to database: {e}", exc_info=True)
            return False

    def find_similar_users(self, query_embedding: List[float], 
                          user_id: Optional[str] = None,
                          top_k: int = 5,
                          similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Получаем всех пользователей
                if user_id:
                    cursor.execute("SELECT user_id, username, description, embedding, game_type FROM users WHERE user_id != ?", (user_id,))
                else:
                    cursor.execute("SELECT user_id, username, description, embedding, game_type FROM users")
                
                all_users = cursor.fetchall()
                
                # Вычисляем схожесть
                similar_users = []
                query_norm = np.linalg.norm(query_embedding)
                
                for db_user in all_users:
                    user_embedding = json.loads(db_user[3]) if db_user[3] else []
                    
                    if user_embedding and len(user_embedding) == len(query_embedding):
                        try:
                            user_norm = np.linalg.norm(user_embedding)
                            if user_norm > 0 and query_norm > 0:
                                similarity = np.dot(query_embedding, user_embedding) / (query_norm * user_norm)
                                
                                
                                logger.info(f"Similarity with {db_user[0]}: {similarity:.3f} - '{db_user[2][:50]}...'")
                                
                                if similarity >= similarity_threshold:
                                    similar_users.append({
                                        'user_id': db_user[0],
                                        'username': db_user[1],
                                        'description': db_user[2],
                                        'game_type': db_user[4],
                                        'similarity_score': float(similarity)
                                    })
                        except Exception as e:
                            logger.warning(f"Error calculating similarity: {e}")
                            continue
                        
                
                similar_users.sort(key=lambda x: x['similarity_score'], reverse=True)
                
                logger.info(f"Found {len(similar_users[:top_k])} similar users (threshold: {similarity_threshold})")
                return similar_users[:top_k]
                
        except Exception as e:
            logger.error(f"Error finding similar users: {e}")
            return []
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Получение пользователя"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT user_id, username, description, embedding, game_type, created_at FROM users WHERE user_id = ?", (user_id,))
                
                result = cursor.fetchone()
                if result:
                    return {
                        'user_id': result[0],
                        'username': result[1],
                        'description': result[2],
                        'embedding': json.loads(result[3]) if result[3] else [],
                        'game_type': result[4],
                        'created_at': result[5]
                    }
                return None
                
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None

    def delete_user(self, user_id: str) -> bool:
        """Удаление пользователя"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"User {user_id} deleted successfully")
                    return True
                else:
                    logger.warning(f"User {user_id} not found for deletion")
                    return False
                    
        except Exception as e:
            logger.error(f"Error deleting user: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM users")
                total_users = cursor.fetchone()[0]
                
                cursor.execute("SELECT game_type, COUNT(*) FROM users GROUP BY game_type")
                users_by_game = dict(cursor.fetchall())
                
                return {
                    'total_users': total_users,
                    'users_by_game_type': users_by_game
                }
                
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'total_users': 0, 'users_by_game_type': {}}

# Глобальный инстанс
vector_db = SQLiteVectorDB()