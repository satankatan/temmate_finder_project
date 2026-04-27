from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import uvicorn

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Поднимаемся на уровень выше (в корень)
sys.path.append(project_root)

from database.sqlite_db import vector_db
from model.inference import TextEmbedder
from preprocessing.text_processor import preprocessor

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

database = vector_db
text_embedder = TextEmbedder()
preprocessor = preprocessor

# Модели данных Pydantic
class UserRequest(BaseModel):
    user_id: str = Field(..., description="Уникальный идентификатор пользователя")
    username: Optional[str] = Field(None, description="Имя пользователя")
    description: str = Field(..., description="Текстовое описание стиля игры", min_length=10)
    game_type: Optional[str] = Field("general", description="Тип игры")

class SimilarityRequest(BaseModel):
    description: str = Field(..., description="Текстовое описание для поиска", min_length=3)
    user_id: Optional[str] = Field(None, description="ID пользователя")
    game_type: Optional[str] = Field(None, description="Фильтр по типу игры")  # ← ДОБАВЬТЕ ЭТО
    top_k: int = Field(5, ge=1, le=20, description="Количество возвращаемых результатов")
    similarity_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Порог схожести")

class SimilarUserResponse(BaseModel):
    user_id: str
    username: Optional[str]
    description: str
    game_type: str
    similarity_score: float

class UserResponse(BaseModel):
    user_id: str
    username: Optional[str]
    description: str
    game_type: str
    created_at: Optional[str]

class StatsResponse(BaseModel):
    total_users: int
    users_by_game_type: Dict[str, int]

class HealthResponse(BaseModel):
    status: str
    database: bool
    model_loaded: bool

# Инициализация приложения
app = FastAPI(
    title="Teammate Finder API",
    description="API для поиска тиммейтов на основе анализа текстовых описаний",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Для разработки, в продакшене укажите конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Зависимости (упрощенные для SQLite)
def get_database():
    """Зависимость для базы данных"""
    return vector_db

def get_embedder():
    """Зависимость для эмбеддера"""
    return text_embedder  

def get_preprocessor():
    """Зависимость для препроцессора"""
    return preprocessor

# Роуты
@app.get("/", response_model=Dict[str, str])
async def root():
    """Корневой эндпоинт"""
    return {"message": "Teammate Finder API", "version": "1.0.0"}

@app.get("/health", response_model=HealthResponse)
async def health_check(
    db = Depends(get_database),
    embedder = Depends(get_embedder)
):
    """Проверка здоровья сервиса"""
    try:
        # Проверяем базу данных
        db_stats = db.get_stats()
        database_ok = db_stats is not None
        
        # Проверяем модель
        model_loaded = embedder is not None
        
        status = "healthy" if database_ok and model_loaded else "degraded"
        
        return HealthResponse(
            status=status,
            database=database_ok,
            model_loaded=model_loaded
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            database=False,
            model_loaded=False
        )

@app.post("/users", response_model=Dict[str, str])
async def add_user(
    user_request: UserRequest,
    db = Depends(get_database),
    embedder = Depends(get_embedder),
    preprocessor = Depends(get_preprocessor)
):  
    try:
        logger.info(f"Adding user: {user_request.user_id}")
        
        # Очистка и подготовка текста
        cleaned_text = preprocessor.clean_text(user_request.description)
        logger.info(f"Cleaned text: {cleaned_text}")
        
        # Получение эмбеддинга С игровым контекстом
        embedding = embedder.get_embedding(cleaned_text)
        logger.info(f"Generated enhanced embedding of length: {len(embedding)}")
        
        # Подготовка данных пользователя
        user_data = {
            'user_id': user_request.user_id,
            'username': user_request.username,
            'description': user_request.description,
            'embedding': embedding,
            'game_type': user_request.game_type
        }
        
        # Сохранение в базу данных
        success = db.add_user(user_data)
        
        if success:
            logger.info(f"User {user_request.user_id} added successfully")
            return {"status": "success", "message": "User added successfully"}
        else:
            logger.error(f"Failed to add user {user_request.user_id}")
            raise HTTPException(status_code=500, detail="Failed to add user to database")
            
    except Exception as e:
        logger.error(f"Error adding user: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/similarity/search", response_model=List[SimilarUserResponse])
async def find_similar_users(
    request: SimilarityRequest,
    db = Depends(get_database),
    embedder = Depends(get_embedder),
    preprocessor = Depends(get_preprocessor)
):
    """Поиск похожих пользователей по текстовому описанию"""
    try:
        logger.info(f"Searching similar users for description: {request.description[:50]}...")
        
        # Очистка и подготовка текста
        cleaned_text = preprocessor.clean_text(request.description)
        logger.info(f"Cleaned search text: {cleaned_text}")
        
        # Получение эмбеддинга
        embedding = embedder.get_embedding(cleaned_text)
        logger.info(f"Search embedding generated, length: {len(embedding)}")
        
        # Поиск похожих пользователей с фильтром по игре
        similar_users = db.find_similar_users(
            query_embedding=embedding,
            user_id=request.user_id,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        
        logger.info(f"Found {len(similar_users)} similar users (threshold: {request.similarity_threshold})")
        
        return similar_users
        
    except Exception as e:
        logger.error(f"Error finding similar users: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    db = Depends(get_database)
):
    """Получение информации о пользователе"""
    try:
        logger.info(f"Getting user: {user_id}")
        user = db.get_user(user_id)
        if not user:
            logger.warning(f"User {user_id} not found")
            raise HTTPException(status_code=404, detail="User not found")
        
        logger.info(f"User {user_id} found")
        return user
        
    except Exception as e:
        logger.error(f"Error getting user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/users/{user_id}", response_model=Dict[str, str])
async def delete_user(
    user_id: str,
    db = Depends(get_database)
):
    """Удаление пользователя из системы"""
    try:
        logger.info(f"Deleting user: {user_id}")
        success = db.delete_user(user_id)
        if success:
            logger.info(f"User {user_id} deleted successfully")
            return {"status": "success", "message": "User deleted successfully"}
        else:
            logger.error(f"Failed to delete user {user_id}")
            raise HTTPException(status_code=500, detail="Failed to delete user")
            
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=StatsResponse)
async def get_stats(db = Depends(get_database)):
    """Получение статистики системы"""
    try:
        logger.info("Getting system stats")
        stats = db.get_stats()
        logger.info(f"Stats: {stats}")
        return StatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test")
async def test_endpoint():
    """Тестовый эндпоинт для проверки работы"""
    return {
        "status": "ok",
        "message": "API is working!",
        "database": "SQLite",
        "features": ["add_user", "find_similar", "get_user", "delete_user", "stats"]
    }

# Запуск приложения
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )