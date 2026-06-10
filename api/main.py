from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import uvicorn
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from database.sqlite_db import vector_db
from model.inference import CustomTextEmbedder
from preprocessing.text_processor import preprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

database = vector_db
text_embedder = CustomTextEmbedder()


class UserRequest(BaseModel):
    user_id: str = Field(..., description="Уникальный идентификатор пользователя")
    username: Optional[str] = Field(None, description="Имя пользователя")
    description: str = Field(..., description="Текстовое описание стиля игры", min_length=10)
    game_type: Optional[str] = Field("general", description="Тип игры")


class SimilarityRequest(BaseModel):
    description: str = Field(..., description="Текстовое описание для поиска", min_length=3)
    user_id: Optional[str] = Field(None, description="ID пользователя")
    game_type: Optional[str] = Field(None, description="Фильтр по типу игры")
    top_k: int = Field(5, ge=1, le=20, description="Количество возвращаемых результатов")
    similarity_threshold: float = Field(0.35, ge=0.0, le=1.0, description="Порог схожести")


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
    model_type: str


class ModelInfoResponse(BaseModel):
    type: str
    architecture: str
    trained_on: str
    embedding_dim: int
    model_loaded: bool
    vocab_size: int


app = FastAPI(
    title="Teammate Finder API",
    description="Семантический матчмейкинг на собственной нейросети и размеченных игровых данных",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(project_root, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


def get_database():
    return vector_db


def get_embedder():
    return text_embedder


def get_preprocessor():
    return preprocessor


@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "Teammate Finder — нейросетевая система семантического матчмейкинга",
        "version": "2.0.0",
        "model": "custom_lstm_attention",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(db=Depends(get_database), embedder=Depends(get_embedder)):
    try:
        db_stats = db.get_stats()
        database_ok = db_stats is not None
        model_loaded = embedder.model_loaded
        status = "healthy" if database_ok and model_loaded else "degraded"

        return HealthResponse(
            status=status,
            database=database_ok,
            model_loaded=model_loaded,
            model_type="custom_lstm_attention",
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            database=False,
            model_loaded=False,
            model_type="custom_lstm_attention",
        )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info(embedder=Depends(get_embedder)):
    info = embedder.get_model_info()
    return ModelInfoResponse(**info)


@app.post("/users", response_model=Dict[str, str])
async def add_user(
    user_request: UserRequest,
    db=Depends(get_database),
    embedder=Depends(get_embedder),
    text_preprocessor=Depends(get_preprocessor),
):
    try:
        cleaned_text = text_preprocessor.clean_text(user_request.description)
        embedding = embedder.get_embedding(cleaned_text)

        user_data = {
            "user_id": user_request.user_id,
            "username": user_request.username,
            "description": user_request.description,
            "embedding": embedding,
            "game_type": user_request.game_type,
        }

        success = db.add_user(user_data)
        if success:
            return {"status": "success", "message": "User added successfully"}
        raise HTTPException(status_code=500, detail="Failed to add user to database")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding user: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similarity/search", response_model=List[SimilarUserResponse])
async def find_similar_users(
    request: SimilarityRequest,
    db=Depends(get_database),
    embedder=Depends(get_embedder),
    text_preprocessor=Depends(get_preprocessor),
):
    try:
        cleaned_text = text_preprocessor.clean_text(request.description)
        embedding = embedder.get_embedding(cleaned_text)

        similar_users = db.find_similar_users(
            query_embedding=embedding,
            user_id=request.user_id,
            game_type=request.game_type,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
        )

        return similar_users
    except Exception as e:
        logger.error(f"Error finding similar users: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: str, db=Depends(get_database)):
    user = db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.delete("/users/{user_id}", response_model=Dict[str, str])
async def delete_user(user_id: str, db=Depends(get_database)):
    success = db.delete_user(user_id)
    if success:
        return {"status": "success", "message": "User deleted successfully"}
    raise HTTPException(status_code=404, detail="User not found")


@app.get("/stats", response_model=StatsResponse)
async def get_stats(db=Depends(get_database)):
    stats = db.get_stats()
    return StatsResponse(**stats)


@app.get("/test")
async def test_endpoint(embedder=Depends(get_embedder)):
    return {
        "status": "ok",
        "model": embedder.get_model_info(),
        "features": ["add_user", "find_similar", "get_user", "delete_user", "stats", "model_info"],
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
