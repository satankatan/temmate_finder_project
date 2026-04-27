import pytest
import asyncio
from fastapi.testclient import TestClient
import numpy as np
import sys
import os

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app, get_database
from database.sqlite_db import vector_db
from preprocessing.text_processor import TextPreprocessor

# Test client
client = TestClient(app)

# Mock данные для тестов
TEST_USERS = [
    {
        "user_id": "test_user_1",
        "username": "AggressivePlayer",
        "description": "Агрессивный игрок, люблю атаковать и давить на врага. Всегда впереди команды!",
        "game_type": "dota2"
    },
    {
        "user_id": "test_user_2", 
        "username": "StrategicMind",
        "description": "Стратегический игрок, продумываю тактики и контрпики. Люблю умную игру.",
        "game_type": "dota2"
    },
    {
        "user_id": "test_user_3",
        "username": "SupportMain",
        "description": "Спокойный саппорт, помогаю команде, ставлю варды и контролирую карту.",
        "game_type": "csgo"
    }
]

class TestVectorDatabase:
    """Тесты для базы данных"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.db = vector_db()
        self.preprocessor = TextPreprocessor()
        
    def teardown_method(self):
        """Очистка после каждого теста"""
        # Удаляем тестовых пользователей
        for user in TEST_USERS:
            self.db.delete_user(user['user_id'])
        self.db.close()
    
    def test_add_and_get_user(self):
        """Тест добавления и получения пользователя"""
        test_user = TEST_USERS[0]
        test_embedding = [0.1] * 128
        
        user_data = {
            **test_user,
            'embedding': test_embedding
        }
        
        # Добавляем пользователя
        result = self.db.add_user(user_data)
        assert result == True
        
        # Получаем пользователя
        retrieved_user = self.db.get_user(test_user['user_id'])
        assert retrieved_user is not None
        assert retrieved_user['user_id'] == test_user['user_id']
        assert retrieved_user['description'] == test_user['description']
    
    def test_find_similar_users(self):
        """Тест поиска похожих пользователей"""
        # Добавляем тестовых пользователей
        for user in TEST_USERS:
            user_data = {
                **user,
                'embedding': [0.1 * i for i in range(128)]  # Простой эмбеддинг
            }
            self.db.add_user(user_data)
        
        # Поиск похожих пользователей
        query_embedding = [0.1 * i for i in range(128)]
        similar_users = self.db.find_similar_users(
            query_embedding, 
            top_k=2,
            similarity_threshold=0.5
        )
        
        assert len(similar_users) > 0
        assert 'similarity_score' in similar_users[0]
    
    def test_delete_user(self):
        """Тест удаления пользователя"""
        test_user = TEST_USERS[0]
        user_data = {
            **test_user,
            'embedding': [0.1] * 128
        }
        
        self.db.add_user(user_data)
        
        # Удаляем пользователя
        result = self.db.delete_user(test_user['user_id'])
        assert result == True
        
        # Проверяем, что пользователь удален
        retrieved_user = self.db.get_user(test_user['user_id'])
        assert retrieved_user is None
    
    def test_get_stats(self):
        """Тест получения статистики"""
        stats = self.db.get_stats()
        assert 'total_users' in stats
        assert 'users_by_game_type' in stats

class TestAPIEndpoints:
    """Тесты для API эндпоинтов"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.client = TestClient(app)
        
        # Очищаем тестовых пользователей
        self.db = vector_db()
        for user in TEST_USERS:
            self.db.delete_user(user['user_id'])
        self.db.close()
    
    def test_health_check(self):
        """Тест проверки здоровья сервиса"""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] in ['healthy', 'degraded', 'unhealthy']
    
    def test_root_endpoint(self):
        """Тест корневого эндпоинта"""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert 'message' in data
        assert 'version' in data
    
    def test_add_user_success(self):
        """Тест успешного добавления пользователя"""
        user_data = TEST_USERS[0]
        response = self.client.post("/users", json=user_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
    
    def test_add_user_validation(self):
        """Тест валидации данных пользователя"""
        invalid_user = {
            "user_id": "test",
            "description": "short"  # Слишком короткое описание
        }
        
        response = self.client.post("/users", json=invalid_user)
        assert response.status_code == 422  # Validation error
    
    def test_find_similar_users(self):
        """Тест поиска похожих пользователей"""
        # Сначала добавляем пользователей
        for user in TEST_USERS:
            self.client.post("/users", json=user)
        
        # Затем ищем похожих
        search_request = {
            "description": "Ищу агрессивного игрока для атаки",
            "top_k": 3,
            "similarity_threshold": 0.1
        }
        
        response = self.client.post("/similarity/search", json=search_request)
        assert response.status_code == 200
        
        results = response.json()
        assert isinstance(results, list)
    
    def test_get_user_success(self):
        """Тест успешного получения пользователя"""
        user_data = TEST_USERS[0]
        self.client.post("/users", json=user_data)
        
        response = self.client.get(f"/users/{user_data['user_id']}")
        assert response.status_code == 200
        
        user_response = response.json()
        assert user_response['user_id'] == user_data['user_id']
    
    def test_get_user_not_found(self):
        """Тест получения несуществующего пользователя"""
        response = self.client.get("/users/nonexistent_user")
        assert response.status_code == 404
    
    def test_delete_user(self):
        """Тест удаления пользователя"""
        user_data = TEST_USERS[0]
        self.client.post("/users", json=user_data)
        
        response = self.client.delete(f"/users/{user_data['user_id']}")
        assert response.status_code == 200
        
        # Проверяем, что пользователь действительно удален
        get_response = self.client.get(f"/users/{user_data['user_id']}")
        assert get_response.status_code == 404
    
    def test_get_stats(self):
        """Тест получения статистики"""
        response = self.client.get("/stats")
        assert response.status_code == 200
        
        stats = response.json()
        assert 'total_users' in stats
        assert 'users_by_game_type' in stats

class TestTextProcessing:
    """Тесты для обработки текста"""
    
    def setup_method(self):
        self.preprocessor = TextPreprocessor()
    
    def test_text_cleaning(self):
        """Тест очистки текста"""
        test_text = "Я агрессивный игрок!!! Люблю давить на врага и контролировать карту."
        cleaned = self.preprocessor.clean_text(test_text)
        
        assert isinstance(cleaned, str)
        assert len(cleaned) > 0
        # Проверяем, что специальные символы удалены
        assert '!!!' not in cleaned
    
    def test_gaming_slang_replacement(self):
        """Тест замены игрового сленга"""
        slang_text = "Я агро мидер, каррю игры, фармлю и ганкаю линии"
        cleaned = self.preprocessor.clean_text(slang_text)
        
        # Проверяем, что сленг заменен на нормальные слова
        assert 'агро' not in cleaned or 'агрессивный' in cleaned
    
    def test_empty_text(self):
        """Тест обработки пустого текста"""
        cleaned = self.preprocessor.clean_text("")
        assert cleaned == ""
    
    def test_special_characters(self):
        """Тест обработки текста со специальными символами"""
        text_with_special = "Игрок@#$% с *()опытом!!!"
        cleaned = self.preprocessor.clean_text(text_with_special)
        
        # Проверяем, что остались только буквы и пробелы
        assert all(c.isalnum() or c.isspace() for c in cleaned)

# Запуск тестов
if __name__ == "__main__":
    pytest.main([__file__, "-v"])