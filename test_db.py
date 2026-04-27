import sys

sys.path.append('database')
from database.sqlite_db import vector_db

# Тест базы данных
def test_database():
    # Добавляем тестового пользователя
    test_user = {
        'user_id': 'test_user_1',
        'username': 'TestPlayer',
        'description': 'Агрессивный игрок, люблю атаковать',
        'embedding': [0.1] * 128,
        'game_type': 'dota2'
    }
    
    success = vector_db.add_user(test_user)
    print(f"User added: {success}")
    
    # Получаем статистику
    stats = vector_db.get_stats()
    print(f"Database stats: {stats}")
    
    # Ищем пользователя
    user = vector_db.get_user('test_user_1')
    print(f"Found user: {user is not None}")

if __name__ == "__main__":
    test_database()