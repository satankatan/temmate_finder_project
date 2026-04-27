from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseVectorDB(ABC):
    """Абстрактный базовый класс для векторных БД"""
    
    @abstractmethod
    def add_user(self, user_data: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    def find_similar_users(self, query_embedding: List[float], 
                          user_id: Optional[str] = None,
                          top_k: int = 5) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        pass