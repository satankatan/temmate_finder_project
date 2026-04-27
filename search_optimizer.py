import numpy as np
from typing import List, Dict, Any
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

logger = logging.getLogger(__name__)

class SearchOptimizer:
    def __init__(self, vector_db, embedder, preprocessor):
        self.vector_db = vector_db
        self.embedder = embedder
        self.preprocessor = preprocessor
        self.quality_threshold = 0.7
        self.fallback_enabled = True
        
        # TF-IDF fallback
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        self.is_tfidf_trained = False
    
    def train_tfidf_fallback(self, descriptions: List[str]):
        """Обучение TF-IDF fallback системы"""
        try:
            cleaned_descriptions = [self.preprocessor.clean_text(desc) for desc in descriptions]
            self.tfidf_vectorizer.fit(cleaned_descriptions)
            self.is_tfidf_trained = True
            logger.info("TF-IDF fallback system trained successfully")
        except Exception as e:
            logger.error(f"Error training TF-IDF: {e}")
    
    def hybrid_search(self, query: str, user_id: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Гибридный поиск: нейросеть + TF-IDF"""
        try:
            # Основной поиск через нейросеть
            cleaned_query = self.preprocessor.clean_text(query)
            embedding = self.embedder.get_embedding(cleaned_query)
            
            neural_results = self.vector_db.find_similar_users(
                embedding, user_id, top_k * 2, 0.3  # Более низкий порог для большего охвата
            )
            
            # Если нейросеть дает плохие результаты, используем fallback
            if self.fallback_enabled and self._is_low_quality(neural_results):
                logger.info("Using TF-IDF fallback due to low neural network quality")
                tfidf_results = self._tfidf_search(cleaned_query, user_id, top_k)
                return self._rerank_results(neural_results, tfidf_results, top_k)
            
            return neural_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fallback to TF-IDF в случае ошибки
            if self.fallback_enabled and self.is_tfidf_trained:
                return self._tfidf_search(query, user_id, top_k)
            return []
    
    def _is_low_quality(self, results: List[Dict]) -> bool:
        """Проверка качества результатов нейросети"""
        if not results:
            return True
        
        avg_similarity = np.mean([r['similarity_score'] for r in results])
        return avg_similarity < self.quality_threshold
    
    def _tfidf_search(self, query: str, user_id: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """TF-IDF based search as fallback"""
        if not self.is_tfidf_trained:
            return []
        
        try:
            # Получаем всех пользователей из базы
            all_users = self._get_all_users()
            if not all_users:
                return []
            
            # Подготавливаем тексты
            user_descriptions = [user['description'] for user in all_users]
            user_ids = [user['user_id'] for user in all_users]
            
            # Вычисляем TF-IDF схожесть
            query_vec = self.tfidf_vectorizer.transform([query])
            descs_vec = self.tfidf_vectorizer.transform(user_descriptions)
            
            similarities = cosine_similarity(query_vec, descs_vec).flatten()
            
            # Сортируем по схожести
            sorted_indices = np.argsort(similarities)[::-1]
            
            results = []
            for idx in sorted_indices[:top_k]:
                if user_ids[idx] != user_id:  # Исключаем текущего пользователя
                    results.append({
                        'user_id': user_ids[idx],
                        'username': all_users[idx].get('username'),
                        'description': all_users[idx]['description'],
                        'game_type': all_users[idx].get('game_type', 'general'),
                        'similarity_score': float(similarities[idx])
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in TF-IDF search: {e}")
            return []
    
    def _get_all_users(self) -> List[Dict]:
        """Получение всех пользователей из базы (упрощенная версия)"""
        # В реальной реализации здесь будет запрос к базе
        return []
    
    def _rerank_results(self, neural_results: List[Dict], tfidf_results: List[Dict], top_k: int) -> List[Dict]:
        """Переранжирование результатов от разных методов"""
        all_results = {}
        
        # Добавляем результаты нейросети с весом
        for result in neural_results:
            all_results[result['user_id']] = {
                **result,
                'combined_score': result['similarity_score'] * 0.7  # Вес нейросети
            }
        
        # Добавляем TF-IDF результаты с весом
        for result in tfidf_results:
            user_id = result['user_id']
            if user_id in all_results:
                # Усредняем если пользователь есть в обоих результатах
                all_results[user_id]['combined_score'] = (
                    all_results[user_id]['combined_score'] + result['similarity_score'] * 0.3
                )
            else:
                all_results[user_id] = {
                    **result,
                    'combined_score': result['similarity_score'] * 0.3  # Вес TF-IDF
                }
        
        # Сортируем по комбинированному score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )
        
        # Убираем combined_score из финального результата
        for result in sorted_results:
            result.pop('combined_score', None)
        
        return sorted_results[:top_k]
    
    def optimize_search_parameters(self, test_queries: List[Dict]):
        """Оптимизация параметров поиска"""
        best_threshold = 0.3
        best_quality = 0.0
        
        for threshold in np.arange(0.1, 0.9, 0.1):
            self.quality_threshold = threshold
            total_quality = 0.0
            
            for query in test_queries:
                results = self.hybrid_search(query['description'])
                if results:
                    avg_similarity = np.mean([r['similarity_score'] for r in results])
                    total_quality += avg_similarity
            
            avg_quality = total_quality / len(test_queries) if test_queries else 0.0
            
            if avg_quality > best_quality:
                best_quality = avg_quality
                best_threshold = threshold
        
        self.quality_threshold = best_threshold
        logger.info(f"Optimized quality threshold: {best_threshold:.2f}")
        
        return best_threshold

class QueryExpander:
    """Расширитель запросов для улучшения поиска"""
    
    def __init__(self):
        self.gaming_synonyms = {
            'агрессивный': ['атакующий', 'напористый', 'давление', 'агрессия'],
            'спокойный': ['расслабленный', 'пассивный', 'терпеливый', 'стабильный'],
            'стратегический': ['тактический', 'продуманный', 'аналитический', 'умный'],
            'саппорт': ['поддержка', 'помощник', 'хилер', 'баффер'],
            'керри': ['носитель', 'дд', 'урон', 'атакующий'],
            'мидер': ['центральный', 'мидлейнер', 'контроль центра'],
            'ранговый': ['рейтинговый', 'competitive', 'рангед']
        }
    
    def expand_query(self, query: str) -> str:
        """Расширение запроса синонимами"""
        words = query.lower().split()
        expanded_words = []
        
        for word in words:
            expanded_words.append(word)
            if word in self.gaming_synonyms:
                expanded_words.extend(self.gaming_synonyms[word][:2])  # Добавляем 2 синонима
        
        return ' '.join(expanded_words)