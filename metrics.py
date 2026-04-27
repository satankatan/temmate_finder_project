import numpy as np
from sklearn.metrics import precision_score, recall_score, ndcg_score
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class SearchMetrics:
    def __init__(self):
        self.metrics_history = []
    
    def calculate_precision_recall(self, results: List[Dict], relevant_users: List[str], k: int = 5):
        """Вычисление Precision@K и Recall@K"""
        if not results or not relevant_users:
            return 0.0, 0.0
        
        # Берем топ-K результатов
        top_k_results = results[:k]
        top_k_ids = [result['user_id'] for result in top_k_results]
        
        # Количество релевантных результатов в топ-K
        relevant_in_top_k = len(set(top_k_ids) & set(relevant_users))
        
        precision = relevant_in_top_k / len(top_k_results) if top_k_results else 0.0
        recall = relevant_in_top_k / len(relevant_users) if relevant_users else 0.0
        
        return precision, recall
    
    def calculate_ndcg(self, results: List[Dict], relevant_users: List[str], k: int = 5):
        """Вычисление NDCG@K"""
        if not results or not relevant_users:
            return 0.0
        
        # Создаем идеальный ranking
        ideal_ranking = [1.0] * min(len(relevant_users), k) + [0.0] * (k - min(len(relevant_users), k))
        
        # Создаем фактический ranking
        actual_relevance = []
        for result in results[:k]:
            relevance = 1.0 if result['user_id'] in relevant_users else 0.0
            actual_relevance.append(relevance)
        
        # Дополняем до длины K если нужно
        actual_relevance.extend([0.0] * (k - len(actual_relevance)))
        
        try:
            ndcg = ndcg_score([ideal_ranking], [actual_relevance])
            return ndcg
        except:
            return 0.0
    
    def calculate_mean_reciprocal_rank(self, results: List[Dict], relevant_users: List[str]):
        """Вычисление Mean Reciprocal Rank"""
        if not results or not relevant_users:
            return 0.0
        
        for rank, result in enumerate(results, 1):
            if result['user_id'] in relevant_users:
                return 1.0 / rank
        
        return 0.0
    
    def evaluate_search_quality(self, test_queries: List[Dict]):
        """Оценка качества поиска на тестовых запросах"""
        metrics = {
            'precision@5': [],
            'recall@5': [],
            'ndcg@5': [],
            'mrr': [],
            'avg_similarity_score': []
        }
        
        for query in test_queries:
            results = query['results']
            relevant_users = query['relevant_users']
            
            precision, recall = self.calculate_precision_recall(results, relevant_users, 5)
            ndcg = self.calculate_ndcg(results, relevant_users, 5)
            mrr = self.calculate_mean_reciprocal_rank(results, relevant_users)
            avg_similarity = np.mean([r['similarity_score'] for r in results]) if results else 0.0
            
            metrics['precision@5'].append(precision)
            metrics['recall@5'].append(recall)
            metrics['ndcg@5'].append(ndcg)
            metrics['mrr'].append(mrr)
            metrics['avg_similarity_score'].append(avg_similarity)
        
        # Агрегируем метрики
        aggregated_metrics = {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
            for metric, values in metrics.items()
        }
        
        self.metrics_history.append(aggregated_metrics)
        return aggregated_metrics
    
    def generate_quality_report(self, test_queries: List[Dict]):
        """Генерация отчета  качестве"""
        metrics = self.evaluate_search_quality(test_queries)
        
        report = """
        📊 ОТЧЕТ  КАЧЕСТВЕ ПОИСКА
        ===========================
        
        Основные метрики:
        • Precision@5: {precision_mean:.3f} ± {precision_std:.3f}
        • Recall@5: {recall_mean:.3f} ± {recall_std:.3f}
        • NDCG@5: {ndcg_mean:.3f} ± {ndcg_std:.3f}
        • MRR: {mrr_mean:.3f} ± {mrr_std:.3f}
        • Средняя схожесть: {similarity_mean:.3f} ± {similarity_std:.3f}
        
        Интерпретация:
        • Precision > 0.7 - отличное качество
        • Recall > 0.6 - хорошее покрытие
        • NDCG > 0.8 - отличное ранжирование
        """.format(
            precision_mean=metrics['precision@5']['mean'],
            precision_std=metrics['precision@5']['std'],
            recall_mean=metrics['recall@5']['mean'],
            recall_std=metrics['recall@5']['std'],
            ndcg_mean=metrics['ndcg@5']['mean'],
            ndcg_std=metrics['ndcg@5']['std'],
            mrr_mean=metrics['mrr']['mean'],
            mrr_std=metrics['mrr']['std'],
            similarity_mean=metrics['avg_similarity_score']['mean'],
            similarity_std=metrics['avg_similarity_score']['std']
        )
        
        return report

class ABTest:
    def __init__(self):
        self.variants = {}
    
    def add_variant(self, name, model):
        """Добавление варианта для тестирования"""
        self.variants[name] = {
            'model': model,
            'metrics': [],
            'users': set()
        }
    
    def run_comparison(self, test_queries: List[Dict], variant_name: str):
        """Запуск сравнения для конкретного варианта"""
        metrics_calculator = SearchMetrics()
        
        # Симулируем результаты для варианта
        simulated_results = []
        for query in test_queries:
            # Здесь будет реальный вызов модели
            results = []  # Заглушка
            simulated_results.append({
                'results': results,
                'relevant_users': query['relevant_users']
            })
        
        metrics = metrics_calculator.evaluate_search_quality(simulated_results)
        self.variants[variant_name]['metrics'].append(metrics)
        
        return metrics
    
    def get_best_variant(self):
        """Определение лучшего варианта"""
        if not self.variants:
            return None
        
        best_variant = None
        best_score = -1
        
        for name, data in self.variants.items():
            if data['metrics']:
                avg_precision = np.mean([m['precision@5']['mean'] for m in data['metrics']])
                if avg_precision > best_score:
                    best_score = avg_precision
                    best_variant = name
        
        return best_variant