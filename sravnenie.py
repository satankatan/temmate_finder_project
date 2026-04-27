import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.manifold import TSNE
from sklearn.metrics import precision_score, ndcg_score
import scipy.stats as stats
from sentence_transformers import SentenceTransformer
import logging
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Настройка логирования и стиля графиков
logging.basicConfig(level=logging.INFO)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MethodComparator:
    def __init__(self):
        """Инициализация моделей для сравнения"""
        self.models = {
            'TF-IDF': None,
            'Sentence-BERT': SentenceTransformer('all-MiniLM-L6-v2'),
            'Multilingual-BERT': SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        }
        self.results = {}
        
    def load_and_prepare_data(self, file_path: str) -> pd.DataFrame:
        """Загрузка и подготовка данных"""
        # Предполагаем CSV с колонками: description1, description2, compatibility_score
        data = pd.read_csv(file_path)
        
        # Очистка данных
        data = data.dropna()
        data['compatibility_score'] = data['compatibility_score'].clip(0, 1)
        
        logging.info(f"Загружено {len(data)} пар описаний")
        return data
    
    def compute_embeddings(self, descriptions: List[str]) -> Dict[str, np.ndarray]:
        """Вычисление эмбеддингов разными методами"""
        embeddings = {}
        
        # TF-IDF
        tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_embeddings = tfidf.fit_transform(descriptions).toarray()
        embeddings['TF-IDF'] = tfidf_embeddings
        self.models['TF-IDF'] = tfidf
        
        # Sentence-BERT
        embeddings['Sentence-BERT'] = self.models['Sentence-BERT'].encode(descriptions)
        
        # Multilingual-BERT
        embeddings['Multilingual-BERT'] = self.models['Multilingual-BERT'].encode(descriptions)
        
        logging.info("Эмбеддинги успешно вычислены")
        return embeddings
    
    def calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray, method: str) -> float:
        """Расчет схожести между векторами"""
        if method == 'cosine':
            return cosine_similarity([emb1], [emb2])[0][0]
        elif method == 'euclidean':
            distance = np.linalg.norm(emb1 - emb2)
            return 1 / (1 + distance)  # Преобразование расстояния в схожесть
        else:
            raise ValueError(f"Неизвестный метод: {method}")
    
    def evaluate_method(self, method_name: str, embeddings: np.ndarray, 
                       data: pd.DataFrame, similarity_method: str = 'cosine') -> Dict:
        """Оценка одного метода"""
        predictions = []
        actual_scores = []
        
        for idx, row in data.iterrows():
            idx1 = idx  # Предполагаем, что индексы соответствуют порядку в embeddings
            idx2 = idx  # В реальности нужна правильная индексация
            
            emb1 = embeddings[idx1]
            emb2 = embeddings[idx2]
            
            similarity = self.calculate_similarity(emb1, emb2, similarity_method)
            predictions.append(similarity)
            actual_scores.append(row['compatibility_score'])
        
        # Бинаризация для precision@k
        threshold = 0.7  # Порог для бинарной классификации
        binary_actual = [1 if score >= threshold else 0 for score in actual_scores]
        binary_pred = [1 if pred >= threshold else 0 for pred in predictions]
        
        # Метрики
        precision = precision_score(binary_actual, binary_pred, zero_division=0)
        
        # NDCG
        try:
            ndcg = ndcg_score([actual_scores], [predictions])
        except:
            ndcg = 0
        
        # Корреляция
        correlation = np.corrcoef(predictions, actual_scores)[0, 1]
        
        return {
            'predictions': predictions,
            'actual_scores': actual_scores,
            'precision': precision,
            'ndcg': ndcg,
            'correlation': correlation,
            'binary_actual': binary_actual,
            'binary_pred': binary_pred
        }
    
    def compare_all_methods(self, data: pd.DataFrame):
        """Сравнение всех методов"""
        # Собираем все уникальные описания
        all_descriptions = list(set(data['description1'].tolist() + data['description2'].tolist()))
        
        # Вычисляем эмбеддинги
        embeddings_dict = self.compute_embeddings(all_descriptions)
        
        # Создаем маппинг описаний к индексам
        desc_to_idx = {desc: idx for idx, desc in enumerate(all_descriptions)}
        
        # Оцениваем каждый метод
        for method_name, embeddings in embeddings_dict.items():
            logging.info(f"Оценка метода: {method_name}")
            
            # Создаем копию данных с правильными индексами
            eval_data = data.copy()
            eval_data['idx1'] = eval_data['description1'].map(desc_to_idx)
            eval_data['idx2'] = eval_data['description2'].map(desc_to_idx)
            eval_data = eval_data.dropna()
            
            self.results[method_name] = self.evaluate_method(
                method_name, embeddings, eval_data
            )
    
    def create_comparison_plot(self):
        """Создание гистограммы сравнения методов"""
        methods = list(self.results.keys())
        precision_scores = [self.results[method]['precision'] for method in methods]
        ndcg_scores = [self.results[method]['ndcg'] for method in methods]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Гистограмма precision
        bars1 = ax1.bar(methods, precision_scores, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        ax1.set_title('Сравнение Precision методов', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Precision Score')
        ax1.set_ylim(0, 1)
        
        for bar, value in zip(bars1, precision_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Гистограмма NDCG
        bars2 = ax2.bar(methods, ndcg_scores, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        ax2.set_title('Сравнение NDCG методов', fontsize=14, fontweight='bold')
        ax2.set_ylabel('NDCG Score')
        ax2.set_ylim(0, 1)
        
        for bar, value in zip(bars2, ndcg_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('methods_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_tsne_visualization(self, embeddings_dict: Dict[str, np.ndarray]):
        """t-SNE визуализация семантических пространств"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (method_name, embeddings) in enumerate(embeddings_dict.items()):
            if embeddings.shape[1] > 2:  # Если размерность > 2, применяем t-SNE
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
                embeddings_2d = tsne.fit_transform(embeddings)
            else:
                embeddings_2d = embeddings
            
            scatter = axes[idx].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                      alpha=0.6, s=50)
            axes[idx].set_title(f'{method_name} - t-SNE проекция', fontweight='bold')
            axes[idx].set_xlabel('Component 1')
            axes[idx].set_ylabel('Component 2')
        
        plt.tight_layout()
        plt.savefig('tsne_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def statistical_significance_test(self):
        """Проверка статистической значимости различий"""
        methods = list(self.results.keys())
        
        print("=" * 50)
        print("СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ РАЗЛИЧИЙ")
        print("=" * 50)
        
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method1, method2 = methods[i], methods[j]
                
                pred1 = self.results[method1]['predictions']
                pred2 = self.results[method2]['predictions']
                actual = self.results[method1]['actual_scores']
                
                # T-test для предсказаний
                t_stat, p_value = stats.ttest_rel(pred1, pred2)
                
                # Корреляции с истинными значениями
                corr1 = np.corrcoef(pred1, actual)[0, 1]
                corr2 = np.corrcoef(pred2, actual)[0, 1]
                
                print(f"{method1} vs {method2}:")
                print(f"  T-test: t={t_stat:.3f}, p={p_value:.5f}")
                print(f"  Корреляции: {method1}={corr1:.3f}, {method2}={corr2:.3f}")
                print(f"  Стат. значимость: {'ДА' if p_value < 0.05 else 'НЕТ'}")
                print("-" * 40)
    
    def print_detailed_results(self):
        """Вывод детальных результатов"""
        print("=" * 60)
        print("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ СРАВНЕНИЯ МЕТОДОВ")
        print("=" * 60)
        
        for method_name, result in self.results.items():
            print(f"\n{method_name}:")
            print(f"  Precision: {result['precision']:.4f}")
            print(f"  NDCG: {result['ndcg']:.4f}")
            print(f"  Корреляция: {result['correlation']:.4f}")
            print(f"  Количество предсказаний: {len(result['predictions'])}")
    
    def run_complete_analysis(self, data_file: str):
        """Полный анализ"""
        # Загрузка данных
        data = self.load_and_prepare_data(data_file)
        
        # Сравнение методов
        self.compare_all_methods(data)
        
        # Визуализации
        all_descriptions = list(set(data['description1'].tolist() + data['description2'].tolist()))
        embeddings_dict = self.compute_embeddings(all_descriptions)
        
        self.create_comparison_plot()
        self.create_tsne_visualization(embeddings_dict)
        
        # Статистический анализ
        self.statistical_significance_test()
        self.print_detailed_results()
        
        return self.results

# Пример использования
if __name__ == "__main__":
    # Создаем тестовые данные (в реальности загрузите из CSV)
    sample_data = pd.DataFrame({
        'description1': [
            "Агрессивный саппорт, люблю давить лайн",
            "Тактический керри, фокусируюсь на фарме", 
            "Стратегический мидер, контролирую руны",
            "Пассивный саппорт, играю от защиты",
            "Хард-керри, нуждаюсь в поддержке саппорта",
            "Роам-саппорт, постоянно гангаю лайны",
            "Фарм-мидер, фокусируюсь на своем экспириенсе",
            "Снайпер в CS:GO, играю на AWP",
            "Энтри-фраггер, иду первым на сайт",
            "Дуэлянт в Valorant, открываю фраги",
            "Контроллер, ставлю смоки и контролирую зоны",
            "Ингейм-лидер, разрабатываю тактики",
            "Агрессивный керри, начинаю драться с первых минут",
            "Защитный саппорт, спасаю тиммейтов в драках",
            "Темпо-мидер, контролирую руны и гангаю",
            "Луркер в CS, режу ротации противника",
            "Страж в Valorant, защищаю тылы команды",
            "Универсальный керри, адаптируюсь под ситуацию",
            "Инициатор-саппорт, начинаю выгодные драки",
            "Пуш-мидер, разрушаю башни и создаю спейс",
            "Сupport-рифер, играю на подстраховке",
            "Смоук-мастер, блокирую обзор противнику",
            "Спейс-керри, создаю давление на карте",
            "Вардящий саппорт, контролирую карту",
            "Стратег, читаю игру противника",
            "Агрессивный дуэлянт, создаю пространство",
            "Сейв-саппорт, играю на спасение союзников",
            "Фраггер, набираю много убийств за раунд",
            "Темпо-керри, контролирую темп игры",
            "Роам-мидер, создаю давление на лайнах"
        ],
        'description2': [
            "Активный поддержка, постоянно harass противника",
            "Фармящий керри, играю на позднюю стадию",
            "Роам-мидер, постоянно гангаю лайны", 
            "Защитный саппорт, ставлю варды и хиллю",
            "Сейв-саппорт, играю на спасение союзников",
            "Агрессивный керри, начинаю драться с первых минут",
            "Темпо-мидер, контролирую руны и гангаю",
            "Сupport-рифер, играю на подстраховке снайпера",
            "Луркер, режу ротации и создаю неожиданности",
            "Контроллер, помогаю дуэлянту заходить на сайт",
            "Дуэлянт, работаю в паре с контроллером",
            "Рифлер, выполняю тактические указания лидера",
            "Агрессивный саппорт, давим лайн с первых секунд",
            "Хард-керри, нуждаюсь в защите саппорта",
            "Роам-саппорт, помогаю мидеру гангать",
            "Снайпер, занимаю позиции на второй линии",
            "Инициатор, разведываю позиции противника",
            "Вардящий саппорт, обеспечиваю видимость для керри",
            "Универсальный керри, подстраиваюсь под инициативы",
            "Защитный саппорт, прикрываю пуш-мидера",
            "Энтри-фраггер, открываю сайт для команды",
            "Дуэлянт, использую смоки для захода",
            "Роам-саппорт, создаю спейс для керри",
            "Стратегический керри, использую информацию с карты",
            "Ингейм-лидер, координирую действия стратега",
            "Контроллер, создаю условия для дуэлянта",
            "Хард-саппорт, покупаю все необходимое для команды",
            "Страж, обеспечиваю безопасность фраггеру",
            "Тактический саппорт, помогаю контролировать темп",
            "Агрессивный саппорт, поддерживаю роам-мидера"
        ],
        'compatibility_score': [
            0.85,  # агрессивный саппорт + активный поддержка
            0.75,  # тактический керри + фармящий керри  
            0.80,  # стратегический мидер + роам-мидер
            0.90,  # пассивный саппорт + защитный саппорт
            0.95,  # хард-керри + сейв-саппорт
            0.70,  # роам-саппорт + агрессивный керри
            0.65,  # фарм-мидер + темпо-мидер
            0.75,  # снайпер + support-рифер
            0.80,  # энтри-фраггер + луркер
            0.85,  # дуэлянт + контроллер
            0.80,  # контроллер + дуэлянт
            0.75,  # ингейм-лидер + рифлер
            0.90,  # агрессивный керри + агрессивный саппорт
            0.85,  # защитный саппорт + хард-керри
            0.80,  # темпо-мидер + роам-саппорт
            0.70,  # луркер + снайпер
            0.75,  # страж + инициатор
            0.80,  # универсальный керри + вардящий саппорт
            0.85,  # инициатор-саппорт + универсальный керри
            0.75,  # пуш-мидер + защитный саппорт
            0.80,  # support-рифер + энтри-фраггер
            0.85,  # смоук-мастер + дуэлянт
            0.80,  # спейс-керри + роам-саппорт
            0.75,  # вардящий саппорт + стратегический керри
            0.70,  # стратег + ингейм-лидер
            0.85,  # агрессивный дуэлянт + контроллер
            0.90,  # сейв-саппорт + хард-керри
            0.75,  # фраггер + страж
            0.80,  # темпо-керри + тактический саппорт
            0.85   # роам-мидер + агрессивный саппорт
        ]
    })
    
    # Сохраняем sample data для демонстрации
    sample_data.to_csv('sample_gaming_data.csv', index=False)
    
    # Запускаем анализ
    comparator = MethodComparator()
    results = comparator.run_complete_analysis('sample_gaming_data.csv')