import pandas as pd
import numpy as np
from model.inference import TextEmbedder
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score

# Ваши данные
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

def calculate_predictions(data):
    """Рассчитывает predictions для всех пар"""
    
    # Инициализируем модель
    embedder = TextEmbedder()
    
    # Получаем эмбеддинги для ВСЕХ описаний
    all_descriptions = list(data['description1']) + list(data['description2'])
    print("Получаем эмбеддинги...")
    
    all_embeddings = []
    for desc in all_descriptions:
        embedding = embedder.get_embedding(desc)
        all_embeddings.append(embedding)
    
    # Рассчитываем схожести для каждой пары
    predictions = []
    
    for i in range(len(data)):
        # Берем эмбеддинги для текущей пары
        emb1 = all_embeddings[i]  # description1
        emb2 = all_embeddings[i + len(data)]  # description2
        
        # Вычисляем косинусную схожесть
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        predictions.append(similarity)
        
        print(f"Пара {i+1}: prediction = {similarity:.3f}, actual = {data.iloc[i]['compatibility_score']}")
    
    return predictions

def plot_precision_vs_threshold(predictions, actual_scores):
    """Строит график зависимости точности от порога"""
    
    # Бинаризуем истинные оценки (порог 0.7)
    binary_actual = [1 if score >= 0.7 else 0 for score in actual_scores]
    
    # Перебираем разные пороги
    thresholds = np.arange(0.1, 1.0, 0.05)
    precision_scores = []
    
    for threshold in thresholds:
        # Бинаризуем предсказания по текущему порогу
        binary_pred = [1 if pred >= threshold else 0 for pred in predictions]
        
        # Считаем precision
        precision = precision_score(binary_actual, binary_pred, zero_division=0)
        precision_scores.append(precision)
    
    # Строим график
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision_scores, 'b-o', linewidth=2, markersize=6)
    plt.title('Зависимость точности от порога схожести', fontsize=14, fontweight='bold')
    plt.xlabel('Порог схожести', fontsize=12)
    plt.ylabel('Точность (Precision)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Находим оптимальный порог
    optimal_idx = np.argmax(precision_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_precision = precision_scores[optimal_idx]
    
    # Подсвечиваем оптимальную точку
    plt.scatter(optimal_threshold, optimal_precision, color='red', s=100, zorder=5)
    plt.annotate(f'Оптимум: {optimal_threshold:.2f}', 
                xy=(optimal_threshold, optimal_precision),
                xytext=(optimal_threshold + 0.1, optimal_precision - 0.1),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    plt.savefig('precision_vs_threshold.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return optimal_threshold, optimal_precision

if __name__ == "__main__":
    # 1. Получаем predictions
    predictions = calculate_predictions(sample_data)
    actual_scores = sample_data['compatibility_score'].tolist()
    
    # 2. Сохраняем результаты
    results_df = pd.DataFrame({
        'description1': sample_data['description1'],
        'description2': sample_data['description2'], 
        'prediction': predictions,
        'actual_score': actual_scores
    })
    results_df.to_csv('predictions_results.csv', index=False, encoding='utf-8')
    
    print(f"\n✅ Predictions сохранены в predictions_results.csv")
    print(f"📊 Примеры predictions: {predictions[:5]}")
    
    # 3. Строим график
    optimal_threshold, max_precision = plot_precision_vs_threshold(predictions, actual_scores)
    print(f"🎯 Оптимальный порог: {optimal_threshold:.3f}")
    print(f"📈 Максимальная точность: {max_precision:.3f}")