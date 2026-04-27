import re
from typing import List

class TextPreprocessor:
    def __init__(self):
        self.gaming_slang_dict = {
            'агр': 'агрессивный', 'агро': 'агрессивный', 'агрисьный': 'агрессивный',
            'сап': 'саппорт', 'сапп': 'саппорт', 'сапорт': 'саппорт',
            'мид': 'мидер', 'мидер': 'мидер', 'лайн': 'линия', 'лайне': 'линия',
            'каррю': 'нести', 'карри': 'нести',
            'фармлю': 'зарабатывать', 'фарм': 'зарабатывание',
            'ганк': 'нападение', 'ганкаю': 'нападать',
            'роам': 'перемещение', 'роамлю': 'перемещаться',
            'керри': 'носитель', 'абу': 'прокачка', 'апаю': 'прокачивать',
            'нерф': 'ослабление', 'нерфнуть': 'ослаблять',
            'крашить': 'убивать', 'крашу': 'убивать',
            'катка': 'игра', 'катку': 'игру',
            'ммр': 'рейтинг', 'рангов': 'рейтинговый', 'рангом': 'рейтинг',
            'сапаю': 'играть саппортом', 'драфт': 'выбор героев',
            'дотка': 'dota', 'доту': 'dota', 'контра': 'контрпик',
            'пик': 'выбор героя', 'хиллю': 'лечить', 'хил': 'лечение'
        }
        
        self.stop_words = {
            'игра', 'игре', 'игрой', 'игру', 'гейм', 'гам', 'гаминг',
            'просто', 'очень', 'сильно', 'вообще', 'типа', 'как бы', 'это',
            'что', 'который', 'такой', 'еще', 'уже', 'можно', 'нужно', 'очень',
            'я', 'ты', 'он', 'она', 'они', 'мы', 'вы', 'и', 'в', 'на', 'с', 'по',
            'у', 'о', 'об', 'из', 'от', 'до', 'за', 'же', 'ли', 'бы', 'то', 'не',
            'ну', 'вот', 'как', 'так', 'там', 'здесь', 'прям', 'просто', 'люблю'
        }
        
        self.gaming_keywords = {
            'агрессивный', 'пассивный', 'стратегический', 'тактический',
            'саппорт', 'керри', 'мидер', 'оффлейнер', 'роамер', 'джанглер',
            'ранговый', 'рейтинговый', 'командный', 'одиночный', 'атакующий',
            'защитный', 'контроль', 'карта', 'линия', 'лес', 'башня', 'герой',
            'нести', 'зарабатывать', 'нападать', 'перемещаться', 'лечить'
        }
    
    def clean_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""

        text = text.lower()

        text = re.sub(r'[^\w\s]', ' ', text)

        text = self._replace_gaming_slang(text)

        text = self._remove_stopwords_preserve_keywords(text)

        text = ' '.join(text.split())
        
        return text
    
    def _replace_gaming_slang(self, text: str) -> str:

        words = text.split()
        normalized_words = []
        
        for word in words:

            if word in self.gaming_slang_dict:
                normalized_words.append(self.gaming_slang_dict[word])
            else:
                normalized_words.append(word)
        
        text = ' '.join(normalized_words)

        for slang, normal in self.gaming_slang_dict.items():

            text = re.sub(r'\b' + slang + r'\b', normal, text)
        
        return text
    
    def _remove_stopwords_preserve_keywords(self, text: str) -> str:
        """Удаляет стоп-слова, но сохраняет игровые ключевые слова"""
        words = text.split()
        filtered_words = [
            word for word in words 
            if (word not in self.stop_words or word in self.gaming_keywords) 
            and len(word) > 2
        ]
        return ' '.join(filtered_words)
    
    def preprocess_for_training(self, texts: List[str]) -> List[str]:
        """Пакетная обработка текстов для обучения"""
        return [self.clean_text(text) for text in texts if text]

# Глобальный инстанс для использования в API
preprocessor = TextPreprocessor()