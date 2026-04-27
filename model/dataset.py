import torch
from torch.utils.data import Dataset, DataLoader
import random
from collections import Counter
import json

class Vocabulary:
    """Класс для управления словарем"""
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_freq = Counter()
    
    def build_vocab(self, texts, min_freq=2):
        """Строит словарь из списка текстов"""
        for text in texts:
            self.word_freq.update(text.split())
        
        # Добавляем слова, встречающиеся достаточно часто
        for word, freq in self.word_freq.items():
            if freq >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def __len__(self):
        return len(self.word2idx)
    
    def text_to_indices(self, text, max_length=50):
        """Преобразует текст в последовательность индексов"""
        words = text.split()[:max_length]
        indices = [self.word2idx.get(word, 1) for word in words]  # 1 = <UNK>
        
        # Добавляем padding если нужно
        if len(indices) < max_length:
            indices += [0] * (max_length - len(indices))  # 0 = <PAD>
        
        return indices

class TeammateDataset(Dataset):
    def __init__(self, texts, vocabulary, max_length=50, mode='triplet'):
        """
        Args:
            texts: список текстовых описаний
            vocabulary: объект Vocabulary
            max_length: максимальная длина последовательности
            mode: 'triplet' или 'contrastive'
        """
        self.texts = texts
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.mode = mode
        
        # Для triplet learning создаем "похожие" и "непохожие" пары
        if mode == 'triplet':
            self._prepare_triplets()
    
    def _prepare_triplets(self):
        """Подготавливает триплеты (anchor, positive, negative)"""
        self.triplets = []
        
        # Простая стратегия: считаем тексты похожими если у них есть общие ключевые слова
        for i, anchor_text in enumerate(self.texts):
            anchor_words = set(anchor_text.split())
            
            # Ищем похожий текст (positive)
            positive_candidates = []
            for j, candidate_text in enumerate(self.texts):
                if i != j:
                    candidate_words = set(candidate_text.split())
                    similarity = len(anchor_words & candidate_words) / len(anchor_words | candidate_words)
                    if similarity > 0.3:  # порог похожести
                        positive_candidates.append((j, similarity))
            
            # Ищем непохожий текст (negative)
            negative_candidates = []
            for j, candidate_text in enumerate(self.texts):
                if i != j:
                    candidate_words = set(candidate_text.split())
                    similarity = len(anchor_words & candidate_words) / len(anchor_words | candidate_words)
                    if similarity < 0.1:  # порог непохожести
                        negative_candidates.append((j, similarity))
            
            # Создаем триплеты
            if positive_candidates and negative_candidates:
                # Берем самый похожий и самый непохожий
                positive_idx = max(positive_candidates, key=lambda x: x[1])[0]
                negative_idx = min(negative_candidates, key=lambda x: x[1])[0]
                
                self.triplets.append((i, positive_idx, negative_idx))
    
    def __len__(self):
        if self.mode == 'triplet':
            return len(self.triplets)
        else:
            return len(self.texts)
    
    def __getitem__(self, idx):
        if self.mode == 'triplet':
            anchor_idx, positive_idx, negative_idx = self.triplets[idx]
            
            anchor_text = self.texts[anchor_idx]
            positive_text = self.texts[positive_idx]
            negative_text = self.texts[negative_idx]
            
            anchor_indices = self.vocabulary.text_to_indices(anchor_text, self.max_length)
            positive_indices = self.vocabulary.text_to_indices(positive_text, self.max_length)
            negative_indices = self.vocabulary.text_to_indices(negative_text, self.max_length)
            
            return {
                'anchor': torch.tensor(anchor_indices, dtype=torch.long),
                'positive': torch.tensor(positive_indices, dtype=torch.long),
                'negative': torch.tensor(negative_indices, dtype=torch.long)
            }
        
        else:
            text = self.texts[idx]
            indices = self.vocabulary.text_to_indices(text, self.max_length)
            return torch.tensor(indices, dtype=torch.long)

def create_data_loaders(texts, batch_size=32, train_ratio=0.8, max_length=50):
    """Создает DataLoader'ы для обучения и валидации"""
    
    # Разделяем данные
    split_idx = int(len(texts) * train_ratio)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    # Создаем и обучаем словарь на тренировочных данных
    vocab = Vocabulary()
    vocab.build_vocab(train_texts, min_freq=2)
    
    # Создаем datasets
    train_dataset = TeammateDataset(train_texts, vocab, max_length, mode='triplet')
    val_dataset = TeammateDataset(val_texts, vocab, max_length, mode='triplet')
    
    # Создаем data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, vocab