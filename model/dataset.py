import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class Vocabulary:
    """Словарь для собственной нейросети."""

    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.word_freq = Counter()

    def build_vocab(self, texts: List[str], min_freq: int = 1):
        for text in texts:
            self.word_freq.update(text.split())

        for word, freq in self.word_freq.items():
            if freq >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def __len__(self):
        return len(self.word2idx)

    def text_to_indices(self, text: str, max_length: int = 50) -> List[int]:
        words = text.split()[:max_length]
        indices = [self.word2idx.get(word, 1) for word in words]

        if len(indices) < max_length:
            indices += [0] * (max_length - len(indices))

        return indices

    def save(self, path: str):
        payload = {
            "word2idx": self.word2idx,
            "idx2word": {int(k): v for k, v in self.idx2word.items()},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        vocab = cls()
        vocab.word2idx = payload["word2idx"]
        vocab.idx2word = {int(k): v for k, v in payload["idx2word"].items()}
        return vocab


class LabeledPairDataset(Dataset):
    """Датасет триплетов из размеченных пар совместимости."""

    def __init__(
        self,
        triplets: List[Tuple[str, str, str]],
        vocabulary: Vocabulary,
        max_length: int = 50,
    ):
        self.triplets = triplets
        self.vocabulary = vocabulary
        self.max_length = max_length

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_text, positive_text, negative_text = self.triplets[idx]

        return {
            "anchor": torch.tensor(
                self.vocabulary.text_to_indices(anchor_text, self.max_length),
                dtype=torch.long,
            ),
            "positive": torch.tensor(
                self.vocabulary.text_to_indices(positive_text, self.max_length),
                dtype=torch.long,
            ),
            "negative": torch.tensor(
                self.vocabulary.text_to_indices(negative_text, self.max_length),
                dtype=torch.long,
            ),
        }


def load_compatibility_dataset(
    csv_path: str,
    preprocessor,
    positive_threshold: float = 0.7,
    negative_threshold: float = 0.45,
) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """Загружает размеченный датасет и строит триплеты для обучения."""
    df = pd.read_csv(csv_path).dropna()
    df["compatibility_score"] = df["compatibility_score"].clip(0, 1)

    texts = set()
    positive_pairs: List[Tuple[str, str]] = []
    negative_pairs: List[Tuple[str, str]] = []

    for _, row in df.iterrows():
        text1 = preprocessor.clean_text(str(row["description1"]))
        text2 = preprocessor.clean_text(str(row["description2"]))
        score = float(row["compatibility_score"])

        if not text1 or not text2:
            continue

        texts.add(text1)
        texts.add(text2)

        if score >= positive_threshold:
            positive_pairs.append((text1, text2))
            positive_pairs.append((text2, text1))
        elif score <= negative_threshold:
            negative_pairs.append((text1, text2))
            negative_pairs.append((text2, text1))

    triplets: List[Tuple[str, str, str]] = []
    all_texts = list(texts)

    for anchor, positive in positive_pairs:
        negatives = [neg for a, neg in negative_pairs if a == anchor]
        if not negatives:
            negatives = [
                candidate
                for candidate in all_texts
                if candidate not in {anchor, positive}
            ]
        if negatives:
            triplets.append((anchor, positive, random.choice(negatives)))

    return list(texts), triplets


def create_data_loaders(
    texts: List[str],
    triplets: List[Tuple[str, str, str]],
    batch_size: int = 16,
    train_ratio: float = 0.85,
    max_length: int = 50,
    min_freq: int = 1,
):
    """Создает DataLoader'ы на основе размеченных триплетов."""
    random.shuffle(triplets)
    split_idx = max(1, int(len(triplets) * train_ratio))
    train_triplets = triplets[:split_idx]
    val_triplets = triplets[split_idx:] or triplets[:1]

    vocab = Vocabulary()
    vocab.build_vocab(texts, min_freq=min_freq)

    train_dataset = LabeledPairDataset(train_triplets, vocab, max_length)
    val_dataset = LabeledPairDataset(val_triplets, vocab, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, vocab


def load_dataset_from_csv(
    csv_path: str,
    preprocessor,
    batch_size: int = 16,
    train_ratio: float = 0.85,
    max_length: int = 50,
):
    texts, triplets = load_compatibility_dataset(csv_path, preprocessor)
    if not triplets:
        raise ValueError("Не удалось построить триплеты из датасета")

    return create_data_loaders(
        texts=texts,
        triplets=triplets,
        batch_size=batch_size,
        train_ratio=train_ratio,
        max_length=max_length,
    )
