import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from model.dataset import Vocabulary
from model.network import TextEmbeddingModel
from preprocessing.text_processor import preprocessor

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
DEFAULT_VOCAB_PATH = PROJECT_ROOT / "models" / "vocab.json"


class CustomTextEmbedder:
    """Инференс собственной LSTM+Attention модели, обученной на игровом сленге."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self.vocab_path = Path(vocab_path or DEFAULT_VOCAB_PATH)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = preprocessor
        self.model: Optional[TextEmbeddingModel] = None
        self.vocab: Optional[Vocabulary] = None
        self.max_length = 40
        self.output_dim = 128
        self.model_loaded = False
        self._load()

    def _load(self):
        if not self.model_path.exists() or not self.vocab_path.exists():
            logger.warning(
                "Модель не найдена (%s). Запустите: python -m model.train",
                self.model_path,
            )
            return

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.vocab = Vocabulary.load(str(self.vocab_path))
        self.max_length = int(checkpoint.get("max_length", 40))
        self.output_dim = int(checkpoint.get("output_dim", 128))

        self.model = TextEmbeddingModel(
            vocab_size=checkpoint["vocab_size"],
            embedding_dim=checkpoint["embedding_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            output_dim=self.output_dim,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.model_loaded = True
        logger.info("Загружена собственная модель: %s", self.model_path.name)

    def get_model_info(self) -> dict:
        return {
            "type": "custom_lstm_attention",
            "architecture": "Embedding + BiLSTM + Attention + FC",
            "trained_on": "gaming slang compatibility dataset",
            "embedding_dim": self.output_dim,
            "model_loaded": self.model_loaded,
            "vocab_size": len(self.vocab) if self.vocab else 0,
        }

    def _encode_indices(self, indices: List[int]) -> List[float]:
        tensor = torch.tensor([indices], dtype=torch.long, device=self.device)
        with torch.no_grad():
            embedding = self.model(tensor).cpu().numpy()[0]
        return embedding.tolist()

    def get_embedding(self, text: str) -> List[float]:
        cleaned = self.preprocessor.clean_text(text)
        if not cleaned:
            cleaned = self.preprocessor.clean_text(text.lower())

        if self.model_loaded and self.vocab:
            indices = self.vocab.text_to_indices(cleaned, self.max_length)
            return self._encode_indices(indices)

        logger.warning("Модель не загружена, используется fallback-вектор")
        return self._fallback_embedding(cleaned)

    def _fallback_embedding(self, text: str) -> List[float]:
        """Простой хэш-вектор для демо, если модель ещё не обучена."""
        vector = np.zeros(self.output_dim, dtype=np.float32)
        for token in text.split():
            idx = hash(token) % self.output_dim
            vector[idx] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector.tolist()


# Совместимость со старым именем
TextEmbedder = CustomTextEmbedder
