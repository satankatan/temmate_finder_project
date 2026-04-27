import torch
import numpy as np
from typing import List
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class TextEmbedder:
    def __init__(self, model_name: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'):
        try:
            logger.info(f"Loading model: {model_name}")
            self.model = SentenceTransformer(model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def enhance_embedding_with_game_context(self, text: str, base_embedding: List[float]) -> List[float]:
        """Усиливает эмбеддинг информацией об игре"""
        enhanced = base_embedding.copy()
        

        game_weights = {
            'dota': [0.2] * 50 + [0.0] * (len(enhanced)-50),         
            'дота': [0.2] * 50 + [0.0] * (len(enhanced)-50),
            'дот': [0.2] * 50 + [0.0] * (len(enhanced)-50),
            'cs:go': [0.0] * 50 + [0.2] * 50 + [0.0] * (len(enhanced)-100),  
            'csgo': [0.0] * 50 + [0.2] * 50 + [0.0] * (len(enhanced)-100),
            'кс': [0.0] * 50 + [0.2] * 50 + [0.0] * (len(enhanced)-100),
            'counter': [0.0] * 50 + [0.2] * 50 + [0.0] * (len(enhanced)-100),
            'valorant': [0.0] * 100 + [0.2] * 50 + [0.0] * (len(enhanced)-150),  
            'вало': [0.0] * 100 + [0.2] * 50 + [0.0] * (len(enhanced)-150),
            'валорент': [0.0] * 100 + [0.2] * 50 + [0.0] * (len(enhanced)-150),
            'lol': [0.0] * 150 + [0.2] * 50 + [0.0] * (len(enhanced)-200),
            'league': [0.0] * 150 + [0.2] * 50 + [0.0] * (len(enhanced)-200),
            'wot': [0.0] * 200 + [0.2] * 50 + [0.0] * (len(enhanced)-250),
            'tanks': [0.0] * 200 + [0.2] * 50 + [0.0] * (len(enhanced)-250),
        }
        
        text_lower = text.lower()
        for game, weight in game_weights.items():
            if game in text_lower:
                logger.info(f"Enhancing embedding with game context: {game}")
                enhanced = [e + w for e, w in zip(enhanced, weight)]
        
        # Нормализуем обратно чтобы не сломать косинусную схожесть
        enhanced_np = np.array(enhanced)
        if np.linalg.norm(enhanced_np) > 0:
            enhanced_np = enhanced_np / np.linalg.norm(enhanced_np)
        
        return enhanced_np.tolist()
    
    def get_embedding(self, text: str) -> List[float]:
        """Получение семантического эмбеддинга для текста с игровым контекстом"""
        try:
            if self.model:
                # Получаем базовый эмбеддинг
                base_embedding = self.model.encode(text, convert_to_tensor=False)
                
                # Усиливаем игровым контекстом
                enhanced_embedding = self.enhance_embedding_with_game_context(text, base_embedding.tolist())
                
                logger.info(f"Base embedding norm: {np.linalg.norm(base_embedding):.3f}, "
                           f"Enhanced: {np.linalg.norm(enhanced_embedding):.3f}")
                
                return enhanced_embedding
            else:
                logger.warning("Using fallback embedding")
                return np.random.rand(768).tolist()
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return np.random.rand(768).tolist()