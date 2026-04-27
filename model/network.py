import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionLayer(nn.Module):
    """Слой внимания для выделения важных слов"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, lstm_out):
        # lstm_out: [batch_size, seq_len, hidden_dim]
        attention_weights = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        weighted = torch.sum(lstm_out * attention_weights, dim=1)  # [batch_size, hidden_dim]
        return weighted

class TextEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, 
                 num_layers=2, dropout=0.3, output_dim=128):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # Слой эмбеддингов
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # индекс для padding
        )
        
        # LSTM слои
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Слой внимания
        self.attention = AttentionLayer(hidden_dim * 2)  # *2 из-за bidirectional
        
        # Полносвязные слои
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, output_dim)
        )
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов для лучшей сходимости"""
        # Эмбеддинги
        nn.init.xavier_uniform_(self.embedding.weight)
        
        # LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                # Устанавливаем forget gate bias в 1 для лучшего обучения
                if len(param) > 1:
                    param.data[self.hidden_dim:2*self.hidden_dim] = 1.0
        
        # FC слои
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, input_ids, attention_mask=None):
        # input_ids: [batch_size, seq_length]
        
        # Получаем эмбеддинги слов
        embedded = self.embedding(input_ids)  # [batch_size, seq_length, embedding_dim]
        
        # Применяем LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)  # [batch_size, seq_length, hidden_dim*2]
        
        # Применяем механизм внимания
        weighted = self.attention(lstm_out)  # [batch_size, hidden_dim*2]
        
        # Полносвязные слои
        output = self.fc_layers(weighted)  # [batch_size, output_dim]
        
        # L2 нормализация для косинусного сходства
        output = F.normalize(output, p=2, dim=1)
        
        return output

class TripletLoss(nn.Module):
    """Triplet Loss для обучения модели сравнения"""
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        # anchor, positive, negative: [batch_size, embedding_dim]
        
        pos_distance = F.pairwise_distance(anchor, positive, p=2)
        neg_distance = F.pairwise_distance(anchor, negative, p=2)
        
        losses = F.relu(pos_distance - neg_distance + self.margin)
        return losses.mean()

class ContrastiveLoss(nn.Module):
    """Contrastive Loss для парных сравнений"""
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        # label = 1 для похожих, 0 для непохожих
        euclidean_distance = F.pairwise_distance(output1, output2, p=2)
        
        loss_similar = label * torch.pow(euclidean_distance, 2)
        loss_dissimilar = (1 - label) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2
        )
        
        loss = torch.mean(loss_similar + loss_dissimilar)
        return loss