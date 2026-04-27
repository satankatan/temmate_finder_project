import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import time
import os
from tqdm import tqdm
import random  # ✅ ДОБАВЛЕНО
from model.network import TextEmbeddingModel, TripletLoss  # ✅ ДОБАВЛЕНО
from model.dataset import create_data_loaders  # ✅ ДОБАВЛЕНО

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Оптимизатор и функция потерь
        self.optimizer = Adam(
            model.parameters(), 
            lr=0.001,
            weight_decay=1e-5
        )
        self.criterion = TripletLoss(margin=0.2)  # ✅ Теперь TripletLoss импортирован
        
        # Для логирования
        self.writer = SummaryWriter('logs/teammate_matcher')
        self.best_val_loss = float('inf')
        
    def train_epoch(self, epoch):
        """Одна эпоха обучения"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Перемещаем данные на устройство
            anchor = batch['anchor'].to(self.device)
            positive = batch['positive'].to(self.device)
            negative = batch['negative'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            anchor_emb = self.model(anchor)
            positive_emb = self.model(positive)
            negative_emb = self.model(negative)
            
            # Вычисляем loss
            loss = self.criterion(anchor_emb, positive_emb, negative_emb)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping для стабильности
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Обновляем progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
            
            # Логируем в tensorboard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('train/batch_loss', loss.item(), global_step)
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self, epoch):
        """Валидация модели"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                anchor = batch['anchor'].to(self.device)
                positive = batch['positive'].to(self.device)
                negative = batch['negative'].to(self.device)
                
                anchor_emb = self.model(anchor)
                positive_emb = self.model(positive)
                negative_emb = self.model(negative)
                
                loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar('val/epoch_loss', avg_loss, epoch)
        
        return avg_loss
    
    def train(self, num_epochs, save_dir='models'):
        """Полный цикл обучения"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting training for {num_epochs} epochs on {self.device}")
        print(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Обучение
            train_loss = self.train_epoch(epoch)
            
            # Валидация
            val_loss = self.validate(epoch)
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch}/{num_epochs} | '
                  f'Train Loss: {train_loss:.4f} | '
                  f'Val Loss: {val_loss:.4f} | '
                  f'Time: {epoch_time:.2f}s')
            
            # Сохраняем лучшую модель
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(os.path.join(save_dir, 'best_model.pth'))
                print(f'New best model saved with val_loss: {val_loss:.4f}')
            
            # Сохраняем чекпоинт каждые 5 эпох
            if epoch % 5 == 0:
                self.save_model(os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        self.writer.close()
        print("Training completed!")
    
    def save_model(self, path):
        """Сохраняет модель и метаданные"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'vocab_size': self.model.vocab_size,
            'embedding_dim': self.model.embedding_dim,
            'hidden_dim': self.model.hidden_dim,
            'best_val_loss': self.best_val_loss
        }, path)
    
    @classmethod
    def load_model(cls, path, vocab, device='cuda'):
        """Загружает модель"""
        checkpoint = torch.load(path, map_location=device)
        
        model = TextEmbeddingModel(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(device)

def generate_synthetic_data(num_samples=1000):
    """Генерирует синтетические данные для демонстрации"""
    gaming_styles = [
        "агрессивный атакующий давление враг контроль карта",
        "спокойный защитный поддержка команда помощь",
        "стратегический тактический анализ план мышление",
        "быстрый реакция скорость мобильность динамичный",
        "техничный точный скилл умение мастерство",
        "командный взаимодействие коммуникация совместный",
        "одиночный независимый самостоятельный индивидуальный"
    ]
    
    roles = ["керри", "саппорт", "мидер", "оффлейнер", "джанглер"]
    
    texts = []
    for _ in range(num_samples):
        style = random.choice(gaming_styles)  # ✅ Теперь random импортирован
        role = random.choice(roles)  # ✅ Теперь random импортирован
        texts.append(f"{style} {role}")
    
    return texts

def main():
    """Основная функция для запуска обучения"""
    # Генерируем синтетические данные
    print("Generating synthetic data...")
    texts = generate_synthetic_data(2000)
    
    # Создаем data loaders
    print("Creating data loaders...")
    train_loader, val_loader, vocab = create_data_loaders(  # ✅ Теперь create_data_loaders импортирован
        texts, 
        batch_size=32,
        train_ratio=0.8,
        max_length=30
    )
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Создаем модель
    model = TextEmbeddingModel(  # ✅ Теперь TextEmbeddingModel импортирован
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_dim=256,
        output_dim=128
    )
    
    # Инициализируем тренер
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = ModelTrainer(model, train_loader, val_loader, device)
    
    # Запускаем обучение
    trainer.train(num_epochs=20)

if __name__ == "__main__":
    main()