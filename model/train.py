import argparse
import os
import time
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from model.dataset import Vocabulary, load_dataset_from_csv
from model.network import TextEmbeddingModel, TripletLoss
from preprocessing.text_processor import preprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "train.csv"
FULL_DATASET = PROJECT_ROOT / "data" / "gaming_compatibility_dataset.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"


class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, device="cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = StepLR(self.optimizer, step_size=8, gamma=0.8)
        self.criterion = TripletLoss(margin=0.25)
        self.best_val_loss = float("inf")

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            anchor = batch["anchor"].to(self.device)
            positive = batch["positive"].to(self.device)
            negative = batch["negative"].to(self.device)

            self.optimizer.zero_grad()
            anchor_emb = self.model(anchor)
            positive_emb = self.model(positive)
            negative_emb = self.model(negative)
            loss = self.criterion(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        self.scheduler.step()
        return total_loss / max(len(self.train_loader), 1)

    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                anchor = batch["anchor"].to(self.device)
                positive = batch["positive"].to(self.device)
                negative = batch["negative"].to(self.device)

                anchor_emb = self.model(anchor)
                positive_emb = self.model(positive)
                negative_emb = self.model(negative)
                total_loss += self.criterion(anchor_emb, positive_emb, negative_emb).item()

        return total_loss / max(len(self.val_loader), 1)

    def train(self, num_epochs: int, save_dir: Path, vocab: Vocabulary, max_length: int):
        save_dir.mkdir(parents=True, exist_ok=True)
        vocab.save(save_dir / "vocab.json")

        print(f"Обучение на {self.device}, параметров: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | {elapsed:.1f}s"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(save_dir / "best_model.pth", vocab, max_length)
                print(f"  -> сохранена лучшая модель (val_loss={val_loss:.4f})")

        self.save_checkpoint(save_dir / "last_model.pth", vocab, max_length)
        print("Обучение завершено.")

    def save_checkpoint(self, path: Path, vocab: Vocabulary, max_length: int):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "vocab_size": self.model.vocab_size,
                "embedding_dim": self.model.embedding_dim,
                "hidden_dim": self.model.hidden_dim,
                "output_dim": self.model.output_dim,
                "max_length": max_length,
                "best_val_loss": self.best_val_loss,
            },
            path,
        )


def train_model(
    dataset_path: Path = DEFAULT_DATASET,
    model_dir: Path = DEFAULT_MODEL_DIR,
    epochs: int = 30,
    batch_size: int = 8,
    max_length: int = 40,
):
    if not dataset_path.exists():
        hint = ""
        if dataset_path == DEFAULT_DATASET and FULL_DATASET.exists():
            hint = " Сначала выполните: python scripts/split_dataset.py"
        raise FileNotFoundError(f"Датасет не найден: {dataset_path}.{hint}")

    train_loader, val_loader, vocab = load_dataset_from_csv(
        str(dataset_path),
        preprocessor,
        batch_size=batch_size,
        max_length=max_length,
    )

    model = TextEmbeddingModel(
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_dim=192,
        num_layers=2,
        dropout=0.25,
        output_dim=128,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = ModelTrainer(model, train_loader, val_loader, device)
    trainer.train(epochs, model_dir, vocab, max_length)
    return model_dir / "best_model.pth"


def main():
    parser = argparse.ArgumentParser(description="Обучение собственной модели матчмейкинга")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=40)
    parser.add_argument("--output", type=str, default=str(DEFAULT_MODEL_DIR))
    args = parser.parse_args()

    train_model(
        dataset_path=Path(args.dataset),
        model_dir=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
