# -*- coding: utf-8 -*-
"""
Разделение полного датасета на train / test (модель не видит test при обучении).

Запуск:
  python scripts/split_dataset.py
  python scripts/split_dataset.py --test-ratio 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "gaming_compatibility_dataset.csv"
TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "test.csv"
MANIFEST_PATH = PROJECT_ROOT / "data" / "split_manifest.json"


def score_bucket(score: float) -> str:
    if score >= 0.7:
        return "compatible"
    if score <= 0.45:
        return "incompatible"
    return "neutral"


def stratified_split(df: pd.DataFrame, test_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy().reset_index(drop=True)
    df["_bucket"] = df["compatibility_score"].astype(float).map(score_bucket)

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for bucket, group in df.groupby("_bucket"):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_test = max(1, round(len(group) * test_ratio))
        if len(group) <= 2:
            n_test = 1
        test_parts.append(group.iloc[:n_test])
        train_parts.append(group.iloc[n_test:])

    train_df = pd.concat(train_parts, ignore_index=True).drop(columns=["_bucket"])
    test_df = pd.concat(test_parts, ignore_index=True).drop(columns=["_bucket"])

    return train_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/test split для честной оценки модели")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--train-out", type=Path, default=TRAIN_PATH)
    parser.add_argument("--test-out", type=Path, default=TEST_PATH)
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Доля пар в test (0.15–0.25)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Исходный датасет не найден: {args.source}")

    df = pd.read_csv(args.source, encoding="utf-8").dropna(
        subset=["description1", "description2", "compatibility_score"]
    )
    df["compatibility_score"] = df["compatibility_score"].astype(float).clip(0, 1)

    train_df, test_df = stratified_split(df, args.test_ratio, args.seed)

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(args.train_out, index=False, encoding="utf-8-sig")
    test_df.to_csv(args.test_out, index=False, encoding="utf-8-sig")

    manifest = {
        "source": str(args.source),
        "train_file": str(args.train_out),
        "test_file": str(args.test_out),
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "total_pairs": len(df),
        "train_pairs": len(train_df),
        "test_pairs": len(test_df),
        "train_buckets": train_df["compatibility_score"].apply(
            lambda s: score_bucket(float(s))
        ).value_counts().to_dict(),
        "test_buckets": test_df["compatibility_score"].apply(
            lambda s: score_bucket(float(s))
        ).value_counts().to_dict(),
        "note": "Модель обучается только на train.csv; метрики считаются на test.csv",
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Всего пар: {len(df)}")
    print(f"Train: {len(train_df)} -> {args.train_out}")
    print(f"Test:  {len(test_df)} -> {args.test_out}")
    print(f"Манифест: {MANIFEST_PATH}")
    print("\nДальше:")
    print("  python -m model.train --dataset data/train.csv --epochs 30")
    print("  python scripts/evaluate_custom_model.py --data data/test.csv")


if __name__ == "__main__":
    main()
