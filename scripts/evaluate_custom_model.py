# -*- coding: utf-8 -*-
"""
Оценка собственной модели TextEmbeddingModel (LSTM + Attention).

Считает метрики на размеченном датасете и сохраняет графики для отчёта:
  - precision_vs_threshold.png   — зависимость Precision от порога (Рис. 6)
  - predicted_vs_actual.png      — scatter предсказание vs экспертная оценка
  - metrics_summary.png          — Precision / NDCG / корреляция одной моделью
  - all_methods_comparison.png   — сравнение с baseline (если есть baseline_metrics.json)

Запуск из корня проекта (по умолчанию оценка на data/test.csv — пары, не виденные при обучении):
  pip install matplotlib
  python scripts/split_dataset.py
  python -m model.train --dataset data/train.csv
  python scripts/evaluate_custom_model.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score, precision_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.inference import CustomTextEmbedder

DEFAULT_DATA = PROJECT_ROOT / "data" / "test.csv"
FULL_DATASET = PROJECT_ROOT / "data" / "gaming_compatibility_dataset.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "custom_model"
BASELINE_METRICS_FILE = PROJECT_ROOT / "results" / "baseline_metrics.json"

# Значения из раздела «Оценка эффективности» (30 пар, baseline без дообучения)
DEFAULT_BASELINES = {
    "TF-IDF": {"precision": 0.6333, "ndcg": 0.8912, "correlation": -0.1245},
    "Sentence-BERT": {"precision": 0.9667, "ndcg": 0.9731, "correlation": 0.9124},
    "Multilingual-BERT": {"precision": 0.9000, "ndcg": 0.9456, "correlation": 0.8567},
}

MODEL_LABEL = "Custom LSTM+Attention"
ACTUAL_THRESHOLD = 0.7  # порог «совместимы» в разметке


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    return float(np.dot(v1, v2) / norm)


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    required = {"description1", "description2", "compatibility_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В CSV нет столбцов: {missing}")
    df = df.dropna(subset=list(required)).copy()
    df["compatibility_score"] = df["compatibility_score"].astype(float).clip(0, 1)
    return df.reset_index(drop=True)


def predict_pairs(embedder: CustomTextEmbedder, df: pd.DataFrame) -> list[float]:
    predictions: list[float] = []
    for _, row in df.iterrows():
        emb1 = np.array(embedder.get_embedding(str(row["description1"])), dtype=np.float32)
        emb2 = np.array(embedder.get_embedding(str(row["description2"])), dtype=np.float32)
        predictions.append(cosine_similarity(emb1, emb2))
    return predictions


def compute_metrics(
    predictions: list[float],
    actual: list[float],
    pred_threshold: float = 0.5,
    actual_threshold: float = ACTUAL_THRESHOLD,
) -> dict:
    pred_arr = np.array(predictions, dtype=np.float64)
    actual_arr = np.array(actual, dtype=np.float64)

    binary_actual = (actual_arr >= actual_threshold).astype(int)
    binary_pred = (pred_arr >= pred_threshold).astype(int)

    precision = float(precision_score(binary_actual, binary_pred, zero_division=0))
    try:
        ndcg = float(ndcg_score([actual_arr], [pred_arr]))
    except ValueError:
        ndcg = 0.0

    if len(pred_arr) > 1 and np.std(pred_arr) > 0 and np.std(actual_arr) > 0:
        correlation = float(np.corrcoef(pred_arr, actual_arr)[0, 1])
    else:
        correlation = 0.0

    mae = float(np.mean(np.abs(pred_arr - actual_arr)))
    rmse = float(np.sqrt(np.mean((pred_arr - actual_arr) ** 2)))

    return {
        "precision": precision,
        "ndcg": ndcg,
        "correlation": correlation,
        "mae": mae,
        "rmse": rmse,
        "pred_threshold": pred_threshold,
        "actual_threshold": actual_threshold,
        "pairs_count": len(predictions),
    }


def find_best_threshold(
    predictions: list[float],
    actual: list[float],
    actual_threshold: float = ACTUAL_THRESHOLD,
) -> tuple[float, float]:
    binary_actual = [1 if score >= actual_threshold else 0 for score in actual]
    best_threshold, best_precision = 0.5, 0.0
    for threshold in np.arange(0.05, 1.0, 0.05):
        binary_pred = [1 if pred >= threshold else 0 for pred in predictions]
        precision = precision_score(binary_actual, binary_pred, zero_division=0)
        if precision >= best_precision:
            best_precision = float(precision)
            best_threshold = float(threshold)
    return best_threshold, best_precision


def plot_precision_vs_threshold(
    predictions: list[float],
    actual: list[float],
    output_path: Path,
    actual_threshold: float = ACTUAL_THRESHOLD,
) -> tuple[float, float]:
    binary_actual = [1 if score >= actual_threshold else 0 for score in actual]
    thresholds = np.arange(0.05, 1.0, 0.05)
    precision_scores = []

    for threshold in thresholds:
        binary_pred = [1 if pred >= threshold else 0 for pred in predictions]
        precision_scores.append(precision_score(binary_actual, binary_pred, zero_division=0))

    precision_scores = np.array(precision_scores, dtype=float)
    optimal_idx = int(np.argmax(precision_scores))
    optimal_threshold = float(thresholds[optimal_idx])
    optimal_precision = float(precision_scores[optimal_idx])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precision_scores, "b-o", linewidth=2, markersize=5, label=MODEL_LABEL)
    ax.scatter(
        [optimal_threshold],
        [optimal_precision],
        color="red",
        s=100,
        zorder=5,
        label=f"Оптимум: {optimal_threshold:.2f}, P={optimal_precision:.3f}",
    )
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.6, label="Порог 0.5 (как у SBERT)")
    ax.set_title(f"Зависимость Precision от порога — {MODEL_LABEL}", fontweight="bold")
    ax.set_xlabel("Порог схожести (prediction threshold)")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return optimal_threshold, optimal_precision


def plot_predicted_vs_actual(
    predictions: list[float],
    actual: list[float],
    output_path: Path,
    correlation: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(actual, predictions, alpha=0.75, s=60, edgecolors="white", linewidth=0.5)
    ax.plot([0, 1], [0, 1], "r--", alpha=0.5, label="Идеальное совпадение")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Экспертная оценка совместимости")
    ax.set_ylabel("Предсказание модели (косинусная схожесть)")
    ax.set_title(f"{MODEL_LABEL}: предсказание vs разметка\nr = {correlation:.3f}", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_summary(metrics: dict, output_path: Path) -> None:
    labels = ["Precision", "NDCG", f"Корреляция\n(r={metrics['correlation']:.3f})"]
    values = [metrics["precision"], metrics["ndcg"], abs(metrics["correlation"])]
    colors = ["#6c5ce7", "#00b894", "#0984e3"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Метрики модели — {MODEL_LABEL}", fontweight="bold")
    ax.set_ylabel("Значение")

    for bar, value, label in zip(bars, values, labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    note = (
        f"Precision @ pred>={metrics['pred_threshold']:.1f}, "
        f"actual>={metrics['actual_threshold']:.1f} | "
        f"MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f} | "
        f"n={metrics['pairs_count']}"
    )
    ax.text(0.5, -0.15, note, transform=ax.transAxes, ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_baselines() -> dict[str, dict]:
    if BASELINE_METRICS_FILE.exists():
        with open(BASELINE_METRICS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_BASELINES


def plot_all_methods_comparison(custom_metrics: dict, output_path: Path) -> None:
    baselines = load_baselines()
    methods = list(baselines.keys()) + [MODEL_LABEL]
    precision_vals = [baselines[m]["precision"] for m in baselines] + [custom_metrics["precision"]]
    ndcg_vals = [baselines[m]["ndcg"] for m in baselines] + [custom_metrics["ndcg"]]

    colors = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#6c5ce7"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bars1 = ax1.bar(methods, precision_vals, color=colors)
    ax1.set_title("Сравнение Precision", fontweight="bold")
    ax1.set_ylabel("Precision")
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis="x", rotation=15)
    for bar, value in zip(bars1, precision_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.3f}", ha="center", fontweight="bold")

    bars2 = ax2.bar(methods, ndcg_vals, color=colors)
    ax2.set_title("Сравнение NDCG", fontweight="bold")
    ax2.set_ylabel("NDCG")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="x", rotation=15)
    for bar, value in zip(bars2, ndcg_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.3f}", ha="center", fontweight="bold")

    fig.suptitle("Baseline-методы vs собственная модель", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(metrics: dict, optimal: tuple[float, float], output_path: Path, eval_file: str) -> None:
    opt_threshold, opt_precision = optimal
    split_note = metrics.get("eval_split", "hold-out test (модель не видела эти пары при обучении)")
    lines = [
        "=" * 60,
        f"ОЦЕНКА МОДЕЛИ: {MODEL_LABEL}",
        "=" * 60,
        f"Файл оценки: {eval_file}",
        f"Тип выборки: {split_note}",
        f"Пар в выборке: {metrics['pairs_count']}",
        "",
        f"Precision (порог pred>={metrics['pred_threshold']}, actual>={metrics['actual_threshold']}): {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)",
        f"NDCG:  {metrics['ndcg']:.4f}",
        f"Корреляция Пирсона: {metrics['correlation']:.4f}",
        f"MAE:   {metrics['mae']:.4f}",
        f"RMSE:  {metrics['rmse']:.4f}",
        "",
        f"Оптимальный порог pred: {opt_threshold:.2f} -> Precision = {opt_precision:.4f} ({opt_precision*100:.2f}%)",
        "",
        "Формулировка для отчёта:",
        f"Собственная модель TextEmbeddingModel оценена на отложенной тестовой выборке "
        f"({metrics['pairs_count']} пар, файл {Path(eval_file).name}). "
        f"Precision {metrics['precision']*100:.2f}% при пороге {metrics['pred_threshold']}, "
        f"NDCG {metrics['ndcg']:.4f}, корреляция с экспертными оценками {metrics['correlation']:.4f}. "
        f"Обучение выполнялось только на train.csv без включения тестовых пар.",
        "=" * 60,
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Оценка собственной модели TextEmbeddingModel")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="CSV с парами описаний")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Папка для графиков")
    parser.add_argument("--pred-threshold", type=float, default=0.5, help="Порог pred для Precision (как у SBERT)")
    parser.add_argument("--actual-threshold", type=float, default=ACTUAL_THRESHOLD, help="Порог actual для «совместимы»")
    parser.add_argument("--no-comparison", action="store_true", help="Не строить график с baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.data.exists():
        hint = " Сначала: python scripts/split_dataset.py"
        if args.data == DEFAULT_DATA and FULL_DATASET.exists():
            print(f"Файл {args.data} не найден.{hint}")
        else:
            print(f"Файл не найден: {args.data}")
        sys.exit(1)

    print(f"Загрузка данных: {args.data}")
    if args.data.name == "test.csv":
        print("Оценка на hold-out test: модель не обучалась на этих парах.")
    df = load_dataset(args.data)
    print(f"Пар для оценки: {len(df)}")

    print("Загрузка модели...")
    embedder = CustomTextEmbedder()
    if not embedder.model_loaded:
        print("ОШИБКА: models/best_model.pth не найден. Сначала: python -m model.train")
        sys.exit(1)

    info = embedder.get_model_info()
    print(f"Модель: {info['architecture']}, vocab={info['vocab_size']}, dim={info['embedding_dim']}")

    print("Расчёт предсказаний...")
    predictions = predict_pairs(embedder, df)
    actual = df["compatibility_score"].tolist()

    metrics = compute_metrics(
        predictions,
        actual,
        pred_threshold=args.pred_threshold,
        actual_threshold=args.actual_threshold,
    )
    optimal = find_best_threshold(predictions, actual, args.actual_threshold)
    metrics["optimal_pred_threshold"] = optimal[0]
    metrics["optimal_precision"] = optimal[1]
    metrics["model"] = MODEL_LABEL
    metrics["model_info"] = info
    metrics["eval_file"] = str(args.data)
    metrics["eval_split"] = (
        "hold-out test (модель не видела эти пары при обучении)"
        if args.data.name == "test.csv"
        else "указанный датасет"
    )

    results_df = df.copy()
    results_df["prediction"] = predictions
    results_df["error"] = results_df["prediction"] - results_df["compatibility_score"]
    results_df.to_csv(args.output_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    plt.style.use("seaborn-v0_8-whitegrid")
    plot_precision_vs_threshold(
        predictions,
        actual,
        args.output_dir / "precision_vs_threshold.png",
        args.actual_threshold,
    )
    plot_predicted_vs_actual(
        predictions,
        actual,
        args.output_dir / "predicted_vs_actual.png",
        metrics["correlation"],
    )
    plot_metrics_summary(metrics, args.output_dir / "metrics_summary.png")

    if not args.no_comparison:
        plot_all_methods_comparison(metrics, args.output_dir / "all_methods_comparison.png")

    write_report(metrics, optimal, args.output_dir / "metrics_report.txt", str(args.data))

    report = (args.output_dir / "metrics_report.txt").read_text(encoding="utf-8")
    try:
        print("\n" + report)
    except UnicodeEncodeError:
        print("\n" + report.encode("ascii", errors="replace").decode("ascii"))
    print(f"\nГрафики сохранены в: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
