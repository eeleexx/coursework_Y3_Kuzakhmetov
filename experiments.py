#!/usr/bin/env python3
"""
Run baseline and ablation experiments for the coursework project.

The script compares four models on the same chronological split:
  - price_only: LSTM over OHLCV windows
  - text_only: MLP over cached FinBERT embeddings
  - static_fusion: simple concatenation of text and price encoders
  - full_drl: disentangled model with MoCo and gated fusion

Outputs:
  - results/baseline_metrics.csv
  - results/baseline_metrics.png
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import build_dataloaders
from model import MultimodalCausalModel, TemporalEncoder, TextEncoder


class PriceOnlyModel(nn.Module):
    """Unimodal baseline using only OHLCV history."""

    def __init__(self, price_input_dim: int = 5, latent_dim: int = 128):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(price_input_dim, 128, 2, latent_dim)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, text_emb: torch.Tensor, price_window: torch.Tensor):
        del text_emb
        return self.predictor(self.temporal_encoder(price_window)).squeeze(-1)


class TextOnlyModel(nn.Module):
    """Unimodal baseline using only FinBERT news embeddings."""

    def __init__(self, text_input_dim: int = 768, latent_dim: int = 128):
        super().__init__()
        self.text_encoder = TextEncoder(text_input_dim, latent_dim)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, text_emb: torch.Tensor, price_window: torch.Tensor):
        del price_window
        return self.predictor(self.text_encoder(text_emb)).squeeze(-1)


class StaticFusionModel(nn.Module):
    """Ablation baseline: concatenate modalities without disentanglement/gating."""

    def __init__(
        self,
        text_input_dim: int = 768,
        price_input_dim: int = 5,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.text_encoder = TextEncoder(text_input_dim, latent_dim)
        self.temporal_encoder = TemporalEncoder(price_input_dim, 128, 2, latent_dim)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, text_emb: torch.Tensor, price_window: torch.Tensor):
        h_text = self.text_encoder(text_emb)
        h_price = self.temporal_encoder(price_window)
        return self.predictor(torch.cat([h_text, h_price], dim=1)).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run coursework model baselines")
    parser.add_argument("--tickers", nargs="+", default=["AMD"], help="Tickers to evaluate")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models", nargs="+", default=["price_only", "text_only", "static_fusion", "full_drl"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(name: str, text_dim: int) -> nn.Module:
    if name == "price_only":
        return PriceOnlyModel()
    if name == "text_only":
        return TextOnlyModel(text_dim)
    if name == "static_fusion":
        return StaticFusionModel(text_dim)
    if name == "full_drl":
        return MultimodalCausalModel(text_input_dim=text_dim)
    raise ValueError(f"Unknown model: {name}")


def forward_loss(
    model_name: str,
    model: nn.Module,
    text_emb: torch.Tensor,
    price_window: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if model_name == "full_drl":
        prediction, loss_dict = model(text_emb, price_window, target)
        return prediction.squeeze(-1), loss_dict["total_loss"]
    prediction = model(text_emb, price_window)
    return prediction, F.mse_loss(prediction, target)


def train_model(
    model_name: str,
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: str,
    epochs: int,
    lr: float,
) -> List[float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history: List[float] = []

    for _ in range(epochs):
        total = 0.0
        n = 0
        for text_emb, price_window, target in train_loader:
            text_emb = text_emb.to(device)
            price_window = price_window.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            _, loss = forward_loss(model_name, model, text_emb, price_window, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total += loss.item()
            n += 1
        history.append(total / max(n, 1))
    return history


@torch.no_grad()
def collect_predictions(
    model_name: str,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    for text_emb, price_window, target in loader:
        text_emb = text_emb.to(device)
        price_window = price_window.to(device)
        target = target.to(device)
        if model_name == "full_drl":
            prediction, _ = model(text_emb, price_window)
            prediction = prediction.squeeze(-1)
        else:
            prediction = model(text_emb, price_window)
        preds.append(prediction.detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    error = pred - target
    strategy_returns = np.sign(pred) * target
    strategy_std = float(np.std(strategy_returns))
    return {
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
        "directional_accuracy": float(np.mean(np.sign(pred) == np.sign(target))),
        "information_coefficient": safe_corr(pred, target),
        "strategy_mean_return": float(np.mean(strategy_returns)),
        "strategy_sharpe": float(np.sqrt(252) * np.mean(strategy_returns) / (strategy_std + 1e-12)),
    }


def save_metrics(rows: List[Dict[str, object]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "ticker",
        "model",
        "train_final_loss",
        "mse",
        "mae",
        "directional_accuracy",
        "information_coefficient",
        "strategy_mean_return",
        "strategy_sharpe",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_summary(rows: List[Dict[str, object]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    models = []
    for row in rows:
        model = str(row["model"])
        if model not in models:
            models.append(model)

    summary_rows = []
    for model in models:
        model_rows = [r for r in rows if r["model"] == model]
        wins = 0
        for ticker in sorted({str(r["ticker"]) for r in rows}):
            ticker_rows = [r for r in rows if r["ticker"] == ticker]
            best_mse = min(float(r["mse"]) for r in ticker_rows)
            own = [r for r in ticker_rows if r["model"] == model][0]
            if abs(float(own["mse"]) - best_mse) < 1e-12:
                wins += 1
        summary_rows.append(
            {
                "model": model,
                "mean_mse": float(np.mean([float(r["mse"]) for r in model_rows])),
                "mean_mae": float(np.mean([float(r["mae"]) for r in model_rows])),
                "mean_directional_accuracy": float(
                    np.mean([float(r["directional_accuracy"]) for r in model_rows])
                ),
                "mean_information_coefficient": float(
                    np.mean([float(r["information_coefficient"]) for r in model_rows])
                ),
                "mse_wins": wins,
            }
        )

    fields = [
        "model",
        "mean_mse",
        "mean_mae",
        "mean_directional_accuracy",
        "mean_information_coefficient",
        "mse_wins",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)


def plot_metrics(rows: List[Dict[str, object]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    models = []
    tickers = []
    for row in rows:
        model = str(row["model"])
        ticker = str(row["ticker"])
        if model not in models:
            models.append(model)
        if ticker not in tickers:
            tickers.append(ticker)
    metrics = ["mse", "directional_accuracy", "information_coefficient", "strategy_sharpe"]
    titles = ["MSE (lower is better)", "Directional Accuracy", "Information Coefficient", "Strategy Sharpe"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]
    for ax, metric, title in zip(axes.flat, metrics, titles):
        vals = []
        labels = []
        for model in models:
            model_rows = [r for r in rows if r["model"] == model]
            vals.append(float(np.mean([float(r[metric]) for r in model_rows])))
            labels.append(model)
        ax.bar(labels, vals, color=colors[: len(labels)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(
        f"Baseline comparison (mean over {', '.join(tickers)})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    rows: List[Dict[str, object]] = []

    print("=" * 72)
    print("Baseline and Ablation Experiments")
    print("=" * 72)
    print(f"Tickers: {', '.join(args.tickers)}")
    print(f"Models:  {', '.join(args.models)}")
    print(f"Epochs:  {args.epochs}")
    print(f"Device:  {device}")

    for ticker in args.tickers:
        print(f"\n--- {ticker} ---")
        train_loader, val_loader, text_dim = build_dataloaders(
            ticker=ticker,
            window=args.window,
            batch_size=args.batch_size,
            device=device,
        )

        for model_name in args.models:
            set_seed(args.seed)
            print(f"[{ticker}] training {model_name}...")
            model = make_model(model_name, text_dim).to(device)
            history = train_model(model_name, model, train_loader, device, args.epochs, args.lr)
            pred, target = collect_predictions(model_name, model, val_loader, device)
            metrics = regression_metrics(pred, target)
            row: Dict[str, object] = {
                "ticker": ticker,
                "model": model_name,
                "train_final_loss": history[-1],
                **metrics,
            }
            rows.append(row)
            print(
                f"  mse={metrics['mse']:.6f} "
                f"mae={metrics['mae']:.6f} "
                f"dir={metrics['directional_accuracy']:.3f} "
                f"ic={metrics['information_coefficient']:.3f} "
                f"sharpe={metrics['strategy_sharpe']:.2f}"
            )

    csv_path = Path("results/baseline_metrics.csv")
    summary_path = Path("results/baseline_summary.csv")
    png_path = Path("results/baseline_metrics.png")
    save_metrics(rows, csv_path)
    save_summary(rows, summary_path)
    plot_metrics(rows, png_path)
    print(f"\nSaved metrics -> {csv_path}")
    print(f"Saved summary -> {summary_path}")
    print(f"Saved plot    -> {png_path}")


if __name__ == "__main__":
    main()
