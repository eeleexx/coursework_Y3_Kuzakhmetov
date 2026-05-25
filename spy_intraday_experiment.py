#!/usr/bin/env python3
"""
SPY intraday extension using 1-minute OHLCV bars plus tweet sentiment features.

This is intentionally separate from the daily equity/news pipeline. It tests
whether the same multimodal fusion idea works in a higher-frequency event-study
setting where social/news-like signals are already aggregated per minute.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from experiments import regression_metrics, set_seed
from model import MultimodalCausalModel, TemporalEncoder, TextEncoder


PRICE_COLS = ["open", "high", "low", "close", "volume"]
TEXT_COLS = [
    "tweet_count",
    "labeled_count",
    "bullish_count",
    "bearish_count",
    "sentiment_sum",
    "sentiment_mean",
    "labeled_sentiment_mean",
]


class SpyIntradayDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        window: int = 30,
        stride: int = 5,
        target_col: str = "forward_return_5m",
        max_samples: int | None = 50000,
    ):
        df = pd.read_csv(csv_path, parse_dates=["bar_time_utc", "price_time_local"])
        df = df.sort_values("bar_time_utc").reset_index(drop=True)
        df = df.dropna(subset=[target_col]).reset_index(drop=True)
        self.meta = df[["bar_time_utc", "price_time_local", "close", "tweet_count", "sentiment_mean", target_col]].copy()

        price = df[PRICE_COLS].astype(float).copy()
        text = df[TEXT_COLS].astype(float).fillna(0.0).copy()

        for col in PRICE_COLS:
            price[col] = (price[col] - price[col].mean()) / (price[col].std() + 1e-8)
        for col in ["tweet_count", "labeled_count", "bullish_count", "bearish_count", "sentiment_sum"]:
            text[col] = (text[col] - text[col].mean()) / (text[col].std() + 1e-8)

        self.price = price.to_numpy(dtype=np.float32)
        self.text = text.to_numpy(dtype=np.float32)
        self.target = df[target_col].to_numpy(dtype=np.float32)
        self.window = window

        indices = list(range(window, len(df), stride))
        if max_samples is not None and len(indices) > max_samples:
            # Keep chronological coverage while limiting CPU training time.
            pick = np.linspace(0, len(indices) - 1, max_samples, dtype=int)
            indices = [indices[i] for i in pick]
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = self.indices[idx]
        text = torch.from_numpy(self.text[i])
        price_window = torch.from_numpy(self.price[i - self.window + 1 : i + 1])
        target = torch.tensor(self.target[i], dtype=torch.float32)
        return text, price_window, target


class PriceOnlyIntraday(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(input_dim=len(PRICE_COLS), hidden_dim=64, num_layers=1, latent_dim=64)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, text: torch.Tensor, price: torch.Tensor):
        del text
        return self.head(self.temporal_encoder(price)).squeeze(-1)


class TextOnlyIntraday(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder(input_dim=len(TEXT_COLS), latent_dim=64)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, text: torch.Tensor, price: torch.Tensor):
        del price
        return self.head(self.text_encoder(text)).squeeze(-1)


class StaticFusionIntraday(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder(input_dim=len(TEXT_COLS), latent_dim=64)
        self.temporal_encoder = TemporalEncoder(input_dim=len(PRICE_COLS), hidden_dim=64, num_layers=1, latent_dim=64)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, text: torch.Tensor, price: torch.Tensor):
        h_text = self.text_encoder(text)
        h_price = self.temporal_encoder(price)
        return self.head(torch.cat([h_text, h_price], dim=1)).squeeze(-1)


def make_model(name: str) -> nn.Module:
    if name == "price_only":
        return PriceOnlyIntraday()
    if name == "tweet_only":
        return TextOnlyIntraday()
    if name == "static_fusion":
        return StaticFusionIntraday()
    if name == "full_drl":
        return MultimodalCausalModel(
            text_input_dim=len(TEXT_COLS),
            price_input_dim=len(PRICE_COLS),
            latent_dim=64,
            subspace_dim=32,
            lstm_hidden=64,
            lstm_layers=1,
            moco_queue_size=256,
            lambda_diff=0.01,
            lambda_moco=0.01,
        )
    raise ValueError(name)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SPY intraday multimodal experiment")
    parser.add_argument("--csv", default="data/spy_1min_tweet_price_dataset.csv")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=50000)
    parser.add_argument("--target_col", default="forward_return_5m")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def forward_loss(name: str, model: nn.Module, text, price, target):
    if name == "full_drl":
        pred, loss_dict = model(text, price, target)
        return pred.squeeze(-1), loss_dict["total_loss"]
    pred = model(text, price)
    return pred, F.mse_loss(pred, target)


def train(name: str, model: nn.Module, loader: DataLoader, optimizer, device: str) -> float:
    model.train()
    total = 0.0
    n = 0
    for text, price, target in loader:
        text, price, target = text.to(device), price.to(device), target.to(device)
        optimizer.zero_grad()
        _, loss = forward_loss(name, model, text, price, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(name: str, model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    gates: List[np.ndarray] = []
    for text, price, target in loader:
        text, price = text.to(device), price.to(device)
        if name == "full_drl":
            pred, loss_dict = model(text, price)
            pred = pred.squeeze(-1)
            gates.append(loss_dict["gate"].detach().cpu().numpy().reshape(1))
        else:
            pred = model(text, price)
        preds.append(pred.detach().cpu().numpy())
        targets.append(target.numpy())
    pred = np.concatenate(preds)
    target = np.concatenate(targets)
    metrics = regression_metrics(pred, target)
    metrics["prediction_std"] = float(np.std(pred))
    if gates:
        metrics["mean_gate"] = float(np.mean(np.concatenate(gates)))
    return metrics


def save_rows(rows: List[Dict[str, object]], path: Path):
    path.parent.mkdir(exist_ok=True)
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: List[Dict[str, object]], path: Path):
    models = [str(r["model"]) for r in rows]
    metrics = ["mse", "mae", "directional_accuracy", "information_coefficient"]
    titles = ["MSE", "MAE", "Directional Accuracy", "Information Coefficient"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]
    for ax, metric, title in zip(axes, metrics, titles):
        ax.bar(models, [float(r[metric]) for r in rows], color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("SPY 1-minute tweet/price multimodal experiment", fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    dataset = SpyIntradayDataset(
        args.csv,
        window=args.window,
        stride=args.stride,
        target_col=args.target_col,
        max_samples=args.max_samples,
    )
    train_n = int(len(dataset) * 0.8)
    val_n = len(dataset) - train_n
    train_ds = torch.utils.data.Subset(dataset, list(range(train_n)))
    val_ds = torch.utils.data.Subset(dataset, list(range(train_n, train_n + val_n)))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print("=" * 72)
    print("SPY Intraday Tweet/Price Experiment")
    print("=" * 72)
    print(f"Samples: {len(dataset)} ({len(train_ds)} train / {len(val_ds)} val)")
    print(f"Window: {args.window} min | Stride: {args.stride} | Target: {args.target_col}")

    rows: List[Dict[str, object]] = []
    for name in ["price_only", "tweet_only", "static_fusion", "full_drl"]:
        set_seed(args.seed)
        model = make_model(name).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        loss = 0.0
        for _ in range(args.epochs):
            loss = train(name, model, train_loader, optimizer, args.device)
        metrics = evaluate(name, model, val_loader, args.device)
        row: Dict[str, object] = {"model": name, "train_final_loss": loss, **metrics}
        rows.append(row)
        print(
            f"{name:14s} mse={metrics['mse']:.8f} mae={metrics['mae']:.6f} "
            f"dir={metrics['directional_accuracy']:.3f} ic={metrics['information_coefficient']:+.3f}"
        )

    save_rows(rows, Path("results/spy_intraday_metrics.csv"))
    plot(rows, Path("results/spy_intraday_metrics.png"))
    print("Saved -> results/spy_intraday_metrics.csv")
    print("Saved -> results/spy_intraday_metrics.png")


if __name__ == "__main__":
    main()
