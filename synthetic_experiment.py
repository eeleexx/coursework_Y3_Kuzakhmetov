#!/usr/bin/env python3
"""
Controlled simulation for the Headline-vs-Content hypothesis.

Real market data cannot provide ground-truth labels for "content" and
"headline noise". This script creates a small synthetic market where the
true latent factors are known:
  - content_signal causally affects next-step returns
  - headline_noise affects text embeddings but has no causal price effect

It then evaluates whether the same shared/private/gated modelling idea
recovers this distinction better than simple baselines.
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from experiments import StaticFusionModel, regression_metrics, set_seed
from model import MultimodalCausalModel


class SyntheticMarketDataset(Dataset):
    """Synthetic multimodal events with known content/noise factors."""

    def __init__(
        self,
        n_samples: int = 4000,
        text_dim: int = 64,
        window: int = 20,
        price_dim: int = 5,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        self.text_dim = text_dim
        self.window = window
        self.price_dim = price_dim

        content = rng.normal(0, 1, size=n_samples).astype(np.float32)
        noise = rng.normal(0, 1.3, size=n_samples).astype(np.float32)
        market = rng.normal(0, 0.5, size=n_samples).astype(np.float32)

        text = rng.normal(0, 0.25, size=(n_samples, text_dim)).astype(np.float32)
        text[:, 0] += 1.8 * content
        text[:, 1] += -1.2 * content
        text[:, 2] += 2.5 * noise
        text[:, 3] += -1.8 * noise

        price = rng.normal(0, 0.2, size=(n_samples, window, price_dim)).astype(np.float32)
        time = np.linspace(-1, 1, window, dtype=np.float32)
        price[:, :, 3] += content[:, None] * (time[None, :] + 1.2) * 0.25
        price[:, :, 4] += np.abs(noise[:, None]) * 0.10 + market[:, None] * 0.15
        price[:, :, 0] += market[:, None] * 0.2
        price[:, :, 1] += content[:, None] * 0.15

        target = (0.045 * content + 0.012 * market + rng.normal(0, 0.02, size=n_samples)).astype(np.float32)

        self.text = text
        self.price = price
        self.target = target
        self.content = content
        self.noise = noise

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.text[idx]),
            torch.from_numpy(self.price[idx]),
            torch.tensor(self.target[idx]),
            torch.tensor(self.content[idx]),
            torch.tensor(self.noise[idx]),
        )


class PriceOnlySynthetic(nn.Module):
    def __init__(self, price_dim: int = 5):
        super().__init__()
        self.lstm = nn.LSTM(price_dim, 64, batch_first=True, num_layers=1)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, text: torch.Tensor, price: torch.Tensor):
        del text
        _, (h, _) = self.lstm(price)
        return self.head(h[-1]).squeeze(-1)


class TextOnlySynthetic(nn.Module):
    def __init__(self, text_dim: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(text_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, text: torch.Tensor, price: torch.Tensor):
        del price
        return self.head(text).squeeze(-1)


def parse_args():
    parser = argparse.ArgumentParser(description="Run synthetic content-vs-noise experiment")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def make_model(name: str, text_dim: int) -> nn.Module:
    if name == "price_only":
        return PriceOnlySynthetic()
    if name == "text_only":
        return TextOnlySynthetic(text_dim)
    if name == "static_fusion":
        return StaticFusionModel(text_input_dim=text_dim, latent_dim=64)
    if name == "full_drl":
        return MultimodalCausalModel(
            text_input_dim=text_dim,
            latent_dim=64,
            subspace_dim=32,
            lstm_hidden=64,
            lstm_layers=1,
            moco_queue_size=128,
            lambda_diff=0.01,
            lambda_moco=0.05,
        )
    raise ValueError(name)


def train_epoch(name: str, model: nn.Module, loader: DataLoader, optimizer, device: str) -> float:
    model.train()
    total = 0.0
    n = 0
    for text, price, target, _, _ in loader:
        text = text.to(device)
        price = price.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        if name == "full_drl":
            pred, loss_dict = model(text, price, target)
            loss = loss_dict["total_loss"]
        else:
            pred = model(text, price)
            loss = F.mse_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(name: str, model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    contents: List[np.ndarray] = []
    noises: List[np.ndarray] = []
    shared_norms: List[np.ndarray] = []
    private_norms: List[np.ndarray] = []
    gates: List[np.ndarray] = []

    for text, price, target, content, noise in loader:
        text = text.to(device)
        price = price.to(device)
        if name == "full_drl":
            h_text = model.text_encoder(text)
            h_price = model.temporal_encoder(price)
            s_text, p_text, _ = model.text_disentangle(h_text)
            s_price, _, _ = model.price_disentangle(h_price)
            pred, gate = model.fusion(s_text, s_price, p_text)
            pred = pred.squeeze(-1)
            shared_norms.append(torch.linalg.norm(s_text, dim=1).cpu().numpy())
            private_norms.append(torch.linalg.norm(p_text, dim=1).cpu().numpy())
            gates.append(gate.squeeze(-1).cpu().numpy())
        else:
            pred = model(text, price)
        preds.append(pred.cpu().numpy())
        targets.append(target.numpy())
        contents.append(content.numpy())
        noises.append(noise.numpy())

    pred = np.concatenate(preds)
    target = np.concatenate(targets)
    content = np.concatenate(contents)
    noise = np.concatenate(noises)
    metrics = regression_metrics(pred, target)
    metrics["pred_content_corr"] = corr(pred, content)
    metrics["pred_noise_corr"] = corr(pred, noise)
    if shared_norms:
        shared = np.concatenate(shared_norms)
        private = np.concatenate(private_norms)
        gate = np.concatenate(gates)
        metrics["shared_content_corr"] = corr(shared, np.abs(content))
        metrics["private_noise_corr"] = corr(private, np.abs(noise))
        metrics["gate_noise_corr"] = corr(gate, np.abs(noise))
    return metrics


def corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def save_rows(rows: List[Dict[str, object]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()})
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: List[Dict[str, object]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    models = [str(r["model"]) for r in rows]
    metrics = ["mse", "pred_content_corr", "pred_noise_corr", "shared_content_corr", "private_noise_corr"]
    titles = ["MSE", "Prediction vs true content", "Prediction vs true noise", "Shared vs content", "Private vs noise"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4))
    for ax, metric, title in zip(axes, metrics, titles):
        vals = [float(r.get(metric, 0.0)) for r in rows]
        ax.bar(models, vals, color=["#264653", "#2a9d8f", "#e9c46a", "#e76f51"])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Synthetic content-vs-noise recovery", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    dataset = SyntheticMarketDataset(n_samples=args.n_samples, seed=args.seed)
    train_n = int(len(dataset) * 0.8)
    val_n = len(dataset) - train_n
    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(dataset, [train_n, val_n], generator=generator)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    rows: List[Dict[str, object]] = []
    for name in ["price_only", "text_only", "static_fusion", "full_drl"]:
        set_seed(args.seed)
        model = make_model(name, dataset.text_dim).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        loss = 0.0
        for _ in range(args.epochs):
            loss = train_epoch(name, model, train_loader, optimizer, args.device)
        metrics = evaluate(name, model, val_loader, args.device)
        row: Dict[str, object] = {"model": name, "train_final_loss": loss, **metrics}
        rows.append(row)
        print(
            f"{name:14s} mse={metrics['mse']:.6f} "
            f"pred_content={metrics['pred_content_corr']:+.3f} "
            f"pred_noise={metrics['pred_noise_corr']:+.3f}"
        )

    save_rows(rows, Path("results/synthetic_metrics.csv"))
    plot(rows, Path("results/synthetic_metrics.png"))
    print("Saved -> results/synthetic_metrics.csv")
    print("Saved -> results/synthetic_metrics.png")


if __name__ == "__main__":
    main()
