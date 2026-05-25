#!/usr/bin/env python3
"""
train_prototype.py — Proof-of-Concept training script for the
Multimodal Causal Model.

Runs N training epochs on a single ticker (default: AMD), logs
per-batch and per-epoch losses, and saves a loss curve plot.

Usage
-----
    python train_prototype.py                    # defaults: AMD, 5 epochs
    python train_prototype.py --ticker INTC --epochs 10 --lr 1e-3

Demonstrates that the composite loss (MSE + λ·diff + λ·MoCo) decreases
over training, indicating the model is learning to disentangle shared
and private latent factors.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import torch

from data_loader import build_dataloaders
from model import MultimodalCausalModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Multimodal Causal Model (PoC)")
    p.add_argument("--ticker", type=str, default="AMD", help="Stock ticker")
    p.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--window", type=int, default=20, help="OHLCV look-back window")
    p.add_argument("--latent_dim", type=int, default=128, help="Encoder latent dim")
    p.add_argument("--subspace_dim", type=int, default=64, help="Shared/private dim")
    p.add_argument("--lambda_diff", type=float, default=0.01, help="Diff loss weight")
    p.add_argument("--lambda_moco", type=float, default=0.01, help="MoCo loss weight")
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def train_one_epoch(
    model: MultimodalCausalModel,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
) -> dict:
    """Train for one epoch; returns averaged loss dict."""
    model.train()
    running = {"total_loss": 0.0, "mse_loss": 0.0, "diff_loss": 0.0, "moco_loss": 0.0, "gate": 0.0}
    n_batches = 0

    for batch_idx, (text_emb, price_win, target) in enumerate(loader):
        text_emb = text_emb.to(device)
        price_win = price_win.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        _, loss_dict = model(text_emb, price_win, target)
        loss_dict["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in running:
            running[k] += loss_dict[k].item()
        n_batches += 1

        if batch_idx % 5 == 0:
            print(
                f"  [Epoch {epoch+1} | Batch {batch_idx:>3d}] "
                f"total={loss_dict['total_loss'].item():.6f}  "
                f"mse={loss_dict['mse_loss'].item():.6f}  "
                f"diff={loss_dict['diff_loss'].item():.6f}  "
                f"moco={loss_dict['moco_loss'].item():.4f}  "
                f"gate={loss_dict['gate'].item():.3f}"
            )

    return {k: v / max(n_batches, 1) for k, v in running.items()}


@torch.no_grad()
def evaluate(
    model: MultimodalCausalModel,
    loader: torch.utils.data.DataLoader,
    device: str,
) -> dict:
    """Evaluate on validation set; returns averaged loss dict."""
    model.eval()
    running = {"total_loss": 0.0, "mse_loss": 0.0, "diff_loss": 0.0, "moco_loss": 0.0, "gate": 0.0}
    n_batches = 0

    for text_emb, price_win, target in loader:
        text_emb = text_emb.to(device)
        price_win = price_win.to(device)
        target = target.to(device)

        _, loss_dict = model(text_emb, price_win, target)
        for k in running:
            running[k] += loss_dict[k].item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in running.items()}


def plot_losses(history: list[dict], save_path: Path):
    """Save a multi-panel loss curve plot."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training Loss Curves — Multimodal Causal Model", fontsize=14)

    keys = ["total_loss", "mse_loss", "diff_loss", "moco_loss"]
    titles = ["Total Loss", "MSE (Prediction)", "Diff (Orthogonality)", "MoCo (Contrastive)"]
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]

    epochs = list(range(1, len(history) + 1))
    for ax, key, title, color in zip(axes.flat, keys, titles, colors):
        train_vals = [h["train"][key] for h in history]
        val_vals = [h["val"][key] for h in history]
        ax.plot(epochs, train_vals, "o-", color=color, label="Train", linewidth=2)
        ax.plot(epochs, val_vals, "s--", color=color, alpha=0.6, label="Val", linewidth=2)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[train] Loss curve saved → {save_path}")
    plt.close(fig)


def main():
    args = parse_args()
    print("=" * 60)
    print("  Multimodal Causal Model — Proof of Concept")
    print("=" * 60)
    print(f"  Ticker:      {args.ticker}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  LR:          {args.lr}")
    print(f"  Device:      {args.device}")
    print(f"  Window:      {args.window}")
    print(f"  Latent dim:  {args.latent_dim}")
    print(f"  Subspace dim:{args.subspace_dim}")
    print("=" * 60)

    # ---- Data ----
    train_loader, val_loader, text_dim = build_dataloaders(
        ticker=args.ticker,
        window=args.window,
        batch_size=args.batch_size,
        device=args.device,
    )

    # ---- Model ----
    model = MultimodalCausalModel(
        text_input_dim=text_dim,
        price_input_dim=5,
        latent_dim=args.latent_dim,
        subspace_dim=args.subspace_dim,
        lambda_diff=args.lambda_diff,
        lambda_moco=args.lambda_moco,
    ).to(args.device)

    print(f"\n[train] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ---- Training loop ----
    history = []
    for epoch in range(args.epochs):
        print(f"\n{'─'*50}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'─'*50}")

        train_metrics = train_one_epoch(model, train_loader, optimizer, args.device, epoch)
        val_metrics = evaluate(model, val_loader, args.device)
        scheduler.step()

        history.append({"train": train_metrics, "val": val_metrics})

        print(
            f"\n  ► Train: total={train_metrics['total_loss']:.6f}  "
            f"mse={train_metrics['mse_loss']:.6f}  "
            f"gate={train_metrics['gate']:.3f}"
        )
        print(
            f"  ► Val:   total={val_metrics['total_loss']:.6f}  "
            f"mse={val_metrics['mse_loss']:.6f}  "
            f"gate={val_metrics['gate']:.3f}"
        )

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  Training Complete — Epoch Summary")
    print("=" * 60)
    for i, h in enumerate(history):
        t = h["train"]
        print(
            f"  Epoch {i+1}: total={t['total_loss']:.6f}  "
            f"mse={t['mse_loss']:.6f}  "
            f"diff={t['diff_loss']:.6f}  "
            f"moco={t['moco_loss']:.4f}"
        )

    first_loss = history[0]["train"]["total_loss"]
    last_loss = history[-1]["train"]["total_loss"]
    pct_change = (last_loss - first_loss) / (first_loss + 1e-8) * 100
    print(f"\n  Loss change: {first_loss:.6f} → {last_loss:.6f} ({pct_change:+.1f}%)")
    if last_loss < first_loss:
        print("  ✅ Loss decreased — model is learning!")
    else:
        print("  ⚠️  Loss did not decrease — consider tuning hyperparameters.")

    # ---- Plot ----
    plot_losses(history, Path("results/loss_curve.png"))


if __name__ == "__main__":
    main()
