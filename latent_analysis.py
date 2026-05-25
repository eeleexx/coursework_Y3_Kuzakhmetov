#!/usr/bin/env python3
"""
Train the full DRL model and export interpretable latent diagnostics.

This script produces report-ready artifacts:
  - per-event content/noise/gate table
  - gate timeline plot
  - content-vs-noise scatter plot
  - lagged predictive proxy table for the "content vs noise" hypothesis

The lagged proxy is deliberately conservative: it is not presented as a
formal Granger causality test, but as evidence that learned content features
carry more temporal predictive signal than private/noise features.
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
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_loader import DATA_DIR, MultimodalDataset, load_news_data, load_stock_data
from experiments import forward_loss, regression_metrics, set_seed, train_model
from model import MultimodalCausalModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export latent diagnostics for full DRL model")
    parser.add_argument("--ticker", default="AMD")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_cached_dataset(ticker: str, window: int) -> Tuple[MultimodalDataset, pd.DataFrame, pd.DataFrame]:
    cache_path = DATA_DIR / f"{ticker}_finbert_embeddings.npy"
    cache_dates_path = DATA_DIR / f"{ticker}_finbert_dates.npy"
    if not cache_path.exists() or not cache_dates_path.exists():
        raise FileNotFoundError(
            f"Missing cached embeddings for {ticker}. Run data_loader/build_dataloaders first."
        )

    stock_df = load_stock_data(ticker)
    news_df = load_news_data(ticker)
    embeddings = np.load(cache_path)
    news_dates = pd.to_datetime(np.load(cache_dates_path, allow_pickle=True)).tolist()
    dataset = MultimodalDataset(embeddings, stock_df, news_dates, window=window)
    return dataset, stock_df.reset_index(drop=True), news_df.reset_index(drop=True)


def split_dataset(dataset: MultimodalDataset, val_split: float):
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n_train + n_val))
    train_ds = torch.utils.data.Subset(dataset, train_indices)
    val_ds = torch.utils.data.Subset(dataset, val_indices)
    return train_ds, val_ds, train_indices, val_indices


@torch.no_grad()
def collect_latents(
    model: MultimodalCausalModel,
    dataset: MultimodalDataset,
    indices: List[int],
    stock_df: pd.DataFrame,
    news_df: pd.DataFrame,
    device: str,
) -> pd.DataFrame:
    model.eval()
    rows: List[Dict[str, object]] = []

    for ds_idx in indices:
        emb_i, stock_i = dataset.samples[ds_idx]
        text_emb, price_window, target = dataset[ds_idx]
        text_emb_b = text_emb.unsqueeze(0).to(device)
        price_window_b = price_window.unsqueeze(0).to(device)

        h_text = model.text_encoder(text_emb_b)
        h_price = model.temporal_encoder(price_window_b)
        s_text, p_text, _ = model.text_disentangle(h_text)
        s_price, p_price, _ = model.price_disentangle(h_price)
        prediction, gate = model.fusion(s_text, s_price, p_text)
        price_only_prediction = model.fusion.predictor(s_price)
        text_only_prediction = model.fusion.predictor(s_text)

        news_row = news_df.iloc[emb_i]
        stock_row = stock_df.iloc[stock_i]
        text_content_norm = float(torch.linalg.norm(s_text).item())
        text_noise_norm = float(torch.linalg.norm(p_text).item())
        price_content_norm = float(torch.linalg.norm(s_price).item())
        price_noise_norm = float(torch.linalg.norm(p_price).item())

        rows.append(
            {
                "date": pd.to_datetime(stock_row["Date"]).date().isoformat(),
                "ticker": stock_row.get("symbol", ""),
                "title": news_row.get("title", ""),
                "summary": news_row.get("summary", ""),
                "target_return": float(target.item()),
                "predicted_return": float(prediction.squeeze().item()),
                "counterfactual_price_only_return": float(price_only_prediction.squeeze().item()),
                "counterfactual_text_only_return": float(text_only_prediction.squeeze().item()),
                "abs_target_return": abs(float(target.item())),
                "gate_text_confidence": float(gate.squeeze().item()),
                "text_content_norm": text_content_norm,
                "text_noise_norm": text_noise_norm,
                "price_content_norm": price_content_norm,
                "price_noise_norm": price_noise_norm,
                "text_noise_ratio": text_noise_norm / (text_content_norm + text_noise_norm + 1e-12),
                "volatility_20d": float(stock_row.get("Volatility_20d", np.nan)),
                "same_day_return": float(stock_row.get("Return", np.nan)),
            }
        )

    return pd.DataFrame(rows)


def ols_r2(y: np.ndarray, x: np.ndarray) -> float:
    if len(y) < 5:
        return 0.0
    x_aug = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
    pred = x_aug @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def lagged_predictive_proxy(events: pd.DataFrame) -> Dict[str, float]:
    daily = (
        events.groupby("date", as_index=False)
        .agg(
            target_return=("target_return", "mean"),
            text_content_norm=("text_content_norm", "mean"),
            text_noise_norm=("text_noise_norm", "mean"),
            gate_text_confidence=("gate_text_confidence", "mean"),
        )
        .sort_values("date")
    )
    daily["return_lag1"] = daily["target_return"].shift(1)
    daily["content_lag1"] = daily["text_content_norm"].shift(1)
    daily["noise_lag1"] = daily["text_noise_norm"].shift(1)
    daily["gate_lag1"] = daily["gate_text_confidence"].shift(1)
    daily = daily.dropna().reset_index(drop=True)

    y = daily["target_return"].to_numpy(dtype=float)
    base = daily[["return_lag1"]].to_numpy(dtype=float)
    content = daily[["return_lag1", "content_lag1"]].to_numpy(dtype=float)
    noise = daily[["return_lag1", "noise_lag1"]].to_numpy(dtype=float)
    gate = daily[["return_lag1", "gate_lag1"]].to_numpy(dtype=float)
    both = daily[["return_lag1", "content_lag1", "noise_lag1", "gate_lag1"]].to_numpy(dtype=float)

    base_r2 = ols_r2(y, base)
    content_r2 = ols_r2(y, content)
    noise_r2 = ols_r2(y, noise)
    gate_r2 = ols_r2(y, gate)
    both_r2 = ols_r2(y, both)
    return {
        "n_daily_events": float(len(daily)),
        "ar1_r2": base_r2,
        "ar1_plus_content_r2": content_r2,
        "ar1_plus_noise_r2": noise_r2,
        "ar1_plus_gate_r2": gate_r2,
        "full_latent_proxy_r2": both_r2,
        "content_delta_r2": content_r2 - base_r2,
        "noise_delta_r2": noise_r2 - base_r2,
        "gate_delta_r2": gate_r2 - base_r2,
    }


def save_proxy(proxy: Dict[str, float], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(proxy.keys()))
        writer.writeheader()
        writer.writerow(proxy)


def plot_gate_timeline(events: pd.DataFrame, out_path: Path):
    daily = (
        events.groupby("date", as_index=False)
        .agg(
            gate_text_confidence=("gate_text_confidence", "mean"),
            abs_target_return=("abs_target_return", "mean"),
            text_noise_ratio=("text_noise_ratio", "mean"),
        )
        .sort_values("date")
    )
    dates = pd.to_datetime(daily["date"])
    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax1.plot(dates, daily["gate_text_confidence"], color="#2a9d8f", linewidth=2, label="Text gate")
    ax1.plot(dates, daily["text_noise_ratio"], color="#e76f51", linewidth=1.5, alpha=0.8, label="Text noise ratio")
    ax1.set_ylabel("Gate / noise ratio")
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.bar(dates, daily["abs_target_return"], color="#264653", alpha=0.2, width=4, label="Abs next-day return")
    ax2.set_ylabel("Absolute next-day return")

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()
    fig.suptitle("Gate dynamics vs. realized reaction", fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_content_noise(events: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        events["text_content_norm"],
        events["text_noise_norm"],
        c=events["abs_target_return"],
        cmap="viridis",
        alpha=0.75,
        s=35,
    )
    ax.set_xlabel("Text shared/content norm")
    ax.set_ylabel("Text private/noise norm")
    ax.set_title("Content-noise decomposition of news events", fontweight="bold")
    ax.grid(alpha=0.25)
    plt.colorbar(sc, ax=ax, label="Absolute next-day return")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset, stock_df, news_df = load_cached_dataset(args.ticker, args.window)
    train_ds, val_ds, train_indices, val_indices = split_dataset(dataset, args.val_split)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print("=" * 72)
    print("Latent Diagnostics")
    print("=" * 72)
    print(f"Ticker: {args.ticker}")
    print(f"Dataset: {len(dataset)} samples ({len(train_ds)} train / {len(val_ds)} val)")
    print(f"Device: {args.device}")

    model = MultimodalCausalModel(text_input_dim=dataset.embeddings.shape[1]).to(args.device)
    history = train_model("full_drl", model, train_loader, args.device, args.epochs, args.lr)

    preds = []
    targets = []
    model.eval()
    with torch.no_grad():
        for text_emb, price_window, target in val_loader:
            text_emb = text_emb.to(args.device)
            price_window = price_window.to(args.device)
            prediction, _ = model(text_emb, price_window)
            preds.append(prediction.squeeze(-1).cpu().numpy())
            targets.append(target.numpy())
    metrics = regression_metrics(np.concatenate(preds), np.concatenate(targets))
    print(
        f"Validation: mse={metrics['mse']:.6f} mae={metrics['mae']:.6f} "
        f"dir={metrics['directional_accuracy']:.3f} ic={metrics['information_coefficient']:.3f}"
    )
    print(f"Final train loss: {history[-1]:.6f}")

    events = collect_latents(model, dataset, val_indices, stock_df, news_df, args.device)
    events_path = Path(f"results/latent_events_{args.ticker}.csv")
    events.to_csv(events_path, index=False)

    proxy = lagged_predictive_proxy(events)
    proxy_path = Path(f"results/latent_predictive_proxy_{args.ticker}.csv")
    save_proxy(proxy, proxy_path)

    plot_gate_timeline(events, Path(f"results/gate_timeline_{args.ticker}.png"))
    plot_content_noise(events, Path(f"results/content_noise_scatter_{args.ticker}.png"))

    top_content = events.sort_values("text_content_norm", ascending=False).head(3)
    top_noise = events.sort_values("text_noise_ratio", ascending=False).head(3)
    print("\nTop content-like events:")
    for _, row in top_content.iterrows():
        print(f"  {row['date']} | gate={row['gate_text_confidence']:.2f} | {row['title'][:90]}")
    print("\nTop noise-like events:")
    for _, row in top_noise.iterrows():
        print(f"  {row['date']} | noise={row['text_noise_ratio']:.2f} | {row['title'][:90]}")

    print(f"\nSaved events -> {events_path}")
    print(f"Saved proxy  -> {proxy_path}")
    print("Saved plots  -> results/gate_timeline_*.png, results/content_noise_scatter_*.png")


if __name__ == "__main__":
    main()
