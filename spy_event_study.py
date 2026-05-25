#!/usr/bin/env python3
"""
SPY intraday event-study diagnostics.

The neural intraday experiment is intentionally conservative because 5-minute
directional returns are close to noise. This script tests the more defensible
market-microstructure claim: information-flow intensity is associated with
higher realised short-horizon reaction and trading volume.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    df = pd.read_csv("data/spy_1min_tweet_price_dataset.csv", parse_dates=["bar_time_utc", "price_time_local"])
    df = df.sort_values("bar_time_utc").reset_index(drop=True)
    df["abs_forward_return_5m"] = df["forward_return_5m"].abs()
    df["abs_sentiment"] = df["sentiment_mean"].abs()

    corr_cols = [
        "tweet_count",
        "labeled_count",
        "bullish_count",
        "bearish_count",
        "sentiment_mean",
        "abs_sentiment",
        "volume",
        "abs_forward_return_5m",
        "forward_return_5m",
    ]
    corr = df[corr_cols].corr()
    corr[["abs_forward_return_5m", "volume", "forward_return_5m"]].to_csv(out_dir / "spy_intraday_correlations.csv")

    thresholds = [0, 7, 11, 13, 19]
    rows = []
    for threshold in thresholds:
        subset = df[df["tweet_count"] >= threshold]
        rows.append(
            {
                "filter": f"tweet_count >= {threshold}",
                "minutes": len(subset),
                "mean_abs_forward_return_5m": subset["abs_forward_return_5m"].mean(),
                "median_abs_forward_return_5m": subset["abs_forward_return_5m"].median(),
                "mean_volume": subset["volume"].mean(),
                "positive_forward_return_share": (subset["forward_return_5m"] > 0).mean(),
                "mean_tweet_count": subset["tweet_count"].mean(),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "spy_event_study_summary.csv", index=False)

    buckets = df.copy()
    buckets["tweet_bucket"] = pd.qcut(buckets["tweet_count"], q=10, duplicates="drop")
    bucket_summary = (
        buckets.groupby("tweet_bucket", observed=True)
        .agg(
            minutes=("tweet_count", "size"),
            mean_tweet_count=("tweet_count", "mean"),
            mean_abs_forward_return_5m=("abs_forward_return_5m", "mean"),
            mean_volume=("volume", "mean"),
            direction_up_share=("forward_return_5m", lambda x: (x > 0).mean()),
        )
        .reset_index()
    )
    bucket_summary["tweet_bucket"] = bucket_summary["tweet_bucket"].astype(str)
    bucket_summary.to_csv(out_dir / "spy_attention_buckets.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(len(bucket_summary))
    axes[0].plot(x, bucket_summary["mean_tweet_count"], marker="o", color="#264653")
    axes[0].set_title("Tweet intensity bucket")
    axes[0].set_ylabel("Mean tweets / minute")
    axes[1].plot(x, bucket_summary["mean_abs_forward_return_5m"] * 10000, marker="o", color="#e76f51")
    axes[1].set_title("5-min absolute reaction")
    axes[1].set_ylabel("Basis points")
    axes[2].plot(x, bucket_summary["mean_volume"], marker="o", color="#2a9d8f")
    axes[2].set_title("Trading volume")
    axes[2].set_ylabel("Shares / minute")
    for ax in axes:
        ax.set_xlabel("Tweet-count decile")
        ax.grid(alpha=0.25)
    fig.suptitle("SPY intraday information-flow intensity vs market reaction", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "spy_event_study_buckets.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    top = df.sort_values("tweet_count", ascending=False).head(6).copy()
    top.to_csv(out_dir / "spy_top_attention_minutes.csv", index=False)
    plot_top_windows(df, top, out_dir / "spy_top_event_windows.png")

    print("Saved -> results/spy_intraday_correlations.csv")
    print("Saved -> results/spy_event_study_summary.csv")
    print("Saved -> results/spy_attention_buckets.csv")
    print("Saved -> results/spy_event_study_buckets.png")
    print("Saved -> results/spy_top_event_windows.png")


def plot_top_windows(df: pd.DataFrame, top: pd.DataFrame, path: Path, window: int = 30):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, (_, event) in zip(axes.flat, top.iterrows()):
        idx = int(event.name)
        start = max(0, idx - window)
        end = min(len(df), idx + window + 1)
        subset = df.iloc[start:end].copy()
        rel = np.arange(len(subset)) - (idx - start)
        base = float(event["close"])
        norm_close = subset["close"].astype(float) / base * 100
        ax.plot(rel, norm_close, color="#264653", linewidth=1.8)
        ax.axvline(0, color="#e76f51", linestyle="--", linewidth=1)
        ax.set_title(
            f"{event['bar_time_utc']} | tweets={event['tweet_count']:.0f}",
            fontsize=9,
        )
        ax.set_xlabel("Minutes around attention spike")
        ax.set_ylabel("Close, event=100")
        ax.grid(alpha=0.25)
    fig.suptitle("Top SPY tweet-attention minutes and local price windows", fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
