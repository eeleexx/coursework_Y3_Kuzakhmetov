#!/usr/bin/env python3
"""
SPY tweet-attention lead-lag response curve.

This script treats tweets as a social information-flow channel. It compares
high-attention minutes against matched low-attention control minutes and
measures how absolute returns and trading volume evolve after the event.
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


HORIZONS = [1, 2, 5, 10, 15, 30]


def non_overlapping_events(df: pd.DataFrame, threshold: float, min_gap: int = 30) -> pd.DataFrame:
    candidates = df[df["tweet_count"] >= threshold].copy()
    candidates = candidates.sort_values("tweet_count", ascending=False)
    selected: list[int] = []
    blocked = np.zeros(len(df), dtype=bool)

    for idx in candidates.index:
        if blocked[idx]:
            continue
        selected.append(idx)
        start = max(0, idx - min_gap)
        end = min(len(df), idx + min_gap + 1)
        blocked[start:end] = True

    return df.loc[sorted(selected)].copy()


def matched_controls(df: pd.DataFrame, events: pd.DataFrame, max_tweet_count: float) -> pd.DataFrame:
    controls = []
    used: set[int] = set()
    low_attention = df[df["tweet_count"] <= max_tweet_count].copy()

    for _, event in events.iterrows():
        same_hour = low_attention[low_attention["hour"] == event["hour"]]
        pool = same_hour if len(same_hour) else low_attention
        pool = pool[~pool.index.isin(used)]
        if len(pool) == 0:
            break
        idx = int((pool["bar_time_utc"] - event["bar_time_utc"]).abs().idxmin())
        controls.append(idx)
        used.add(idx)

    return df.loc[controls].copy()


def summarize_response(df: pd.DataFrame, events: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    close = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)

    for horizon in HORIZONS:
        abs_returns = []
        signed_returns = []
        future_volumes = []
        base_volumes = []

        for idx in events.index:
            if idx + horizon >= len(df) or idx < 30:
                continue
            signed = close[idx + horizon] / close[idx] - 1.0
            signed_returns.append(signed)
            abs_returns.append(abs(signed))
            future_volumes.append(volume[idx + 1 : idx + horizon + 1].sum())
            base_volumes.append(volume[idx - 30 : idx].sum() * horizon / 30.0)

        base = np.asarray(base_volumes, dtype=float)
        future = np.asarray(future_volumes, dtype=float)
        rows.append(
            {
                "group": label,
                "horizon_min": horizon,
                "events": len(abs_returns),
                "mean_abs_return_bps": float(np.mean(abs_returns) * 10000),
                "median_abs_return_bps": float(np.median(abs_returns) * 10000),
                "mean_signed_return_bps": float(np.mean(signed_returns) * 10000),
                "mean_future_volume": float(np.mean(future)),
                "mean_volume_ratio_vs_prior_30m": float(np.mean(future / (base + 1e-8))),
            }
        )

    return pd.DataFrame(rows)


def plot_response(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"tweet_spike": "#e76f51", "matched_control": "#264653"}

    for group, group_df in summary.groupby("group"):
        axes[0].plot(
            group_df["horizon_min"],
            group_df["mean_abs_return_bps"],
            marker="o",
            linewidth=2,
            label=group.replace("_", " "),
            color=colors.get(group),
        )
        axes[1].plot(
            group_df["horizon_min"],
            group_df["mean_volume_ratio_vs_prior_30m"],
            marker="o",
            linewidth=2,
            label=group.replace("_", " "),
            color=colors.get(group),
        )

    axes[0].set_title("Absolute price reaction")
    axes[0].set_xlabel("Minutes after event")
    axes[0].set_ylabel("Mean absolute return, bps")
    axes[1].set_title("Volume expansion")
    axes[1].set_xlabel("Minutes after event")
    axes[1].set_ylabel("Future volume / prior 30-min baseline")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("SPY intraday response after tweet-attention spikes", fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv("data/spy_1min_tweet_price_dataset.csv", parse_dates=["bar_time_utc", "price_time_local"])
    df = df.sort_values("bar_time_utc").reset_index(drop=True)
    df = df.dropna(subset=["forward_return_5m"]).reset_index(drop=True)
    df["hour"] = df["price_time_local"].dt.hour

    spike_threshold = float(df["tweet_count"].quantile(0.95))
    control_threshold = float(df["tweet_count"].quantile(0.50))
    events = non_overlapping_events(df, spike_threshold, min_gap=30)
    controls = matched_controls(df, events, control_threshold)

    summary = pd.concat(
        [
            summarize_response(df, events, "tweet_spike"),
            summarize_response(df, controls, "matched_control"),
        ],
        ignore_index=True,
    )
    summary.to_csv(out_dir / "spy_response_curve.csv", index=False)
    events.head(200).to_csv(out_dir / "spy_response_events.csv", index=False)
    controls.head(200).to_csv(out_dir / "spy_response_controls.csv", index=False)
    plot_response(summary, out_dir / "spy_response_curve.png")

    print(f"Spike threshold: tweet_count >= {spike_threshold:.0f}")
    print(f"Events: {len(events)} | Controls: {len(controls)}")
    print("Saved -> results/spy_response_curve.csv")
    print("Saved -> results/spy_response_curve.png")


if __name__ == "__main__":
    main()
