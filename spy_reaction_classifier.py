#!/usr/bin/env python3
"""
SPY intraday reaction classification.

The directional-return task is intentionally hard and close to random. This
script evaluates a more defensible target: whether the next 5-minute absolute
reaction falls into the top decile. It compares price-only, tweet-only, and
fusion feature sets using a chronological split.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PRICE_FEATURES = [
    "return_1m",
    "return_5m",
    "volume",
    "barCount",
    "average",
    "intraday_range",
    "close_to_open",
]

TWEET_FEATURES = [
    "tweet_count",
    "labeled_count",
    "unlabeled_count",
    "bullish_count",
    "bearish_count",
    "sentiment_sum",
    "sentiment_mean",
    "labeled_sentiment_mean",
    "abs_sentiment",
    "bull_bear_imbalance",
]


def prepare() -> pd.DataFrame:
    df = pd.read_csv("data/spy_1min_tweet_price_dataset.csv", parse_dates=["bar_time_utc", "price_time_local"])
    df = df.sort_values("bar_time_utc").reset_index(drop=True)
    df = df.dropna(subset=["forward_return_5m", "return_1m", "return_5m"]).copy()
    df["abs_forward_return_5m"] = df["forward_return_5m"].abs()
    df["intraday_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["close_to_open"] = df["close"] / df["open"].replace(0, np.nan) - 1.0
    df["abs_sentiment"] = df["sentiment_mean"].abs()
    df["bull_bear_imbalance"] = (df["bullish_count"] - df["bearish_count"]) / (df["labeled_count"] + 1.0)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    threshold = df["abs_forward_return_5m"].quantile(0.90)
    df["top_decile_abs_reaction"] = (df["abs_forward_return_5m"] >= threshold).astype(int)
    return df


def precision_at_k(y_true: np.ndarray, score: np.ndarray, k_share: float = 0.10) -> float:
    k = max(1, int(len(score) * k_share))
    order = np.argsort(score)[::-1][:k]
    return float(np.mean(y_true[order]))


def evaluate_feature_set(df: pd.DataFrame, name: str, features: list[str]) -> dict[str, float | str | int]:
    split = int(len(df) * 0.8)
    train = df.iloc[:split]
    test = df.iloc[split:]
    x_train = train[features].to_numpy(dtype=float)
    y_train = train["top_decile_abs_reaction"].to_numpy(dtype=int)
    x_test = test[features].to_numpy(dtype=float)
    y_test = test["top_decile_abs_reaction"].to_numpy(dtype=int)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    score = model.predict_proba(x_test)[:, 1]
    pred = (score >= 0.5).astype(int)

    return {
        "model": name,
        "features": len(features),
        "train_samples": len(train),
        "test_samples": len(test),
        "positive_rate_test": float(np.mean(y_test)),
        "roc_auc": float(roc_auc_score(y_test, score)),
        "average_precision": float(average_precision_score(y_test, score)),
        "f1_at_0_5": float(f1_score(y_test, pred)),
        "precision_at_top_10pct": precision_at_k(y_test, score, 0.10),
        "precision_at_top_5pct": precision_at_k(y_test, score, 0.05),
    }


def plot(rows: list[dict[str, float | str | int]], path: Path) -> None:
    labels = [str(r["model"]).replace("_", " ") for r in rows]
    auc = [float(r["roc_auc"]) for r in rows]
    ap = [float(r["average_precision"]) for r in rows]
    p10 = [float(r["precision_at_top_10pct"]) for r in rows]
    baseline = float(rows[0]["positive_rate_test"])

    x = np.arange(len(rows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - width, auc, width, label="ROC AUC", color="#264653")
    ax.bar(x, ap, width, label="Avg. precision", color="#2a9d8f")
    ax.bar(x + width, p10, width, label="Precision@top10%", color="#e76f51")
    ax.axhline(baseline, color="#6c757d", linestyle="--", linewidth=1, label="Base positive rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(max(auc), max(ap), max(p10), baseline) * 1.25)
    ax.set_title("SPY top-decile 5-minute reaction classification")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    plt.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    df = prepare()
    rows = [
        evaluate_feature_set(df, "price_only", PRICE_FEATURES),
        evaluate_feature_set(df, "tweet_only", TWEET_FEATURES),
        evaluate_feature_set(df, "price_tweet_fusion", PRICE_FEATURES + TWEET_FEATURES),
    ]
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "spy_reaction_classifier.csv", index=False)
    plot(rows, out_dir / "spy_reaction_classifier.png")

    print(result.to_string(index=False))
    print("Saved -> results/spy_reaction_classifier.csv")
    print("Saved -> results/spy_reaction_classifier.png")


if __name__ == "__main__":
    main()
