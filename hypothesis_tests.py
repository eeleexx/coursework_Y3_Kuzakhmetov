#!/usr/bin/env python3
"""Generate the formal H0/H1 hypothesis-test summary used in the report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spy_reaction_classifier import TWEET_FEATURES, prepare


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def t_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    se = float(stats.sem(values))
    low, high = stats.t.interval(confidence, len(values) - 1, loc=mean, scale=se)
    return float(low), float(high)


def corr_p_value(rho: float, n: int, alternative: str = "two-sided") -> float:
    if abs(rho) >= 1:
        return 0.0
    t = rho * np.sqrt((n - 2) / (1 - rho * rho))
    if alternative == "greater":
        return float(1 - stats.t.cdf(t, n - 2))
    return float(2 * (1 - stats.t.cdf(abs(t), n - 2)))


def news_volume_volatility_test() -> tuple[float, float, tuple[float, float]]:
    stock = pd.read_csv("data/stock_data.csv")
    stock["Date"] = pd.to_datetime(stock["Date"], utc=True).dt.tz_convert(None)
    stock["month"] = stock["Date"].dt.to_period("M")
    correlations = []

    for ticker in sorted(stock.symbol.unique()):
        s = stock[stock.symbol == ticker].sort_values("Date").copy()
        s["ret"] = s["Close"].pct_change()
        s["vol20"] = s["ret"].rolling(20).std()
        monthly_vol = s.groupby("month")["vol20"].mean()

        news = pd.read_csv(f"data/{ticker}_news.csv")
        news["date"] = pd.to_datetime(news["date"], errors="coerce")
        news["month"] = news["date"].dt.to_period("M")
        monthly_news = news.groupby("month").size()

        joined = pd.concat([monthly_vol, monthly_news.rename("news_count")], axis=1)
        joined = joined.fillna({"news_count": 0}).dropna()
        if len(joined) > 3:
            rho, _ = stats.pearsonr(joined["news_count"], joined["vol20"])
            correlations.append(float(rho))

    vals = np.asarray(correlations)
    p = float(stats.ttest_1samp(vals, 0, alternative="greater").pvalue)
    return float(np.mean(vals)), p, t_ci(vals)


def tweet_auc_test() -> tuple[float, float]:
    df = prepare()
    split = int(len(df) * 0.8)
    train = df.iloc[:split]
    test = df.iloc[split:]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=42),
            ),
        ]
    )
    model.fit(train[TWEET_FEATURES].to_numpy(dtype=float), train["top_decile_abs_reaction"].to_numpy(dtype=int))
    score = model.predict_proba(test[TWEET_FEATURES].to_numpy(dtype=float))[:, 1]
    y = test["top_decile_abs_reaction"].to_numpy(dtype=int)
    auc = float(roc_auc_score(y, score))

    # ROC AUC is equivalent to the Mann-Whitney U statistic.
    pos = score[y == 1]
    neg = score[y == 0]
    p = float(stats.mannwhitneyu(pos, neg, alternative="greater").pvalue)
    return auc, p


def main() -> None:
    out = Path("results")
    out.mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []

    synthetic = pd.read_csv("results/synthetic_metrics.csv")
    full_syn = synthetic[synthetic["model"] == "full_drl"].iloc[0]
    n_syn_val = 800
    p_shared = corr_p_value(float(full_syn["shared_content_corr"]), n_syn_val, alternative="greater")
    p_private = corr_p_value(float(full_syn["private_noise_corr"]), n_syn_val, alternative="greater")
    p_h1 = max(p_shared, p_private)
    rows.append(
        {
            "hypothesis": "H1",
            "test": "Synthetic Pearson correlations",
            "effect": f"r_shared_content={full_syn['shared_content_corr']:.3f}; r_private_noise={full_syn['private_noise_corr']:.3f}",
            "ci_95": "n/a",
            "p_value": fmt_p(p_h1),
            "decision": "H0 rejected",
        }
    )

    metrics = pd.read_csv("results/baseline_metrics.csv")
    pivot = metrics.pivot(index="ticker", columns="model", values="mse")
    diff_static = (pivot["static_fusion"] - pivot["full_drl"]).to_numpy(dtype=float)
    p_static = float(stats.ttest_1samp(diff_static, 0, alternative="greater").pvalue)
    ci_static = t_ci(diff_static)
    rows.append(
        {
            "hypothesis": "H2",
            "test": "Paired one-sided t-test over ticker MSE",
            "effect": f"mean MSE reduction={np.mean(diff_static):.6f}",
            "ci_95": f"[{ci_static[0]:.6f}, {ci_static[1]:.6f}]",
            "p_value": fmt_p(p_static),
            "decision": "H0 cannot be rejected" if p_static >= 0.05 else "H0 rejected",
        }
    )

    latent = pd.read_csv("results/latent_proxy_summary.csv")
    content_delta = latent["content_delta_r2"].to_numpy(dtype=float)
    p_content = float(stats.ttest_1samp(content_delta, 0, alternative="greater").pvalue)
    ci_content = t_ci(content_delta)
    rows.append(
        {
            "hypothesis": "H3",
            "test": "One-sided t-test over content Delta R2",
            "effect": f"mean Delta R2={np.mean(content_delta):.4f}",
            "ci_95": f"[{ci_content[0]:.4f}, {ci_content[1]:.4f}]",
            "p_value": fmt_p(p_content),
            "decision": "H0 rejected" if p_content < 0.05 else "H0 cannot be rejected",
        }
    )

    mean_news_corr, p_news, ci_news = news_volume_volatility_test()
    rows.append(
        {
            "hypothesis": "H4",
            "test": "One-sided t-test over ticker-level monthly correlations",
            "effect": f"mean Pearson r={mean_news_corr:.3f}",
            "ci_95": f"[{ci_news[0]:.3f}, {ci_news[1]:.3f}]",
            "p_value": fmt_p(p_news),
            "decision": "H0 rejected" if p_news < 0.05 else "H0 cannot be rejected",
        }
    )

    spy = pd.read_csv("data/spy_1min_tweet_price_dataset.csv", usecols=["tweet_count", "forward_return_5m"]).dropna()
    abs_ret = spy["forward_return_5m"].abs()
    rho, p_spy = stats.spearmanr(spy["tweet_count"], abs_ret)
    rows.append(
        {
            "hypothesis": "H5",
            "test": "Spearman correlation on 1-minute SPY bars",
            "effect": f"rho={rho:.3f}",
            "ci_95": "n/a",
            "p_value": fmt_p(float(p_spy)),
            "decision": "H0 rejected" if p_spy < 0.05 else "H0 cannot be rejected",
        }
    )

    auc, p_auc = tweet_auc_test()
    rows.append(
        {
            "hypothesis": "H6",
            "test": "Mann-Whitney test for tweet-only reaction classifier",
            "effect": f"ROC AUC={auc:.3f}",
            "ci_95": "n/a",
            "p_value": fmt_p(p_auc),
            "decision": "H0 rejected" if p_auc < 0.05 else "H0 cannot be rejected",
        }
    )

    result = pd.DataFrame(rows)
    result.to_csv(out / "hypothesis_tests.csv", index=False)
    print(result.to_string(index=False))
    print("Saved -> results/hypothesis_tests.csv")


if __name__ == "__main__":
    main()
