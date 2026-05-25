#!/usr/bin/env python3
"""
generate_eda_plots.py — Generates EDA figures for the CP1 report:
  1. Per-ticker descriptive statistics
  2. Returns distribution histograms
  3. OHLCV correlation heatmaps
  4. News volume vs. price volatility scatter
  5. News coverage timeline
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_DIR = Path("data")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

TICKERS = ["AMD", "INTC", "PFE", "JNJ", "BAC", "JPM"]
SECTORS = {"AMD": "Tech", "INTC": "Tech", "PFE": "Pharma", "JNJ": "Pharma", "BAC": "Banking", "JPM": "Banking"}
COLORS = {"AMD": "#e63946", "INTC": "#457b9d", "PFE": "#2a9d8f", "JNJ": "#e9c46a", "BAC": "#264653", "JPM": "#f4a261"}


# ── Load data ──
print("Loading data...")
stock = pd.read_csv(DATA_DIR / "stock_data.csv")
stock["Date"] = pd.to_datetime(stock["Date"], utc=True).dt.tz_localize(None)
stock = stock.sort_values(["symbol", "Date"]).reset_index(drop=True)

news_dfs = {}
for t in TICKERS:
    df = pd.read_csv(DATA_DIR / f"{t}_news.csv")
    df["date"] = pd.to_datetime(df["date"])
    news_dfs[t] = df

# Compute returns
stock["Return"] = stock.groupby("symbol")["Close"].pct_change()
stock["Volatility_20d"] = stock.groupby("symbol")["Return"].transform(lambda x: x.rolling(20).std())
stock["Log_Volume"] = np.log1p(stock["Volume"])


# ═══════════════════════════════════════════════════════════
# 1. Descriptive statistics table (printed to console for LaTeX)
# ═══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TABLE: Per-Ticker Descriptive Statistics")
print("="*70)
stats_rows = []
for t in TICKERS:
    s = stock[stock["symbol"] == t]
    n = news_dfs[t]
    stats_rows.append({
        "Ticker": t,
        "Sector": SECTORS[t],
        "Price Records": len(s),
        "News Articles": len(n),
        "Date Range": f"{s['Date'].min().strftime('%Y-%m')}-{s['Date'].max().strftime('%Y-%m')}",
        "Mean Close": f"{s['Close'].mean():.2f}",
        "Std Close": f"{s['Close'].std():.2f}",
        "Mean Return": f"{s['Return'].mean()*100:.4f}%",
        "Std Return": f"{s['Return'].std()*100:.2f}%",
        "Mean Vol (20d)": f"{s['Volatility_20d'].mean()*100:.2f}%",
    })
stats_df = pd.DataFrame(stats_rows)
print(stats_df.to_string(index=False))


# ═══════════════════════════════════════════════════════════
# 2. Return distributions
# ═══════════════════════════════════════════════════════════
print("\nPlotting return distributions...")
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Daily Return Distributions by Ticker", fontsize=14, fontweight="bold")
for ax, t in zip(axes.flat, TICKERS):
    ret = stock[stock["symbol"] == t]["Return"].dropna()
    ax.hist(ret, bins=80, color=COLORS[t], alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"{t} ({SECTORS[t]})", fontsize=11)
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    ax.text(0.95, 0.95, f"μ={ret.mean()*100:.3f}%\nσ={ret.std()*100:.2f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
plt.tight_layout()
fig.savefig(OUT_DIR / "return_distributions.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  → {OUT_DIR / 'return_distributions.png'}")


# ═══════════════════════════════════════════════════════════
# 3. OHLCV Correlation heatmap (cross-ticker)
# ═══════════════════════════════════════════════════════════
print("Plotting cross-ticker return correlation...")
returns_pivot = stock.pivot_table(index="Date", columns="symbol", values="Return")
returns_pivot = returns_pivot[TICKERS]  # order
corr = returns_pivot.corr()

fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(corr.values, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(TICKERS)))
ax.set_yticks(range(len(TICKERS)))
ax.set_xticklabels(TICKERS, fontsize=11)
ax.set_yticklabels(TICKERS, fontsize=11)
# Annotate
for i in range(len(TICKERS)):
    for j in range(len(TICKERS)):
        val = corr.values[i, j]
        color = "white" if abs(val) > 0.6 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10, color=color)
plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson Correlation")
ax.set_title("Cross-Ticker Daily Return Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_DIR / "cross_ticker_correlation.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  → {OUT_DIR / 'cross_ticker_correlation.png'}")


# ═══════════════════════════════════════════════════════════
# 4. Feature correlation heatmap (within AMD)
# ═══════════════════════════════════════════════════════════
print("Plotting within-ticker feature correlation (AMD)...")
amd = stock[stock["symbol"] == "AMD"].copy()
feat_cols = ["Open", "High", "Low", "Close", "Volume", "Return", "Volatility_20d"]
feat_corr = amd[feat_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(feat_corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
labels = ["Open", "High", "Low", "Close", "Volume", "Return", "Volatility\n(20d)"]
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
ax.set_yticklabels(labels, fontsize=9)
for i in range(len(labels)):
    for j in range(len(labels)):
        val = feat_corr.values[i, j]
        color = "white" if abs(val) > 0.6 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)
plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson Correlation")
ax.set_title("Feature Correlation Matrix (AMD)", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_DIR / "amd_feature_correlation.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  → {OUT_DIR / 'amd_feature_correlation.png'}")


# ═══════════════════════════════════════════════════════════
# 5. News volume vs. volatility scatter
# ═══════════════════════════════════════════════════════════
print("Plotting news volume vs. volatility...")
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Monthly News Volume vs. Price Volatility", fontsize=14, fontweight="bold")
for ax, t in zip(axes.flat, TICKERS):
    s = stock[stock["symbol"] == t].set_index("Date")
    n = news_dfs[t].set_index("date")
    # Monthly aggregation
    monthly_vol = s["Volatility_20d"].resample("ME").mean().dropna()
    monthly_news = n.resample("ME").size()
    # Align
    common = monthly_vol.index.intersection(monthly_news.index)
    if len(common) > 5:
        mv = monthly_vol[common]
        mn = monthly_news[common]
        ax.scatter(mn, mv * 100, alpha=0.6, color=COLORS[t], s=30)
        # Trend line
        z = np.polyfit(mn.values, mv.values * 100, 1)
        x_line = np.linspace(mn.min(), mn.max(), 50)
        ax.plot(x_line, np.polyval(z, x_line), "--", color="black", alpha=0.5, linewidth=1)
        r = np.corrcoef(mn.values, mv.values)[0, 1]
        ax.text(0.95, 0.95, f"r = {r:.2f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.set_title(f"{t} ({SECTORS[t]})", fontsize=11)
    ax.set_xlabel("News Articles / Month")
    ax.set_ylabel("Volatility (%)")
plt.tight_layout()
fig.savefig(OUT_DIR / "news_volume_vs_volatility.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  → {OUT_DIR / 'news_volume_vs_volatility.png'}")


# ═══════════════════════════════════════════════════════════
# 6. News coverage timeline
# ═══════════════════════════════════════════════════════════
print("Plotting news coverage timeline...")
fig, ax = plt.subplots(figsize=(14, 5))
for t in TICKERS:
    n = news_dfs[t].set_index("date")
    monthly = n.resample("ME").size()
    ax.plot(monthly.index, monthly.values, label=t, color=COLORS[t], linewidth=1.5, alpha=0.8)
ax.set_title("Monthly News Article Count by Ticker", fontsize=13, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Articles / Month")
ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=10)
ax.grid(True, alpha=0.2)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.tight_layout()
fig.savefig(OUT_DIR / "news_coverage_timeline.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  → {OUT_DIR / 'news_coverage_timeline.png'}")


# ═══════════════════════════════════════════════════════════
# 7. Price time series (normalised)
# ═══════════════════════════════════════════════════════════
print("Plotting normalised price series...")
fig, ax = plt.subplots(figsize=(14, 5))
for t in TICKERS:
    s = stock[stock["symbol"] == t].set_index("Date")["Close"]
    normalised = s / s.iloc[0] * 100
    ax.plot(normalised.index, normalised.values, label=t, color=COLORS[t], linewidth=1.5)
ax.set_title("Normalised Close Price (Base = 100)", fontsize=13, fontweight="bold")
ax.set_ylabel("Normalised Price")
ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=10)
ax.grid(True, alpha=0.2)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.tight_layout()
fig.savefig(OUT_DIR / "normalised_prices.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  → {OUT_DIR / 'normalised_prices.png'}")


print("\n✅ All EDA figures saved to figures/")
