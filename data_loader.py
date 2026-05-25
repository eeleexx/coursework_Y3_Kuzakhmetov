"""
data_loader.py — Data parsing, FinBERT embedding extraction,
and PyTorch Dataset / DataLoader construction for the
Multimodal Causal Model.

Implements:
  • CSV loading for OHLCV prices and news
  • FinBERT-based semantic embeddings (yiyanghkust/finbert-tone)
  • Date-level synchronisation of news ↔ prices (T+0 and T+1)
  • Sliding-window multimodal Dataset for training
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"

# ---------------------------------------------------------------------------
# 1. Stock data
# ---------------------------------------------------------------------------

def load_stock_data(ticker: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load OHLCV data for *ticker* from the combined stock_data.csv."""
    df = pd.read_csv(data_dir / "stock_data.csv")
    df = df[df["symbol"] == ticker].copy()

    # Parse date — strip timezone info and normalise to date
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.date
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Derived features
    df["Return"] = df["Close"].pct_change()
    df["Volatility_20d"] = df["Return"].rolling(20).std()

    # Normalise OHLCV columns per-stock (z-score)
    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in ohlcv_cols:
        mean, std = df[col].mean(), df[col].std()
        df[f"{col}_norm"] = (df[col] - mean) / (std + 1e-8)

    return df


# ---------------------------------------------------------------------------
# 2. News data
# ---------------------------------------------------------------------------

def load_news_data(ticker: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load news headlines/summaries for *ticker*."""
    path = data_dir / f"{ticker}_news.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Combine title + summary into a single text field
    df["text"] = df["title"].fillna("") + ". " + df["summary"].fillna("")
    df["text"] = df["text"].str.strip(". ")
    return df


# ---------------------------------------------------------------------------
# 3. FinBERT embeddings
# ---------------------------------------------------------------------------

def get_finbert_embeddings(
    texts: List[str],
    model,
    tokenizer,
    device: str = "cpu",
    batch_size: int = 32,
    max_length: int = 128,
) -> np.ndarray:
    """
    Encode *texts* with FinBERT and return the [CLS] embedding (768-d)
    for each text.  Runs inference in eval mode with no gradients.
    """
    model.eval()
    all_embeddings: List[np.ndarray] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT encoding"):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

        # [CLS] token embedding — first token of last_hidden_state
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_emb)

    return np.concatenate(all_embeddings, axis=0)


# ---------------------------------------------------------------------------
# 4. Synchronise news ↔ prices
# ---------------------------------------------------------------------------

def synchronize_data(
    news_df: pd.DataFrame,
    stock_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge news with same-day (T+0) price data.
    Also attaches the *next trading day* close (T+1) so we can compute
    the reaction return as the training target.
    """
    stock = stock_df.copy()
    stock["Next_Close"] = stock["Close"].shift(-1)
    stock["Target_Return"] = stock["Next_Close"] / stock["Close"] - 1

    merged = pd.merge(
        news_df,
        stock[
            ["Date", "Open_norm", "High_norm", "Low_norm", "Close_norm",
             "Volume_norm", "Return", "Volatility_20d", "Target_Return"]
        ],
        left_on="date",
        right_on="Date",
        how="inner",
    )
    merged = merged.dropna(subset=["Target_Return"]).reset_index(drop=True)
    return merged


# ---------------------------------------------------------------------------
# 5. PyTorch Dataset
# ---------------------------------------------------------------------------

class MultimodalDataset(Dataset):
    """
    Yields (text_embedding, price_window, target_return) tuples.

    Parameters
    ----------
    embeddings : np.ndarray   — (N, 768) FinBERT [CLS] vectors
    stock_df   : pd.DataFrame — full stock dataframe (sorted by date)
    news_dates : list[pd.Timestamp] — date for each embedding row
    window     : int          — look-back window for price features
    """

    OHLCV_NORM_COLS = [
        "Open_norm", "High_norm", "Low_norm", "Close_norm", "Volume_norm"
    ]

    def __init__(
        self,
        embeddings: np.ndarray,
        stock_df: pd.DataFrame,
        news_dates: list,
        window: int = 20,
    ):
        super().__init__()
        self.window = window

        # Build a date → row-index lookup for the stock dataframe
        stock_df = stock_df.reset_index(drop=True)
        self.stock_dates = stock_df["Date"].tolist()
        self.date_to_idx = {d: i for i, d in enumerate(self.stock_dates)}

        # Pre-extract the normalised OHLCV matrix (T, 5)
        self.price_matrix = stock_df[self.OHLCV_NORM_COLS].values.astype(np.float32)

        # Target returns aligned to stock dates
        stock_df["Next_Close"] = stock_df["Close"].shift(-1)
        self.target_returns = (
            stock_df["Next_Close"] / stock_df["Close"] - 1
        ).values.astype(np.float32)

        # Filter to samples where we have enough look-back AND a valid target
        self.samples: list[Tuple[int, int]] = []  # (emb_idx, stock_row_idx)
        for emb_i, dt in enumerate(news_dates):
            if dt not in self.date_to_idx:
                continue
            stock_i = self.date_to_idx[dt]
            if stock_i < window:
                continue
            if stock_i >= len(self.target_returns) - 1:
                continue
            if np.isnan(self.target_returns[stock_i]):
                continue
            self.samples.append((emb_i, stock_i))

        self.embeddings = embeddings.astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        emb_i, stock_i = self.samples[idx]

        text_emb = torch.from_numpy(self.embeddings[emb_i])              # (768,)
        price_win = torch.from_numpy(
            self.price_matrix[stock_i - self.window : stock_i]            # (W, 5)
        )
        target = torch.tensor(self.target_returns[stock_i], dtype=torch.float32)

        return text_emb, price_win, target


# ---------------------------------------------------------------------------
# 6. Convenience: build DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(
    ticker: str = "AMD",
    window: int = 20,
    batch_size: int = 32,
    val_split: float = 0.2,
    device: str = "cpu",
    data_dir: Path = DATA_DIR,
    cache_embeddings: bool = True,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    End-to-end helper: load data → embed news → build train/val loaders.

    Returns (train_loader, val_loader, text_embed_dim).
    """
    from transformers import BertTokenizer, BertModel

    print(f"[data_loader] Loading data for {ticker} …")
    stock_df = load_stock_data(ticker, data_dir)
    news_df = load_news_data(ticker, data_dir)

    # ----- FinBERT embeddings (cached to disk for speed) -----
    cache_path = data_dir / f"{ticker}_finbert_embeddings.npy"
    cache_dates_path = data_dir / f"{ticker}_finbert_dates.npy"

    if cache_embeddings and cache_path.exists() and cache_dates_path.exists():
        print("[data_loader] Loading cached FinBERT embeddings …")
        embeddings = np.load(cache_path)
        news_dates = pd.to_datetime(np.load(cache_dates_path, allow_pickle=True))
    else:
        print("[data_loader] Downloading / loading FinBERT …")
        tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        bert_model = BertModel.from_pretrained("yiyanghkust/finbert-tone").to(device)

        texts = news_df["text"].tolist()
        embeddings = get_finbert_embeddings(
            texts, bert_model, tokenizer, device=device
        )
        news_dates = news_df["date"]

        if cache_embeddings:
            np.save(cache_path, embeddings)
            np.save(cache_dates_path, news_dates.values)
            print(f"[data_loader] Cached embeddings → {cache_path}")

        # Free BERT memory
        del bert_model
        if device != "cpu":
            torch.cuda.empty_cache()

    # ----- Dataset / split -----
    dataset = MultimodalDataset(
        embeddings=embeddings,
        stock_df=stock_df,
        news_dates=news_dates.tolist(),
        window=window,
    )
    print(f"[data_loader] Dataset size: {len(dataset)} samples")

    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    # Chronological split (no shuffle) — take first n_train as train
    train_ds = torch.utils.data.Subset(dataset, list(range(n_train)))
    val_ds = torch.utils.data.Subset(dataset, list(range(n_train, n_train + n_val)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, embeddings.shape[1]  # 768
