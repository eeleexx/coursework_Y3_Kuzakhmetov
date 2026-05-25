"""
model.py — Multimodal Causal Model for disentangled analysis of
information flow on stock prices.

Architecture
------------
1. TextEncoder        — projection head on top of pre-computed FinBERT [CLS]
2. TemporalEncoder    — 2-layer LSTM over OHLCV sliding windows
3. DisentanglementModule — shared / private subspace projections
4. MoCoHead           — Momentum Contrastive Learning on shared subspaces
5. GatedFusionLayer   — Reliability-aware adaptive fusion
6. MultimodalCausalModel — top-level model tying everything together

Implements the "Headline vs. Content" hypothesis:
  • *Shared* latent subspace captures market-invariant *content* signals
    that are correlated across text and prices.
  • *Private* subspace captures modality-specific *noise* (headline hype,
    microstructure noise).
  • The Gated Fusion layer learns to down-weight text when the private
    (noise) component is large, implementing reliability-aware fusion.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# 1. Text Encoder  (operates on *pre-computed* FinBERT embeddings)
# ═══════════════════════════════════════════════════════════════════════════

class TextEncoder(nn.Module):
    """
    Lightweight projection head that maps pre-computed FinBERT [CLS]
    embeddings (768-d) into the model's latent dimension.

    FinBERT itself is frozen and only used at data-loading time (see
    data_loader.py).  This keeps training fast and memory-efficient.
    """

    def __init__(self, input_dim: int = 768, latent_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 768) → (B, latent_dim)"""
        return self.projection(x)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Temporal Encoder  (LSTM over OHLCV windows)
# ═══════════════════════════════════════════════════════════════════════════

class TemporalEncoder(nn.Module):
    """
    2-layer LSTM that encodes a sliding window of normalised OHLCV
    features into a fixed-size latent vector.
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 2,
        latent_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, W, 5) → (B, latent_dim)"""
        _, (h_n, _) = self.lstm(x)          # h_n: (num_layers, B, hidden)
        last_hidden = h_n[-1]               # (B, hidden)
        return self.projection(last_hidden)  # (B, latent_dim)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Disentanglement Module
# ═══════════════════════════════════════════════════════════════════════════

class DisentanglementModule(nn.Module):
    """
    Projects an encoder output into two orthogonal subspaces:
      • **Shared** — market-invariant features (content signal)
      • **Private** — modality-specific features (noise)

    Orthogonality is enforced via a *difference loss*:
        L_diff = ‖S^T · P‖_F^2
    where S and P are the batch matrices of shared / private vectors.
    """

    def __init__(self, input_dim: int = 128, subspace_dim: int = 64):
        super().__init__()
        self.shared_proj = nn.Sequential(
            nn.Linear(input_dim, subspace_dim),
            nn.LayerNorm(subspace_dim),
            nn.GELU(),
        )
        self.private_proj = nn.Sequential(
            nn.Linear(input_dim, subspace_dim),
            nn.LayerNorm(subspace_dim),
            nn.GELU(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, input_dim)
        Returns (shared, private, diff_loss)
        """
        shared = self.shared_proj(x)    # (B, subspace_dim)
        private = self.private_proj(x)  # (B, subspace_dim)

        # Normalised decorrelation loss.  A raw Frobenius penalty can be
        # minimised by collapsing one subspace to zero, so we centre and
        # normalise rows before measuring cross-subspace correlation.
        shared_norm = F.normalize(shared - shared.mean(dim=0, keepdim=True), dim=1)
        private_norm = F.normalize(private - private.mean(dim=0, keepdim=True), dim=1)
        cross_corr = torch.mm(shared_norm.t(), private_norm)
        diff_loss = torch.mean(cross_corr**2)

        return shared, private, diff_loss


# ═══════════════════════════════════════════════════════════════════════════
# 4. Momentum Contrastive (MoCo) Head
# ═══════════════════════════════════════════════════════════════════════════

class MoCoHead(nn.Module):
    """
    Simplified MoCo objective applied to the *shared* subspaces of
    both modalities.

    Goal: maximise agreement between shared_text and shared_price for
    the same sample while contrasting against a momentum queue of
    negatives.

    Uses InfoNCE loss:
        L = -log( exp(q·k+ / τ) / Σ exp(q·k / τ) )
    """

    def __init__(
        self,
        dim: int = 64,
        queue_size: int = 256,
        momentum: float = 0.999,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.dim = dim
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        # Small MLP projectors for query / key
        self.q_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.k_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        # Initialise key projector from query projector
        for param_q, param_k in zip(
            self.q_proj.parameters(), self.k_proj.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Queue of negative keys
        self.register_buffer("queue", torch.randn(dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self):
        for param_q, param_k in zip(
            self.q_proj.parameters(), self.k_proj.parameters()
        ):
            param_k.data = (
                self.momentum * param_k.data + (1.0 - self.momentum) * param_q.data
            )

    @torch.no_grad()
    def _enqueue(self, keys: torch.Tensor):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        remaining = self.queue_size - ptr

        if batch_size <= remaining:
            self.queue[:, ptr : ptr + batch_size] = keys.T
        else:
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, : batch_size - remaining] = keys[remaining:].T

        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def forward(
        self, shared_text: torch.Tensor, shared_price: torch.Tensor
    ) -> torch.Tensor:
        """
        shared_text:  (B, dim) — query
        shared_price: (B, dim) — positive key

        Returns InfoNCE loss (scalar).
        """
        q = F.normalize(self.q_proj(shared_text), dim=1)   # (B, dim)

        with torch.no_grad():
            self._momentum_update()
            k = F.normalize(self.k_proj(shared_price), dim=1)  # (B, dim)

        # Positive logits: (B, 1)
        l_pos = torch.einsum("bd,bd->b", q, k).unsqueeze(1)
        # Negative logits: (B, queue_size)
        l_neg = torch.einsum("bd,dk->bk", q, self.queue.clone().detach())

        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature  # (B, 1+K)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)

        # Update queue
        self._enqueue(k)

        return loss


# ═══════════════════════════════════════════════════════════════════════════
# 5. Gated Fusion Layer  (Reliability-Aware Adaptive Fusion)
# ═══════════════════════════════════════════════════════════════════════════

class GatedFusionLayer(nn.Module):
    """
    Adaptive fusion inspired by AMFNet.

    Computes a confidence gate g ∈ [0, 1] from the *concatenation* of
    shared_text and private_text.  When the private (noise) component
    is large, the gate learns to down-weight textual input:

        g = σ(W · [shared_text ‖ private_text] + b)
        fused = g · shared_text + (1 − g) · shared_price

    A final MLP maps the fused representation → scalar prediction.
    """

    def __init__(self, subspace_dim: int = 64):
        super().__init__()
        # Gate
        self.gate = nn.Sequential(
            nn.Linear(subspace_dim * 2, subspace_dim),
            nn.ReLU(),
            nn.Linear(subspace_dim, 1),
            nn.Sigmoid(),
        )
        # Predictor head
        self.predictor = nn.Sequential(
            nn.Linear(subspace_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        shared_text: torch.Tensor,
        shared_price: torch.Tensor,
        private_text: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (prediction, gate_value).
        prediction: (B, 1) — predicted next-day return
        gate_value: (B, 1) — textual confidence ∈ [0, 1]
        """
        gate_input = torch.cat([shared_text, private_text], dim=1)
        g = self.gate(gate_input)  # (B, 1)

        fused = g * shared_text + (1 - g) * shared_price  # (B, subspace_dim)
        prediction = self.predictor(fused)                 # (B, 1)

        return prediction, g


# ═══════════════════════════════════════════════════════════════════════════
# 6. Top-Level Model
# ═══════════════════════════════════════════════════════════════════════════

class MultimodalCausalModel(nn.Module):
    """
    Full multimodal architecture:

        FinBERT [CLS]  ──► TextEncoder ──► DisentanglementModule(text)
                                                  │  shared_text
                                                  │  private_text
                                                  ▼
        OHLCV window ──► TemporalEncoder ──► DisentanglementModule(price)
                                                  │  shared_price
                                                  │  private_price
                                                  ▼
                              MoCoHead(shared_text, shared_price)  → L_moco
                              GatedFusionLayer(shared_text, shared_price,
                                               private_text)       → prediction

    Loss = MSE(prediction, target)
         + λ_diff  · (diff_loss_text + diff_loss_price)
         + λ_moco  · moco_loss
    """

    def __init__(
        self,
        text_input_dim: int = 768,
        price_input_dim: int = 5,
        latent_dim: int = 128,
        subspace_dim: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        moco_queue_size: int = 256,
        lambda_diff: float = 0.01,
        lambda_moco: float = 0.1,
    ):
        super().__init__()
        self.lambda_diff = lambda_diff
        self.lambda_moco = lambda_moco

        # Encoders
        self.text_encoder = TextEncoder(text_input_dim, latent_dim)
        self.temporal_encoder = TemporalEncoder(
            price_input_dim, lstm_hidden, lstm_layers, latent_dim
        )

        # Disentanglement
        self.text_disentangle = DisentanglementModule(latent_dim, subspace_dim)
        self.price_disentangle = DisentanglementModule(latent_dim, subspace_dim)

        # Contrastive head
        self.moco = MoCoHead(subspace_dim, moco_queue_size)

        # Fusion + prediction
        self.fusion = GatedFusionLayer(subspace_dim)

    def forward(
        self,
        text_emb: torch.Tensor,
        price_window: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        text_emb     : (B, 768)   pre-computed FinBERT embedding
        price_window : (B, W, 5)  normalised OHLCV window
        target       : (B,)       next-day return (optional, for loss)

        Returns
        -------
        prediction : (B, 1)
        loss_dict  : dict with keys 'mse_loss', 'diff_loss', 'moco_loss',
                     'total_loss', 'gate'
        """
        # ---- Encode ----
        h_text = self.text_encoder(text_emb)           # (B, latent)
        h_price = self.temporal_encoder(price_window)  # (B, latent)

        # ---- Disentangle ----
        s_text, p_text, diff_text = self.text_disentangle(h_text)
        s_price, p_price, diff_price = self.price_disentangle(h_price)

        # ---- MoCo contrastive loss ----
        moco_loss = self.moco(s_text, s_price)

        # ---- Gated fusion → prediction ----
        prediction, gate = self.fusion(s_text, s_price, p_text)

        # ---- Loss computation ----
        loss_dict: Dict[str, torch.Tensor] = {}
        diff_loss = diff_text + diff_price

        if target is not None:
            mse_loss = F.mse_loss(prediction.squeeze(-1), target)
            total_loss = (
                mse_loss
                + self.lambda_diff * diff_loss
                + self.lambda_moco * moco_loss
            )
            loss_dict["mse_loss"] = mse_loss.detach()
            loss_dict["diff_loss"] = diff_loss.detach()
            loss_dict["moco_loss"] = moco_loss.detach()
            loss_dict["total_loss"] = total_loss
            loss_dict["gate"] = gate.mean().detach()
        else:
            loss_dict["gate"] = gate.mean().detach()

        return prediction, loss_dict
