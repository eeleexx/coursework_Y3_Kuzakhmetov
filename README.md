# Multimodal Model for Causal Analysis of Information Flow on Stock Prices

**HSE University — 3rd-Year Research Project (Kuzakhmetov I.R.)**

A PyTorch-based pipeline implementing Disentangled Representation Learning (DRL) to separate fundamental market signals from "headline noise" in financial news.

---

## Architecture Overview

```
FinBERT [CLS] ──► TextEncoder ──► DisentanglementModule
                                        │ shared (content)
                                        │ private (noise)
                                        ▼
OHLCV window  ──► TemporalEncoder ──► DisentanglementModule
                   (2-layer LSTM)       │ shared (content)
                                        │ private (noise)
                                        ▼
                MoCo contrastive loss (shared_text ↔ shared_price)
                                        ▼
                GatedFusionLayer ──► Prediction (next-day return)
```

### How It Implements the "Headline vs. Content" Hypothesis

The key insight is the **Disentanglement Module**, which projects each modality's features into two orthogonal subspaces:

| Subspace | Meaning | What it captures |
|---|---|---|
| **Shared** | Market-invariant | True economic *content* — signals correlated across both text and prices |
| **Private** | Modality-specific | *Headline noise* (text) or microstructure noise (prices) |

- An orthogonality loss (`diff_loss`) enforces that shared and private vectors are decorrelated.
- **MoCo (Momentum Contrastive Learning)** pulls the shared representations of text and prices together, ensuring the shared subspace captures genuinely correlated market signals.
- The **Gated Fusion Layer** estimates a confidence score for the textual modality. When the private (noise) component is large, the gate automatically down-weights text input.

### Design Decisions

| Decision | Rationale |
|---|---|
| **Frozen FinBERT** | The transformer is only used for embedding extraction at data-load time. Saves GPU memory; only the projection head is trained. |
| **LSTM** (not TCN) | Simpler to implement and debug for PoC; sufficient for capturing temporal dependencies in 20-day windows. |
| **MoCo queue = 256** | Balances diversity of negative samples with memory. Can be increased for larger datasets. |
| **Cached embeddings** | FinBERT embeddings are saved to `.npy` files after first extraction, making subsequent runs instant. |

---

## Project Structure

```
coursework_Y3/
├── data/                      # Raw CSV data
│   ├── stock_data.csv         # OHLCV for 6 tickers (2019–2025)
│   ├── AMD_news.csv           # News for each ticker
│   ├── INTC_news.csv
│   ├── ...
├── data_loader.py             # Data parsing, FinBERT embeddings, Dataset
├── model.py                   # Full model architecture
├── train_prototype.py         # PoC training script
├── experiments.py             # Baselines and ablation metrics
├── latent_analysis.py         # Gate/content/noise diagnostics
├── synthetic_experiment.py    # Controlled content-vs-noise simulation
├── build_event_lens.py        # Static HTML interpretability demo
├── spy_intraday_experiment.py # SPY 1-minute tweet/price model extension
├── spy_event_study.py         # SPY intraday attention/reaction analysis
├── requirements.txt           # Python dependencies
├── results/                   # Generated plots
│   └── loss_curve.png
└── overview.md                # Research project overview
```

---

## Installation & Usage

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the proof-of-concept training
python train_prototype.py --ticker AMD --epochs 5

# 4. Run baseline / ablation study
python experiments.py --tickers AMD INTC --epochs 5

# 5. Export latent diagnostics for EventLens
python latent_analysis.py --ticker AMD --epochs 5
python build_event_lens.py --ticker AMD

# 6. Run controlled synthetic validation
python synthetic_experiment.py --epochs 8

# 7. SPY intraday social information-flow study
python spy_intraday_experiment.py --epochs 3 --max_samples 30000
python spy_event_study.py
python spy_response_curve.py
python spy_reaction_classifier.py

# Optional: customise parameters
python train_prototype.py --ticker INTC --epochs 10 --lr 1e-3 --batch_size 64
```

### Expected Output

- Per-batch loss logging showing decreasing total loss
- Epoch summary table with MSE, diff, and MoCo loss components
- `results/loss_curve.png` — multi-panel plot of all loss components
- `results/baseline_metrics.csv/.png` — price-only, text-only, static-fusion, and full-DRL comparison
- `results/baseline_summary.csv` — aggregate metrics over all evaluated tickers
- `results/latent_events_AMD.csv` — per-event text gate, content norm, noise norm, and predictions
- `results/latent_proxy_summary.csv` — latent predictive proxy across tickers
- `results/event_lens_<TICKER>.html` — static interpretability dashboard with price windows and text/no-text counterfactuals
- `results/synthetic_metrics.csv/.png` — controlled validation where true content/noise factors are known
- `results/spy_intraday_metrics.csv/.png` — SPY 1-minute tweet/price modelling study
- `results/spy_event_study_summary.csv` and `results/spy_event_study_buckets.png` — intraday information-flow event study
- `results/spy_response_curve.csv/.png` — matched lead-lag response after tweet-attention spikes
- `results/spy_reaction_classifier.csv/.png` — top-decile 5-minute reaction classification

---

## Dependencies

- Python 3.10+
- PyTorch ≥ 2.0
- HuggingFace Transformers ≥ 4.30
- pandas, numpy, matplotlib, tqdm

---

## Phases Covered

- **Phase 1**: Dual-Stream Disentangled Encoder with MoCo contrastive regularisation ✅
- **Phase 2**: Reliability-Aware Adaptive Fusion (Gated Fusion Layer) ✅
- **Phase 3**: Interpretable validation via ablations, latent predictive proxy, synthetic ground-truth simulation, and EventLens dashboard ✅/ongoing

## Current Validation Snapshot

- **Six-ticker baselines**: compares price-only LSTM, text-only FinBERT, static fusion, and full DRL across AMD, INTC, PFE, JNJ, BAC, and JPM.
- **Aggregate performance**: full DRL has the best mean MSE/MAE and wins MSE on 4 of 6 tickers in the current run.
- **Latent proxy**: learned content/noise/gate diagnostics are exported for every ticker; AMD and BAC show especially clear content-feature gains over an AR(1) return proxy.
- **Controlled simulation**: predictions correlate strongly with true synthetic content and weakly with injected headline noise, supporting the content/noise separation hypothesis.
- **SPY intraday social information-flow study**: 1-minute SPY OHLCV plus tweet sentiment shows that social attention is associated with absolute 5-minute reaction and trading volume. Tweet-only features also identify high-reaction regimes above the base rate, while directional predictability remains weak. This is framed as an intraday event-study and reaction-risk task, not a full HFT/order-book claim.
