# Finalisation Plan — Daily Validation + SPY Social Information Flow

## Current Status (24 May 2026)

| Component | Status | Evidence |
|---|---|---|
| Phase 1: Disentangled encoders | Complete | `model.py`, `train_prototype.py` |
| Phase 2: Reliability-aware fusion | Complete | `GatedFusionLayer`, gate diagnostics |
| Ablation study | Complete for all 6 tickers | `results/baseline_metrics.csv`, `results/baseline_summary.csv` |
| Latent content/noise analysis | Complete for all 6 tickers | `results/latent_events_<TICKER>.csv`, `results/latent_proxy_summary.csv` |
| Controlled synthetic validation | Complete | `results/synthetic_metrics.csv` |
| EventLens demo | Complete | `results/event_lens_AMD.html` |
| SPY intraday social information-flow study | Complete | `results/spy_event_study_summary.csv`, `results/spy_response_curve.csv`, `results/spy_reaction_classifier.csv` |
| Full HFT/order-book validation | Deferred | Requires timestamped news/order-book source |

## Final Coursework Scope

The defensible final scope is:

> An interpretable multimodal daily/event-study model that separates market-relevant news content from headline-driven noise, validates the architecture through ablations, latent predictive proxies, and a controlled simulation, and adds a SPY 1-minute social information-flow study based on tweet attention and sentiment.

This avoids overstating daily OHLCV results as HFT while still preserving the original research direction.

## Remaining Work Before Submission

### 1. Report Integration

- Add the ablation table from `results/baseline_summary.csv` and ticker-level details from `results/baseline_metrics.csv`.
- Add the latent proxy table from `results/latent_proxy_summary.csv`.
- Add `results/gate_timeline_AMD.png` and `results/content_noise_scatter_AMD.png`.
- Add the synthetic validation table/figure from `results/synthetic_metrics.csv` and `results/synthetic_metrics.png`.
- Add a short EventLens case-study section referencing `results/event_lens_AMD.html`.
- Add the SPY intraday event-study table/figures from `results/spy_event_study_summary.csv`, `results/spy_event_study_buckets.png`, `results/spy_response_curve.png`, `results/spy_reaction_classifier.png`, and `results/spy_top_event_windows.png`.

### 2. Claims to Keep

- The model trains end-to-end on real financial news and OHLCV data.
- Full DRL is competitive with unimodal/static baselines, has the best mean MSE/MAE, and wins MSE on 4 of 6 tickers in the current run.
- Learned shared/content features add predictive signal in the latent proxy for several tickers, especially AMD and BAC; noise features are weaker on average but not uniformly zero.
- Controlled simulation supports the architecture's ability to separate content from headline noise when ground truth is known.
- SPY intraday data shows that tweet/social-information-flow intensity is associated with larger absolute 5-minute reactions and much higher trading volume.
- Tweet-only features identify top-decile 5-minute reaction regimes above the base rate; fusion slightly improves average precision and top-ranked alert precision over price-only features.

### 3. Claims to Avoid

- Do not claim full HFT validation.
- Do not claim formal causal proof.
- Do not claim SHAP was implemented unless it is added later.
- Do not overinterpret paper-trading Sharpe because the validation set is small and transaction costs are ignored.
- Do not claim robust SPY directional alpha: the intraday signal is stronger for reaction/volume than for direction.

### 4. Intraday / SPY Study

Possible future extensions:

| Option | Use | Risk |
|---|---|---|
| SPY 1-minute + tweets | Already available as social information-flow study | Directional predictability is weak |
| Seeking Alpha RapidAPI parser | News collection for SPY | API limits, timestamp quality |
| LOBSTER / Refinitiv / Bloomberg | True HFT validation | Licence/access barrier |
| Synthetic intraday process | Controlled stress test | Not empirical market evidence |

Current intraday target: **SPY, 2020--2021 tweets + 1-minute bars**. Expand only if a timestamped news/order-book source becomes available.

## Priority Order

1. Finish final report sections.
2. Polish EventLens screenshots/case-study narrative.
3. Polish EventLens screenshots/case-study narrative for one or two tickers.
4. Keep SPY claims focused on social attention, reaction intensity, and reaction-risk classification.
