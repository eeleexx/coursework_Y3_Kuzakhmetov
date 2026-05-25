# Multimodal Model for Causal Analysis of Information Flow on Stock Prices

HSE University, 3rd-Year Research Project, Kuzakhmetov I.R.

This repository contains a PyTorch-based research prototype for interpretable multimodal analysis of financial information flow and stock-price reactions. The project combines financial news, daily OHLCV data, FinBERT text embeddings, LSTM price encoders, disentangled shared/private representations, contrastive alignment, and reliability-aware gated fusion.

The final empirical scope is intentionally conservative: the project is a daily equity/news event-study with an additional SPY 1-minute social information-flow extension. It does not claim to be a full high-frequency trading or order-book system.

## Architecture Overview

```text
FinBERT [CLS] -> TextEncoder -> DisentanglementModule
                                    | shared text content
                                    | private text noise
                                    v
OHLCV window  -> LSTM Encoder -> DisentanglementModule
                                    | shared price content
                                    | private price noise
                                    v
MoCo contrastive alignment between shared text and shared price
                                    v
Reliability-aware gated fusion -> next-return prediction
```

The main modelling idea is to separate information that is shared across text and price dynamics from modality-specific noise. The shared subspace is treated as a proxy for content-like information, while the private subspace captures headline noise or price-only variation. A gated fusion layer then estimates how much the model should rely on text for a given event.

## Project Structure

```text
coursework_Y3/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── stock_data.csv
│   ├── AMD_news.csv, INTC_news.csv, ...
│   ├── *_finbert_embeddings.npy
│   └── *_finbert_dates.npy
├── figures/
│   └── EDA figures used in the report
├── results/
│   ├── baseline and ablation outputs
│   ├── latent diagnostics
│   ├── EventLens HTML dashboards
│   ├── synthetic validation outputs
│   └── SPY intraday study outputs
├── tex/
│   ├── report_final.tex
│   └── report_final.pdf
├── data_loader.py
├── model.py
├── train_prototype.py
├── experiments.py
├── latent_analysis.py
├── synthetic_experiment.py
├── build_event_lens.py
├── hypothesis_tests.py
├── generate_eda_plots.py
├── spy_intraday_experiment.py
├── spy_event_study.py
├── spy_response_curve.py
└── spy_reaction_classifier.py
```

The authoritative LaTeX source is `tex/report_final.tex`. There is intentionally no duplicate report source in the repository root.

## Data Notes

The lightweight daily equity/news data and cached FinBERT embeddings are intended to be committed with the repository. Two raw SPY tweet/intraday CSV files are intentionally ignored by `.gitignore` because they are too large for normal Git remotes:

```text
data/SPY_concat_files.csv
data/spy_1min_tweet_price_dataset.csv
```

These files are required only for rerunning the SPY intraday extension. If they are not present, the daily equity/news experiments and report source still remain available.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Main Experiments

```bash
# Prototype training for one ticker
python train_prototype.py --ticker AMD --epochs 5

# Baselines and ablations
python experiments.py --epochs 5

# Latent diagnostics and EventLens dashboards
python latent_analysis.py --ticker AMD --epochs 5
python build_event_lens.py --ticker AMD

# Controlled synthetic validation
python synthetic_experiment.py --epochs 8

# Statistical hypothesis-test summary
python hypothesis_tests.py
```

## SPY Intraday Extension

The SPY scripts require the ignored raw SPY CSV files listed above.

```bash
python spy_intraday_experiment.py --epochs 3 --max_samples 30000
python spy_event_study.py
python spy_response_curve.py
python spy_reaction_classifier.py
```

This extension is framed as a social information-flow and reaction-risk study. The main result is that tweet attention is associated with larger absolute 5-minute reactions and higher volume, while directional predictability remains weak.

## Report Compilation

Compile the report from the repository root so that relative paths to `figures/` and `results/` resolve correctly:

```bash
latexmk -g -pdf -interaction=nonstopmode -halt-on-error -outdir=tex tex/report_final.tex
```

The compiled report is written to:

```text
tex/report_final.pdf
```

## Expected Outputs

Key generated outputs include:

```text
results/baseline_metrics.csv
results/baseline_summary.csv
results/baseline_metrics.png
results/latent_proxy_summary.csv
results/event_lens_<TICKER>.html
results/synthetic_metrics.csv
results/synthetic_metrics.png
results/hypothesis_tests.csv
results/spy_event_study_summary.csv
results/spy_event_study_buckets.png
results/spy_response_curve.csv
results/spy_response_curve.png
results/spy_reaction_classifier.csv
results/spy_reaction_classifier.png
tex/report_final.pdf
```

## Validation Status

Phase 1, the dual-stream disentangled encoder with MoCo-style contrastive regularisation, is implemented.

Phase 2, reliability-aware adaptive fusion through a gated fusion layer, is implemented.

Phase 3, interpretability and validation, is implemented through baseline comparisons, latent predictive proxies, controlled synthetic validation, hypothesis-test summaries, SPY intraday event studies, and EventLens dashboards.

## Current Empirical Snapshot

The six-ticker daily experiment compares price-only LSTM, text-only FinBERT, static fusion, and full DRL models across AMD, INTC, PFE, JNJ, BAC, and JPM. In the current run, the full DRL model has the best mean MSE and MAE and wins the MSE comparison on four out of six tickers. The report states the statistical limitations explicitly: the paired MSE improvement over static fusion is not significant at the 95% level.

The synthetic validation supports the content/noise distinction under controlled ground-truth factors. The SPY intraday extension supports an attention/reaction-risk interpretation rather than a directional trading-alpha claim.

## Repository

https://github.com/eeleexx/coursework_Y3_Kuzakhmetov
