#!/usr/bin/env python3
"""
Build a self-contained EventLens HTML report from latent diagnostics.

The report is intentionally static so it can be opened without Streamlit,
Plotly, or internet access.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd

from data_loader import load_stock_data


def parse_args():
    parser = argparse.ArgumentParser(description="Build static EventLens report")
    parser.add_argument("--ticker", default="AMD")
    parser.add_argument("--events", default=None)
    parser.add_argument("--metrics", default="results/baseline_metrics.csv")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def fmt_pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def bar(width: float, color: str) -> str:
    width = max(0.0, min(100.0, width))
    return f"<span class='bar'><span style='width:{width:.1f}%;background:{color}'></span></span>"


def price_svg(stock: pd.DataFrame, date: str, window: int = 10) -> str:
    stock = stock.sort_values("Date").reset_index(drop=True)
    target = pd.to_datetime(date)
    matches = stock.index[stock["Date"] == target].tolist()
    if not matches:
        return "<div class='sparkline empty'>No price window</div>"
    idx = matches[0]
    start = max(0, idx - window)
    end = min(len(stock), idx + window + 2)
    subset = stock.iloc[start:end].copy()
    closes = subset["Close"].astype(float).to_numpy()
    if len(closes) < 2:
        return "<div class='sparkline empty'>No price window</div>"
    xs = [i * 100 / (len(closes) - 1) for i in range(len(closes))]
    lo, hi = float(closes.min()), float(closes.max())
    ys = [70 - ((float(v) - lo) / (hi - lo + 1e-12)) * 55 for v in closes]
    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))
    event_x = xs[idx - start]
    first = closes[0]
    labels = f"{subset.iloc[0]['Date'].date()} -> {subset.iloc[-1]['Date'].date()}"
    return f"""
      <div class="sparkline">
        <svg viewBox="0 0 100 78" preserveAspectRatio="none" role="img">
          <line x1="{event_x:.2f}" y1="6" x2="{event_x:.2f}" y2="74" class="event-line" />
          <polyline points="{points}" />
        </svg>
        <small>{html.escape(labels)} · close {first:.2f} -> {closes[-1]:.2f}</small>
      </div>
    """


def event_card(row: pd.Series, rank_label: str, stock: pd.DataFrame) -> str:
    content = float(row["text_content_norm"])
    noise = float(row["text_noise_norm"])
    total = content + noise + 1e-12
    content_share = content / total * 100
    noise_share = noise / total * 100
    gate = float(row["gate_text_confidence"])
    pred = float(row["predicted_return"])
    price_only = float(row.get("counterfactual_price_only_return", 0.0))
    text_only = float(row.get("counterfactual_text_only_return", 0.0))
    actual = float(row["target_return"])
    verdict = "trusted text" if gate >= 0.5 else "price-dominant"
    if abs(pred - actual) < 0.01:
        accuracy = "close"
    elif pred * actual > 0:
        accuracy = "right direction"
    else:
        accuracy = "missed direction"

    return f"""
    <article class="event-card">
      <div class="event-meta">{html.escape(rank_label)} · {html.escape(str(row['date']))}</div>
      <h3>{html.escape(str(row['title']))}</h3>
      <p>{html.escape(str(row.get('summary', '')))[:360]}</p>
      <div class="chips">
        <span>{verdict}</span>
        <span>{accuracy}</span>
        <span>gate {gate:.2f}</span>
      </div>
      {price_svg(stock, str(row['date']))}
      <div class="metric-grid">
        <div><b>Predicted</b><strong>{fmt_pct(pred)}</strong></div>
        <div><b>Actual T+1</b><strong>{fmt_pct(actual)}</strong></div>
        <div><b>No text</b><strong>{fmt_pct(price_only)}</strong></div>
        <div><b>Text only</b><strong>{fmt_pct(text_only)}</strong></div>
      </div>
      <label>Content share {content_share:.1f}%</label>
      {bar(content_share, "#2a9d8f")}
      <label>Noise share {noise_share:.1f}%</label>
      {bar(noise_share, "#e76f51")}
    </article>
    """


def metrics_table(metrics: pd.DataFrame, ticker: str) -> str:
    subset = metrics[metrics["ticker"] == ticker].copy()
    if subset.empty:
        subset = metrics.copy()
    rows = []
    for _, r in subset.iterrows():
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(r['model']))}</td>"
            f"<td>{float(r['mse']):.6f}</td>"
            f"<td>{float(r['mae']):.6f}</td>"
            f"<td>{float(r['directional_accuracy']):.3f}</td>"
            f"<td>{float(r['information_coefficient']):+.3f}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def main():
    args = parse_args()
    ticker = args.ticker
    events_path = Path(args.events or f"results/latent_events_{ticker}.csv")
    metrics_path = Path(args.metrics)
    out_path = Path(args.out or f"results/event_lens_{ticker}.html")

    events = pd.read_csv(events_path)
    metrics = pd.read_csv(metrics_path)
    stock = load_stock_data(ticker)

    biggest_reactions = events.sort_values("abs_target_return", ascending=False).head(6)
    content_heavy = events.sort_values("text_content_norm", ascending=False).head(4)
    noise_heavy = events.sort_values("text_noise_ratio", ascending=False).head(4)

    avg_gate = events["gate_text_confidence"].mean()
    avg_noise = events["text_noise_ratio"].mean()
    best_model = (
        metrics[metrics["ticker"] == ticker]
        .sort_values("mse", ascending=True)
        .iloc[0]["model"]
    )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EventLens · {html.escape(ticker)}</title>
  <style>
    :root {{
      --ink: #16211f;
      --muted: #64706d;
      --paper: #f7f1e5;
      --card: #fffaf0;
      --green: #2a9d8f;
      --orange: #e76f51;
      --blue: #264653;
      --line: rgba(22, 33, 31, .14);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 12% 18%, rgba(231, 111, 81, .18), transparent 28rem),
        radial-gradient(circle at 84% 4%, rgba(42, 157, 143, .18), transparent 24rem),
        linear-gradient(135deg, #fbf7ed, var(--paper));
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 48px 22px 72px; }}
    header {{ display: grid; grid-template-columns: 1.4fr .9fr; gap: 28px; align-items: end; }}
    h1 {{ font-size: clamp(48px, 8vw, 110px); line-height: .9; margin: 0; letter-spacing: -0.06em; }}
    h2 {{ font-size: 30px; margin: 48px 0 18px; }}
    h3 {{ font-size: 22px; margin: 8px 0 10px; line-height: 1.12; }}
    p {{ color: var(--muted); line-height: 1.55; }}
    .hero-note {{
      border-left: 5px solid var(--orange);
      padding-left: 18px;
      font-size: 18px;
    }}
    .kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 34px 0; }}
    .kpi, .event-card, .panel {{
      background: rgba(255, 250, 240, .78);
      border: 1px solid var(--line);
      box-shadow: 0 18px 45px rgba(38, 70, 83, .08);
      border-radius: 24px;
      padding: 20px;
      backdrop-filter: blur(8px);
    }}
    .kpi b {{ display: block; color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: .08em; }}
    .kpi strong {{ display: block; font-size: 30px; margin-top: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .event-meta {{ color: var(--orange); text-transform: uppercase; font-size: 12px; letter-spacing: .12em; font-weight: 700; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 16px; }}
    .chips span {{ border: 1px solid var(--line); border-radius: 999px; padding: 6px 10px; background: white; font-size: 13px; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 12px 0 16px; }}
    .metric-grid div {{ border-top: 1px solid var(--line); padding-top: 8px; }}
    .metric-grid b {{ display: block; color: var(--muted); font-size: 12px; }}
    .metric-grid strong {{ font-size: 19px; }}
    label {{ display: block; font-size: 12px; color: var(--muted); margin: 10px 0 4px; }}
    .bar {{ display: block; height: 10px; background: rgba(38, 70, 83, .12); border-radius: 999px; overflow: hidden; }}
    .bar span {{ display: block; height: 100%; border-radius: 999px; }}
    table {{ width: 100%; border-collapse: collapse; background: rgba(255,255,255,.42); border-radius: 18px; overflow: hidden; }}
    th, td {{ padding: 12px 14px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
    .image-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    img {{ width: 100%; border-radius: 20px; border: 1px solid var(--line); background: white; }}
    .sparkline {{ margin: 12px 0 14px; padding: 10px; background: rgba(255,255,255,.52); border-radius: 16px; border: 1px solid var(--line); }}
    .sparkline svg {{ width: 100%; height: 92px; display: block; }}
    .sparkline polyline {{ fill: none; stroke: var(--blue); stroke-width: 2.3; vector-effect: non-scaling-stroke; }}
    .sparkline .event-line {{ stroke: var(--orange); stroke-width: 1.5; stroke-dasharray: 4 3; vector-effect: non-scaling-stroke; }}
    .sparkline small {{ color: var(--muted); display: block; margin-top: 6px; }}
    @media (max-width: 860px) {{
      header, .grid, .kpis, .image-row {{ grid-template-columns: 1fr; }}
      .metric-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <div class="event-meta">Multimodal causal model demo</div>
      <h1>EventLens<br>{html.escape(ticker)}</h1>
    </div>
    <p class="hero-note">
      Static dashboard for the Headline vs Content hypothesis. It shows where the model
      treated news as market-relevant content and where it routed text into private noise.
    </p>
  </header>

  <section class="kpis">
    <div class="kpi"><b>Validation events</b><strong>{len(events)}</strong></div>
    <div class="kpi"><b>Average text gate</b><strong>{avg_gate:.2f}</strong></div>
    <div class="kpi"><b>Average noise ratio</b><strong>{avg_noise:.2f}</strong></div>
    <div class="kpi"><b>Best MSE model</b><strong>{html.escape(str(best_model))}</strong></div>
  </section>

  <section class="panel">
    <h2>Baseline Comparison</h2>
    <table>
      <thead><tr><th>Model</th><th>MSE</th><th>MAE</th><th>Direction</th><th>IC</th></tr></thead>
      <tbody>{metrics_table(metrics, ticker)}</tbody>
    </table>
  </section>

  <h2>Largest Market Reactions</h2>
  <section class="grid">
    {"".join(event_card(row, f"reaction #{i+1}", stock) for i, (_, row) in enumerate(biggest_reactions.iterrows()))}
  </section>

  <h2>Most Content-Like Headlines</h2>
  <section class="grid">
    {"".join(event_card(row, f"content #{i+1}", stock) for i, (_, row) in enumerate(content_heavy.iterrows()))}
  </section>

  <h2>Most Noise-Heavy Headlines</h2>
  <section class="grid">
    {"".join(event_card(row, f"noise #{i+1}", stock) for i, (_, row) in enumerate(noise_heavy.iterrows()))}
  </section>

  <h2>Diagnostics</h2>
  <section class="image-row">
    <img src="gate_timeline_{html.escape(ticker)}.png" alt="Gate timeline">
    <img src="content_noise_scatter_{html.escape(ticker)}.png" alt="Content noise scatter">
  </section>
</main>
</body>
</html>
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    print(f"Saved EventLens -> {out_path}")


if __name__ == "__main__":
    main()
