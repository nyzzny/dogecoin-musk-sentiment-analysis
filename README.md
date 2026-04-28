# Dogecoin Price Volatility and Social Media Sentiment Analysis
### An Empirical Study Based on Elon Musk's Tweets and Reddit Community Sentiment (2017–2024)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **In a nutshell:** Musk's high‑impact tweets produce a statistically significant ~10% cumulative abnormal return on the event day, yet all daily‑frequency trading strategies built on these signals fail out‑of‑sample.

---

## Key Findings

- 📈 **Event Study**: Market‑adjusted Cumulative Abnormal Return (CAR) reaches **10.00%** on the event day (t=0). Block bootstrap (10,000 iterations) one‑sided **p = 0.0018**, confirming a robust "Musk Effect".
- 🔁 **Granger Causality**: Musk sentiment shows only marginal predictive power for returns (p < 0.1 at lag 3), while **returns significantly Granger‑cause Musk sentiment** (p < 0.1 at lags 3–4) – reverse causality is stronger. Reddit sentiment has no predictive power in either direction.
- 📉 **Out‑of‑Sample Backtesting**: Three progressively stricter trading strategies (simple hold, SMA200 filter, composite signal) **all lose money** in the 2022–2024 test period, with total returns ranging from **–0.99% to –53.17%** (after 0.5% one‑way friction). The benchmark buy‑and‑hold gained between +0.90% and +2.41%.
- 🔍 **Limitations**: Daily granularity hides intraday shocks; VADER cannot read visual memes (e.g., a photo of Musk’s dog caused a 5.4% DOGE rally that text sentiment missed); real‑world slippage likely far exceeds modelled costs.

---

## Repository Structure

├── data/
│ └── README.md # Data source & variable descriptions
├── code/
│ ├── Data_Preparation.py
│ ├── Analysis.py
│ ├── Event_Study.py
│ ├── Backtest_Simple.py
│ ├── Backtest_Filtered.py
│ └── Backtest_Composite.py
├── figures/
│ ├── time_series_overview.png
│ ├── correlation_heatmap.png
│ ├── event_study_high_impact.png
│ ├── event_study_market_model_bootstrap.png
│ ├── strategy_backtest_out_of_sample.png
│ ├── filtered_strategy.png
│ ├── composite_strategy_out_of_sample.png
│ └── data_pipeline_flowchart.png


---

## Data Sources

| Data | Source | Description |
|------|--------|-------------|
| Dogecoin Price | Yahoo Finance (`yfinance`) | DOGE‑USD daily OHLCV, 2017-11-10 to 2024-04-12 (2,346 trading days) |
| Elon Musk Tweets | [Kaggle Dataset](https://www.kaggle.com/code/mehmetutkubala/elon-musk-tweets-sentiment-classified-via-roberta/input) (Bala, 2022) | Archive of ~54k tweets; filtered to 67 high‑impact event days |
| Reddit Posts | Pushshift API (`r/dogecoin`) | ~1.13M posts; daily aggregated post counts & sentiment |
| Bitcoin Price | Yahoo Finance (`yfinance`) | BTC‑USD daily returns (market proxy for the event study) |

The final merged dataset (`Master_Analysis_Dataset_Daily_v2_FIXED.csv`) contains 2,346 rows and 10 variables. On days with zero Reddit posts, `reddit_avg_sentiment` is set to `NaN` to avoid false neutrality.

---

## Methodology Overview

### 1. Sentiment Analysis
- **Tool**: VADER with a custom `CRYPTO_LEXICON` (`moon`, `hodl`, `🚀`, `rekt`, etc.)
- **Output**: Daily mean `compound` sentiment for Musk tweets and Reddit posts separately

### 2. Event Study
- **Event**: Days with ≥1 high‑impact tweet (Doge‑related, likes > 100k)
- **Market model**: Bitcoin returns as the market proxy; estimation window [-200, -11]
- **Abnormal Return**: `AR_t = R_t - (α + β·R_BTC,t)`, event window [-5, +15]
- **Inference**: Block bootstrap (10,000 iterations) for 95% confidence intervals; overlapping events removed

### 3. Granger Causality
- Bivariate VAR models, lags 1–5

### 4. Out‑of‑Sample Trading Strategies
- **Split**: Training (2017–2021) / Testing (2022–2024)
- **Cost**: 0.5% one‑way friction
- **Strategy 1 – Simple Hold**: Buy at close on signal, hold 14 days (optimised)
- **Strategy 2 – SMA200 Filter**: Only enter if price > 200‑day SMA
- **Strategy 3 – Composite Signal**: Requires (a) high‑impact tweet, (b) Reddit Z > 1.5, (c) BTC return > 0

---

## Scripts Description

| Script | Purpose |
|--------|---------|
| `1_Data_Preparation_FIXED.py` | Downloads price data, cleans tweets/Reddit, applies sentiment, aggregates daily, exports the master dataset |
| `2_EDA.py` | Time‑series plot, correlation heatmap, extreme‑return analysis, initial raw‑return event study |
| `3_Event_Study.py` | Market‑model event study with block bootstrap; outputs Figures 4A/4B and CAR significance |
| `4_Backtest_Simple.py` | Simple event‑driven holding strategy backtest |
| `4_Backtest_Filtered.py` | SMA200 trend‑filtered strategy backtest |
| `4_Backtest_Composite.py` | Composite signal strategy backtest |

---

## Key Figures

| Figure | Description |
|--------|-------------|
| `time_series_overview.png` | Four panels: log price, daily returns, Musk tweet count, Reddit post count |
| `correlation_heatmap.png` | Correlation matrix – reveals the weak‑signal environment |
| `event_study_high_impact.png` | Raw return event study (unadjusted) |
| `event_study_market_model_bootstrap.png` | Market‑model adjusted AR and CAR with bootstrap CI – **core result** |
| `strategy_backtest_out_of_sample.png` | Simple hold strategy: equity curve, drawdown, trade markers |
| `filtered_strategy.png` | SMA200 filtered strategy: equity curve |
| `composite_strategy_out_of_sample.png` | Composite signal strategy: equity curve |

---

## Performance Summary

| Strategy | Trades | Total Return | Benchmark Return | Win Rate | Max Drawdown |
|----------|--------|--------------|------------------|----------|--------------|
| Simple Hold (14d) | 26 | –47.32% | +2.41% | 42.31% | –60.76% |
| SMA200 Filter (20d) | 10 | –53.17% | +0.90% | 10.0% | –55.57% |
| Composite Signal (4d) | 1 | –0.99% | +0.90% | 0.0% | –7.54% |

*The composite strategy triggered only once in the 28‑month test period (14 Jan 2022 – Tesla merchandise announcement), losing –10.34% on that single trade.*

---

## Setup & Reproduction

### Prerequisites
- Python 3.8+
- Packages listed in `code/requirements.txt`

### Installation
```bash
git clone https://github.com/nyzzny/dogecoin-musk-sentiment-analysis.git
cd dogecoin-musk-sentiment-analysis
pip install -r code/requirements.txt
