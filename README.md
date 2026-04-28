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
