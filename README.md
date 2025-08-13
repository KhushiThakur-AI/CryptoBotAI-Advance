<!-- Banner (replace with your own image at docs/banner.png) -->
<p align="center">
  <img src="docs/banner.png" alt="CryptoBotAI-Advance banner" width="100%" />
</p>

<h1 align="center">CryptoBotAI‑Advance</h1>
<p align="center">AI‑assisted crypto trading bot with multi‑indicator signals, news & Reddit sentiment, Telegram controls, and risk management.</p>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue">
  <img alt="Last Commit" src="https://img.shields.io/github/last-commit/KhushiThakur-AI/CryptoBotAI-Advance">
  <img alt="Issues" src="https://img.shields.io/github/issues/KhushiThakur-AI/CryptoBotAI-Advance">
  <img alt="Stars" src="https://img.shields.io/github/stars/KhushiThakur-AI/CryptoBotAI-Advance">
</p>

---

## Table of Contents
- [Features](#features)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Keywords (SEO)](#keywords-seo)

---

## Features
- **Multi‑indicator strategy:** RSI, EMA, MACD, Bollinger Bands, Stochastic RSI, ADX  
- **Sentiment fusion:** CryptoPanic API + top 5 crypto RSS feeds (CoinDesk, CoinTelegraph, CryptoSlate, Bitcoin Magazine, Decrypt) + Reddit (r/cryptocurrency, r/bitcoin, r/cryptomarkets) with NLP scoring  
- **Risk controls:** Daily max loss/cap, trailing stop‑loss (TSL), profit targets  
- **Capital allocation:** %‑based sizing, diversification across top signals  
- **Multi‑timeframe confirmation** (e.g., 15m + 1h)  
- **Telegram control:** Inline Buy/Sell, manual overrides, run‑state messages  
- **Logging:** Firestore + (planned) Google Sheets with weekly summaries  
- **Backtesting:** Price‑based simulator, win/loss tracking  
- **Modular configs:** JSON/YAML for symbols, thresholds, filters  
- **Security & resilience:** HMAC signing, input validation, retry/failover logic

---

## Quick Start
```bash
git clone https://github.com/KhushiThakur-AI/CryptoBotAI-Advance.git
cd CryptoBotAI-Advance
pip install -r requirements.txt
python main.py
