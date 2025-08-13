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

🚀 Features
📊 Multi-Indicator Strategy
RSI – Overbought/oversold detection

MACD – Momentum shift confirmation

EMA – Trend following

ADX – Trend strength

Bollinger Bands – Volatility breakout

Stochastic RSI – Entry timing (where used)

🛡 Risk Management
✅ Stop-Loss (SL), ✅ Take-Profit (TP), ✅ Trailing Stop-Loss (TSL)

✅ Daily Max Loss Guard

✅ Cooldown + Duplicate-Trade Blocker

🧠 Smart Trade Logic
Trade Confidence Score (weighted multi-indicator confirmation)

Per-coin configuration via config.json

Multi-timeframe confirmation (e.g., 15m + 1h)

📤 Telegram Alerts
📈 Trade Executed (BUY/SELL)

🚨 SL/TP/TSL Triggered

🧾 Trade Summary (daily/weekly)

⚠️ Capital issues / loss guard

📄 Google Sheets Logging
✅ Trade history

✅ P&L tracking

✅ Per-symbol worksheets (e.g., BTCUSDT, ETHUSDT)

Optional: Firestore for real-time logging/state.
---

## Quick Start
```bash
git clone https://github.com/KhushiThakur-AI/CryptoBotAI-Advance.git
cd CryptoBotAI-Advance
pip install -r requirements.txt
python main.py
