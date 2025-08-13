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
-  - [Screenshots](#screenshots)
    - [All project files](#all-project-files)
    - [Console output](#console-output)
    - [Streamlit dashboard](#streamlit-dashboard)
    - [Telegram upgrades](#telegram-upgrades)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Keywords (SEO)](#keywords-seo)
- [Tips & Disclaimer](#tips--disclaimer)
---

🚀 Features (Old Bot)

1. Multi-Indicator Strategy
RSI – Overbought/oversold detection
MACD – Momentum shift confirmation
EMA – Trend following
ADX – Trend strength
Bollinger Bands – Volatility breakout

2. Risk Management
✅ Stop Loss (SL)
✅ Take Profit (TP)
✅ Trailing Stop Loss (TSL)
✅ Daily Max Loss Guard
✅ Cooldown + Duplicate Trade Blocker

3. Smart Trade Logic
Trade Confidence Score (based on multiple indicators)
Per-coin configuration using config.json

4. Telegram Alerts
Trade Executed (BUY/SELL)
SL/TP/TSL Triggered
 Trade Summary (daily/weekly)
Capital issues / loss guard
Google Sheets Logging
Trade history
P&L tracking
Per-symbol worksheets (BTCUSDT, ETHUSDT, etc.)

## ✅ Latest 25+ Updates (Complete & Recent Work)

| #  | Feature / Development                      | Description                                                  | Why It’s Important                     | Priority / Status       | Notes                                           |
|----|---------------------------------------------|--------------------------------------------------------------|------------------------------------------|-------------------------|-------------------------------------------------|
| 1  | Add More Indicators                        | ADX + Bollinger added; use top-2 performers for confirmation/points | Improves signal accuracy                | DONE                    | ADX = trend strength; BB = breakouts            |
| 2  | Real Balance Awareness                     | Check USDT wallet before trading                             | Prevents over-trading                    | DONE                    | `get_usdt_balance()` pre-trade                  |
| 3  | Smart Capital Allocation                   | % of wallet per trade (e.g., 20–30%)                         | Safer, scales with account               | DONE                    | Dynamic sizing by available balance             |
| 4  | Diversification Logic                       | Invest across 2–3 strongest coins                            | Lowers risk, more opportunities          | DONE                    | Pick top scores across symbols                  |
| 5  | Trailing Stop-Loss (TSL)                   | Lock profits as price rises                                  | Avoids giving back gains                 | DONE                    | Dynamic TSL vs static SL/TP                      |
| 6  | Daily Max Loss Guard                       | Halt if loss > threshold/day                                 | Avoid wipeouts                           | DONE                    | e.g., stop if loss > $3/day                      |
| 7  | Trade Confidence Scoring                   | Weighted indicators + thresholds                             | Better trade quality                     | DONE                    | EMA crossover + ADX>20, etc.                     |
| 8  | Trade Journal Logging                      | Log full reasons per trade                                   | Audit & debugging                        | DONE                    | Google Sheets / Firestore                        |
| 9  | Profit Target Exit                         | Auto-exit after X% profit                                    | Locks in wins                            | DONE                    | Fixed or dynamic targets                         |
| 10 | Dynamic Rebalancing                        | Weekly realloc by performance                                | Optimizes portfolio                      | DONE                    | Track performance & update weights               |
| 11 | Multi-Timeframe Confirmation               | Align 1h + 15m, etc.                                         | Avoid choppy entries                     | DONE                    | Timeframe confluence                            |
| 12 | Basic & Advanced Signal Scoring            | Indicator weighting + trend filters                          | Smarter decisions                        | DONE                    | Thresholded weighted scores                      |
| 13 | Market Filtering (Vol/Volume)              | Filter low-liquidity/high-volatility coins                   | Reduce risk/false signals                 | DONE                    | `min_volume_usd`, `max_volatility`               |
| 14 | No-Trade Detection & Logging               | Log “No trade this cycle”                                    | Confirms bot alive                       | DONE                    | Telegram + Sheets                               |
| 15 | Active Trade Hours Config                  | Run only in set windows (e.g., 14:00–20:00)                  | Target liquidity                         | DONE                    | Configurable in settings                         |
| 16 | Live Exchange Execution                    | Move to real authenticated orders                            | Real profit capture                      | TO DO / LATER           | `python-binance` + confirms/safeguards           |
| 17 | Adaptive Position Sizing                   | Size by score/volatility                                     | Optimize R/R                             | DONE                    | Confidence/ATR-aware sizing                      |
| 18 | Enhanced Sentiment Scoring                 | VADER/TextBlob style scoring                                 | Better sentiment integration             | DONE                    | Beyond binary sentiment                          |
| 19 | Sell Logic Upgrade                         | TSL/indicator-based exits                                    | Better exits                             | DONE                    | Beyond static TP/SL                              |
| 20 | Volatility / Volume Filter                 | Block extreme vol/low volume                                 | Reduce false signals                     | DONE                    | Works with market filtering                      |
| 21 | ML-Based Signal Booster (Exp.)             | XGBoost to score/boost signals                               | Potential win-rate gain                  | DONE (Experimental)     | Needs more data                                  |
| 22 | External Config Files                      | JSON/YAML for settings                                       | Easier updates                           | DONE                    | Decoupled from code                              |
| 23 | Streamlit Dashboard Enhancements           | Real-time charts, manual overrides, P&L                      | Better monitoring                        | IN PROGRESS / DONE      | Firestore live visualization                     |
| 24 | Security Enhancements                      | HMAC, input validation, errors                               | Protect keys & stability                 | DONE                    | Critical for live trading                        |
| 25 | Backtesting Logic                          | Historical simulation engine                                 | Test before live                         | DONE                    | Use price (and future sentiment)                 |
| 26 | Modularization + Unit Testing              | Split modules + add tests                                    | Maintainable & reliable                  | DONE (baseline)         | More tests planned                               |
| 27 | Retry & Failover Logic                     | Auto-retry, fallback caching, Telegram alerts                | More uptime                              | DONE                    | Handle transient API errors                      |
| 28 | Real Symbol-Based News                     | CryptoPanic + RSS per symbol                                 | Real-time news sentiment                 | DONE                    | Proper integration, Replit-ready                 |
---
## 🔭 Future Updates & Backlog (Planned / In Progress)

### A. High-Impact Next Steps (29–35)

| #  | Feature                          | What It Does                                         | Benefit                          | Status    | Notes                               |
|----|----------------------------------|------------------------------------------------------|-----------------------------------|-----------|-------------------------------------|
| 29 | Async Execution                  | `asyncio` + `aiohttp` for concurrent fetch           | Faster cycles, lower latency      | 🔜 Planned| Best for multi-API workloads        |
| 30 | Daily Max Trade Cap               | Enforce daily budget (e.g., $100)                    | Risk control                      | ✅ Done   | Already implemented                 |
| 31 | Track Win/Loss Ratio              | Rolling last 10+ trades win% & avg PnL                | Go-live readiness                  | 🔜 Planned| Decision aid                         |
| 32 | Manual Trade via Telegram         | Inline/manual BUY/SELL with logging                   | Human override                     | ✅ Done   | Restricted access                    |
| 33 | Historical Sentiment for Backtest | Use archived sentiment data                           | Realistic backtests                | 🔜 Planned| Needs archive/3rd-party              |
| 34 | Market Depth & Slippage           | Model bid/ask & liquidity                             | Realistic fills                    | 🔜 Planned| Order-book aware                     |
| 35 | AI Integration                    | GPT explains/labels signals                           | Transparency & debugging           | 🔜 Planned| Auto trade labeling                  |
---
## Quick Start

git clone https://github.com/KhushiThakur-AI/CryptoBotAI-Advance.git
cd CryptoBotAI-Advance
pip install -r requirements.txt
python main.py
---

## Configuration
To configure the project for your environment:

1. **Clone the Repository**
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   
2. Install Dependencies
python command (Run in Replit Shell)
pip install -r requirements.txt
(or use npm install if your project uses Node.js)

3. Set Environment Variables
Create a .env file in the root directory.
Add your API keys, tokens, and configuration settings: API_KEY=your_api_key
SECRET_KEY=your_secret_key

4. Run the Application
python main.py
---

## Architecture
The project follows a **modular architecture** for scalability and maintainability:

- **Core Modules** – Main application logic and services.
- **API Layer** – Handles requests, responses, and third-party API integration.
- **Database Layer** – Manages data storage and retrieval.
- **Utilities** – Helper functions and reusable scripts.
- **Configuration** – Environment-specific settings.
- **UI / Dashboard** *(if applicable)* – Frontend or monitoring interface.

**Flow Overview:**
1. Input is received from API/Frontend.
2. Processed through core logic modules.
3. Data is fetched/stored in the database.
4. Output is returned or displayed in the UI.

---
## Screenshots

### All project files
![All project files list](https://github.com/KhushiThakur-AI/CryptoBotAI-Advance/raw/main/ALL%20FILES%20LIST%20PNG.png)

### Console output
![Console output](https://github.com/KhushiThakur-AI/CryptoBotAI-Advance/raw/main/Console%20Outcomes.png)

### Streamlit dashboard
![Streamlit dashboard](https://github.com/KhushiThakur-AI/CryptoBotAI-Advance/raw/main/Streamlit%20Dashboard.png)

### Telegram upgrades
![Telegram upgrades](https://github.com/KhushiThakur-AI/CryptoBotAI-Advance/raw/main/Telegram%20Updates.png)

---

## Roadmap
### ✅ Done (Highlights)
- **Daily Max Loss & Trade Cap** – Risk control to limit daily exposure
- **Manual Trade via Telegram** – Inline buttons, restricted access
- **Trailing Stop-Loss (Basic)** – Protects profits and limits losses
- **Profit Target Exits** – Automated take-profit triggers
- **Diversification & Multi-Timeframe Confirmation** – Safer trade entries
- **Backtesting (Price-Based)** – Strategy testing with win/loss tracking

---

### 🔜 Planned & In-Progress
- **Async Execution** – `asyncio` + `aiohttp` for concurrent I/O
- **Win/Loss Analytics** – Rolling win%, average PnL
- **Historical Sentiment for Backtests** – Archived RSS, Reddit, CryptoPanic sentiment
- **Market Depth & Slippage** – More realistic order fills
- **AI Signal Explanation** – GPT-generated trade labels & reasoning
- **Streamlit Dashboard Enhancements** – Live charts, manual overrides, P&L (in progress)
- **Live Exchange Execution** – Binance authenticated orders with safeguards
- **Advanced News Sentiment AI** – Weighted fusion of RSS + Reddit + CryptoPanic
- **Multi-Exchange Support** – KuCoin, Bybit, Kraken
- **ML Signal Boost** – XGBoost / LightGBM model integration
- **Smart Exits** – ATR-based trailing stop-loss, adaptive rules
- **Portfolio Optimization** – Performance & volatility-aware allocation
- **Alert System** – Breakouts, whale moves, sentiment spikes
- **Config Overhaul** – Full JSON/YAML support, hot-reload capability
- **Error Recovery & Retry** – Resilient 24/7 runtime

---

## Contributing
We welcome contributions! 🎉

**Steps to Contribute:**
1. **Fork** the repository.
2. **Create a new branch**:
   ```bash
   git checkout -b feature/YourFeature
3. Commit changes:
     ```bash
     git commit -m "Add your message here"
4. Push to your fork:
   ```bash
   git push origin feature/YourFeature
   Open a Pull Request on GitHub.

IMP = Please ensure your code follows the existing style and includes documentation for new features.
---
## License
This project is licensed under the **MIT License**. You can view the full terms in the [LICENSE file](https://github.com/KhushiThakur-AI/CryptoBotAI-Advance/blob/main/LICENSE).
---
### Keyword SEO
advanced AI crypto trading bot, automated crypto trading, cryptocurrency bot, algorithmic trading software, Binance trading bot, crypto market analysis, sentiment-based trading, AI trading system, machine learning crypto bot, real-time trading dashboard
---
## Tips & Disclaimer

### Tips for Non-Technical Users
If you are **new to coding**, **crypto trading**, or both, it is highly recommended to use an **LLM (Large Language Model)** such as:
- [ChatGPT](https://chat.openai.com/)
- [Replit AI Assistant](https://replit.com/ai)
- [Gemini AI](https://gemini.google.com/)

These tools can:
- Help you understand the project structure.
- Guide you through configuration and setup.
- Explain crypto-related concepts in plain language.
- Assist in customizing this bot for your own needs.

> **Important:** This repository should be treated as a **reference project**.  
> Do **not** blindly copy-paste the code without understanding how it works.  
> Your own **research, testing, and strategic thinking** are essential for building a successful and sustainable trading bot.

---

### Disclaimer
- I am **not** a licensed financial advisor.  
- This project is for **educational purposes only**.  
- Trading cryptocurrencies involves **high risk**, and you should only invest what you can afford to lose.  
- The author(s) are **not responsible** for any financial losses incurred from using this code.

> By using this repository, you acknowledge that you are solely responsible for your trading decisions.


