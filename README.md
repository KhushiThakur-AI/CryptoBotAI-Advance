# CryptoBotAI-Advance
This is a advance crypto trading bot that is an update of my old bot. This current bot has around 25 + more features added to it and still advancement and multiple test and needed. It is currently intelligent but need more work.

The main.py file has more than 2500+ lines of code that are responsible for running the entire crypto trading bot smoothly. 

# CryptoBotAI-Advance 🚀

## Features

- **Multi-indicator strategy**: RSI, EMA, MACD, Bollinger Bands, Stochastic RSI, ADX
- **Adaptive Position Sizing**: Trade size adjusts to risk level and sentiment
- **Daily Budget Cap**: Restrict max investment per day
- **Trailing Stop-Loss** & Take-Profit for smarter exits
- **Sentiment Analysis** from multiple sources:
- 
  - **Reddit (crypto subreddits)**
  - **RSS feeds from top 5 crypto trading sites**:
    1. CoinDesk
    2. CoinTelegraph
    3. CryptoSlate
    4. Bitcoin Magazine
    5. Decrypt
       
  - **CryptoPanic API** news sentiment
  - **Hugging Face AI** for advanced sentiment scoring
- **Google Sheets Logging**: Auto-log trades and weekly summaries
- **Telegram Control**:
  - Inline **Buy/Sell** buttons
  - `/buy SYMBOL AMOUNT Reason`
  - `/sell SYMBOL AMOUNT Reason`
  - `/cancel SYMBOL` to exit position
  - Restricted access to your user ID
- **Chart Generation** & Telegram alerts
- **Modular Codebase** for easy maintenance
- **Error Logging** with Google Sheets + Telegram alerts
- **Firebase Firestore** integration for config & state saving

---

## 🛠️ File Structure
CryptoBotAI-Advance/
│── main.py # Main bot execution script
│── config.json # Config for symbols, SL/TP/TSL, indicator settings
│── indicators.py # All technical indicator calculations
│── sentiment.py # Reddit, RSS, CryptoPanic, Hugging Face sentiment logic
│── telegram_bot.py # Telegram bot commands & inline buttons
│── binance_client.py # Binance trading logic
│── google_sheets.py # Google Sheets API logging
│── firebase_client.py # Firestore database integration
│── utils.py # Helper functions
│── requirements.txt # Python dependencies
│── README.md # Project documentation
│── /charts # Auto-generated trade charts

 
---

## ⚡ How Sentiment Analysis Works
1. **Fetch News**  
   - From **top 5 crypto RSS feeds** (CoinDesk, CoinTelegraph, CryptoSlate, Bitcoin Magazine, Decrypt)
   - From **Reddit** (subreddits like r/cryptocurrency, r/bitcoin, r/cryptomarkets)
   - From **CryptoPanic API**
2. **Combine Headlines + Summaries**
3. **Run through Hugging Face Sentiment Model**
4. **Boost/Reduce Signal Strength**:
   - Strongly positive → increase buy confidence
   - Strongly negative → increase sell/avoid confidence

---

## 📊 Example Google Sheets Logs
**Trade Log Columns**:
`Symbol | Time (UTC) | Action | Amount | Invested (USD) | Quantity | Reason | RSI Value | Current Price | Unrealized P/L | Profit/Loss | High Price | Low Price | Avg Price | Entry Price | 1D Low | 1D High | Prev Close`

**Weekly Summary Columns**:
`Week Start | Week End | Total Trades | Winning Trades | Losing Trades | Win Rate (%) | Gross Profits | Gross Losses | Net Profit | Symbols Traded`

---
📅 Bot Roadmap & Planned Features
This bot is actively developed and evolving toward a fully autonomous, AI-enhanced crypto trading system.
Below is the roadmap of completed and planned features:

✅ Completed Features
Daily Max Trade Cap → Prevents over-trading and controls daily risk.

Manual Trade Controls via Telegram → Supports /buy /sell /cancel commands, inline Buy/Sell buttons, and restricted access.

🔜 Planned & In-Progress Features
⚡ Async Execution – Use asyncio + aiohttp for concurrent API calls → Faster data gathering, lower latency.

📊 Win/Loss Tracking – Maintain stats for last N trades (win %, average PnL) → Evaluate strategy performance.

📈 Historical Sentiment for Backtesting – Integrate archived sentiment feeds for realistic backtests.

💧 Market Depth & Slippage Handling – Factor in bid-ask spread & liquidity for better fills.

🤖 AI Signal Explanation – GPT-powered trade reasoning & decision labeling for transparency.

📊 Streamlit Dashboard Enhancements – Real-time charts, manual override controls, P&L dashboard linked to Firestore. (In Progress)

🔑 Live Exchange Execution – Execute authenticated trades on Binance with error handling.

📰 Advanced News Sentiment AI – Weighted sentiment scoring from CryptoPanic + top RSS feeds + Reddit trends.

🌐 Multi-Exchange Support – Add KuCoin, Bybit, Kraken for diversified opportunities.

📚 Machine Learning Signal Boost – Train ML models (XGBoost, LightGBM) for higher-confidence trade detection.

📊 Live Web Dashboard – Heatmaps, live signals, and manual trades from browser interface.

📉 Trailing Stop-Loss & Smart Exit – Volatility-based TSL and adaptive exit rules for profit protection.

🧪 Backtesting Module – Run strategy simulations with adjustable parameters on historical data.

📂 Portfolio Optimization – Auto-adjust allocations based on asset performance & volatility.

🚨 Alert System – Telegram notifications for price breakouts, whale activity, and sentiment spikes.

⚙ Config File System – Store settings in JSON/YAML for safer non-code changes.

🛠 Error Recovery & Retry Logic – Auto retries, failover handling, and Telegram error alerts for 24/7 uptime.

---

## 🚀 Quick Start
```bash
pip install -r requirements.txt
python main.py

**LICENSE**
MIT License

Copyright (c) 2025 Khushi Thakur

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
