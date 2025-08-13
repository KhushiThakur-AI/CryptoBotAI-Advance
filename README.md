# CryptoBotAI-Advance
This is a advance crypto trading bot that is an update of my old bot. This current bot has around 25 + features added to it and more to go. It is currently intelligent but need more work.
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
#	Feature Name	Description	Purpose / Benefit	Status	Notes
1	Async Execution	Use asyncio + aiohttp for concurrent data fetching	Faster data gathering, lower latency	🔜 Planned	Enhances performance with multiple API calls.
2	Daily Max Trade Cap	Enforce daily trade budget limit (e.g., $100)	Prevents over-trading, controls risk	✅ Done	Already implemented.
3	Track Win/Loss Ratio	Analyze last 10+ trades win% and average PnL	Measures strategy effectiveness	🔜 Planned	Helps decide when to go live from paper mode.
4	Manual Trade Controls via Telegram	Allow manual buy/sell commands with logging and confirmation	Human override & accountability	✅ Done	Inline buttons + restricted access implemented.
5	Historical Sentiment Data for Backtesting	Integrate archived sentiment data for realistic backtests	Improves strategy validation	🔜 Planned	Requires data archive or paid API.
6	Market Depth & Slippage Consideration	Factor bid-ask spread & liquidity into order execution	Reduces bad fills & slippage	🔜 Planned	Useful for low-liquidity assets.
7	AI Integration	Use GPT/AI for live signal explanation & labeling	Smarter trade reasoning & debugging	🔜 Planned	Could improve trade transparency.
8	Streamlit Dashboard Enhancements	Real-time charts, manual override, P&L dashboards	Improves monitoring & control	⏳ In Progress	Connects to Firestore for live data.
9	Live Exchange Execution	Move from paper to real authenticated exchange orders	Enables real profits	🔜 Planned	Using Binance API with full error handling.
10	Advanced News Sentiment AI	Include top 5 crypto RSS feeds, Reddit trends, weighted scoring	More accurate sentiment analysis	🔜 Planned	Expands beyond CryptoPanic.
11	Multi-Exchange Support	Add KuCoin, Bybit, Kraken alongside Binance	Increases trading opportunities	🔜 Planned	Improves diversification.
12	Machine Learning Signal Boost	Use ML (XGBoost, LightGBM) to detect high-confidence trades	Increases win rate	🔜 Planned	Needs labeled historical data.
13	Live Web Dashboard	Streamlit dashboard with charts, heatmaps, manual trades	Real-time control	🔜 Planned	Integrates with sentiment system.
14	Trailing Stop-Loss & Smart Exit	Volatility-based TSL & adaptive exit rules	Maximizes profits, cuts losses	🔜 Planned	Replaces static SL/TP.
15	Backtesting Module	Run strategy on historical data	Test before going live	🔜 Planned	Adjustable parameters for experiments.
16	Portfolio Optimization	Auto-balance based on performance & risk	Improves capital allocation	🔜 Planned	Sentiment & volatility aware.
17	Alert System	Telegram alerts for breakouts, whale activity, sentiment shifts	Faster reaction to market changes	🔜 Planned	Can include price targets.
18	Config File System	Store all settings in JSON/YAML	Easy edits without touching code	🔜 Planned	Safer than editing Python files.
19	Error Recovery & Retry Logic	Auto retries + Telegram error alerts	Improves bot uptime	🔜 Planned	Critical for 24/7 running.

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
