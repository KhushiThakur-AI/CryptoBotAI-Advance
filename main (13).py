import firebase_admin
from firebase_admin import credentials, firestore
import os
import json
import time
import logging
import requests
import pandas as pd
import ta
import datetime
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from binance.client import Client
from binance.enums import *
from dotenv import load_dotenv
import telegram
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import math
import tempfile
import asyncio
import hmac
import hashlib
import csv
import random
import unittest
from retry_handler import setup_robust_apis, health_check_and_recovery
from circuit_breaker import APICircuitBreakerManager
from news_sentiment_integration import NewsSentimentIntegration

# ML dependencies
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Load config
CONFIG_FILE = "config.json"
try:
    with open(CONFIG_FILE, "r") as f:
        config_data = json.load(f)
except FileNotFoundError:
    logger.error(f"Error: {CONFIG_FILE} not found. Please create it.")
    exit(1)
except json.JSONDecodeError as e:
    logger.error(f"Error: Could not decode JSON from {CONFIG_FILE}: {e}. Please check its format.")
    exit(1)

SYMBOLS_CONFIG = config_data.get("symbols", {})
SETTINGS = config_data.get("settings", {})
INDICATORS_SETTINGS = SETTINGS.get("indicators", {})

# New multi-timeframe settings
MULTI_TIMEFRAME_SETTINGS = SETTINGS.get("multi_timeframe_confirmation", {})
PRIMARY_TIMEFRAME = MULTI_TIMEFRAME_SETTINGS.get("primary_timeframe", "15m")
CONFIRMATION_TIMEFRAME = MULTI_TIMEFRAME_SETTINGS.get("confirmation_timeframe", "1h")

# New signal scoring settings
SIGNAL_SCORING_SETTINGS = SETTINGS.get("signal_scoring", {})
SIGNAL_WEIGHTS = SIGNAL_SCORING_SETTINGS.get("signal_weights", {})
MIN_SIGNAL_SCORE_BUY = SIGNAL_SCORING_SETTINGS.get("min_signal_score_buy", 1.0)
MIN_SIGNAL_SCORE_SELL = SIGNAL_SCORING_SETTINGS.get("min_signal_score_sell", 1.0)

# New active trading hours settings
ACTIVE_TRADE_HOURS = SETTINGS.get("active_trade_hours", [])

# New market filtering settings
MARKET_FILTERS = SETTINGS.get("market_filters", {})
MIN_VOLUME_USD = MARKET_FILTERS.get("min_volume_usd", 1000000.0)
MAX_VOLATILITY = MARKET_FILTERS.get("max_volatility", 0.05)

# New adaptive position sizing settings
ADAPTIVE_SIZING_SETTINGS = SETTINGS.get("adaptive_sizing", {})
ADAPTIVE_SIZING_ENABLED = ADAPTIVE_SIZING_SETTINGS.get("enabled", False)
MIN_TRADE_AMOUNT_USD = ADAPTIVE_SIZING_SETTINGS.get("min_trade_amount_usd", 10.0)
MAX_TRADE_AMOUNT_USD = ADAPTIVE_SIZING_SETTINGS.get("max_trade_amount_usd", 500.0)
MAX_VOLATILITY_FOR_SIZING = ADAPTIVE_SIZING_SETTINGS.get("max_volatility_for_sizing", 0.05)
BASE_TRADE_AMOUNT_USD = SETTINGS.get("trade_amount_usd", 10.0)

# Daily Max Loss Guard setting
DAILY_MAX_LOSS_USD = SETTINGS.get("daily_max_loss_usd", 2.0)

# New ML-Based Signal Booster settings
ML_BOOSTER_SETTINGS = config_data.get("ml_booster", {})
ML_BOOSTER_ENABLED = ML_BOOSTER_SETTINGS.get("enabled", False)
ML_MODEL_PATH = ML_BOOSTER_SETTINGS.get("model_path", "ml_model.joblib")
ML_CONFIDENCE_THRESHOLD = ML_BOOSTER_SETTINGS.get("confidence_threshold", 0.7)
ML_SIGNAL_BOOST_WEIGHT = ML_BOOSTER_SETTINGS.get("signal_boost_weight", 1.0)

ACTIVE_SYMBOLS = list(SYMBOLS_CONFIG.keys())
logger.info(f"Active Symbols configured: {ACTIVE_SYMBOLS}")

# Replit secret usage
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")

if not all([BINANCE_API_KEY, BINANCE_API_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
    logger.error("Error: One or more Replit Secrets (BINANCE_API_KEY, BINANCE_API_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID) are not set. Please set them in Replit.")
    exit(1)

# Setup clients
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)

# Google Sheets setup (moved before robust_apis setup)
gsheet = None
try:
    if GOOGLE_SHEET_ID:
        scope = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
        gsheet = gspread.authorize(creds)
        logger.info("Google Sheets connected successfully.")
    else:
        logger.info("GOOGLE_SHEET_ID not provided in Replit Secrets. Google Sheets integration disabled.")
except Exception as e:
    logger.warning(f"Could not connect to Google Sheets: {e}. Trading logs will not be written to sheet.")

# Setup robust API wrapper
robust_apis = setup_robust_apis(
    binance_client=client,
    gsheet_client=gsheet,
    telegram_bot_token=TELEGRAM_BOT_TOKEN,
    telegram_chat_id=TELEGRAM_CHAT_ID,
    spreadsheet_id=GOOGLE_SHEET_ID
)

# Setup circuit breakers
circuit_breaker_manager = APICircuitBreakerManager()

# --- Firebase Admin SDK setup for Firestore ---
FIREBASE_CRED_PATH = os.path.join(os.path.dirname(__file__), "firebase_credentials.json")
firebase_db = None

if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred)
        print("✅ Firebase initialized.")
        firebase_db = firestore.client()
    except Exception as e:
        print(f"❌ Firebase init failed: {e}")
else:
    print("⚠️ Firebase already initialized.")
    firebase_db = firestore.client()

APP_ID = "myApp"
USER_ID = "user123"

# --- Utility functions ---
def get_klines(symbol, interval=PRIMARY_TIMEFRAME, lookback=SETTINGS.get("lookback", 250)):
    """
    Fetches klines data for a given symbol and interval.
    The 'lookback' parameter is now dynamically read from SETTINGS in config.json.
    """
    try:
        return robust_apis.get_klines(symbol, interval, lookback)
    except Exception as e:
        logger.error(f"Error fetching klines for {symbol}: {e}")
        return None

def get_latest_price(symbol):
    """Fetches the current ticker price for a given symbol."""
    try:
        return robust_apis.get_latest_price(symbol)
    except Exception as e:
        logger.error(f"Error fetching latest price for {symbol}: {e}")
        return None

def get_usdt_balance():
    """Fetches the available USDT balance from the Binance spot wallet."""
    try:
        return robust_apis.get_usdt_balance()
    except Exception as e:
        logger.error(f"Error fetching USDT balance from Binance: {e}")
        return 0.0

async def send_telegram_message(message, image_path=None):
    """Sends a message to Telegram, optionally with an image."""
    try:
        if image_path:
            with open(image_path, 'rb') as f:
                await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=f, caption=message)
        else:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
    except Exception as e:
        logging.error(f"Failed to send Telegram message: {e}")

async def send_daily_trade_hours_message():
    """
    Sends a daily message to Telegram with the configured active trading hours,
    converted to IST. This message is sent only once per day.
    """
    if not ACTIVE_TRADE_HOURS:
        message = "🗓️ **Daily Active Hours:**\n\nNo active trading hours are configured. The bot will run continuously."
        await send_telegram_message(message)
        return

    # Assuming the hours in the config are in UTC for server consistency
    utc_tz = pytz.timezone('UTC')
    ist_tz = pytz.timezone('Asia/Kolkata')

    hours_list = []
    for interval in ACTIVE_TRADE_HOURS:
        try:
            # Parse the time from the config, assuming it's in a server's local time (UTC is a good assumption)
            start_time_local = datetime.datetime.strptime(interval["start"], "%H:%M").time()
            end_time_local = datetime.datetime.strptime(interval["end"], "%H:%M").time()

            # Create dummy datetime objects for conversion
            dummy_date = datetime.date.today()
            start_dt_utc = utc_tz.localize(datetime.datetime.combine(dummy_date, start_time_local))
            end_dt_utc = utc_tz.localize(datetime.datetime.combine(dummy_date, end_time_local))

            # Convert to IST
            start_dt_ist = start_dt_utc.astimezone(ist_tz)
            end_dt_ist = end_dt_utc.astimezone(ist_tz)

            hours_list.append(f"• **{start_dt_ist.strftime('%I:%M %p')} - {end_dt_ist.strftime('%I:%M %p')} IST**")
        except (ValueError, KeyError) as e:
            logger.error(f"Invalid time format in config.json for active_trade_hours: {e}. Please use 'HH:MM'.")
            return

    hours_message = "\n".join(hours_list)
    today = datetime.date.today().strftime('%A, %B %d')
    message = (
        f"🗓️ **Daily Active Hours for {today}:**\n\n"
        f"The bot is configured to run during the following windows (in IST) to target high-activity periods:\n\n"
        f"{hours_message}\n\n"
        f"Happy trading! I'll let you know when the bot wakes up for the next window."
    )
    await send_telegram_message(message)
    logger.info("Daily active trading hours message sent to Telegram.")

def add_indicators(df):
    """Adds technical indicators to the DataFrame based on settings."""
    if INDICATORS_SETTINGS.get("rsi_enabled", False):
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=INDICATORS_SETTINGS.get("rsi_period", 14)).rsi()
    else:
        df['rsi'] = None

    if INDICATORS_SETTINGS.get("ema_enabled", False):
        df['ema_fast'] = ta.trend.EMAIndicator(df['close'], window=INDICATORS_SETTINGS.get("ema_fast", 50)).ema_indicator()
        df['ema_slow'] = ta.trend.EMAIndicator(df['close'], window=INDICATORS_SETTINGS.get("ema_slow", 200)).ema_indicator()
    else:
        df['ema_fast'] = None
        df['ema_slow'] = None

    if INDICATORS_SETTINGS.get("macd_enabled", False):
        macd = ta.trend.MACD(
            close=df['close'],
            window_fast=INDICATORS_SETTINGS.get("macd_fast", 12),
            window_slow=INDICATORS_SETTINGS.get("macd_slow", 26),
            window_sign=INDICATORS_SETTINGS.get("macd_signal", 9)
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
    else:
        df['macd'] = None
        df['macd_signal'] = None
        df['macd_hist'] = None

    if INDICATORS_SETTINGS.get("bollinger_enabled", False):
        bollinger = ta.volatility.BollingerBands(
            close=df['close'],
            window=INDICATORS_SETTINGS.get("bollinger_period", 20),
            window_dev=INDICATORS_SETTINGS.get("bollinger_std", 2)
        )
        df['bb_bbm'] = bollinger.bollinger_mavg()
        df['bb_bbh'] = bollinger.bollinger_hband()
        df['bb_bbl'] = bollinger.bollinger_lband()
    else:
        df['bb_bbm'] = None
        df['bb_bbh'] = None
        df['bb_bbl'] = None

    if INDICATORS_SETTINGS.get("stoch_rsi_enabled", False):
        stoch_rsi = ta.momentum.StochRSIIndicator(
            close=df['close'],
            window=INDICATORS_SETTINGS.get("stoch_rsi_period", 14),
        )
        df['stoch_rsi_k'] = stoch_rsi.stochrsi_k()
        df['stoch_rsi_d'] = stoch_rsi.stochrsi_d()
    else:
        df['stoch_rsi_k'] = None
        df['stoch_rsi_d'] = None

    if INDICATORS_SETTINGS.get("adx_enabled", False):
        adx_indicator = ta.trend.ADXIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=INDICATORS_SETTINGS.get("adx_period", 14)
        )
        df['adx'] = adx_indicator.adx()
        df['adx_pos'] = adx_indicator.adx_pos()
        df['adx_neg'] = adx_indicator.adx_neg()
    else:
        df['adx'] = None
        df['adx_pos'] = None
        df['adx_neg'] = None

    return df

def get_symbol_info(symbol):
    """Fetches symbol exchange information for precision, minQty, and minNotional."""
    try:
        info = client.get_symbol_info(symbol)
        if not info:
            logger.error(f"Could not get symbol info for {symbol}")
            return None

        price_precision = 0
        quantity_precision = 0
        min_notional = 0.0
        min_qty = 0.0

        for f in info['filters']:
            if f['filterType'] == 'PRICE_FILTER':
                price_precision = int(round(-math.log10(float(f['tickSize']))))
            elif f['filterType'] == 'LOT_SIZE':
                quantity_precision = int(round(-math.log10(float(f['stepSize']))))
                min_qty = float(f['minQty'])
            elif f['filterType'] == 'MIN_NOTIONAL':
                min_notional = float(f['minNotional'])

        return {
            'pricePrecision': price_precision,
            'quantityPrecision': quantity_precision,
            'minNotional': min_notional,
            'minQty': min_qty
        }
    except Exception as e:
        logger.error(f"Error getting symbol info for {symbol}: {e}")
        return None

def generate_chart(df, symbol, timeframe, indicator_settings):
    """
    Generates a Matplotlib chart of price action and selected indicators.
    Saves the chart to a temporary file and returns its path.
    """
    if df.empty:
        logger.warning(f"Cannot generate chart for {symbol}: DataFrame is empty.")
        return None

    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image_path = temp_file.name
    temp_file.close()

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1, 1]})

    axes[0].plot(df['timestamp'], df['close'], label='Close Price', color='blue')
    if indicator_settings.get("ema_enabled", False) and 'ema_fast' in df.columns and 'ema_slow' in df.columns:
        axes[0].plot(df['timestamp'], df['ema_fast'], label=f'EMA {indicator_settings.get("ema_fast", 50)}', color='orange')
        axes[0].plot(df['timestamp'], df['ema_slow'], label=f'EMA {indicator_settings.get("ema_slow", 200)}', color='purple')
    if indicator_settings.get("bollinger_enabled", False) and 'bb_bbm' in df.columns:
        axes[0].plot(df['timestamp'], df['bb_bbm'], label='BB Middle', color='green', linestyle='--')
        axes[0].plot(df['timestamp'], df['bb_bbh'], label='BB Upper', color='red', linestyle=':')
        axes[0].plot(df['timestamp'], df['bb_bbl'], label='BB Lower', color='red', linestyle=':')

    axes[0].set_title(f'{symbol} Price Chart ({timeframe})')
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].grid(True)

    if indicator_settings.get("rsi_enabled", False) and 'rsi' in df.columns and df['rsi'].iloc[-1] is not None:
        axes[1].plot(df['timestamp'], df['rsi'], label='RSI', color='green')
        axes[1].axhline(70, color='red', linestyle='--', alpha=0.7)
        axes[1].axhline(30, color='red', linestyle='--', alpha=0.7)
        axes[1].set_title('RSI')
        axes[1].set_ylabel('RSI')
        axes[1].set_ylim(0, 100)
        axes[1].grid(True)
    else:
        fig.delaxes(axes[1])

    if indicator_settings.get("macd_enabled", False) and 'macd' in df.columns and df['macd'].iloc[-1] is not None:
        axes[2].plot(df['timestamp'], df['macd'], label='MACD', color='blue')
        axes[2].plot(df['timestamp'], df['macd_signal'], label='Signal', color='red')
        axes[2].bar(df['timestamp'], df['macd_hist'], label='Histogram', color='gray', alpha=0.7)
        axes[2].set_title('MACD')
        axes[2].set_ylabel('Value')
        axes[2].legend()
        axes[2].grid(True)
    elif indicator_settings.get("adx_enabled", False) and 'adx' in df.columns and df['adx'].iloc[-1] is not None:
        axes[2].plot(df['timestamp'], df['adx'], label='ADX', color='purple')
        axes[2].plot(df['timestamp'], df['adx_pos'], label='+DI', color='green')
        axes[2].plot(df['timestamp'], df['adx_neg'], label='-DI', color='red')
        axes[2].axhline(25, color='gray', linestyle='--', alpha=0.7)
        axes[2].set_title('ADX')
        axes[2].set_ylabel('ADX Value')
        axes[2].legend()
        axes[2].grid(True)
    else:
        fig.delaxes(axes[2])

    fig.autofmt_xdate()
    axes[0].set_xlim(df['timestamp'].iloc[0], df['timestamp'].iloc[-1])

    plt.tight_layout()
    plt.savefig(image_path)
    plt.close(fig)
    return image_path

def format_signal_summary(symbol, timeframe, latest_data, current_price, trade_manager, trade_signal="No Trade Signal", ml_score=None):
    """Formats the signal summary message."""
    rsi_val = f"{latest_data['rsi']:.2f}" if latest_data['rsi'] is not None else "N/A"
    ema_fast_val = f"{latest_data['ema_fast']:.2f}" if latest_data['ema_fast'] is not None else "N/A"
    macd_hist_val = f"{latest_data['macd_hist']:.4f}" if latest_data['macd_hist'] is not None else "N/A"
    stoch_rsi_k_val = f"{latest_data['stoch_rsi_k']:.2f}" if latest_data['stoch_rsi_k'] is not None else "N/A"
    stoch_rsi_d_val = f"{latest_data['stoch_rsi_d']:.2f}" if latest_data['stoch_rsi_d'] is not None else "N/A"
    adx_val = f"{latest_data['adx']:.2f}" if latest_data['adx'] is not None else "N/A"
    adx_pos_val = f"{latest_data['adx_pos']:.2f}" if latest_data['adx_pos'] is not None else "N/A"
    adx_neg_val = f"{latest_data['adx_neg']:.2f}" if latest_data['adx_neg'] is not None else "N/A"
    bb_bbh_val = f"{latest_data['bb_bbh']:.2f}" if latest_data['bb_bbh'] is not None else "N/A"
    bb_bbl_val = f"{latest_data['bb_bbl']:.2f}" if latest_data['bb_bbl'] is not None else "N/A"

    ml_score_line = ""
    if ML_BOOSTER_ENABLED and ml_score is not None:
        ml_score_line = f"• ML Score: {ml_score:.2f}\n"

    return (
        f"📊 [{symbol}] Signal Summary ({timeframe})\n\n"
        f"• Price: {current_price:.2f}\n"
        f"• RSI (14): {rsi_val}\n"
        f"• EMA (Fast 50): {ema_fast_val}\n"
        f"• MACD Hist: {macd_hist_val}\n"
        f"• Stoch RSI K/D: {stoch_rsi_k_val}/{stoch_rsi_d_val}\n"
        f"• ADX (14): {adx_val} (+DI: {adx_pos_val}, -DI: {adx_neg_val})\n"
        f"• BBands (20,2) Upper/Lower: {bb_bbh_val}/{bb_bbl_val}\n\n"
        f"{ml_score_line}"
        f"• Current Paper Balance: {trade_manager.paper_balance:,.2f} USD\n\n"
        f" {trade_signal}"
    )

def format_bot_status(trade_manager, current_prices):
    """Formats the overall bot status message."""
    total_unrealized_pnl, open_positions_summary = trade_manager.get_open_positions_pnl(current_prices)
    current_total_balance = trade_manager.paper_balance + total_unrealized_pnl
    total_realized_pnl = trade_manager.paper_balance - trade_manager.initial_paper_balance
    real_usdt_balance = get_usdt_balance() if trade_manager.real_mode else "N/A"

    # Pre-format the real_usdt_balance to handle the 'N/A' string case
    real_usdt_balance_str = f"{real_usdt_balance:,.2f}" if isinstance(real_usdt_balance, (int, float)) else real_usdt_balance

    pause_reason = ""
    if trade_manager.bot_is_paused_permanent:
        pause_reason = "No - Paused due to permanent max loss limit"
    elif trade_manager.bot_is_paused_daily:
        pause_reason = f"No - Paused due to daily max loss limit of ${trade_manager.daily_max_loss_usd:.2f}"
    else:
        pause_reason = "Yes"

    return (
        f"\n\n📋 Bot Status Update\n\n"
        f"• Mode: {'REAL' if trade_manager.real_mode else 'PAPER'}\n"
        f"• Bot Active: {pause_reason}\n"
        f"• Paper Balance: {trade_manager.paper_balance:,.2f} USD\n"
        f"• Real USDT Balance: {real_usdt_balance_str} USD\n"
        f"• Daily Realized PnL: {trade_manager.daily_pnl:,.2f} USD\n"
        f"• Unrealized PnL: {total_unrealized_pnl:,.2f} USD\n"
        f"• Total Effective Balance: {current_total_balance:,.2f} USD\n"
        f"• Total Realized PnL: {total_realized_pnl:,.2f} USD\n"
        f"• Open Positions:\n{open_positions_summary}"
    )

def is_trading_hour():
    """
    Checks if the current time is within the configured active trade hours.
    Returns True if it is, False otherwise.
    """
    if not ACTIVE_TRADE_HOURS:
        return True # If no hours are configured, bot is always active.

    now = datetime.datetime.now().time()
    for interval in ACTIVE_TRADE_HOURS:
        try:
            start_time = datetime.datetime.strptime(interval["start"], "%H:%M").time()
            end_time = datetime.datetime.strptime(interval["end"], "%H:%M").time()
            if start_time <= now <= end_time:
                return True
        except (ValueError, KeyError) as e:
            logger.error(f"Invalid time format in config.json for active_trade_hours: {e}. Please use 'HH:MM'.")
            return False
    return False

def check_market_filters(symbol, ticker_data, min_volume_usd, max_volatility):
    """
    Checks if a symbol meets the minimum volume and maximum volatility criteria.
    Returns True if it passes, False otherwise.
    """
    try:
        # Get 24-hour volume in USD
        quote_volume = float(ticker_data.get('quoteVolume', 0))
        volume_usd = quote_volume

        # Calculate 24-hour volatility
        high = float(ticker_data.get('highPrice', 0))
        low = float(ticker_data.get('lowPrice', 0))
        close = float(ticker_data.get('lastPrice', 0))

        if close == 0:
            volatility = 1.0 # Or some large value to fail the filter
        else:
            volatility = (high - low) / close

        # Apply filters
        if volume_usd < min_volume_usd:
            logger.info(f"Skipping {symbol}: 24h volume ({volume_usd:,.2f} USD) is below min_volume_usd ({min_volume_usd:,.2f} USD).")
            return False
        if volatility > max_volatility:
            logger.info(f"Skipping {symbol}: 24h volatility ({volatility:.2%}) is above max_volatility ({max_volatility:.2%}).")
            return False

        logger.info(f"✅ {symbol} passed market filters. Volume: {volume_usd:,.2f} USD, Volatility: {volatility:.2%}.")
        return True

    except Exception as e:
        logger.error(f"Error checking market filters for {symbol}: {e}")
        return False


# --- ML Signal Booster Class ---
class MLSignalBooster:
    def __init__(self, model_path):
        self.model = None
        self.model_path = model_path
        self.features = [
            'close', 'volume', 'rsi', 'ema_fast', 'ema_slow', 'macd',
            'macd_signal', 'macd_hist', 'bb_bbm', 'bb_bbh', 'bb_bbl',
            'stoch_rsi_k', 'stoch_rsi_d', 'adx', 'adx_pos', 'adx_neg'
        ]
        self._load_model()

    def _load_model(self):
        """Loads the pre-trained XGBoost model if it exists."""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                logger.info(f"✅ ML model loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load ML model from {self.model_path}: {e}")
                self.model = None
        else:
            logger.warning(f"⚠️ ML model not found at {self.model_path}. ML booster will be disabled.")
            self.model = None

    def _prepare_data(self, df):
        """
        Prepares the data for the ML model by selecting features and handling NaNs.
        This is a crucial step for training.
        """
        # Drop rows with NaN values in the features
        df_ml = df[self.features].copy().dropna()
        return df_ml

    def train_model(self, symbol, timeframe, lookback):
        """
        (EXPERIMENTAL) Trains a new XGBoost model.
        This method is for one-time training, not part of the main bot loop.

        To train, you would need to:
        1. Fetch sufficient historical data.
        2. Define a 'target' variable (e.g., did the price increase by > 1% in the next 4 hours?).
        3. Split the data into training and testing sets.
        4. Train and save the model.
        """
        logger.warning(f"Starting ML model training for {symbol} ({timeframe}). This is an offline process and will not be run during the main trading loop.")
        df = get_klines(symbol, interval=timeframe, lookback=lookback)
        if df is None or df.empty:
            logger.error("Could not fetch data for ML training. Aborting.")
            return

        # Add indicators to the dataframe
        df = add_indicators(df)

        # Define the target variable: a simplified example for demonstration
        # `target = 1` if the price increases by more than 0.5% in the next 12 hours, `0` otherwise.
        # This is a key part of your strategy and should be customized.
        df['future_price_high'] = df['close'].shift(-12).rolling(window=12).max()
        df['target'] = (df['future_price_high'] > df['close'] * 1.005).astype(int)

        # Prepare the features and target
        df_ml = self._prepare_data(df)

        if df_ml.empty:
            logger.error("Dataframe is empty after dropping NaNs. Cannot train model.")
            return

        X = df_ml[self.features]
        y = df_ml['target']

        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        # Initialize and train the XGBoost classifier
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

        # Evaluate the model
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        logger.info(f"Model trained with accuracy: {accuracy:.2%}")

        # Save the trained model
        joblib.dump(model, self.model_path)
        logger.info(f"✅ ML model saved to {self.model_path}")

    def get_ml_score(self, df):
        """
        Predicts a confidence score for a new data point using the trained model.
        Returns a score between 0 and 1, or None if the model is not loaded.
        """
        if self.model is None:
            return None

        # Take the latest data point
        latest_data = df.iloc[-1:]

        # Make sure the data has the required features and no NaNs
        features_to_predict = self._prepare_data(latest_data)

        if features_to_predict.empty:
            return None

        try:
            # Predict the probability of the positive class (i.e., a good signal)
            prediction_proba = self.model.predict_proba(features_to_predict[self.features])
            return prediction_proba[0][1] # Get the probability of the '1' class
        except Exception as e:
            logger.error(f"Error predicting with ML model: {e}")
            return None


# --- Trade Manager Class ---
class TradeManager:
    """Manages paper and real trades, position tracking, and logging."""
    def __init__(self, client, realtime_mode, paper_balance_initial, base_trade_amount_usd,
                 min_trade_amount_usd, max_trade_amount_usd, max_total_loss_usd, daily_max_loss_usd, symbols_config,
                 gsheet_client, spreadsheet_id, firebase_db_client, max_volatility_for_sizing, max_diversified_positions):
        self.client = client
        self.real_mode = realtime_mode
        self.paper_balance = paper_balance_initial
        self.initial_paper_balance = paper_balance_initial
        self.base_trade_amount_usd = base_trade_amount_usd
        self.min_trade_amount_usd = min_trade_amount_usd
        self.max_trade_amount_usd = max_trade_amount_usd
        self.max_volatility_for_sizing = max_volatility_for_sizing
        self.max_total_loss_usd = max_total_loss_usd
        self.daily_max_loss_usd = daily_max_loss_usd
        self.symbols_config = symbols_config
        self.paper_positions = {}
        self.gsheet_client = gsheet_client
        self.spreadsheet_id = spreadsheet_id
        self.bot_is_paused_permanent = False
        self.bot_is_paused_daily = False
        self.max_diversified_positions = max_diversified_positions

        self.db = firebase_db_client
        self.app_id = APP_ID
        self.user_id = USER_ID
        self.bot_state_doc_ref = None
        self.trade_history_collection_ref = None

        # New variables for daily loss tracking
        self.daily_pnl = 0.0
        self.last_trade_date = datetime.date.today().isoformat()

        # Initialize ML Booster
        self.ml_booster = MLSignalBooster(ML_MODEL_PATH) if ML_BOOSTER_ENABLED else None
        
        # Initialize News Sentiment Integration
        self.sentiment_integration = NewsSentimentIntegration(config_data)

        if self.db:
            self.bot_state_doc_ref = self.db.collection(f"artifacts/{self.app_id}/users/{self.user_id}/settings").document("bot_state")
            self.trade_history_collection_ref = self.db.collection(f"artifacts/{self.app_id}/users/{self.user_id}/trades")
            self._load_bot_state()

        self.last_daily_summary_date = datetime.date.today()
        self.last_hourly_summary_time = datetime.datetime.now()

        # Flag to prevent spamming Telegram during sleep mode
        self.bot_is_sleeping = False
        # Daily PnL accumulator for the summary sheet, separate from the loss guard
        self.daily_pnl_accumulator = 0.0

        logger.info(f"TradeManager initialized. Realtime Mode: {self.real_mode}, Paper Balance (Initial): {self.initial_paper_balance:.2f} USD")
        logger.info(f"Current Paper Balance (after load): {self.paper_balance:,.2f} USD")
        logger.info(f"Bot Active: {'Yes' if not (self.bot_is_paused_permanent or self.bot_is_paused_daily) else 'No'}")
        logger.info(f"Daily Realized PnL: {self.daily_pnl:,.2f} USD. Daily Max Loss Guard: -{self.daily_max_loss_usd:.2f} USD")


    def _save_bot_state(self):
        """Saves the current bot state (balance, positions) to Firestore."""
        if not self.db or not self.bot_state_doc_ref:
            logger.warning("Firestore client not initialized. Cannot save bot state.")
            return

        try:
            state_data = {
                "paper_balance": self.paper_balance,
                "initial_paper_balance": self.initial_paper_balance,
                "paper_positions": json.dumps(self.paper_positions),
                "bot_is_paused_permanent": self.bot_is_paused_permanent,
                "daily_pnl": self.daily_pnl,
                "last_trade_date": self.last_trade_date,
                "last_updated": firestore.SERVER_TIMESTAMP
            }
            self.bot_state_doc_ref.set(state_data)
            logger.debug("Bot state saved to Firestore.")
        except Exception as e:
            logger.error(f"Error saving bot state to Firestore: {e}")

    def _load_bot_state(self):
        """Loads the bot state from Firestore at startup."""
        if not self.db or not self.bot_state_doc_ref:
            logger.warning("Firestore client not initialized. Cannot load bot state.")
            return

        try:
            doc = self.bot_state_doc_ref.get()
            if doc.exists:
                state_data = doc.to_dict()
                self.paper_balance = state_data.get("paper_balance", self.initial_paper_balance)
                self.initial_paper_balance = state_data.get("initial_paper_balance", self.initial_paper_balance)
                positions_json = state_data.get("paper_positions", "{}")
                self.paper_positions = json.loads(positions_json)
                self.bot_is_paused_permanent = state_data.get("bot_is_paused_permanent", False)
                self.daily_pnl = state_data.get("daily_pnl", 0.0)
                self.last_trade_date = state_data.get("last_trade_date", datetime.date.today().isoformat())
                logger.info("Bot state loaded from Firestore.")

                # Check if it's a new day and reset daily PnL if needed
                current_date = datetime.date.today().isoformat()
                if current_date != self.last_trade_date:
                    logger.info("New day detected. Resetting daily PnL.")
                    self.daily_pnl = 0.0
                    self.last_trade_date = current_date
                    self._save_bot_state() # Save the reset PnL
                else:
                    # Check if the daily loss limit was hit on a previous run today
                    if self.daily_pnl <= -self.daily_max_loss_usd:
                        self.bot_is_paused_daily = True
                        logger.warning(f"Daily loss limit was hit earlier today. Bot is paused for the day.")

            else:
                logger.info("No existing bot state found in Firestore. Starting with initial balance.")
                self._save_bot_state()
        except Exception as e:
            logger.error(f"Error loading bot state from Firestore: {e}")

    def _log_trade_to_firestore(self, trade_details):
        """Logs individual trade details to a Firestore collection."""
        if not self.db or not self.trade_history_collection_ref:
            logger.warning("Firestore client not initialized. Cannot log trade to Firestore.")
            return

        try:
            trade_details["timestamp_server"] = firestore.SERVER_TIMESTAMP
            self.trade_history_collection_ref.add(trade_details)
            logger.debug(f"Trade logged to Firestore: {trade_details.get('symbol', 'N/A')} {trade_details.get('type', 'N/A')}")
        except Exception as e:
            logger.error(f"Error logging trade to Firestore: {e}")

    def log_trade_to_sheet(self, trade_data):
        """Logs trade details to the configured Google Sheet."""
        if not self.gsheet_client or not self.spreadsheet_id:
            logger.warning("Google Sheet client not initialized or Spreadsheet ID missing. Cannot log trade.")
            return

        try:
            worksheet = self.gsheet_client.open_by_key(self.spreadsheet_id).sheet1

            # Convert timestamp to IST and 24-hour format before logging
            ist_tz = pytz.timezone('Asia/Kolkata')
            now_ist = datetime.datetime.now(pytz.utc).astimezone(ist_tz)
            formatted_timestamp = now_ist.strftime('%d-%m-%Y %H:%M:%S')

            indicator_details = trade_data.get('indicator_details', {})

            # The log data to append
            row_to_append = [
                formatted_timestamp,
                trade_data.get('symbol', 'N/A'),
                trade_data.get('type', 'N/A'),
                f"{trade_data.get('price', 'N/A'):.4f}" if isinstance(trade_data.get('price'), (int, float)) else trade_data.get('price', 'N/A'),
                f"{trade_data.get('quantity', 'N/A'):.4f}" if isinstance(trade_data.get('quantity'), (int, float)) else trade_data.get('quantity', 'N/A'),
                f"{trade_data.get('pnl', 'N/A'):.4f}" if isinstance(trade_data.get('pnl'), (int, float)) else trade_data.get('pnl', 'N/A'),
                trade_data.get('reason', 'N/A'),
                # Indicator values
                f"{indicator_details.get('rsi', 'N/A'):.2f}" if isinstance(indicator_details.get('rsi'), (int, float)) else indicator_details.get('rsi', 'N/A'),
                f"{indicator_details.get('ema_fast', 'N/A'):.2f}" if isinstance(indicator_details.get('ema_fast'), (int, float)) else indicator_details.get('ema_fast', 'N/A'),
                f"{indicator_details.get('macd_hist', 'N/A'):.4f}" if isinstance(indicator_details.get('macd_hist'), (int, float)) else indicator_details.get('macd_hist', 'N/A'),
                f"{indicator_details.get('stoch_rsi_k', 'N/A'):.2f}" if isinstance(indicator_details.get('stoch_rsi_k'), (int, float)) else indicator_details.get('stoch_rsi_k', 'N/A'),
                f"{indicator_details.get('stoch_rsi_d', 'N/A'):.2f}" if isinstance(indicator_details.get('stoch_rsi_d'), (int, float)) else indicator_details.get('stoch_rsi_d', 'N/A'),
                f"{indicator_details.get('bb_bbh', 'N/A'):.2f}" if isinstance(indicator_details.get('bb_bbh'), (int, float)) else indicator_details.get('bb_bbh', 'N/A'),
                f"{indicator_details.get('bb_bbl', 'N/A'):.2f}" if isinstance(indicator_details.get('bb_bbl'), (int, float)) else indicator_details.get('bb_bbl', 'N/A'),
                f"{indicator_details.get('bb_bbm', 'N/A'):.2f}" if isinstance(indicator_details.get('bb_bbm'), (int, float)) else indicator_details.get('bb_bbm', 'N/A'),
                f"{indicator_details.get('adx', 'N/A'):.2f}" if isinstance(indicator_details.get('adx'), (int, float)) else indicator_details.get('adx', 'N/A'),
                f"{indicator_details.get('adx_pos', 'N/A'):.2f}" if isinstance(indicator_details.get('adx_pos'), (int, float)) else indicator_details.get('adx_pos', 'N/A'),
                f"{indicator_details.get('adx_neg', 'N/A'):.2f}" if isinstance(indicator_details.get('adx_neg'), (int, float)) else indicator_details.get('adx_neg', 'N/A'),
            ]

            worksheet.append_row(row_to_append)
            logger.info(f"Trade logged to Google Sheet: {row_to_append}")
        except Exception as e:
            logger.error(f"Error logging trade to Google Sheet: {e}")

    def log_ml_training_data(self, symbol, df_primary, df_confirm, signal_data, market_data, trade_outcome=None):
        """
        Logs comprehensive data to ML Training Data sheet for model training.
        This captures market conditions, signals, and outcomes for learning.
        """
        if not self.gsheet_client or not self.spreadsheet_id:
            logger.warning("Google Sheet client not initialized. Cannot log ML training data.")
            return

        try:
            # Try to access ML_Training_Data worksheet, create if doesn't exist
            try:
                ml_worksheet = self.gsheet_client.open_by_key(self.spreadsheet_id).worksheet("ML_Training_Data")
            except gspread.exceptions.WorksheetNotFound:
                # Create the worksheet with headers
                spreadsheet = self.gsheet_client.open_by_key(self.spreadsheet_id)
                ml_worksheet = spreadsheet.add_worksheet(title="ML_Training_Data", rows="1000", cols="50")
                
                # Add comprehensive headers for ML training
                headers = [
                    'timestamp', 'symbol', 'primary_timeframe', 'confirm_timeframe',
                    # Market Data
                    'current_price', 'volume_24h', 'volatility_24h', 'high_24h', 'low_24h',
                    # Primary Timeframe Indicators
                    'pri_rsi', 'pri_ema_fast', 'pri_ema_slow', 'pri_macd', 'pri_macd_signal', 'pri_macd_hist',
                    'pri_bb_upper', 'pri_bb_middle', 'pri_bb_lower', 'pri_stoch_rsi_k', 'pri_stoch_rsi_d',
                    'pri_adx', 'pri_adx_pos', 'pri_adx_neg',
                    # Confirmation Timeframe Indicators
                    'conf_rsi', 'conf_ema_fast', 'conf_ema_slow', 'conf_macd', 'conf_macd_signal', 'conf_macd_hist',
                    'conf_bb_upper', 'conf_bb_middle', 'conf_bb_lower', 'conf_stoch_rsi_k', 'conf_stoch_rsi_d',
                    'conf_adx', 'conf_adx_pos', 'conf_adx_neg',
                    # Signal Data
                    'primary_signal', 'primary_score', 'confirm_signal', 'confirm_score', 'combined_score',
                    'ml_confidence_score', 'signal_strength',
                    # Market Context
                    'existing_position', 'portfolio_balance', 'open_positions_count', 'daily_pnl',
                    # Trade Execution Data
                    'trade_executed', 'trade_type', 'trade_price', 'trade_quantity', 'trade_reason',
                    # Outcome Data (filled later for closed positions)
                    'outcome_pnl', 'outcome_duration_minutes', 'outcome_classification', 'max_drawdown', 'max_profit'
                ]
                ml_worksheet.append_row(headers)
                logger.info("Created ML_Training_Data worksheet with headers")

            # Convert timestamp to IST
            ist_tz = pytz.timezone('Asia/Kolkata')
            now_ist = datetime.datetime.now(pytz.utc).astimezone(ist_tz)
            formatted_timestamp = now_ist.strftime('%d-%m-%Y %H:%M:%S')

            # Get latest data from both timeframes
            pri_latest = df_primary.iloc[-1] if not df_primary.empty else {}
            conf_latest = df_confirm.iloc[-1] if not df_confirm.empty else {}

            # Prepare the comprehensive data row
            ml_data_row = [
                # Basic info
                formatted_timestamp, symbol, PRIMARY_TIMEFRAME, CONFIRMATION_TIMEFRAME,
                
                # Market data
                f"{market_data.get('current_price', 0):.4f}",
                f"{market_data.get('volume_24h', 0):.2f}",
                f"{market_data.get('volatility_24h', 0):.4f}",
                f"{market_data.get('high_24h', 0):.4f}",
                f"{market_data.get('low_24h', 0):.4f}",
                
                # Primary timeframe indicators
                f"{pri_latest.get('rsi', 0):.2f}" if pri_latest.get('rsi') is not None else "N/A",
                f"{pri_latest.get('ema_fast', 0):.4f}" if pri_latest.get('ema_fast') is not None else "N/A",
                f"{pri_latest.get('ema_slow', 0):.4f}" if pri_latest.get('ema_slow') is not None else "N/A",
                f"{pri_latest.get('macd', 0):.4f}" if pri_latest.get('macd') is not None else "N/A",
                f"{pri_latest.get('macd_signal', 0):.4f}" if pri_latest.get('macd_signal') is not None else "N/A",
                f"{pri_latest.get('macd_hist', 0):.4f}" if pri_latest.get('macd_hist') is not None else "N/A",
                f"{pri_latest.get('bb_bbh', 0):.4f}" if pri_latest.get('bb_bbh') is not None else "N/A",
                f"{pri_latest.get('bb_bbm', 0):.4f}" if pri_latest.get('bb_bbm') is not None else "N/A",
                f"{pri_latest.get('bb_bbl', 0):.4f}" if pri_latest.get('bb_bbl') is not None else "N/A",
                f"{pri_latest.get('stoch_rsi_k', 0):.2f}" if pri_latest.get('stoch_rsi_k') is not None else "N/A",
                f"{pri_latest.get('stoch_rsi_d', 0):.2f}" if pri_latest.get('stoch_rsi_d') is not None else "N/A",
                f"{pri_latest.get('adx', 0):.2f}" if pri_latest.get('adx') is not None else "N/A",
                f"{pri_latest.get('adx_pos', 0):.2f}" if pri_latest.get('adx_pos') is not None else "N/A",
                f"{pri_latest.get('adx_neg', 0):.2f}" if pri_latest.get('adx_neg') is not None else "N/A",
                
                # Confirmation timeframe indicators
                f"{conf_latest.get('rsi', 0):.2f}" if conf_latest.get('rsi') is not None else "N/A",
                f"{conf_latest.get('ema_fast', 0):.4f}" if conf_latest.get('ema_fast') is not None else "N/A",
                f"{conf_latest.get('ema_slow', 0):.4f}" if conf_latest.get('ema_slow') is not None else "N/A",
                f"{conf_latest.get('macd', 0):.4f}" if conf_latest.get('macd') is not None else "N/A",
                f"{conf_latest.get('macd_signal', 0):.4f}" if conf_latest.get('macd_signal') is not None else "N/A",
                f"{conf_latest.get('macd_hist', 0):.4f}" if conf_latest.get('macd_hist') is not None else "N/A",
                f"{conf_latest.get('bb_bbh', 0):.4f}" if conf_latest.get('bb_bbh') is not None else "N/A",
                f"{conf_latest.get('bb_bbm', 0):.4f}" if conf_latest.get('bb_bbm') is not None else "N/A",
                f"{conf_latest.get('bb_bbl', 0):.4f}" if conf_latest.get('bb_bbl') is not None else "N/A",
                f"{conf_latest.get('stoch_rsi_k', 0):.2f}" if conf_latest.get('stoch_rsi_k') is not None else "N/A",
                f"{conf_latest.get('stoch_rsi_d', 0):.2f}" if conf_latest.get('stoch_rsi_d') is not None else "N/A",
                f"{conf_latest.get('adx', 0):.2f}" if conf_latest.get('adx') is not None else "N/A",
                f"{conf_latest.get('adx_pos', 0):.2f}" if conf_latest.get('adx_pos') is not None else "N/A",
                f"{conf_latest.get('adx_neg', 0):.2f}" if conf_latest.get('adx_neg') is not None else "N/A",
                
                # Signal data
                signal_data.get('primary_signal', 'No Trade Signal'),
                f"{signal_data.get('primary_score', 0):.2f}",
                signal_data.get('confirm_signal', 'No Trade Signal'),
                f"{signal_data.get('confirm_score', 0):.2f}",
                f"{signal_data.get('combined_score', 0):.2f}",
                f"{signal_data.get('ml_confidence', 0):.3f}" if signal_data.get('ml_confidence') is not None else "N/A",
                signal_data.get('signal_strength', 'WEAK'),
                
                # Market context
                'YES' if symbol in self.paper_positions else 'NO',
                f"{self.paper_balance:.2f}",
                len(self.paper_positions),
                f"{self.daily_pnl:.2f}",
                
                # Trade execution data
                'YES' if signal_data.get('trade_executed', False) else 'NO',
                signal_data.get('trade_type', 'N/A'),
                f"{signal_data.get('trade_price', 0):.4f}" if signal_data.get('trade_price') else "N/A",
                f"{signal_data.get('trade_quantity', 0):.6f}" if signal_data.get('trade_quantity') else "N/A",
                signal_data.get('trade_reason', 'N/A'),
                
                # Outcome data (will be updated later for trades that complete)
                f"{trade_outcome.get('pnl', 0):.4f}" if trade_outcome else "PENDING",
                f"{trade_outcome.get('duration_minutes', 0)}" if trade_outcome else "PENDING",
                trade_outcome.get('classification', 'PENDING') if trade_outcome else "PENDING",
                f"{trade_outcome.get('max_drawdown', 0):.4f}" if trade_outcome else "PENDING",
                f"{trade_outcome.get('max_profit', 0):.4f}" if trade_outcome else "PENDING"
            ]

            ml_worksheet.append_row(ml_data_row)
            logger.info(f"ML training data logged for {symbol}")

        except Exception as e:
            logger.error(f"Error logging ML training data: {e}")

    def update_ml_outcome_data(self, symbol, trade_entry_timestamp, outcome_data):
        """
        Updates the outcome data for a completed trade in the ML training sheet.
        This allows the bot to learn from the actual results of its decisions.
        """
        if not self.gsheet_client or not self.spreadsheet_id:
            return

        try:
            ml_worksheet = self.gsheet_client.open_by_key(self.spreadsheet_id).worksheet("ML_Training_Data")
            
            # Find the row with matching symbol and timestamp
            all_records = ml_worksheet.get_all_records()
            
            for i, record in enumerate(all_records, start=2):  # Start at 2 because of header row
                if (record['symbol'] == symbol and 
                    record['timestamp'] == trade_entry_timestamp and 
                    record['outcome_classification'] == 'PENDING'):
                    
                    # Update the outcome columns
                    outcome_range = f"AX{i}:BB{i}"  # Columns AX to BB are the outcome columns
                    outcome_values = [[
                        f"{outcome_data.get('pnl', 0):.4f}",
                        f"{outcome_data.get('duration_minutes', 0)}",
                        outcome_data.get('classification', 'UNKNOWN'),
                        f"{outcome_data.get('max_drawdown', 0):.4f}",
                        f"{outcome_data.get('max_profit', 0):.4f}"
                    ]]
                    
                    ml_worksheet.update(outcome_range, outcome_values)
                    logger.info(f"Updated ML outcome data for {symbol} trade")
                    break

        except Exception as e:
            logger.error(f"Error updating ML outcome data: {e}")

    def log_no_trade_to_sheet(self):
        """Logs a 'No Trade' event to the Google Sheet."""
        if not self.gsheet_client or not self.spreadsheet_id:
            logger.warning("Google Sheet client not initialized or Spreadsheet ID missing. Cannot log 'no trade' event.")
            return

        # Convert timestamp to IST and 24-hour format
        ist_tz = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.datetime.now(pytz.utc).astimezone(ist_tz)
        formatted_timestamp = now_ist.strftime('%d-%m-%Y %H:%M:%S')

        # Create a row with 'NO_TRADE' and empty values for other columns
        no_trade_data = [formatted_timestamp, "N/A", "NO_TRADE", "N/A", "N/A", "N/A", "N/A"] + ["N/A"] * 12
        try:
            worksheet = self.gsheet_client.open_by_key(self.spreadsheet_id).sheet1
            worksheet.append_row(no_trade_data)
            logger.info("Logged 'No Trade' event to Google Sheet.")
        except Exception as e:
            logger.error(f"Error logging 'No Trade' to Google Sheet: {e}")

    def log_daily_summary(self):
        """Logs daily summary of paper trading balance and PnL to Google Sheet and saves bot state."""
        if not self.gsheet_client or not self.spreadsheet_id:
            logger.warning("Google Sheet client not initialized or Spreadsheet ID missing. Cannot log daily summary.")
            return

        current_date = datetime.date.today()
        if current_date > self.last_daily_summary_date:
            try:
                summary_sheet = self.gsheet_client.open_by_key(self.spreadsheet_id).worksheet("Summary")
                summary_data = [
                    self.last_daily_summary_date.strftime('%Y-%m-%d'),
                    f"{self.paper_balance:.2f}",
                    f"{self.daily_pnl:.2f}"
                ]
                summary_sheet.append_row(summary_data)
                logger.info(f"Daily Summary logged: Date: {self.last_daily_summary_date}, Balance: {self.paper_balance:.2f}, Daily PnL: {self.daily_pnl:.2f}")

                self.daily_pnl = 0.0
                self.last_daily_summary_date = current_date
                self._save_bot_state()

            except gspread.exceptions.WorksheetNotFound:
                logger.error("Google Sheet 'Summary' tab not found. Please create a sheet named 'Summary'.")
            except Exception as e:
                logger.error(f"Error logging daily summary to Google Sheet: {e}")

    def get_open_positions_pnl(self, current_prices):
        """
        Calculates the total unrealized P&L for all open paper positions
        and returns a formatted string of individual positions.
        """
        total_unrealized_pnl = 0.0
        positions_summary_lines = []

        if not self.paper_positions:
            return 0.0, "No open positions."

        for symbol, position in self.paper_positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                unrealized_pnl = (current_price - position["entry_price"]) * position["quantity"]
                total_unrealized_pnl += unrealized_pnl

                # Add a line for the trailing stop price
                positions_summary_lines.append(
                    f"  • {symbol}: Qty {position['quantity']:.4f}, Entry {position['entry_price']:.2f}, "
                    f"Current {current_price:.2f}, TSL {position['current_trailing_stop_price']:.2f}, "
                    f"PnL {unrealized_pnl:.2f} USD"
                )
            else:
                positions_summary_lines.append(f"  • {symbol}: Current price not available.")

        return total_unrealized_pnl, "\n".join(positions_summary_lines)


    def _format_quantity(self, symbol, quantity):
        """Formats quantity to the correct precision for the symbol."""
        info = get_symbol_info(symbol)
        if info:
            precision = info['quantityPrecision']
            return float(f"{quantity:.{precision}f}")
        return quantity

    def _format_price(self, symbol, price):
        """Formats price to the correct precision for the symbol."""
        info = get_symbol_info(symbol)
        if info:
            precision = info['pricePrecision']
            return float(f"{price:.{precision}f}")
        return price

    def calculate_adaptive_trade_size(self, symbol, signal_score, volatility):
        """
        Calculates the trade size in USD based on signal score and volatility.
        A higher signal score and lower volatility lead to a larger trade size.
        """
        if not ADAPTIVE_SIZING_ENABLED:
            logger.info(f"Adaptive sizing is disabled. Using base trade amount: {self.base_trade_amount_usd} USD")
            return self.base_trade_amount_usd

        # Normalize signal score
        # The range of signal scores is roughly 1.0 to 4.0 based on the weights.
        # We can map this to a multiplier.
        min_score = 1.0
        max_score = sum(SIGNAL_WEIGHTS.values()) # Max possible score

        # Avoid division by zero if weights are all zero
        if max_score > min_score:
            score_normalized = (signal_score - min_score) / (max_score - min_score)
            score_multiplier = 1 + score_normalized * 1.5 # Example multiplier from 1.0 to 2.5
        else:
            score_multiplier = 1.0

        # Normalize volatility
        # We want to decrease trade size as volatility increases
        # We cap the volatility at MAX_VOLATILITY_FOR_SIZING to avoid a tiny trade size on extreme spikes
        volatility_capped = min(volatility, self.max_volatility_for_sizing)

        # Inverse relationship: higher volatility -> smaller multiplier
        # Max volatility maps to a multiplier of 0.5 (e.g.)
        # Min volatility (close to 0) maps to a multiplier of 1.5 (e.g.)
        volatility_multiplier = 1.5 - (volatility_capped / self.max_volatility_for_sizing)
        volatility_multiplier = max(0.5, volatility_multiplier) # Ensure a floor

        # Combine factors
        trade_amount_usd = self.base_trade_amount_usd * score_multiplier * volatility_multiplier

        # Clamp the trade amount to the defined min/max range
        trade_amount_usd = max(self.min_trade_amount_usd, min(self.max_trade_amount_usd, trade_amount_usd))

        logger.info(f"Adaptive Sizing for {symbol}: Score={signal_score:.2f}, Volatility={volatility:.2%}. Calculated trade size: {trade_amount_usd:.2f} USD")

        return trade_amount_usd

    async def execute_trade(self, symbol, signal_type, current_price, reason="SIGNAL", indicator_details=None, signal_score=0, volatility=0.0):
        """
        Executes a trade (real or paper) based on the signal.
        Returns True on success, False otherwise.
        """
        if indicator_details is None:
            indicator_details = {}

        # The trade amount is now dynamically calculated
        trade_amount_usd = self.calculate_adaptive_trade_size(symbol, signal_score, volatility)
        trade_quantity_raw = trade_amount_usd / current_price

        symbol_info = get_symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Could not get symbol info for {symbol}. Cannot execute trade.")
            return False

        formatted_quantity = self._format_quantity(symbol, trade_quantity_raw)

        if formatted_quantity < symbol_info['minQty']:
            logger.warning(f"Trade quantity {formatted_quantity} for {symbol} is less than minimum quantity {symbol_info['minQty']}. Adjusting to min_qty for paper/logging, but might fail real trade.")
            formatted_quantity = symbol_info['minQty']

        trade_notional = formatted_quantity * current_price
        if trade_notional < symbol_info['minNotional']:
            logger.warning(f"Trade notional value for {symbol} ({trade_notional:.2f}) is less than minimum notional {symbol_info['minNotional']}. Cannot execute trade.")
            await send_telegram_message(f"🚫 Trade Failed for {symbol} ({signal_type}): Notional value too low ({trade_notional:.2f} USD). Min: {symbol_info['minNotional']} USD.")
            return False

        symbol_params = self.symbols_config.get(symbol, {})
        sl_percent = symbol_params.get("sl", 0.02)
        # Note: The fixed TP percentage is no longer used for the exit logic.
        tsl_percent = symbol_params.get("tsl", 0.01)

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Updated to create the dictionary once
        common_trade_log_details = {
            'rsi': indicator_details.get('rsi', 'N/A'),
            'ema_fast': indicator_details.get('ema_fast', 'N/A'),
            'macd_hist': indicator_details.get('macd_hist', 'N/A'),
            'stoch_rsi_k': indicator_details.get('stoch_rsi_k', 'N/A'),
            'stoch_rsi_d': indicator_details.get('stoch_rsi_d', 'N/A'),
            'bb_bbh': indicator_details.get('bb_bbh', 'N/A'),
            'bb_bbl': indicator_details.get('bb_bbl', 'N/A'),
            'bb_bbm': indicator_details.get('bb_bbm', 'N/A'),
            'adx': indicator_details.get('adx', 'N/A'),
            'adx_pos': indicator_details.get('adx_pos', 'N/A'),
            'adx_neg': indicator_details.get('adx_neg', 'N/A')
        }

        if self.real_mode:
            logger.info(f"Attempting REAL {signal_type} order for {symbol} at {current_price:.2f} with quantity {formatted_quantity}")

            cost = formatted_quantity * current_price
            available_balance = get_usdt_balance()
            if available_balance < cost:
                error_message = f"❌ REAL BUY FAILED for {symbol}: Insufficient USDT balance. Needed: {cost:.2f}, Have: {available_balance:.2f}"
                logger.error(error_message)
                await send_telegram_message(error_message)
                return False

            try:
                if signal_type == "BUY":
                    order = robust_apis.create_order(
                        symbol=symbol,
                        side=SIDE_BUY,
                        order_type=ORDER_TYPE_MARKET,
                        quantity=formatted_quantity
                    )
                elif signal_type == "SELL":
                    order = robust_apis.create_order(
                        symbol=symbol,
                        side=SIDE_SELL,
                        order_type=ORDER_TYPE_MARKET,
                        quantity=formatted_quantity
                    )

                trade_message = (
                    f"✅ REAL TRADE EXECUTED! (Reason: {reason})\n"
                    f"Symbol: {symbol}\n"
                    f"Type: {signal_type}\n"
                    f"Order ID: {order['orderId']}\n"
                    f"Executed Qty: {order['executedQty']}\n"
                    f"Price: {order['fills'][0]['price'] if order['fills'] else 'N/A'}\n"
                    f"Status: {order['status']}\n"
                    f"Signal Score: {signal_score:.2f}"
                )
                logger.info(trade_message)
                await send_telegram_message(trade_message)

                trade_data_log = {
                    "symbol": symbol,
                    "type": f"{signal_type} (Real) - {reason}",
                    "price": float(order['fills'][0]['price']) if order['fills'] else current_price,
                    "quantity": float(order['executedQty']),
                    "pnl": 0.0,
                    "reason": reason,
                    "indicator_details": common_trade_log_details
                }
                self.log_trade_to_sheet(trade_data_log)

                trade_data_firestore = {
                    "timestamp": timestamp, "symbol": symbol, "type": signal_type,
                    "quantity": float(order['executedQty']),
                    "price": float(order['fills'][0]['price']) if order['fills'] else current_price,
                    "pnl": 0.0, "reason": reason, "indicator_details": common_trade_log_details,
                    "entry_price": float(order['fills'][0]['price']) if order['fills'] else current_price,
                    "sl_price_at_entry": "N/A", "tp_price_at_entry": "N/A",
                    "tsl_price_at_hit": "N/A", "real_trade": True, "signal_score": signal_score
                }
                self._log_trade_to_firestore(trade_data_firestore)
                self._save_bot_state()
                return True

            except Exception as e:
                error_message = f"❌ REAL TRADE FAILED for {symbol} ({signal_type}): {e}"
                logger.error(error_message)
                await send_telegram_message(error_message)
                return False

        else: # Paper trading logic
            logger.info(f"Executing PAPER {signal_type} order for {symbol} at {current_price:.2f} with quantity {formatted_quantity}. Reason: {reason}")
            cost = formatted_quantity * current_price

            if signal_type == "BUY":
                if self.paper_balance >= cost:
                    self.paper_balance -= cost

                    initial_stop_loss_price = current_price * (1 - sl_percent)
                    # Note: The fixed take profit price is no longer set here.
                    initial_trailing_stop_price = current_price * (1 - tsl_percent)

                    self.paper_positions[symbol] = {
                        "side": "BUY", "quantity": formatted_quantity, "entry_price": current_price,
                        "stop_loss": initial_stop_loss_price,
                        "highest_price_since_entry": current_price, "current_trailing_stop_price": initial_trailing_stop_price,
                    }
                    trade_message = (
                        f"📈 PAPER BUY Order Executed! (Reason: {reason})\n"
                        f"Symbol: {symbol}\n"
                        f"Price: {current_price:.2f}\n"
                        f"Quantity: {formatted_quantity}\n"
                        f"Virtual SL: {self.paper_positions[symbol]['stop_loss']:.2f}\n"
                        f"Virtual TSL: {self.paper_positions[symbol]['current_trailing_stop_price']:.2f}\n"
                        f"Remaining Paper Balance: {self.paper_balance:.2f} USD\n"
                        f"Signal Score: {signal_score:.2f}"
                    )
                    logger.info(trade_message)
                    await send_telegram_message(trade_message)

                    trade_data_log = {
                        "symbol": symbol,
                        "type": "BUY (Paper)",
                        "price": current_price,
                        "quantity": formatted_quantity,
                        "pnl": 0.0,
                        "reason": reason,
                        "indicator_details": common_trade_log_details
                    }
                    self.log_trade_to_sheet(trade_data_log)

                    trade_data_firestore = {
                        "timestamp": timestamp, "symbol": symbol, "type": "BUY",
                        "quantity": formatted_quantity, "price": current_price,
                        "pnl": 0.0, "reason": reason, "indicator_details": common_trade_log_details,
                        "entry_price": current_price, "sl_price_at_entry": initial_stop_loss_price,
                        "tp_price_at_entry": "N/A",
                        "tsl_price_at_hit": initial_trailing_stop_price, "real_trade": False,
                        "signal_score": signal_score
                    }
                    self._log_trade_to_firestore(trade_data_firestore)
                    self._save_bot_state()
                    return True
                else:
                    message = f"Insufficient paper balance to BUY {symbol}. Needed: {cost:.2f} USD, Have: {self.paper_balance:.2f} USD"
                    logger.warning(message)
                    await send_telegram_message(message)
                    return False

            elif signal_type == "SELL":
                if symbol in self.paper_positions and self.paper_positions[symbol]["side"] == "BUY":
                    position_qty = self.paper_positions[symbol]["quantity"]
                    entry_price = self.paper_positions[symbol]["entry_price"]
                    sl_price_at_entry = self.paper_positions[symbol]["stop_loss"]
                    tsl_price_at_hit = self.paper_positions[symbol]["current_trailing_stop_price"]

                    revenue = position_qty * current_price
                    profit_loss = (current_price - entry_price) * position_qty
                    self.paper_balance += revenue
                    self.daily_pnl += profit_loss

                    trade_message = (
                        f"📉 PAPER SELL Order Executed (Closing Position)! (Reason: {reason})\n"
                        f"Symbol: {symbol}\n"
                        f"Close Price: {current_price:.2f}\n"
                        f"Quantity: {position_qty}\n"
                        f"Entry Price: {entry_price:.2f}\n"
                        f"Profit/Loss: {profit_loss:.2f} USD\n"
                        f"New Paper Balance: {self.paper_balance:.2f} USD\n"
                        f"Daily Realized PnL: {self.daily_pnl:.2f} USD\n"
                        f"Signal Score: {signal_score:.2f}"
                    )
                    logger.info(trade_message)
                    await send_telegram_message(trade_message)

                    trade_data_log = {
                        "symbol": symbol,
                        "type": "SELL (Paper)",
                        "price": current_price,
                        "quantity": position_qty,
                        "pnl": profit_loss,
                        "reason": reason,
                        "indicator_details": common_trade_log_details
                    }
                    self.log_trade_to_sheet(trade_data_log)

                    trade_data_firestore = {
                        "timestamp": timestamp, "symbol": symbol, "type": "SELL",
                        "quantity": position_qty, "price": current_price,
                        "pnl": profit_loss, "reason": reason, "indicator_details": common_trade_log_details,
                        "entry_price": entry_price, "sl_price_at_entry": sl_price_at_entry,
                        "tp_price_at_entry": "N/A",
                        "tsl_price_at_hit": tsl_price_at_hit, "real_trade": False,
                        "signal_score": signal_score
                    }
                    self._log_trade_to_firestore(trade_data_firestore)

                    # Calculate trade metrics for ML learning
                    trade_duration_minutes = 60  # Placeholder - you could track actual time held
                    max_profit_during_trade = max(0, profit_loss)  # Simplified - could track actual max
                    max_drawdown_during_trade = min(0, profit_loss)  # Simplified - could track actual min
                    
                    # Classify the trade outcome for ML learning
                    if profit_loss > 10:  # Profitable trades above $10
                        outcome_classification = "WIN_STRONG"
                    elif profit_loss > 0:  # Small profitable trades
                        outcome_classification = "WIN_SMALL"
                    elif profit_loss > -5:  # Small losses
                        outcome_classification = "LOSS_SMALL"
                    else:  # Significant losses
                        outcome_classification = "LOSS_STRONG"

                    # Prepare outcome data for ML learning
                    outcome_data = {
                        'pnl': profit_loss,
                        'duration_minutes': trade_duration_minutes,
                        'classification': outcome_classification,
                        'max_drawdown': max_drawdown_during_trade,
                        'max_profit': max_profit_during_trade
                    }

                    # Update ML training data with actual outcome
                    trade_entry_time = datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')  # Simplified - ideally track actual entry time
                    self.update_ml_outcome_data(symbol, trade_entry_time, outcome_data)

                    del self.paper_positions[symbol]
                    self.daily_pnl_accumulator += profit_loss # This accumulator is for the summary sheet, separate from daily PnL guard
                    self._save_bot_state()

                    # Check for total loss threshold after a trade closes a position
                    total_realized_pnl = self.paper_balance - self.initial_paper_balance
                    if total_realized_pnl <= -self.max_total_loss_usd:
                        self.bot_is_paused_permanent = True
                        logger.critical(f"Total realized PnL ({total_realized_pnl:.2f} USD) has exceeded the max loss limit of -{self.max_total_loss_usd:.2f} USD. Pausing bot.")
                        final_message = (
                            f"🛑 CRITICAL ALERT: Bot paused due to exceeding max total loss limit.\n"
                            f"Total Realized Loss: {total_realized_pnl:,.2f} USD\n"
                            f"Max Loss Limit: -{self.max_total_loss_usd:,.2f} USD\n"
                            f"Please restart the bot manually after reviewing your strategy."
                        )
                        await send_telegram_message(final_message)
                        self._save_bot_state()
                        exit(1) # Stop the script entirely

                    # Check for daily loss threshold
                    if self.daily_pnl <= -self.daily_max_loss_usd:
                        self.bot_is_paused_daily = True
                        message = (
                            f"🛑 DAILY LOSS GUARD ACTIVATED!\n\n"
                            f"Daily realized loss of {self.daily_pnl:,.2f} USD has met or exceeded the daily max loss of -{self.daily_max_loss_usd:,.2f} USD.\n"
                            f"The bot will pause all trading activity for the rest of the day to protect your capital. It will resume automatically tomorrow."
                        )
                        await send_telegram_message(message)
                        self._save_bot_state()

                    return True

                else:
                    logger.warning(f"Attempted to SELL {symbol} but no BUY position found or side mismatch.")
                    await send_telegram_message(f"🚫 Failed to SELL {symbol}: No active BUY position to close.")
                    return False

    async def send_no_trade_message(self):
        """Sends a 'No Trade' message to Telegram and logs to sheets/firestore."""
        message = (
            f"✅ Bot Status: Active and Running!\n\n"
            f"No trades were triggered this cycle based on the current strategy. The bot is continuing to monitor the markets."
        )
        logger.info("No trades executed this cycle.")
        await send_telegram_message(message)
        self.log_no_trade_to_sheet()

        # Log to Firestore
        no_trade_details = {
            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "type": "NO_TRADE",
            "message": "No trades triggered this cycle.",
            "real_trade": self.real_mode
        }
        self._log_trade_to_firestore(no_trade_details)

    async def check_and_manage_positions(self, current_prices):
        """
        Manages open paper positions:
        - Updates highest price since entry for TSL.
        - Updates trailing stop loss price.
        - Checks for Stop Loss, Take Profit, or Trailing Stop Loss hits.
        - Executes SELL orders for hits.
        """
        positions_to_close = []
        trade_happened_in_cycle = False

        # Optimization: Fetch and calculate indicators once before the loop
        latest_bollinger_bands = {}
        for symbol in self.paper_positions.keys():
            df = get_klines(symbol, interval=PRIMARY_TIMEFRAME, lookback=SETTINGS.get("lookback", 250))
            if df is not None and not df.empty:
                df = add_indicators(df)
                if 'bb_bbh' in df.columns and df['bb_bbh'].iloc[-1] is not None:
                    latest_bollinger_bands[symbol] = df['bb_bbh'].iloc[-1]
            else:
                logger.warning(f"Could not get latest klines for {symbol}. Cannot check for dynamic TP.")

        for symbol, position in list(self.paper_positions.items()):
            if symbol not in current_prices:
                logger.warning(f"Skipping position management for {symbol}: current price not available.")
                continue

            current_price = current_prices[symbol]
            entry_price = position['entry_price']
            pos_quantity = position['quantity']

            # Update highest price and TSL
            if current_price > position['highest_price_since_entry']:
                position['highest_price_since_entry'] = current_price
                tsl_percent = self.symbols_config.get(symbol, {}).get("tsl", 0.01)
                new_trailing_stop = position['highest_price_since_entry'] * (1 - tsl_percent)
                if new_trailing_stop > position['current_trailing_stop_price']:
                    position['current_trailing_stop_price'] = new_trailing_stop
                    logger.debug(f"Updated TSL for {symbol} to {new_trailing_stop:.2f}")

            close_reason = None
            # Check for dynamic Take Profit based on Bollinger Band
            if latest_bollinger_bands.get(symbol) and current_price >= latest_bollinger_bands[symbol]:
                close_reason = "BOLLINGER_BAND_EXIT"
            # Check for Stop Loss
            elif current_price <= position['stop_loss']:
                close_reason = "STOP_LOSS"
            # Check for Trailing Stop Loss
            elif position['current_trailing_stop_price'] and current_price <= position['current_trailing_stop_price']:
                if position['current_trailing_stop_price'] > position['entry_price']:
                    close_reason = "TRAILING_STOP_LOSS (profit protected)"
                else:
                    close_reason = "TRAILING_STOP_LOSS (loss minimized)"

            if close_reason:
                logger.info(f"Position close signal for {symbol} due to {close_reason} at {current_price:.2f}. Entry: {entry_price:.2f}")
                # We need to explicitly check the result of the trade execution
                if await self.execute_trade(symbol, "SELL", current_prices[symbol], reason=close_reason, signal_score=0, volatility=0.0):
                    trade_happened_in_cycle = True

        return trade_happened_in_cycle


    async def run_strategy_for_symbol(self, symbol, df, current_price, sentiment_data=None):
        """
        Implements a multi-indicator trading strategy with sentiment analysis and returns a score.
        This function no longer executes trades directly.
        """
        max_lookback = max(
            INDICATORS_SETTINGS.get("rsi_period", 14),
            INDICATORS_SETTINGS.get("ema_fast", 50),
            INDICATORS_SETTINGS.get("ema_slow", 200),
            INDICATORS_SETTINGS.get("macd_slow", 26) + INDICATORS_SETTINGS.get("macd_signal", 9),
            INDICATORS_SETTINGS.get("bollinger_period", 20),
            INDICATORS_SETTINGS.get("stoch_rsi_period", 14),
            INDICATORS_SETTINGS.get("adx_period", 14) + 14
        )

        if df is None or len(df) < max_lookback:
            logger.warning(f"Not enough data for {symbol} to calculate indicators.")
            return "No Trade Signal", 0, {}, None, "Insufficient data"

        latest_data = df.iloc[-1]
        previous_data = df.iloc[-2]

        indicator_details = {
            'rsi': latest_data['rsi'], 'ema_fast': latest_data['ema_fast'],
            'macd_hist': latest_data['macd_hist'], 'stoch_rsi_k': latest_data['stoch_rsi_k'],
            'stoch_rsi_d': latest_data['stoch_rsi_d'], 'bb_bbh': latest_data['bb_bbh'],
            'bb_bbl': latest_data['bb_bbl'], 'bb_bbm': latest_data['bb_bbm'],
            'adx': latest_data['adx'], 'adx_pos': latest_data['adx_pos'],
            'adx_neg': latest_data['adx_neg']
        }

        has_position = symbol in self.paper_positions
        buy_score = 0
        sell_score = 0

        RSI_BUY_THRESHOLD = 30
        RSI_SELL_THRESHOLD = 70
        ADX_TREND_THRESHOLD = 25
        BOLLINGER_BAND_TOLERANCE = 0.005

        # Check for Buy Signals and add their scores
        if latest_data['rsi'] is not None and latest_data['rsi'] < RSI_BUY_THRESHOLD:
            buy_score += SIGNAL_WEIGHTS.get("rsi_buy", 1.0)

        if latest_data['bb_bbl'] is not None and current_price < latest_data['bb_bbl'] * (1 + BOLLINGER_BAND_TOLERANCE):
            buy_score += SIGNAL_WEIGHTS.get("bollinger_buy", 1.0)

        if latest_data['adx'] is not None and latest_data['adx'] > ADX_TREND_THRESHOLD and latest_data['adx_pos'] > latest_data['adx_neg']:
            buy_score += SIGNAL_WEIGHTS.get("adx_uptrend", 1.0)

        if latest_data['macd'] is not None and latest_data['macd_signal'] is not None and latest_data['macd'] > latest_data['macd_signal'] and previous_data['macd'] <= previous_data['macd_signal']:
            buy_score += SIGNAL_WEIGHTS.get("macd_bullish_cross", 1.0)

        # Check for Sell Signals and add their scores
        if latest_data['rsi'] is not None and latest_data['rsi'] > RSI_SELL_THRESHOLD:
            sell_score += SIGNAL_WEIGHTS.get("rsi_sell", 1.0)

        if latest_data['bb_bbh'] is not None and current_price > latest_data['bb_bbh'] * (1 - BOLLINGER_BAND_TOLERANCE):
            sell_score += SIGNAL_WEIGHTS.get("bollinger_sell", 1.0)

        if latest_data['adx'] is not None and latest_data['adx'] > ADX_TREND_THRESHOLD and latest_data['adx_neg'] > latest_data['adx_pos']:
            sell_score += SIGNAL_WEIGHTS.get("adx_downtrend", 1.0)

        if latest_data['macd'] is not None and latest_data['macd_signal'] is not None and latest_data['macd'] < latest_data['macd_signal'] and previous_data['macd'] >= previous_data['macd_signal']:
            sell_score += SIGNAL_WEIGHTS.get("macd_bearish_cross", 1.0)

        # Get ML score if enabled
        ml_score = None
        if self.ml_booster:
            ml_score = self.ml_booster.get_ml_score(df)
            if ml_score is not None:
                if ml_score >= ML_CONFIDENCE_THRESHOLD:
                    buy_score += ML_SIGNAL_BOOST_WEIGHT
                    logger.info(f"ML booster validated BUY signal for {symbol} with confidence {ml_score:.2f}. Score boosted by {ML_SIGNAL_BOOST_WEIGHT}.")

        # Apply sentiment analysis to technical signals
        sentiment_reasoning = "No sentiment data"
        if sentiment_data and sentiment_data.get('sufficient_data', False):
            # Add sentiment-based signals
            sentiment_rec = sentiment_data.get('recommendation', '')
            
            if 'BULLISH' in sentiment_rec:
                sentiment_boost = SIGNAL_WEIGHTS.get("sentiment_bullish", 0.8)
                if 'STRONG' in sentiment_rec:
                    sentiment_boost *= 1.5
                buy_score += sentiment_boost
                
            elif 'BEARISH' in sentiment_rec:
                sentiment_boost = SIGNAL_WEIGHTS.get("sentiment_bearish", 0.8)
                if 'STRONG' in sentiment_rec:
                    sentiment_boost *= 1.5
                sell_score += sentiment_boost

        # Apply sentiment adjustment to final scores
        if sentiment_data:
            if buy_score > 0:
                buy_score, buy_reasoning = self.sentiment_integration.apply_sentiment_to_technical_signal(
                    buy_score, sentiment_data, "BUY"
                )
                sentiment_reasoning = buy_reasoning
            elif sell_score > 0:
                sell_score, sell_reasoning = self.sentiment_integration.apply_sentiment_to_technical_signal(
                    sell_score, sentiment_data, "SELL"
                )
                sentiment_reasoning = sell_reasoning

        # Return the signal and score
        if buy_score >= MIN_SIGNAL_SCORE_BUY and not has_position:
            return "BUY", buy_score, indicator_details, ml_score, sentiment_reasoning
        elif sell_score >= MIN_SIGNAL_SCORE_SELL and has_position:
            return "SELL", sell_score, indicator_details, ml_score, sentiment_reasoning
        else:
            return "No Trade Signal", 0, indicator_details, ml_score, sentiment_reasoning


    async def run_trading_cycle(self):
        """
        Executes a single trading cycle: fetches data, runs strategy, manages positions,
        and executes trades based on a diversification strategy.
        """
        # Check for new day and reset daily PnL
        current_date = datetime.date.today().isoformat()
        if current_date != self.last_trade_date:
            logger.info(f"New day ({current_date}) detected. Resetting daily PnL.")
            self.daily_pnl = 0.0
            self.last_trade_date = current_date
            self.bot_is_paused_daily = False # Reset daily pause
            self._save_bot_state()

        if self.bot_is_paused_permanent or self.bot_is_paused_daily:
            logger.info("Bot is paused due to a loss limit. Skipping trading cycle.")
            bot_status_message = format_bot_status(self, {}) # Pass empty dict as no prices are fetched
            await send_telegram_message(f"⏸️ **Bot is paused.**\n\n{bot_status_message}")
            return

        logger.info(f"--- Starting Trading Cycle ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")

        # Initialize a flag to detect if any trades were executed
        trade_executed_this_cycle = False

        current_prices = {}
        # Get ticker data for all symbols to use for market filtering and adaptive sizing
        try:
            tickers_map = robust_apis.get_ticker_data()
        except Exception as e:
            logger.error(f"Failed to get ticker data for market filtering: {e}")
            tickers_map = {}

        for symbol in ACTIVE_SYMBOLS:
            price = get_latest_price(symbol)
            if price:
                current_prices[symbol] = price
            else:
                logger.warning(f"Could not get latest price for {symbol}. Skipping this symbol for this cycle.")

        if not current_prices:
            logger.warning("No current prices retrieved for any active symbol. Skipping trading cycle.")
            await self.send_no_trade_message()
            return

        # First, check and manage all open positions for stop-loss or dynamic take-profit
        if await self.check_and_manage_positions(current_prices):
            trade_executed_this_cycle = True

        # Get sentiment analysis for all symbols
        logger.info("Fetching sentiment analysis...")
        sentiment_signals = await self.sentiment_integration.get_sentiment_signals(ACTIVE_SYMBOLS)
        
        # Log sentiment summary
        for symbol in ACTIVE_SYMBOLS:
            if symbol in sentiment_signals:
                sentiment_summary = self.sentiment_integration.log_sentiment_summary(symbol, sentiment_signals[symbol])
                logger.info(f"Sentiment: {sentiment_summary}")

        # Get market sentiment overview
        market_sentiment = await self.sentiment_integration.get_market_sentiment_overview(ACTIVE_SYMBOLS)
        logger.info(f"Market Sentiment: {market_sentiment['market_mood']} (Score: {market_sentiment['market_sentiment_score']:.3f}, News: {market_sentiment['total_news_analyzed']})")

        # Collect all multi-timeframe confirmed buy and sell signals with their scores
        all_buy_signals = []
        all_sell_signals = []

        for symbol in ACTIVE_SYMBOLS:
            if symbol not in current_prices:
                continue

            # Get ticker data for the symbol
            ticker_data = tickers_map.get(symbol, {})
            # Apply market filters first
            if not check_market_filters(symbol, ticker_data, MIN_VOLUME_USD, MAX_VOLATILITY):
                continue

            # Calculate volatility for adaptive sizing
            high = float(ticker_data.get('highPrice', 0))
            low = float(ticker_data.get('lowPrice', 0))
            close = float(ticker_data.get('lastPrice', 0))
            volatility = (high - low) / close if close != 0 else 1.0

            # Fetch data for primary timeframe (e.g., 15m)
            df_primary = get_klines(symbol, interval=PRIMARY_TIMEFRAME, lookback=SETTINGS.get("lookback", 250))
            # Fetch data for confirmation timeframe (e.g., 1h)
            df_confirm = get_klines(symbol, interval=CONFIRMATION_TIMEFRAME, lookback=SETTINGS.get("lookback", 250))

            if df_primary is None or df_confirm is None or df_primary.empty or df_confirm.empty:
                logger.warning(f"Could not retrieve klines for {symbol} for multi-timeframe analysis. Skipping.")
                continue

            df_primary = add_indicators(df_primary)
            df_confirm = add_indicators(df_confirm)

            # Get sentiment data for this symbol
            symbol_sentiment = sentiment_signals.get(symbol, {})

            primary_signal, primary_score, primary_details, ml_score_primary, primary_sentiment_reasoning = await self.run_strategy_for_symbol(symbol, df_primary, current_prices[symbol], symbol_sentiment)
            confirm_signal, confirm_score, confirm_details, ml_score_confirm, confirm_sentiment_reasoning = await self.run_strategy_for_symbol(symbol, df_confirm, current_prices[symbol], symbol_sentiment)

            ml_score = ml_score_primary # We'll use the primary timeframe ML score for the trade

            # Multi-timeframe confirmation logic
            has_position = symbol in self.paper_positions
            combined_score = primary_score + confirm_score

            # Prepare market data for ML logging
            market_data = {
                'current_price': current_prices[symbol],
                'volume_24h': float(ticker_data.get('quoteVolume', 0)),
                'volatility_24h': volatility,
                'high_24h': high,
                'low_24h': low
            }

            # Prepare signal data for ML logging
            signal_data = {
                'primary_signal': primary_signal,
                'primary_score': primary_score,
                'confirm_signal': confirm_signal,
                'confirm_score': confirm_score,
                'combined_score': combined_score,
                'ml_confidence': ml_score,
                'signal_strength': 'STRONG' if combined_score >= 3.0 else 'MEDIUM' if combined_score >= 2.0 else 'WEAK',
                'trade_executed': False,  # Will be updated if trade executes
                'trade_type': 'N/A',
                'trade_price': None,
                'trade_quantity': None,
                'trade_reason': 'N/A'
            }

            # Determine if we should execute trade
            trade_should_execute = False
            trade_type = None

            if primary_signal == "BUY" and confirm_signal == "BUY" and not has_position:
                combined_score = primary_score + confirm_score
                all_buy_signals.append({
                    'symbol': symbol,
                    'score': combined_score,
                    'details': primary_details,
                    'ml_score': ml_score,
                    'price': current_prices[symbol],
                    'volatility': volatility
                })
                trade_should_execute = True
                trade_type = "BUY"
            elif primary_signal == "SELL" and confirm_signal == "SELL" and has_position:
                combined_score = primary_score + confirm_score
                all_sell_signals.append({
                    'symbol': symbol,
                    'score': combined_score,
                    'details': primary_details,
                    'ml_score': ml_score,
                    'price': current_prices[symbol],
                    'volatility': volatility
                })
                trade_should_execute = True
                trade_type = "SELL"

            # Update signal data if trade would execute
            if trade_should_execute:
                signal_data.update({
                    'trade_executed': True,  # We'll confirm this after actual execution
                    'trade_type': trade_type,
                    'trade_price': current_prices[symbol],
                    'trade_quantity': self.calculate_adaptive_trade_size(symbol, combined_score, volatility) / current_prices[symbol] if trade_type == "BUY" else self.paper_positions.get(symbol, {}).get('quantity', 0),
                    'trade_reason': f"Multi-Timeframe {trade_type} (Score: {combined_score:.2f}) + Sentiment: {primary_sentiment_reasoning[:50]}"
                })

            # Log ALL analysis to ML training data (not just executed trades)
            self.log_ml_training_data(symbol, df_primary, df_confirm, signal_data, market_data)

        # Execute all pending SELL signals first
        for sell_signal in all_sell_signals:
            symbol = sell_signal['symbol']
            price = sell_signal['price']
            score = sell_signal['score']
            volatility = sell_signal['volatility']
            
            # Add sentiment info to reason
            symbol_sentiment = sentiment_signals.get(symbol, {})
            sentiment_info = f"Sentiment: {symbol_sentiment.get('recommendation', 'N/A')}"
            reason = f"Multi-Timeframe Sell (Score: {score:.2f}) + {sentiment_info}"
            
            if await self.execute_trade(symbol, "SELL", price, reason=reason, indicator_details=sell_signal['details'], signal_score=score, volatility=volatility):
                trade_executed_this_cycle = True

        # Now, sort the BUY signals by score and execute trades for the top N
        all_buy_signals.sort(key=lambda x: x['score'], reverse=True)

        num_open_positions = len(self.paper_positions)
        available_slots = self.max_diversified_positions - num_open_positions

        if available_slots > 0:
            top_buy_signals = all_buy_signals[:available_slots]
            for buy_signal in top_buy_signals:
                symbol = buy_signal['symbol']
                price = buy_signal['price']
                score = buy_signal['score']
                indicator_details = buy_signal['details']
                volatility = buy_signal['volatility']
                
                # Add sentiment info to reason
                symbol_sentiment = sentiment_signals.get(symbol, {})
                sentiment_info = f"Sentiment: {symbol_sentiment.get('recommendation', 'N/A')}"
                reason = f"Multi-Timeframe Buy (Score: {score:.2f}) + {sentiment_info}"

                if await self.execute_trade(symbol, "BUY", price, reason=reason, indicator_details=indicator_details, signal_score=score, volatility=volatility):
                    trade_executed_this_cycle = True
        else:
            logger.info(f"Max diversified positions ({self.max_diversified_positions}) reached. Skipping new buy orders.")

        # If no trades were executed this cycle, send a "no trade" message
        if not trade_executed_this_cycle:
            await self.send_no_trade_message()

        # Generate and send a single, comprehensive status report to Telegram
        bot_status_message = format_bot_status(self, current_prices)
        await send_telegram_message(bot_status_message)

        # Log to Firestore and Google Sheets
        self.log_daily_summary()
        self.log_hourly_summary_to_firestore()

        logger.info("--- Trading Cycle Completed ---")

    def log_hourly_summary_to_firestore(self):
        """Logs hourly summary of PnL to Firestore."""
        current_time = datetime.datetime.now()
        if (current_time - self.last_hourly_summary_time).total_seconds() >= 3600:
            start_of_last_hour = current_time - datetime.timedelta(hours=1)

            if not self.db or not self.trade_history_collection_ref:
                logger.warning("Firestore client not initialized. Cannot log hourly summary to Firestore.")
                return

            try:
                docs = self.trade_history_collection_ref.order_by("timestamp_server", direction=firestore.Query.DESCENDING).limit(200).stream()
                hourly_pnl = 0.0
                hourly_trade_count = 0

                for doc in docs:
                    trade_data = doc.to_dict()
                    trade_timestamp_str = trade_data.get('timestamp')
                    if trade_timestamp_str:
                        try:
                            trade_dt = datetime.datetime.strptime(trade_timestamp_str, '%Y-%m-%d %H:%M:%S')
                            if start_of_last_hour <= trade_dt <= current_time:
                                pnl = trade_data.get('pnl', 0.0)
                                if isinstance(pnl, (int, float)):
                                    hourly_pnl += pnl
                                else:
                                    try:
                                        hourly_pnl += float(pnl)
                                    except ValueError:
                                        pass
                                hourly_trade_count += 1
                        except ValueError:
                            logger.warning(f"Could not parse timestamp string: {trade_timestamp_str}")
                            continue

                hourly_summary_ref = self.db.collection(f"artifacts/{self.app_id}/users/{self.user_id}/hourly_summaries")
                summary_data = {
                    "start_time": start_of_last_hour, "end_time": current_time,
                    "hourly_pnl": hourly_pnl, "hourly_trade_count": hourly_trade_count,
                    "paper_balance_at_end": self.paper_balance,
                    "last_updated": firestore.SERVER_TIMESTAMP
                }
                hourly_summary_ref.add(summary_data)
                logger.info(f"Hourly summary logged to Firestore: PnL={hourly_pnl:.2f}, Trades={hourly_trade_count}")
                self.last_hourly_summary_time = current_time

            except Exception as e:
                logger.error(f"Error logging hourly summary to Firestore: {e}")

# --- Data Collection Module ---
DATA_FILE = 'trading_data.csv'
HEADERS = ['timestamp', 'signal_type', 'price_at_signal', 'outcome']
SIGNAL_DURATION_SECONDS = 60

def initialize_data_file():
    """
    Creates the CSV file with headers if it does not already exist.
    This prevents overwriting valuable data.
    """
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(HEADERS)
        print(f"Created new data file: {DATA_FILE}")

def get_outcome(price_at_signal, wait_time):
    """
    This function determines the outcome of a trade by waiting for a
    set duration and comparing the initial and final prices.

    Args:
        price_at_signal (float): The price of the asset when the signal was generated.
        wait_time (int): The number of seconds to wait before checking the final price.

    Returns:
        str: 'WIN', 'LOSS', or 'NEUTRAL'.
    """
    print(f"Waiting for {wait_time} seconds to determine outcome...")
    time.sleep(wait_time)

    # IMPORTANT: You must ensure 'get_current_price()' is the correct function
    # name from your existing code that retrieves the live price.
    # The function get_current_price() is not defined in the provided code.
    # We will use get_latest_price as a placeholder.
    final_price = get_latest_price("BTCUSDT") # Placeholder, assuming BTCUSDT for demonstration

    if final_price > price_at_signal:
        return 'WIN'
    elif final_price < price_at_signal:
        return 'LOSS'
    else:
        return 'NEUTRAL'

def save_data(data_row):
    """
    Appends a new row of data to the CSV file.

    Args:
        data_row (list): A list containing the timestamp, signal, price, and outcome.
    """
    try:
        with open(DATA_FILE, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data_row)
        print(f"Data saved: {data_row}")
    except Exception as e:
        print(f"Failed to save data: {e}")

def get_trading_data(symbol):
    """
    Simulates fetching real-time trading data for a given symbol.
    In a real-world scenario, you would replace this with an API call.
    """
    try:
        # Simulate a fluctuating price
        price = round(random.uniform(100.0, 200.0), 2)
        # Simulate a random volume
        volume = random.randint(1000, 100000)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())

        data = {
            "timestamp": timestamp,
            "symbol": symbol,
            "price": price,
            "volume": volume
        }
        print(f"Generated data for {symbol}: {data}")
        return data

    except Exception as e:
        print(f"An error occurred while generating data: {e}")
        return None

def write_data_to_google_sheet(sheet_title, data):
    """
    Authenticates with Google and writes a single row of data to the sheet.
    """
    try:
        # Authenticate using the credentials.json file.
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
        client = gspread.authorize(creds)

        # Open the Google Sheet by its title.
        sheet = client.open(sheet_title).sheet1
        print(f"Successfully connected to Google Sheet: '{sheet_title}'")

        # Append the data as a new row. The keys of the dictionary
        # will be the column headers, and the values will be the data.
        # Ensure the headers in your Google Sheet match these keys.
        row = [data["timestamp"], data["symbol"], data["price"], data["volume"]]
        sheet.append_row(row)
        print("Data successfully appended to Google Sheet.")

    except gspread.exceptions.SpreadsheetNotFound:
        print(f"Error: The Google Sheet with the title '{sheet_title}' was not found.")
        print("Please check the sheet title and ensure the service account has access.")
    except Exception as e:
        print(f"An error occurred while writing to Google Sheet: {e}")
        print("Please check your credentials.json file and internet connection.")

def main_data_collection():
    """
    Main function to collect data and write it to the Google Sheet.
    """
    symbols = ['AAPL', 'GOOG', 'MSFT']

    # We will only run this once to demonstrate.
    # In a real application, you would loop this or run it on a schedule.
    for symbol in symbols:
        data = get_trading_data(symbol)
        if data:
            write_data_to_google_sheet("Trading Data", data) # Hardcoded sheet title for this example
            # Add a small delay to prevent API rate limits
            time.sleep(2)

# --- Security Functions ---
API_KEY = "YOUR_API_KEY" # Placeholder
API_SECRET = "YOUR_API_SECRET".encode('utf-8') # Placeholder, must be bytes for hmac

def create_signed_request(payload: dict) -> dict:
    """
    Creates a request payload with an HMAC-SHA256 signature.
    This is a critical security measure to authenticate your requests
    to the exchange's API.

    Args:
        payload (dict): The data to be sent in the API request.

    Returns:
        dict: The original payload with a 'signature' field added.
    """
    # Convert the payload to a JSON string and encode to bytes.
    # The signature is generated on the payload itself.
    payload_json = json.dumps(payload, separators=(',', ':'))

    # Create the HMAC signature using SHA256 and the API secret.
    signature = hmac.new(
        API_SECRET,
        msg=payload_json.encode('utf-8'),
        digestmod=hashlib.sha256
    ).hexdigest()

    # Add the signature and API key to the request payload.
    payload['api_key'] = API_KEY
    payload['signature'] = signature

    return payload

def validate_trade_order(order: dict) -> bool:
    """
    Validates a trade order to ensure all required fields are present
    and have correct data types and values. This prevents bad data from
    being sent to the API.

    Args:
        order (dict): A dictionary representing the trade order.

    Returns:
        bool: True if the order is valid, False otherwise.
    """
    try:
        # Check for required keys
        if not all(k in order for k in ['symbol', 'type', 'quantity', 'price']):
            print("Error: Missing required fields in trade order.")
            return False

        # Check for data types
        if not isinstance(order['symbol'], str):
            print("Error: 'symbol' must be a string.")
            return False
        if not isinstance(order['type'], str):
            print("Error: 'type' must be a string.")
            return False
        if not isinstance(order['quantity'], (int, float)):
            print("Error: 'quantity' must be a number.")
            return False
        if not isinstance(order['price'], (int, float)):
            print("Error: 'price' must be a number.")
            return False

        # Check for valid values
        if order['quantity'] <= 0:
            print("Error: 'quantity' must be a positive number.")
            return False
        if order['price'] <= 0:
            print("Error: 'price' must be a positive number.")
            return False

        # You can add more checks here, e.g., 'type' in ['buy', 'sell']

        return True

    except Exception as e:
        print(f"Validation Error: An unexpected error occurred during validation: {e}")
        return False

def execute_trade(order_payload: dict):
    """
    Simulates sending a trade order to a trading API with robust
    error handling.

    Args:
        order_payload (dict): The complete payload for the trade,
                              including the signature.
    """
    print("Attempting to execute trade...")
    try:
        # Simulate an API call. In a real bot, you would use a library like `requests`.
        # For this example, we'll simulate a successful and a failed response.

        # Assume a successful API call for demonstration.
        # response = requests.post(url, json=order_payload)

        # Simulate a generic network error
        # if 'error' in response.json():
        #     raise ConnectionError("Simulated API connection failure.")

        print(f"Trade successfully submitted for {order_payload['symbol']}!")
        print(f"Payload sent: {order_payload}")

    except ConnectionError as e:
        # Handle specific connection/network errors
        print(f"Network Error: Could not connect to the trading API. Details: {e}")
    except KeyError as e:
        # Handle errors related to missing keys in the response
        print(f"API Response Error: Missing key in the API response: {e}")
    except Exception as e:
        # Catch any other unexpected errors
        print(f"General Error: An unknown issue occurred during the trade execution: {e}")

# --- Backtesting Logic ---
def backtest_strategy(price_data, sentiment_data, initial_capital=1000):
    """
    Simulates a trading strategy on historical price and sentiment data.

    Args:
        price_data (list): A list of historical prices (e.g., daily closing prices).
        sentiment_data (list): A list of sentiment scores corresponding to each price point.
        initial_capital (float): The starting capital for the simulation.

    Returns:
        A dictionary with backtesting results.
    """
    capital = initial_capital
    position = 0  # 0 = no position, 1 = long position
    trades = []

    print("--- Starting Backtest ---")

    # The core simulation loop, iterating through each historical data point
    for i in range(1, len(price_data)):
        current_price = price_data[i]
        current_sentiment = sentiment_data[i]
        previous_price = price_data[i-1]

        # Simple strategy:
        # Buy if sentiment is positive and price is trending up.
        # Sell if sentiment is negative.
        price_change = current_price - previous_price

        # --- BUY LOGIC ---
        if position == 0 and current_sentiment > 0.6 and price_change > 0:
            # Simulate a buy order
            investment_amount = capital * 0.5  # Invest 50% of capital
            quantity_bought = investment_amount / current_price
            capital -= investment_amount
            position += quantity_bought
            trades.append({'type': 'buy', 'price': current_price, 'quantity': quantity_bought})
            print(f"Time {i}: BUY {quantity_bought:.4f} at {current_price:.2f}. Capital: {capital:.2f}. Position: {position:.4f}")

        # --- SELL LOGIC ---
        elif position > 0 and current_sentiment < 0.4:
            # Simulate a sell order
            sell_value = position * current_price
            capital += sell_value
            trades.append({'type': 'sell', 'price': current_price, 'quantity': position})
            print(f"Time {i}: SELL {position:.4f} at {current_price:.2f}. Capital: {capital:.2f}. Position: 0")
            position = 0 # Reset position to 0

    # Calculate final value and profitability
    final_value = capital + (position * price_data[-1])
    total_profit = final_value - initial_capital

    print("\n--- Backtest Complete ---")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Total Profit/Loss: ${total_profit:.2f}")
    print(f"Number of Trades: {len(trades)}")

    return {
        "initial_capital": initial_capital,
        "final_value": final_value,
        "total_profit": total_profit,
        "num_trades": len(trades)
    }

# --- EXAMPLE USAGE for Backtesting ---
if __name__ == "__main__":
    # Mock data for demonstration purposes. In a real scenario, this would
    # be loaded from a database or a file.
    historical_prices = [
        100, 101, 102, 103, 105, 104, 106, 108, 109, 107,
        105, 104, 103, 102, 101, 100, 99, 98, 97, 96,
        98, 100, 102, 105, 107, 109, 110, 112, 115, 114
    ]

    # Mock sentiment data (e.g., from an NLP model, scaled 0-1)
    historical_sentiment = [
        0.5, 0.6, 0.7, 0.8, 0.9, 0.7, 0.6, 0.8, 0.9, 0.6,
        0.4, 0.3, 0.2, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.9, 0.8, 0.7, 0.6
    ]

    # Run the backtest
    results = backtest_strategy(historical_prices, historical_sentiment)
    print("\nBacktest Results:")
    print(json.dumps(results, indent=4))

# --- Trading Bot Core Logic ---
# news.py - Functions for fetching and processing news/sentiment data
def get_sentiment_data():
    """
    Simulates fetching and processing news sentiment data.
    In a real application, this would connect to a news API and an NLP model.

    Returns:
        list: A list of mock sentiment scores (0-1 scale).
    """
    # For now, we'll return the same mock data as before.
    return [
        0.5, 0.6, 0.7, 0.8, 0.9, 0.7, 0.6, 0.8, 0.9, 0.6,
        0.4, 0.3, 0.2, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.9, 0.8, 0.7, 0.6
    ]

# logger.py - A simple custom logging class
class CustomLogger:
    """
    A simple custom logger class to handle logging to both the console and a file.
    """
    def __init__(self, log_file="app.log", level=logging.INFO):
        # Configure logging
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger()

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)

# test_trading_bot.py - Unit tests for the trading bot
class TestTradingBot(unittest.TestCase):
    """
    Test suite for the core trading bot functions.
    """
    def test_generate_signature(self):
        """
        Tests if the HMAC-SHA256 signature is generated correctly.
        """
        api_secret = "test_secret"
        message = "1678886400POST/api/v1/orders{\"symbol\":\"BTC\",\"side\":\"buy\"}"
        expected_signature = "e102652b36a18d186c3104332997e59b9148d281249b6b9075798c199580a56e"
        self.assertEqual(generate_signature(api_secret, message), expected_signature)

    def test_validate_order_input_valid(self):
        """
        Tests a valid order dictionary.
        """
        valid_order = {
            'symbol': 'BTC',
            'side': 'buy',
            'quantity': 0.1,
            'price': 40000
        }
        self.assertTrue(validate_trade_order(valid_order))

    def test_validate_order_input_invalid(self):
        """
        Tests an invalid order dictionary with missing fields.
        """
        invalid_order = {
            'symbol': 'ETH',
            'side': 'sell'
        }
        self.assertFalse(validate_trade_order(invalid_order))

    def test_backtest_strategy_profitability(self):
        """
        Tests if the backtesting function calculates a profit/loss correctly.
        """
        historical_prices = [100, 110, 105, 120]
        historical_sentiment = [0.5, 0.7, 0.3, 0.8]
        results = backtest_strategy(historical_prices, historical_sentiment, initial_capital=1000)

        # We expect a trade to happen at price 110 (buy), then a trade to happen at 105 (sell),
        # resulting in a small loss. The final value should be less than the initial capital.
        self.assertLess(results['final_value'], 1000)
        self.assertNotEqual(results['total_profit'], 0)
        self.assertEqual(results['num_trades'], 2)

# indicators.py - Functions for technical analysis
# This file contains functions to calculate various technical indicators
# based on price data.

def calculate_moving_average(data, window):
    """
    Calculates the simple moving average for a given dataset and window size.

    Args:
        data (list): A list of numerical data (e.g., prices).
        window (int): The number of data points to include in the average.

    Returns:
        float: The simple moving average, or the last data point if there
               aren't enough data points for the window.
    """
    if len(data) < window:
        # Return the last value if there aren't enough data points for the window
        return data[-1]

    window_data = data[-window:]
    return sum(window_data) / window

def calculate_ema(data, window):
    """
    Calculates the Exponential Moving Average (EMA) for a given dataset and window size.

    Args:
        data (pd.Series): A pandas Series of numerical data (e.g., prices).
        window (int): The number of data points to include in the average.

    Returns:
        float: The EMA value, or NaN if there isn't enough data.
    """
    if len(data) < window:
        return np.nan
    return ta.trend.ema_indicator(data, window=window).iloc[-1]

def calculate_rsi(data, window):
    """
    Calculates the Relative Strength Index (RSI) for a given dataset and window size.

    Args:
        data (pd.Series): A pandas Series of numerical data (e.g., prices).
        window (int): The number of data points to include in the calculation.

    Returns:
        float: The RSI value, or NaN if there isn't enough data.
    """
    if len(data) < window:
        return np.nan
    return ta.momentum.rsi(data, window=window).iloc[-1]

def calculate_macd(data, fast_period, slow_period, signal_period):
    """
    Calculates the MACD line, signal line, and MACD histogram.

    Args:
        data (pd.Series): A pandas Series of numerical data (e.g., prices).
        fast_period (int): The window size for the fast EMA.
        slow_period (int): The window size for the slow EMA.
        signal_period (int): The window size for the signal line EMA.

    Returns:
        tuple: A tuple containing the last MACD line, MACD signal line, and MACD histogram.
               Returns (NaN, NaN, NaN) if there isn't enough data.
    """
    if len(data) < slow_period + signal_period:
        return np.nan, np.nan, np.nan

    macd = ta.trend.MACD(
        close=data,
        window_fast=fast_period,
        window_slow=slow_period,
        window_sign=signal_period
    )
    return macd.macd().iloc[-1], macd.macd_signal().iloc[-1], macd.macd_diff().iloc[-1]

def calculate_bollinger_bands(data, window, window_dev):
    """
    Calculates the Bollinger Bands (middle, upper, and lower bands).

    Args:
        data (pd.Series): A pandas Series of numerical data (e.g., prices).
        window (int): The window size for the moving average.
        window_dev (int): The number of standard deviations for the bands.

    Returns:
        tuple: A tuple containing the last middle, upper, and lower Bollinger Band values.
               Returns (NaN, NaN, NaN) if there isn't enough data.
    """
    if len(data) < window:
        return np.nan, np.nan, np.nan

    bollinger = ta.volatility.BollingerBands(
        close=data,
        window=window,
        window_dev=window_dev
    )
    return bollinger.bollinger_mavg().iloc[-1], bollinger.bollinger_hband().iloc[-1], bollinger.bollinger_lband().iloc[-1]

def calculate_stoch_rsi(data, window, smooth1=3, smooth2=3):
    """
    Calculates the Stochastic RSI (%K and %D).

    Args:
        data (pd.Series): A pandas Series of numerical data (e.g., prices).
        window (int): The window size for the RSI and Stochastic calculation.
        smooth1 (int): The window size for the %K smoothing.
        smooth2 (int): The window size for the %D smoothing.

    Returns:
        tuple: A tuple containing the last Stochastic RSI %K and %D values.
               Returns (NaN, NaN) if there isn't enough data.
    """
    if len(data) < window:
        return np.nan, np.nan

    stoch_rsi = ta.momentum.StochRSIIndicator(
        close=data,
        window=window,
        smooth1=smooth1,
        smooth2=smooth2
    )
    return stoch_rsi.stochrsi_k().iloc[-1], stoch_rsi.stochrsi_d().iloc[-1]

def calculate_adx(high, low, close, window):
    """
    Calculates the Average Directional Index (ADX).

    Args:
        high (pd.Series): A pandas Series of high prices.
        low (pd.Series): A pandas Series of low prices.
        close (pd.Series): A pandas Series of close prices.
        window (int): The window size for the ADX calculation.

    Returns:
        float: The ADX value, or NaN if there isn't enough data.
    """
    if len(close) < window:
        return np.nan

    adx = ta.trend.ADXIndicator(
        high=high,
        low=low,
        close=close,
        window=window
    )
    return adx.adx().iloc[-1]

# --- Main execution block ---
if __name__ == "__main__":
    REALTIME_MODE = SETTINGS.get("realtime_mode", False)
    PAPER_BALANCE_INITIAL = SETTINGS.get("paper_trading_balance", 10000.0)
    # The fixed trade amount is now the base for adaptive sizing
    BASE_TRADE_AMOUNT_USD = SETTINGS.get("trade_amount_usd", 10.0)
    MAX_TOTAL_LOSS_USD = SETTINGS.get("max_total_loss_usd", 2.0)
    # New Daily Max Loss Guard setting as per user request
    DAILY_MAX_LOSS_USD_SETTING = SETTINGS.get("daily_max_loss_usd", 2.0)
    # Using a single, unified sleep interval
    TRADING_INTERVAL_SECONDS = SETTINGS.get("check_interval_seconds", 60)
    MAX_DIVERSIFIED_POSITIONS = SETTINGS.get("max_diversified_positions", 3)
    MIN_VOLUME_USD_SETTING = MARKET_FILTERS.get("min_volume_usd", 1000000.0)
    MAX_VOLATILITY_SETTING = MARKET_FILTERS.get("max_volatility", 0.05)

    # Adaptive sizing settings
    ADAPTIVE_SIZING_ENABLED_SETTING = ADAPTIVE_SIZING_SETTINGS.get("enabled", False)
    MIN_TRADE_AMOUNT_USD_SETTING = ADAPTIVE_SIZING_SETTINGS.get("min_trade_amount_usd", 10.0)
    MAX_TRADE_AMOUNT_USD_SETTING = ADAPTIVE_SIZING_SETTINGS.get("max_trade_amount_usd", 500.0)
    MAX_VOLATILITY_FOR_SIZING_SETTING = ADAPTIVE_SIZING_SETTINGS.get("max_volatility_for_sizing", 0.05)


    trade_manager = TradeManager(
        client=client, realtime_mode=REALTIME_MODE,
        paper_balance_initial=PAPER_BALANCE_INITIAL,
        base_trade_amount_usd=BASE_TRADE_AMOUNT_USD,
        min_trade_amount_usd=MIN_TRADE_AMOUNT_USD_SETTING,
        max_trade_amount_usd=MAX_TRADE_AMOUNT_USD_SETTING,
        max_total_loss_usd=MAX_TOTAL_LOSS_USD,
        daily_max_loss_usd=DAILY_MAX_LOSS_USD_SETTING,
        symbols_config=SYMBOLS_CONFIG, gsheet_client=gsheet,
        spreadsheet_id=GOOGLE_SHEET_ID,
        firebase_db_client=firebase_db,
        max_volatility_for_sizing=MAX_VOLATILITY_FOR_SIZING_SETTING,
        max_diversified_positions=MAX_DIVERSIFIED_POSITIONS
    )

    try:
        async def main():
            # Variables are now defined here, inside the function, which is the correct scope
            last_trade_hours_message_date = None
            last_health_check = datetime.datetime.now()
            health_check_interval = 3600  # 1 hour
            woke_up = False

            while True:
                # Periodic health check
                current_time = datetime.datetime.now()
                if (current_time - last_health_check).total_seconds() >= health_check_interval:
                    try:
                        await health_check_and_recovery()
                        last_health_check = current_time
                    except Exception as e:
                        logger.error(f"Health check failed: {e}")
                # Check and send the daily trade hours message if it's a new day
                current_date = datetime.date.today()
                if last_trade_hours_message_date is None or current_date > last_trade_hours_message_date:
                    await send_daily_trade_hours_message()
                    last_trade_hours_message_date = current_date

                if is_trading_hour():
                    # If the bot was previously sleeping, send a "waking up" message
                    if woke_up == False:
                        await send_telegram_message(f"🟢 **Bot is now active!**\n\nStarting trading cycle for the configured hours.")
                        woke_up = True

                    await trade_manager.run_trading_cycle()
                    logger.info(f"Sleeping for {TRADING_INTERVAL_SECONDS} seconds...")
                    await asyncio.sleep(TRADING_INTERVAL_SECONDS)
                else:
                    # If bot is not in active trading hours, log and wait
                    if woke_up == True:
                        await send_telegram_message(f"😴 **Bot is now in sleep mode.**\n\nWaiting for the next active trading hours to begin.")
                        woke_up = False

                    logger.info("Outside of active trading hours. Sleeping for 60 seconds...")
                    await asyncio.sleep(60) # Short sleep to check the time more frequently

        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped manually by KeyboardInterrupt.")
    except Exception as e:
        logger.critical(f"An unhandled error caused the bot to stop: {e}", exc_info=True)

# --- Initialize data file if it doesn't exist ---
initialize_data_file()

# --- Main execution block for data collection demonstration ---
if __name__ == "__main__":
    # Check and add headers to the Google Sheet if the first row is empty
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
        client = gspread.authorize(creds)
        sheet = client.open("Trading Data").sheet1 # Hardcoded sheet title for this example
        if not sheet.row_values(1):  # Check if the first row is empty
            headers = ["timestamp", "symbol", "price", "volume"]
            sheet.append_row(headers)
            print("Added headers to the Google Sheet.")
    except Exception as e:
        print(f"Could not check or add headers to the Google Sheet: {e}")
        print("Please ensure the sheet exists and the credentials are correct.")

    # Run the main data collection and writing loop
    main_data_collection()

# --- Example of running unit tests ---
if __name__ == '__main__':
    unittest.main()

# --- Indicators Module ---
# This file contains functions to calculate various technical indicators
# based on price data.

# --- You can add more indicator functions here as needed. ---