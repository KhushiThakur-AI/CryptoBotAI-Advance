
import time
import asyncio
import logging
import json
from datetime import datetime, timedelta
from functools import wraps
import requests
from binance.exceptions import BinanceAPIException, BinanceRequestException
import telegram
import gspread
from oauth2client.service_account import ServiceAccountCredentials

logger = logging.getLogger(__name__)

class RetryHandler:
    """Handles retry logic and failover mechanisms for various API calls."""
    
    def __init__(self, max_retries=3, base_delay=1.0, max_delay=60.0, backoff_multiplier=2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.failure_cache = {}
        self.cache_expiry = timedelta(minutes=5)
        
    def exponential_backoff_delay(self, attempt):
        """Calculate exponential backoff delay."""
        delay = min(self.base_delay * (self.backoff_multiplier ** attempt), self.max_delay)
        return delay + (0.1 * attempt)  # Add small jitter

class CacheManager:
    """Manages fallback caching for critical data."""
    
    def __init__(self):
        self.cache = {
            'prices': {},
            'klines': {},
            'balance': None,
            'ticker_data': {}
        }
        self.cache_timestamps = {}
        self.cache_expiry = {
            'prices': timedelta(minutes=2),
            'klines': timedelta(minutes=10),
            'balance': timedelta(minutes=5),
            'ticker_data': timedelta(minutes=3)
        }
    
    def set_cache(self, key, data, subkey=None):
        """Set cached data with timestamp."""
        timestamp = datetime.now()
        if subkey:
            if key not in self.cache:
                self.cache[key] = {}
            if key not in self.cache_timestamps:
                self.cache_timestamps[key] = {}
            self.cache[key][subkey] = data
            self.cache_timestamps[key][subkey] = timestamp
        else:
            self.cache[key] = data
            self.cache_timestamps[key] = timestamp
        logger.debug(f"Cached {key}" + (f"[{subkey}]" if subkey else ""))
    
    def get_cache(self, key, subkey=None):
        """Get cached data if not expired."""
        try:
            if subkey:
                if key not in self.cache or subkey not in self.cache[key]:
                    return None
                timestamp = self.cache_timestamps.get(key, {}).get(subkey)
                data = self.cache[key][subkey]
            else:
                if key not in self.cache:
                    return None
                timestamp = self.cache_timestamps.get(key)
                data = self.cache[key]
            
            if timestamp and datetime.now() - timestamp < self.cache_expiry.get(key, timedelta(minutes=5)):
                logger.debug(f"Using cached {key}" + (f"[{subkey}]" if subkey else ""))
                return data
            else:
                logger.debug(f"Cache expired for {key}" + (f"[{subkey}]" if subkey else ""))
                return None
        except Exception as e:
            logger.error(f"Error retrieving cache for {key}: {e}")
            return None

class TelegramAlerter:
    """Handles Telegram alerts for persistent errors."""
    
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.alert_history = {}
        self.alert_cooldown = timedelta(minutes=10)  # Prevent spam
        
    async def send_alert(self, error_type, message, priority="MEDIUM"):
        """Send Telegram alert with cooldown to prevent spam."""
        current_time = datetime.now()
        alert_key = f"{error_type}_{hash(message)}"
        
        # Check cooldown
        if alert_key in self.alert_history:
            last_sent = self.alert_history[alert_key]
            if current_time - last_sent < self.alert_cooldown:
                return
        
        priority_emoji = {
            "LOW": "⚠️",
            "MEDIUM": "🚨",
            "HIGH": "🔴",
            "CRITICAL": "💥"
        }
        
        formatted_message = (
            f"{priority_emoji.get(priority, '⚠️')} **{priority} ALERT**\n"
            f"**Error Type:** {error_type}\n"
            f"**Time:** {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"**Message:** {message}\n"
            f"**Bot Status:** Attempting recovery..."
        )
        
        try:
            bot = telegram.Bot(token=self.bot_token)
            await bot.send_message(chat_id=self.chat_id, text=formatted_message)
            self.alert_history[alert_key] = current_time
            logger.info(f"Sent {priority} alert for {error_type}")
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")

def retry_on_failure(operation_name, use_cache=False, cache_key=None, alert_priority="MEDIUM"):
    """Decorator for retry logic with failover."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retry_handler = getattr(wrapper, '_retry_handler', RetryHandler())
            cache_manager = getattr(wrapper, '_cache_manager', CacheManager())
            alerter = getattr(wrapper, '_alerter', None)
            
            last_exception = None
            
            for attempt in range(retry_handler.max_retries + 1):
                try:
                    result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                    
                    # Cache successful result if caching is enabled
                    if use_cache and cache_key and result is not None:
                        if len(args) > 0:  # Use first argument as subkey if available
                            cache_manager.set_cache(cache_key, result, str(args[0]))
                        else:
                            cache_manager.set_cache(cache_key, result)
                    
                    if attempt > 0:
                        logger.info(f"{operation_name} succeeded after {attempt} retries")
                    
                    return result
                    
                except (BinanceAPIException, BinanceRequestException, requests.RequestException, 
                        gspread.exceptions.APIError, telegram.error.TelegramError, Exception) as e:
                    last_exception = e
                    
                    # Check if this is a critical error that shouldn't be retried
                    if isinstance(e, BinanceAPIException) and e.code in [-1021, -1022]:  # Timestamp/signature errors
                        logger.error(f"{operation_name} failed with critical error: {e}")
                        break
                    
                    if attempt < retry_handler.max_retries:
                        delay = retry_handler.exponential_backoff_delay(attempt)
                        logger.warning(f"{operation_name} attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s")
                        await asyncio.sleep(delay) if asyncio.iscoroutinefunction(func) else time.sleep(delay)
                    else:
                        logger.error(f"{operation_name} failed after {retry_handler.max_retries} retries: {e}")
                        
                        # Send alert for persistent failure
                        if alerter:
                            await alerter.send_alert(
                                f"{operation_name}_FAILURE",
                                f"Operation failed after {retry_handler.max_retries} retries: {str(e)}",
                                alert_priority
                            )
            
            # Attempt to use cached data as fallback
            if use_cache and cache_key:
                if len(args) > 0:
                    cached_data = cache_manager.get_cache(cache_key, str(args[0]))
                else:
                    cached_data = cache_manager.get_cache(cache_key)
                
                if cached_data is not None:
                    logger.warning(f"{operation_name} using cached fallback data")
                    if alerter:
                        await alerter.send_alert(
                            f"{operation_name}_FALLBACK",
                            f"Using cached data due to API failure: {str(last_exception)}",
                            "LOW"
                        )
                    return cached_data
            
            # If all retries failed and no cache available, raise the last exception
            if last_exception:
                raise last_exception
            else:
                raise Exception(f"{operation_name} failed with unknown error")
        
        return wrapper
    return decorator

class RobustAPIWrapper:
    """Wrapper class for robust API operations with retry and failover."""
    
    def __init__(self, binance_client, gsheet_client, telegram_bot_token, telegram_chat_id, spreadsheet_id):
        self.binance_client = binance_client
        self.gsheet_client = gsheet_client
        self.spreadsheet_id = spreadsheet_id
        
        self.retry_handler = RetryHandler()
        self.cache_manager = CacheManager()
        self.alerter = TelegramAlerter(telegram_bot_token, telegram_chat_id)
        
        # Apply retry decorators to methods
        self._setup_retry_decorators()
    
    def _setup_retry_decorators(self):
        """Setup retry decorators for API methods."""
        # Binance API methods
        self.get_latest_price = self._create_retry_method(
            self._get_latest_price_impl, "get_latest_price", True, "prices", "HIGH"
        )
        self.get_klines = self._create_retry_method(
            self._get_klines_impl, "get_klines", True, "klines", "HIGH"
        )
        self.get_usdt_balance = self._create_retry_method(
            self._get_usdt_balance_impl, "get_usdt_balance", True, "balance", "MEDIUM"
        )
        self.get_ticker_data = self._create_retry_method(
            self._get_ticker_data_impl, "get_ticker_data", True, "ticker_data", "MEDIUM"
        )
        self.create_order = self._create_retry_method(
            self._create_order_impl, "create_order", False, None, "CRITICAL"
        )
        
        # Google Sheets methods
        self.log_to_sheet = self._create_retry_method(
            self._log_to_sheet_impl, "log_to_sheet", False, None, "LOW"
        )
    
    def _create_retry_method(self, impl_method, operation_name, use_cache, cache_key, alert_priority):
        """Create a retry-decorated method."""
        decorated = retry_on_failure(operation_name, use_cache, cache_key, alert_priority)(impl_method)
        decorated._retry_handler = self.retry_handler
        decorated._cache_manager = self.cache_manager
        decorated._alerter = self.alerter
        return decorated
    
    def _get_latest_price_impl(self, symbol):
        """Get latest price with error handling."""
        ticker = self.binance_client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    
    def _get_klines_impl(self, symbol, interval="15m", limit=250):
        """Get klines data with error handling."""
        import pandas as pd
        data = self.binance_client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(data, columns=[
            'timestamp','open','high','low','close','volume','close_time',
            'qav','num_trades','taker_base_vol','taker_quote_vol','ignore'
        ])
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['open'] = pd.to_numeric(df['open'])
        df['volume'] = pd.to_numeric(df['volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    def _get_usdt_balance_impl(self):
        """Get USDT balance with error handling."""
        balance = self.binance_client.get_asset_balance(asset='USDT')
        return float(balance['free']) if balance else 0.0
    
    def _get_ticker_data_impl(self):
        """Get ticker data with error handling."""
        tickers = self.binance_client.get_ticker()
        return {t['symbol']: t for t in tickers}
    
    def _create_order_impl(self, symbol, side, order_type, quantity, **kwargs):
        """Create order with error handling."""
        return self.binance_client.create_order(
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=quantity,
            **kwargs
        )
    
    def _log_to_sheet_impl(self, data):
        """Log to Google Sheets with error handling."""
        if not self.gsheet_client or not self.spreadsheet_id:
            raise Exception("Google Sheets not configured")
        
        worksheet = self.gsheet_client.open_by_key(self.spreadsheet_id).sheet1
        worksheet.append_row(data)
        return True

def setup_robust_apis(binance_client, gsheet_client, telegram_bot_token, telegram_chat_id, spreadsheet_id):
    """Setup robust API wrapper."""
    return RobustAPIWrapper(
        binance_client, 
        gsheet_client, 
        telegram_bot_token, 
        telegram_chat_id, 
        spreadsheet_id
    )

# Health check function
async def health_check_and_recovery():
    """Perform health checks and recovery operations."""
    logger.info("Performing health check...")
    
    health_status = {
        'binance_api': False,
        'google_sheets': False,
        'telegram': False,
        'timestamp': datetime.now().isoformat()
    }
    
    # Add health check implementations here
    return health_status
