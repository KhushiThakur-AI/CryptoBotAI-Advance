
import unittest
import unittest.mock as mock
import json
import pandas as pd
import datetime
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules to test
from main import (
    TradeManager, get_klines, add_indicators, get_latest_price, 
    get_usdt_balance, format_signal_summary, format_bot_status,
    is_trading_hour, check_market_filters, get_symbol_info,
    MLSignalBooster, validate_trade_order, create_signed_request
)
import indicators
import sentiment

class TestTechnicalIndicators(unittest.TestCase):
    """Test suite for technical indicator calculations."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_data = [100, 101, 102, 103, 105, 104, 106, 108, 109, 107]
        self.sample_df = pd.DataFrame({
            'close': self.sample_data,
            'high': [x + 2 for x in self.sample_data],
            'low': [x - 2 for x in self.sample_data],
            'volume': [1000] * len(self.sample_data),
            'timestamp': pd.date_range(start='2024-01-01', periods=len(self.sample_data), freq='1H')
        })
    
    def test_moving_average_calculation(self):
        """Test simple moving average calculation."""
        window = 5
        result = indicators.calculate_moving_average(self.sample_data, window)
        expected = sum(self.sample_data[-window:]) / window
        self.assertEqual(result, expected)
    
    def test_moving_average_insufficient_data(self):
        """Test moving average with insufficient data."""
        result = indicators.calculate_moving_average([100, 101], 5)
        self.assertEqual(result, 101)  # Should return last value
    
    def test_add_indicators_rsi(self):
        """Test RSI indicator addition."""
        config = {
            'rsi_enabled': True,
            'rsi_period': 14,
            'ema_enabled': False,
            'macd_enabled': False,
            'bollinger_enabled': False,
            'stoch_rsi_enabled': False,
            'adx_enabled': False
        }
        
        with patch('main.INDICATORS_SETTINGS', config):
            result_df = add_indicators(self.sample_df)
            self.assertIn('rsi', result_df.columns)
            self.assertIsNotNone(result_df['rsi'].iloc[-1])


class TestTradeManager(unittest.TestCase):
    """Test suite for TradeManager class."""
    
    def setUp(self):
        """Set up TradeManager instance for testing."""
        self.mock_client = MagicMock()
        self.mock_gsheet = MagicMock()
        self.mock_firebase = MagicMock()
        
        self.trade_manager = TradeManager(
            client=self.mock_client,
            realtime_mode=False,
            paper_balance_initial=10000.0,
            base_trade_amount_usd=100.0,
            min_trade_amount_usd=10.0,
            max_trade_amount_usd=500.0,
            max_total_loss_usd=1000.0,
            daily_max_loss_usd=100.0,
            symbols_config={'BTCUSDT': {'sl': 0.02, 'tsl': 0.01}},
            gsheet_client=self.mock_gsheet,
            spreadsheet_id='test_sheet',
            firebase_db_client=self.mock_firebase,
            max_volatility_for_sizing=0.05,
            max_diversified_positions=3
        )
    
    def test_initial_balance(self):
        """Test initial balance setting."""
        self.assertEqual(self.trade_manager.paper_balance, 10000.0)
        self.assertEqual(self.trade_manager.initial_paper_balance, 10000.0)
    
    def test_format_quantity(self):
        """Test quantity formatting."""
        with patch('main.get_symbol_info') as mock_symbol_info:
            mock_symbol_info.return_value = {'quantityPrecision': 4}
            result = self.trade_manager._format_quantity('BTCUSDT', 0.123456)
            self.assertEqual(result, 0.1235)
    
    def test_format_price(self):
        """Test price formatting."""
        with patch('main.get_symbol_info') as mock_symbol_info:
            mock_symbol_info.return_value = {'pricePrecision': 2}
            result = self.trade_manager._format_price('BTCUSDT', 50000.123)
            self.assertEqual(result, 50000.12)
    
    def test_calculate_adaptive_trade_size_disabled(self):
        """Test adaptive trade size when disabled."""
        with patch('main.ADAPTIVE_SIZING_ENABLED', False):
            result = self.trade_manager.calculate_adaptive_trade_size('BTCUSDT', 2.0, 0.02)
            self.assertEqual(result, 100.0)  # Should return base amount
    
    def test_calculate_adaptive_trade_size_enabled(self):
        """Test adaptive trade size when enabled."""
        with patch('main.ADAPTIVE_SIZING_ENABLED', True), \
             patch('main.SIGNAL_WEIGHTS', {'rsi_buy': 1.0, 'macd_bullish_cross': 1.0}):
            result = self.trade_manager.calculate_adaptive_trade_size('BTCUSDT', 2.0, 0.02)
            self.assertGreaterEqual(result, self.trade_manager.min_trade_amount_usd)
            self.assertLessEqual(result, self.trade_manager.max_trade_amount_usd)
    
    async def test_execute_paper_buy_trade(self):
        """Test paper buy trade execution."""
        with patch('main.get_symbol_info') as mock_symbol_info, \
             patch('main.send_telegram_message', new=AsyncMock()):
            
            mock_symbol_info.return_value = {
                'quantityPrecision': 4,
                'pricePrecision': 2,
                'minQty': 0.0001,
                'minNotional': 10.0
            }
            
            result = await self.trade_manager.execute_trade(
                'BTCUSDT', 'BUY', 50000.0, 'TEST_BUY', {}, 2.0, 0.02
            )
            
            self.assertTrue(result)
            self.assertIn('BTCUSDT', self.trade_manager.paper_positions)
            self.assertLess(self.trade_manager.paper_balance, 10000.0)
    
    async def test_execute_paper_sell_trade(self):
        """Test paper sell trade execution."""
        # First, set up a position
        self.trade_manager.paper_positions['BTCUSDT'] = {
            'side': 'BUY',
            'quantity': 0.002,
            'entry_price': 50000.0,
            'stop_loss': 49000.0,
            'highest_price_since_entry': 50000.0,
            'current_trailing_stop_price': 49500.0
        }
        
        with patch('main.send_telegram_message', new=AsyncMock()):
            result = await self.trade_manager.execute_trade(
                'BTCUSDT', 'SELL', 51000.0, 'TEST_SELL', {}, 2.0, 0.02
            )
            
            self.assertTrue(result)
            self.assertNotIn('BTCUSDT', self.trade_manager.paper_positions)
            self.assertGreater(self.trade_manager.daily_pnl, 0)  # Should be profitable
    
    def test_get_open_positions_pnl(self):
        """Test PnL calculation for open positions."""
        self.trade_manager.paper_positions['BTCUSDT'] = {
            'quantity': 0.002,
            'entry_price': 50000.0,
            'current_trailing_stop_price': 49500.0
        }
        
        current_prices = {'BTCUSDT': 51000.0}
        total_pnl, summary = self.trade_manager.get_open_positions_pnl(current_prices)
        
        expected_pnl = (51000.0 - 50000.0) * 0.002
        self.assertEqual(total_pnl, expected_pnl)
        self.assertIn('BTCUSDT', summary)


class TestUtilityFunctions(unittest.TestCase):
    """Test suite for utility functions."""
    
    @patch('main.client')
    def test_get_latest_price(self, mock_client):
        """Test latest price fetching."""
        mock_client.get_symbol_ticker.return_value = {'price': '50000.00'}
        result = get_latest_price('BTCUSDT')
        self.assertEqual(result, 50000.0)
    
    @patch('main.client')
    def test_get_latest_price_error(self, mock_client):
        """Test latest price fetching with error."""
        mock_client.get_symbol_ticker.side_effect = Exception('API Error')
        result = get_latest_price('BTCUSDT')
        self.assertIsNone(result)
    
    @patch('main.client')
    def test_get_usdt_balance(self, mock_client):
        """Test USDT balance fetching."""
        mock_client.get_asset_balance.return_value = {'free': '1000.50'}
        result = get_usdt_balance()
        self.assertEqual(result, 1000.50)
    
    @patch('main.client')
    def test_get_symbol_info(self, mock_client):
        """Test symbol info fetching."""
        mock_client.get_symbol_info.return_value = {
            'filters': [
                {'filterType': 'PRICE_FILTER', 'tickSize': '0.01'},
                {'filterType': 'LOT_SIZE', 'stepSize': '0.0001', 'minQty': '0.0001'},
                {'filterType': 'MIN_NOTIONAL', 'minNotional': '10.0'}
            ]
        }
        
        result = get_symbol_info('BTCUSDT')
        
        self.assertEqual(result['pricePrecision'], 2)
        self.assertEqual(result['quantityPrecision'], 4)
        self.assertEqual(result['minNotional'], 10.0)
        self.assertEqual(result['minQty'], 0.0001)
    
    def test_is_trading_hour_no_config(self):
        """Test trading hour check with no configuration."""
        with patch('main.ACTIVE_TRADE_HOURS', []):
            result = is_trading_hour()
            self.assertTrue(result)  # Should be always active
    
    def test_is_trading_hour_with_config(self):
        """Test trading hour check with configuration."""
        now = datetime.datetime.now()
        current_time = now.strftime('%H:%M')
        
        # Create a time window that includes current time
        active_hours = [{'start': '00:00', 'end': '23:59'}]
        
        with patch('main.ACTIVE_TRADE_HOURS', active_hours):
            result = is_trading_hour()
            self.assertTrue(result)
    
    def test_check_market_filters_pass(self):
        """Test market filters with passing conditions."""
        ticker_data = {
            'quoteVolume': '2000000.0',
            'highPrice': '51000.0',
            'lowPrice': '49000.0',
            'lastPrice': '50000.0'
        }
        
        result = check_market_filters('BTCUSDT', ticker_data, 1000000.0, 0.05)
        self.assertTrue(result)
    
    def test_check_market_filters_fail_volume(self):
        """Test market filters failing volume check."""
        ticker_data = {
            'quoteVolume': '500000.0',  # Below minimum
            'highPrice': '51000.0',
            'lowPrice': '49000.0',
            'lastPrice': '50000.0'
        }
        
        result = check_market_filters('BTCUSDT', ticker_data, 1000000.0, 0.05)
        self.assertFalse(result)
    
    def test_check_market_filters_fail_volatility(self):
        """Test market filters failing volatility check."""
        ticker_data = {
            'quoteVolume': '2000000.0',
            'highPrice': '55000.0',  # High volatility
            'lowPrice': '45000.0',
            'lastPrice': '50000.0'
        }
        
        result = check_market_filters('BTCUSDT', ticker_data, 1000000.0, 0.05)
        self.assertFalse(result)


class TestMLSignalBooster(unittest.TestCase):
    """Test suite for ML Signal Booster."""
    
    def setUp(self):
        """Set up ML booster for testing."""
        self.ml_booster = MLSignalBooster('test_model.joblib')
    
    def test_ml_booster_no_model(self):
        """Test ML booster when no model file exists."""
        self.assertIsNone(self.ml_booster.model)
    
    def test_prepare_data(self):
        """Test data preparation for ML model."""
        df = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200],
            'rsi': [30, 35, 40],
            'ema_fast': [100, 101, 102],
            'ema_slow': [99, 100, 101],
            'macd': [0.1, 0.2, 0.3],
            'macd_signal': [0.05, 0.15, 0.25],
            'macd_hist': [0.05, 0.05, 0.05],
            'bb_bbm': [100, 101, 102],
            'bb_bbh': [105, 106, 107],
            'bb_bbl': [95, 96, 97],
            'stoch_rsi_k': [20, 25, 30],
            'stoch_rsi_d': [15, 20, 25],
            'adx': [25, 30, 35],
            'adx_pos': [20, 25, 30],
            'adx_neg': [15, 20, 25]
        })
        
        result = self.ml_booster._prepare_data(df)
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result.columns), len(self.ml_booster.features))
    
    def test_get_ml_score_no_model(self):
        """Test ML score when no model is loaded."""
        df = pd.DataFrame({'close': [100]})
        result = self.ml_booster.get_ml_score(df)
        self.assertIsNone(result)


class TestSecurityFunctions(unittest.TestCase):
    """Test suite for security functions."""
    
    def test_validate_trade_order_valid(self):
        """Test trade order validation with valid order."""
        valid_order = {
            'symbol': 'BTCUSDT',
            'type': 'BUY',
            'quantity': 0.001,
            'price': 50000.0
        }
        
        result = validate_trade_order(valid_order)
        self.assertTrue(result)
    
    def test_validate_trade_order_missing_fields(self):
        """Test trade order validation with missing fields."""
        invalid_order = {
            'symbol': 'BTCUSDT',
            'type': 'BUY'
            # Missing quantity and price
        }
        
        result = validate_trade_order(invalid_order)
        self.assertFalse(result)
    
    def test_validate_trade_order_invalid_types(self):
        """Test trade order validation with invalid data types."""
        invalid_order = {
            'symbol': 'BTCUSDT',
            'type': 'BUY',
            'quantity': 'invalid',  # Should be number
            'price': 50000.0
        }
        
        result = validate_trade_order(invalid_order)
        self.assertFalse(result)
    
    def test_validate_trade_order_negative_values(self):
        """Test trade order validation with negative values."""
        invalid_order = {
            'symbol': 'BTCUSDT',
            'type': 'BUY',
            'quantity': -0.001,  # Should be positive
            'price': 50000.0
        }
        
        result = validate_trade_order(invalid_order)
        self.assertFalse(result)
    
    def test_create_signed_request(self):
        """Test HMAC signature generation."""
        payload = {'symbol': 'BTCUSDT', 'side': 'BUY', 'quantity': 0.001}
        
        with patch('main.API_KEY', 'test_key'), \
             patch('main.API_SECRET', b'test_secret'):
            result = create_signed_request(payload)
            
            self.assertIn('api_key', result)
            self.assertIn('signature', result)
            self.assertEqual(result['api_key'], 'test_key')


class TestSentimentAnalysis(unittest.TestCase):
    """Test suite for sentiment analysis."""
    
    def test_analyze_sentiment_positive(self):
        """Test positive sentiment analysis."""
        result = sentiment.analyze_sentiment("Bitcoin is going to the moon! Great news!")
        self.assertGreater(result, 0)
    
    def test_analyze_sentiment_negative(self):
        """Test negative sentiment analysis."""
        result = sentiment.analyze_sentiment("Bitcoin is crashing badly. Terrible market.")
        self.assertLess(result, 0)
    
    def test_analyze_sentiment_neutral(self):
        """Test neutral sentiment analysis."""
        result = sentiment.analyze_sentiment("Bitcoin price remains stable.")
        self.assertAlmostEqual(result, 0, delta=0.3)


class TestFormatting(unittest.TestCase):
    """Test suite for message formatting functions."""
    
    def test_format_signal_summary(self):
        """Test signal summary formatting."""
        latest_data = {
            'rsi': 45.5,
            'ema_fast': 50000.0,
            'macd_hist': 0.123,
            'stoch_rsi_k': 25.5,
            'stoch_rsi_d': 20.5,
            'adx': 30.0,
            'adx_pos': 25.0,
            'adx_neg': 20.0,
            'bb_bbh': 51000.0,
            'bb_bbl': 49000.0
        }
        
        mock_trade_manager = MagicMock()
        mock_trade_manager.paper_balance = 10000.0
        
        result = format_signal_summary(
            'BTCUSDT', '15m', latest_data, 50000.0, 
            mock_trade_manager, 'BUY Signal'
        )
        
        self.assertIn('BTCUSDT', result)
        self.assertIn('50000.00', result)
        self.assertIn('45.50', result)
        self.assertIn('BUY Signal', result)
    
    def test_format_bot_status(self):
        """Test bot status formatting."""
        mock_trade_manager = MagicMock()
        mock_trade_manager.real_mode = False
        mock_trade_manager.bot_is_paused_permanent = False
        mock_trade_manager.bot_is_paused_daily = False
        mock_trade_manager.paper_balance = 10000.0
        mock_trade_manager.initial_paper_balance = 10000.0
        mock_trade_manager.daily_pnl = 50.0
        mock_trade_manager.daily_max_loss_usd = 100.0
        mock_trade_manager.get_open_positions_pnl.return_value = (100.0, "No positions")
        
        with patch('main.get_usdt_balance', return_value=1000.0):
            result = format_bot_status(mock_trade_manager, {'BTCUSDT': 50000.0})
            
            self.assertIn('PAPER', result)
            self.assertIn('10,000.00', result)
            self.assertIn('Yes', result)  # Bot active


class TestAsyncFunctions(unittest.TestCase):
    """Test suite for async functions."""
    
    def setUp(self):
        """Set up async test environment."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up async test environment."""
        self.loop.close()
    
    async def async_test_send_telegram_message(self):
        """Test Telegram message sending."""
        with patch('main.bot') as mock_bot:
            mock_bot.send_message = AsyncMock()
            
            from main import send_telegram_message
            await send_telegram_message("Test message")
            
            mock_bot.send_message.assert_called_once()
    
    def test_send_telegram_message(self):
        """Wrapper for async Telegram test."""
        self.loop.run_until_complete(self.async_test_send_telegram_message())


class TestConfigValidation(unittest.TestCase):
    """Test suite for configuration validation."""
    
    def test_config_file_structure(self):
        """Test that config.json has required structure."""
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        required_sections = ['symbols', 'settings']
        for section in required_sections:
            self.assertIn(section, config)
        
        # Test settings subsections
        settings = config['settings']
        required_settings = ['indicators', 'trade_amount_usd', 'paper_trading_balance']
        for setting in required_settings:
            self.assertIn(setting, settings)
    
    def test_symbol_configuration(self):
        """Test symbol configuration validation."""
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        symbols = config['symbols']
        self.assertGreater(len(symbols), 0)
        
        for symbol, params in symbols.items():
            self.assertIn('sl', params)
            self.assertIn('tsl', params)
            self.assertIsInstance(params['sl'], (int, float))
            self.assertIsInstance(params['tsl'], (int, float))


# Integration test for the complete trading cycle
class TestTradingIntegration(unittest.TestCase):
    """Integration tests for complete trading workflows."""
    
    @patch('main.get_klines')
    @patch('main.get_latest_price')
    @patch('main.client')
    async def async_test_complete_trading_cycle(self, mock_client, mock_price, mock_klines):
        """Test a complete trading cycle."""
        # Mock data setup
        mock_df = pd.DataFrame({
            'close': [50000] * 100,
            'high': [50100] * 100,
            'low': [49900] * 100,
            'volume': [1000] * 100,
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1H')
        })
        
        mock_klines.return_value = mock_df
        mock_price.return_value = 50000.0
        mock_client.get_ticker.return_value = [{
            'symbol': 'BTCUSDT',
            'quoteVolume': '2000000.0',
            'highPrice': '51000.0',
            'lowPrice': '49000.0',
            'lastPrice': '50000.0'
        }]
        
        # Create trade manager
        trade_manager = TradeManager(
            client=mock_client,
            realtime_mode=False,
            paper_balance_initial=10000.0,
            base_trade_amount_usd=100.0,
            min_trade_amount_usd=10.0,
            max_trade_amount_usd=500.0,
            max_total_loss_usd=1000.0,
            daily_max_loss_usd=100.0,
            symbols_config={'BTCUSDT': {'sl': 0.02, 'tsl': 0.01}},
            gsheet_client=MagicMock(),
            spreadsheet_id='test_sheet',
            firebase_db_client=MagicMock(),
            max_volatility_for_sizing=0.05,
            max_diversified_positions=3
        )
        
        # Mock the required patches for the trading cycle
        with patch('main.ACTIVE_SYMBOLS', ['BTCUSDT']), \
             patch('main.send_telegram_message', new=AsyncMock()), \
             patch('main.is_trading_hour', return_value=True), \
             patch('main.add_indicators', return_value=mock_df):
            
            # This should complete without errors
            await trade_manager.run_trading_cycle()
            
            # Verify the cycle completed
            self.assertIsNotNone(trade_manager.paper_balance)
    
    def test_complete_trading_cycle(self):
        """Wrapper for async trading cycle test."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.async_test_complete_trading_cycle())
        finally:
            loop.close()


def run_tests():
    """Run all tests and generate a report."""
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestTechnicalIndicators,
        TestTradeManager,
        TestUtilityFunctions,
        TestMLSignalBooster,
        TestSecurityFunctions,
        TestSentimentAnalysis,
        TestFormatting,
        TestAsyncFunctions,
        TestConfigValidation,
        TestTradingIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run all tests
    success = run_tests()
    
    # Exit with appropriate code
    exit(0 if success else 1)
