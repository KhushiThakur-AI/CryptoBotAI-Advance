
import logging
import asyncio
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json

from news_fetcher import NewsAggregator
from sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)

class NewsSentimentIntegration:
    def __init__(self, config_data: dict):
        self.news_aggregator = NewsAggregator()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Load sentiment trading settings from config
        self.sentiment_config = config_data.get("sentiment_analysis", {})
        self.enabled = self.sentiment_config.get("enabled", True)
        self.min_news_count = self.sentiment_config.get("min_news_count", 3)
        self.sentiment_weight = self.sentiment_config.get("sentiment_weight", 0.3)
        self.sentiment_threshold = self.sentiment_config.get("sentiment_threshold", 0.2)
        
        # Cache for sentiment data
        self.sentiment_cache = {}
        self.cache_duration = self.sentiment_config.get("cache_duration_minutes", 15) * 60
        
        # News fetching intervals to manage API limits
        self.last_fetch_time = None
        self.fetch_interval = self.sentiment_config.get("fetch_interval_minutes", 30) * 60

    async def get_sentiment_signals(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get sentiment analysis signals for given symbols
        Returns sentiment scores and signal recommendations
        """
        if not self.enabled:
            logger.info("News sentiment analysis is disabled")
            return {symbol: self._get_neutral_sentiment() for symbol in symbols}

        try:
            # Check if we need to fetch fresh news
            current_time = datetime.now()
            should_fetch = (
                self.last_fetch_time is None or 
                (current_time - self.last_fetch_time).total_seconds() > self.fetch_interval
            )

            if should_fetch:
                logger.info("Fetching fresh news and sentiment data...")
                
                # Fetch news from all sources
                news_items = await self.news_aggregator.fetch_all_news(symbols)
                
                if not news_items:
                    logger.warning("No news items fetched. Using neutral sentiment.")
                    return {symbol: self._get_neutral_sentiment() for symbol in symbols}
                
                # Analyze sentiment
                symbol_sentiments = await self.sentiment_analyzer.analyze_news_sentiment(news_items, symbols)
                
                # Cache the results
                self.sentiment_cache = {
                    'data': symbol_sentiments,
                    'timestamp': current_time,
                    'news_count': len(news_items)
                }
                
                self.last_fetch_time = current_time
                
                logger.info(f"Analyzed sentiment for {len(news_items)} news items across {len(symbols)} symbols")
                
            else:
                # Use cached data
                if 'data' in self.sentiment_cache:
                    symbol_sentiments = self.sentiment_cache['data']
                    cache_age = (current_time - self.sentiment_cache['timestamp']).total_seconds() / 60
                    logger.info(f"Using cached sentiment data (age: {cache_age:.1f} minutes)")
                else:
                    logger.warning("No cached sentiment data available. Using neutral sentiment.")
                    return {symbol: self._get_neutral_sentiment() for symbol in symbols}

            # Process sentiment signals for trading
            trading_signals = {}
            for symbol in symbols:
                if symbol in symbol_sentiments:
                    trading_signals[symbol] = self._process_sentiment_for_trading(
                        symbol, symbol_sentiments[symbol]
                    )
                else:
                    trading_signals[symbol] = self._get_neutral_sentiment()

            return trading_signals

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {symbol: self._get_neutral_sentiment() for symbol in symbols}

    def _process_sentiment_for_trading(self, symbol: str, sentiment_data: Dict) -> Dict:
        """Process raw sentiment data into trading signals"""
        
        # Get signal strength from sentiment analyzer
        signal_type, strength = self.sentiment_analyzer.get_sentiment_signal_strength(sentiment_data)
        
        # Calculate trading score based on sentiment
        sentiment_score = sentiment_data.get('overall_score', 0.0)
        confidence = sentiment_data.get('confidence', 0.0)
        news_count = sentiment_data.get('total_count', 0)
        
        # Apply filters
        sufficient_news = news_count >= self.min_news_count
        meets_threshold = abs(sentiment_score) >= self.sentiment_threshold
        
        # Calculate final sentiment contribution to trading score
        if sufficient_news and meets_threshold:
            # Scale sentiment score by confidence and weight
            sentiment_contribution = sentiment_score * confidence * self.sentiment_weight
        else:
            sentiment_contribution = 0.0
        
        # Determine recommendation
        recommendation = self._get_sentiment_recommendation(signal_type, strength, sufficient_news)
        
        return {
            'sentiment_score': sentiment_score,
            'confidence': confidence,
            'news_count': news_count,
            'signal_type': signal_type,
            'signal_strength': strength,
            'sentiment_contribution': sentiment_contribution,
            'recommendation': recommendation,
            'sufficient_data': sufficient_news,
            'meets_threshold': meets_threshold,
            'positive_news': sentiment_data.get('positive_count', 0),
            'negative_news': sentiment_data.get('negative_count', 0),
            'neutral_news': sentiment_data.get('neutral_count', 0),
            'top_news': sentiment_data.get('news_items', [])[:3]  # Top 3 news items
        }

    def _get_sentiment_recommendation(self, signal_type: str, strength: float, sufficient_news: bool) -> str:
        """Convert sentiment signal to trading recommendation"""
        if not sufficient_news:
            return "INSUFFICIENT_NEWS_DATA"
        
        if signal_type == "STRONG_POSITIVE" and strength > 0.6:
            return "SENTIMENT_BULLISH_STRONG"
        elif signal_type == "MODERATE_POSITIVE" and strength > 0.4:
            return "SENTIMENT_BULLISH_MODERATE"
        elif signal_type == "STRONG_NEGATIVE" and strength > 0.6:
            return "SENTIMENT_BEARISH_STRONG"
        elif signal_type == "MODERATE_NEGATIVE" and strength > 0.4:
            return "SENTIMENT_BEARISH_MODERATE"
        else:
            return "SENTIMENT_NEUTRAL"

    def _get_neutral_sentiment(self) -> Dict:
        """Return neutral sentiment data when no analysis is available"""
        return {
            'sentiment_score': 0.0,
            'confidence': 0.0,
            'news_count': 0,
            'signal_type': 'NEUTRAL',
            'signal_strength': 0.0,
            'sentiment_contribution': 0.0,
            'recommendation': 'NO_SENTIMENT_DATA',
            'sufficient_data': False,
            'meets_threshold': False,
            'positive_news': 0,
            'negative_news': 0,
            'neutral_news': 0,
            'top_news': []
        }

    def apply_sentiment_to_technical_signal(self, technical_score: float, sentiment_data: Dict, 
                                          signal_type: str) -> Tuple[float, str]:
        """
        Combine technical analysis signal with sentiment analysis
        Returns: (adjusted_score, reasoning)
        """
        if not self.enabled or not sentiment_data.get('sufficient_data', False):
            return technical_score, "No sentiment adjustment (insufficient data)"
        
        sentiment_contribution = sentiment_data.get('sentiment_contribution', 0.0)
        sentiment_recommendation = sentiment_data.get('recommendation', 'NO_SENTIMENT_DATA')
        
        # Apply sentiment boost or penalty
        if signal_type == "BUY":
            if "BULLISH" in sentiment_recommendation:
                # Sentiment supports the buy signal
                boost = abs(sentiment_contribution)
                adjusted_score = technical_score + boost
                reasoning = f"Sentiment SUPPORTS buy (+{boost:.2f}): {sentiment_recommendation}"
            elif "BEARISH" in sentiment_recommendation:
                # Sentiment contradicts the buy signal
                penalty = abs(sentiment_contribution) * 0.5  # Reduce penalty to not be too aggressive
                adjusted_score = max(0, technical_score - penalty)
                reasoning = f"Sentiment OPPOSES buy (-{penalty:.2f}): {sentiment_recommendation}"
            else:
                adjusted_score = technical_score
                reasoning = f"Neutral sentiment: {sentiment_recommendation}"
                
        elif signal_type == "SELL":
            if "BEARISH" in sentiment_recommendation:
                # Sentiment supports the sell signal
                boost = abs(sentiment_contribution)
                adjusted_score = technical_score + boost
                reasoning = f"Sentiment SUPPORTS sell (+{boost:.2f}): {sentiment_recommendation}"
            elif "BULLISH" in sentiment_recommendation:
                # Sentiment contradicts the sell signal
                penalty = abs(sentiment_contribution) * 0.5
                adjusted_score = max(0, technical_score - penalty)
                reasoning = f"Sentiment OPPOSES sell (-{penalty:.2f}): {sentiment_recommendation}"
            else:
                adjusted_score = technical_score
                reasoning = f"Neutral sentiment: {sentiment_recommendation}"
        else:
            adjusted_score = technical_score
            reasoning = "No signal type specified"
        
        return adjusted_score, reasoning

    async def get_market_sentiment_overview(self, symbols: List[str]) -> Dict:
        """Get overall market sentiment summary"""
        try:
            sentiment_signals = await self.get_sentiment_signals(symbols)
            return await self.sentiment_analyzer.get_market_sentiment_overview(
                {symbol: data for symbol, data in sentiment_signals.items()}
            )
        except Exception as e:
            logger.error(f"Error getting market sentiment overview: {e}")
            return {
                "market_sentiment_score": 0.0,
                "positive_symbols_count": 0,
                "negative_symbols_count": 0,
                "neutral_symbols_count": len(symbols),
                "total_news_analyzed": 0,
                "market_mood": "NEUTRAL",
                "confidence": 0.0
            }

    def log_sentiment_summary(self, symbol: str, sentiment_data: Dict) -> str:
        """Generate a formatted summary for logging"""
        if not sentiment_data.get('sufficient_data', False):
            return f"{symbol}: No sufficient sentiment data"
        
        score = sentiment_data.get('sentiment_score', 0.0)
        confidence = sentiment_data.get('confidence', 0.0)
        news_count = sentiment_data.get('news_count', 0)
        recommendation = sentiment_data.get('recommendation', 'UNKNOWN')
        
        return (
            f"{symbol}: Sentiment={score:.3f}, Confidence={confidence:.2f}, "
            f"News={news_count}, Rec={recommendation}"
        )
