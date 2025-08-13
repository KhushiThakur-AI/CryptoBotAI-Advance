import logging
import asyncio
import aiohttp
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import os

logger = logging.getLogger(__name__)

class SentimentIntensityAnalyzer:
    """Minimal sentiment analyzer fallback without vaderSentiment dependency."""

    def __init__(self):
        # Simple word lists for basic sentiment analysis
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'fantastic', 'wonderful', 'outstanding',
            'bullish', 'rally', 'surge', 'moon', 'pump', 'breakout', 'profit', 'gain',
            'rise', 'up', 'high', 'strong', 'buy', 'bull', 'positive', 'growth'
        }

        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disaster', 'crash', 'dump',
            'bearish', 'fall', 'drop', 'decline', 'loss', 'sell', 'bear', 'fear',
            'panic', 'down', 'low', 'weak', 'negative', 'collapse', 'plunge'
        }

    def polarity_scores(self, text: str) -> Dict[str, float]:
        """Basic sentiment scoring without vaderSentiment."""
        if not text:
            return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}

        words = re.findall(r'\b\w+\b', text.lower())
        pos_count = sum(1 for word in words if word in self.positive_words)
        neg_count = sum(1 for word in words if word in self.negative_words)
        total_words = len(words)

        if total_words == 0:
            return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}

        pos_score = pos_count / total_words
        neg_score = neg_count / total_words
        neu_score = 1.0 - pos_score - neg_score

        # Simple compound score calculation
        compound = (pos_score - neg_score)
        if compound > 0.1:
            compound = min(compound * 2, 1.0)
        elif compound < -0.1:
            compound = max(compound * 2, -1.0)

        return {
            'neg': neg_score,
            'neu': max(neu_score, 0.0),
            'pos': pos_score,
            'compound': compound
        }

class SentimentAnalyzer:
    """Sentiment analysis with crypto market context."""

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.crypto_keywords = {
            'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency',
            'blockchain', 'defi', 'nft', 'altcoin', 'hodl', 'whale'
        }

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment with crypto context weighting."""
        if not text:
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}

        # Basic sentiment analysis
        scores = self.analyzer.polarity_scores(text)

        # Weight crypto-related content more heavily
        text_lower = text.lower()
        crypto_relevance = sum(1 for keyword in self.crypto_keywords if keyword in text_lower)

        if crypto_relevance > 0:
            # Amplify sentiment for crypto-related content
            multiplier = min(1.0 + (crypto_relevance * 0.1), 1.5)
            scores['compound'] *= multiplier
            scores['pos'] *= multiplier if scores['compound'] > 0 else 1.0
            scores['neg'] *= multiplier if scores['compound'] < 0 else 1.0

        return {
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        }

    def batch_analyze(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyze multiple texts."""
        return [self.analyze_sentiment(text) for text in texts]

    def calculate_market_sentiment(self, news_items: List[Dict]) -> Dict[str, float]:
        """Calculate overall market sentiment from news items."""
        if not news_items:
            return {'overall_sentiment': 0.0, 'confidence': 0.0, 'article_count': 0}

        sentiments = []
        total_weight = 0

        for item in news_items:
            text = f"{item.get('title', '')} {item.get('description', '')}"
            sentiment = self.analyze_sentiment(text)

            # Weight by recency (newer articles have more weight)
            age_hours = item.get('age_hours', 24)
            recency_weight = max(0.1, 1.0 - (age_hours / 24.0))

            sentiments.append(sentiment['compound'] * recency_weight)
            total_weight += recency_weight

        if total_weight == 0:
            return {'overall_sentiment': 0.0, 'confidence': 0.0, 'article_count': len(news_items)}

        overall_sentiment = sum(sentiments) / total_weight

        # Calculate confidence based on agreement between articles
        variance = sum((s/total_weight - overall_sentiment)**2 for s in sentiments) / len(sentiments)
        confidence = max(0.0, 1.0 - variance)

        return {
            'overall_sentiment': overall_sentiment,
            'confidence': confidence,
            'article_count': len(news_items)
        }