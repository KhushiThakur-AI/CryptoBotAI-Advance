
import aiohttp
import asyncio
import feedparser
import json
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import re
from urllib.parse import urljoin
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class NewsItem:
    title: str
    content: str
    source: str
    url: str
    published_time: datetime
    symbols: List[str]
    sentiment_score: float = 0.0
    importance_score: float = 0.0

class NewsConfig:
    # CryptoPanic API settings
    CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY")
    CRYPTOPANIC_BASE_URL = "https://cryptopanic.com/api/v1/posts/"
    CRYPTOPANIC_DAILY_LIMIT = 500  # Adjust based on your plan
    CRYPTOPANIC_REQUESTS_PER_HOUR = 20  # Rate limiting
    
    # RSS Feed sources
    RSS_FEEDS = [
        {
            "name": "CoinDesk",
            "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "importance": 0.9
        },
        {
            "name": "CoinTelegraph",
            "url": "https://cointelegraph.com/rss",
            "importance": 0.8
        },
        {
            "name": "Investing.com Crypto",
            "url": "https://www.investing.com/rss/news_285.rss",
            "importance": 0.7
        },
        {
            "name": "CryptoPotato",
            "url": "https://cryptopotato.com/feed/",
            "importance": 0.6
        },
        {
            "name": "BeInCrypto",
            "url": "https://beincrypto.com/feed/",
            "importance": 0.6
        }
    ]
    
    # Reddit settings
    REDDIT_SUBREDDITS = [
        {"name": "cryptocurrency", "importance": 0.8},
        {"name": "Bitcoin", "importance": 0.9},
        {"name": "ethereum", "importance": 0.8},
        {"name": "CryptoMarkets", "importance": 0.7},
        {"name": "altcoin", "importance": 0.6}
    ]

class RateLimiter:
    def __init__(self, requests_per_hour: int):
        self.requests_per_hour = requests_per_hour
        self.requests = []
        
    async def wait_if_needed(self):
        now = time.time()
        # Remove requests older than 1 hour
        self.requests = [req_time for req_time in self.requests if now - req_time < 3600]
        
        if len(self.requests) >= self.requests_per_hour:
            wait_time = 3600 - (now - self.requests[0]) + 1
            logger.info(f"Rate limit reached. Waiting {wait_time:.0f} seconds...")
            await asyncio.sleep(wait_time)
            
        self.requests.append(now)

class CryptoPanicFetcher:
    def __init__(self):
        self.api_key = NewsConfig.CRYPTOPANIC_API_KEY
        self.rate_limiter = RateLimiter(NewsConfig.CRYPTOPANIC_REQUESTS_PER_HOUR)
        self.daily_requests = 0
        self.last_reset = datetime.now().date()
        
    async def fetch_news(self, symbols: List[str], limit: int = 50) -> List[NewsItem]:
        if not self.api_key:
            logger.warning("CryptoPanic API key not found. Skipping CryptoPanic news.")
            return []
            
        # Reset daily counter if new day
        today = datetime.now().date()
        if today > self.last_reset:
            self.daily_requests = 0
            self.last_reset = today
            
        if self.daily_requests >= NewsConfig.CRYPTOPANIC_DAILY_LIMIT:
            logger.warning("Daily CryptoPanic API limit reached. Skipping request.")
            return []
            
        await self.rate_limiter.wait_if_needed()
        
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "auth_token": self.api_key,
                    "public": "true",
                    "kind": "news",
                    "filter": "hot",
                    "currencies": ",".join([s.replace("USDT", "") for s in symbols])
                }
                
                async with session.get(NewsConfig.CRYPTOPANIC_BASE_URL, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.daily_requests += 1
                        return self._parse_cryptopanic_response(data)
                    else:
                        logger.error(f"CryptoPanic API error: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error fetching CryptoPanic news: {e}")
            return []
    
    def _parse_cryptopanic_response(self, data: dict) -> List[NewsItem]:
        news_items = []
        
        for post in data.get("results", []):
            try:
                # Extract symbols from currencies
                symbols = []
                for currency in post.get("currencies", []):
                    symbol = currency.get("code", "") + "USDT"
                    symbols.append(symbol)
                
                news_item = NewsItem(
                    title=post.get("title", ""),
                    content=post.get("title", ""),  # CryptoPanic doesn't provide full content
                    source=f"CryptoPanic - {post.get('source', {}).get('title', 'Unknown')}",
                    url=post.get("url", ""),
                    published_time=self._parse_time(post.get("published_at")),
                    symbols=symbols,
                    importance_score=0.9  # CryptoPanic is high quality
                )
                news_items.append(news_item)
                
            except Exception as e:
                logger.error(f"Error parsing CryptoPanic post: {e}")
                continue
                
        return news_items
    
    def _parse_time(self, time_str: str) -> datetime:
        try:
            return datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        except:
            return datetime.now()

class RSSFeedFetcher:
    def __init__(self):
        self.feeds = NewsConfig.RSS_FEEDS
        
    async def fetch_news(self, symbols: List[str]) -> List[NewsItem]:
        all_news = []
        
        for feed_config in self.feeds:
            try:
                news_items = await self._fetch_single_feed(feed_config, symbols)
                all_news.extend(news_items)
            except Exception as e:
                logger.error(f"Error fetching RSS feed {feed_config['name']}: {e}")
                continue
                
        return all_news
    
    async def _fetch_single_feed(self, feed_config: dict, symbols: List[str]) -> List[NewsItem]:
        try:
            # Use asyncio to run feedparser in thread pool
            loop = asyncio.get_event_loop()
            feed = await loop.run_in_executor(None, feedparser.parse, feed_config["url"])
            
            news_items = []
            
            for entry in feed.entries[:20]:  # Limit to recent 20 entries
                # Check if entry is relevant to our symbols
                relevant_symbols = self._extract_symbols_from_text(
                    f"{entry.get('title', '')} {entry.get('summary', '')}", 
                    symbols
                )
                
                if relevant_symbols:  # Only include if relevant to our trading symbols
                    news_item = NewsItem(
                        title=entry.get("title", ""),
                        content=entry.get("summary", entry.get("description", "")),
                        source=feed_config["name"],
                        url=entry.get("link", ""),
                        published_time=self._parse_rss_time(entry),
                        symbols=relevant_symbols,
                        importance_score=feed_config["importance"]
                    )
                    news_items.append(news_item)
                    
            return news_items
            
        except Exception as e:
            logger.error(f"Error parsing RSS feed {feed_config['name']}: {e}")
            return []
    
    def _extract_symbols_from_text(self, text: str, symbols: List[str]) -> List[str]:
        found_symbols = []
        text_lower = text.lower()
        
        symbol_keywords = {
            "BTCUSDT": ["bitcoin", "btc"],
            "ETHUSDT": ["ethereum", "eth", "ether"],
            "BNBUSDT": ["binance", "bnb"],
            "XRPUSDT": ["ripple", "xrp"],
            "SOLUSDT": ["solana", "sol"],
            "LINKUSDT": ["chainlink", "link"],
            "DOGEUSDT": ["dogecoin", "doge"],
            "SHIBUSDT": ["shiba", "shib"],
            "OPUSDT": ["optimism", "op"],
            "WIFUSDT": ["dogwifhat", "wif"],
            "PEPEUSDT": ["pepe"]
        }
        
        for symbol in symbols:
            keywords = symbol_keywords.get(symbol, [symbol.replace("USDT", "").lower()])
            for keyword in keywords:
                if keyword in text_lower:
                    found_symbols.append(symbol)
                    break
                    
        return list(set(found_symbols))  # Remove duplicates
    
    def _parse_rss_time(self, entry) -> datetime:
        try:
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                return datetime(*entry.published_parsed[:6])
            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                return datetime(*entry.updated_parsed[:6])
            else:
                return datetime.now()
        except:
            return datetime.now()

class RedditFetcher:
    def __init__(self):
        self.subreddits = NewsConfig.REDDIT_SUBREDDITS
        
    async def fetch_news(self, symbols: List[str]) -> List[NewsItem]:
        all_news = []
        
        for subreddit_config in self.subreddits:
            try:
                news_items = await self._fetch_subreddit_posts(subreddit_config, symbols)
                all_news.extend(news_items)
            except Exception as e:
                logger.error(f"Error fetching Reddit posts from {subreddit_config['name']}: {e}")
                continue
                
        return all_news
    
    async def _fetch_subreddit_posts(self, subreddit_config: dict, symbols: List[str]) -> List[NewsItem]:
        try:
            # Use Reddit JSON API (no authentication required for public posts)
            url = f"https://www.reddit.com/r/{subreddit_config['name']}/hot.json?limit=25"
            
            async with aiohttp.ClientSession() as session:
                headers = {"User-Agent": "CryptoTradingBot/1.0"}
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_reddit_response(data, subreddit_config, symbols)
                    else:
                        logger.error(f"Reddit API error for r/{subreddit_config['name']}: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error fetching Reddit posts: {e}")
            return []
    
    def _parse_reddit_response(self, data: dict, subreddit_config: dict, symbols: List[str]) -> List[NewsItem]:
        news_items = []
        
        try:
            posts = data.get("data", {}).get("children", [])
            
            for post in posts:
                post_data = post.get("data", {})
                
                # Skip stickied posts and low-score posts
                if post_data.get("stickied", False) or post_data.get("score", 0) < 10:
                    continue
                
                title = post_data.get("title", "")
                selftext = post_data.get("selftext", "")
                content = f"{title} {selftext}"
                
                # Check relevance to our symbols
                relevant_symbols = self._extract_symbols_from_text(content, symbols)
                
                if relevant_symbols:
                    news_item = NewsItem(
                        title=title,
                        content=content[:500],  # Limit content length
                        source=f"Reddit - r/{subreddit_config['name']}",
                        url=f"https://reddit.com{post_data.get('permalink', '')}",
                        published_time=datetime.fromtimestamp(post_data.get("created_utc", 0)),
                        symbols=relevant_symbols,
                        importance_score=subreddit_config["importance"] * min(post_data.get("score", 0) / 100, 1.0)
                    )
                    news_items.append(news_item)
                    
        except Exception as e:
            logger.error(f"Error parsing Reddit response: {e}")
            
        return news_items
    
    def _extract_symbols_from_text(self, text: str, symbols: List[str]) -> List[str]:
        found_symbols = []
        text_lower = text.lower()
        
        symbol_keywords = {
            "BTCUSDT": ["bitcoin", "btc", "$btc"],
            "ETHUSDT": ["ethereum", "eth", "$eth", "ether"],
            "BNBUSDT": ["binance", "bnb", "$bnb"],
            "XRPUSDT": ["ripple", "xrp", "$xrp"],
            "SOLUSDT": ["solana", "sol", "$sol"],
            "LINKUSDT": ["chainlink", "link", "$link"],
            "DOGEUSDT": ["dogecoin", "doge", "$doge"],
            "SHIBUSDT": ["shiba", "shib", "$shib"],
            "OPUSDT": ["optimism", "op", "$op"],
            "WIFUSDT": ["dogwifhat", "wif", "$wif"],
            "PEPEUSDT": ["pepe", "$pepe"]
        }
        
        for symbol in symbols:
            keywords = symbol_keywords.get(symbol, [symbol.replace("USDT", "").lower()])
            for keyword in keywords:
                if keyword in text_lower:
                    found_symbols.append(symbol)
                    break
                    
        return list(set(found_symbols))

class NewsAggregator:
    def __init__(self):
        self.cryptopanic_fetcher = CryptoPanicFetcher()
        self.rss_fetcher = RSSFeedFetcher()
        self.reddit_fetcher = RedditFetcher()
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
    async def fetch_all_news(self, symbols: List[str]) -> List[NewsItem]:
        """Fetch news from all sources and combine them"""
        cache_key = ",".join(sorted(symbols))
        
        # Check cache
        if cache_key in self.cache:
            cache_time, cached_news = self.cache[cache_key]
            if time.time() - cache_time < self.cache_duration:
                logger.info("Using cached news data")
                return cached_news
        
        logger.info("Fetching fresh news from all sources...")
        
        # Fetch from all sources concurrently
        tasks = [
            self.cryptopanic_fetcher.fetch_news(symbols),
            self.rss_fetcher.fetch_news(symbols),
            self.reddit_fetcher.fetch_news(symbols)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_news = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error fetching news: {result}")
            else:
                all_news.extend(result)
        
        # Sort by importance and recency
        all_news.sort(key=lambda x: (x.importance_score, x.published_time), reverse=True)
        
        # Remove duplicates based on title similarity
        unique_news = self._remove_duplicates(all_news)
        
        # Cache the results
        self.cache[cache_key] = (time.time(), unique_news)
        
        logger.info(f"Fetched {len(unique_news)} unique news items")
        return unique_news[:50]  # Return top 50 most relevant
    
    def _remove_duplicates(self, news_items: List[NewsItem]) -> List[NewsItem]:
        """Remove duplicate news items based on title similarity"""
        unique_news = []
        seen_titles = set()
        
        for item in news_items:
            # Create a normalized title for comparison
            normalized_title = re.sub(r'[^\w\s]', '', item.title.lower())
            normalized_title = ' '.join(normalized_title.split())
            
            # Check if we've seen a similar title
            is_duplicate = False
            for seen_title in seen_titles:
                if self._similarity(normalized_title, seen_title) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_news.append(item)
                seen_titles.add(normalized_title)
                
        return unique_news
    
    def _similarity(self, a: str, b: str) -> float:
        """Calculate simple similarity between two strings"""
        words_a = set(a.split())
        words_b = set(b.split())
        
        if not words_a or not words_b:
            return 0.0
            
        intersection = words_a.intersection(words_b)
        union = words_a.union(words_b)
        
        return len(intersection) / len(union)
