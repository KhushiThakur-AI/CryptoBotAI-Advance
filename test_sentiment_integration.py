
import asyncio
import logging
from news_fetcher import NewsAggregator
from sentiment_analyzer import SentimentAnalyzer
from news_sentiment_integration import NewsSentimentIntegration
import json

logging.basicConfig(level=logging.INFO)

async def test_sentiment_integration():
    """Test the complete sentiment analysis pipeline"""
    
    # Load config
    with open("config.json", "r") as f:
        config_data = json.load(f)
    
    # Test symbols
    test_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    
    print("🔍 Testing News & Sentiment Analysis Integration...")
    print("=" * 60)
    
    # Initialize components
    news_aggregator = NewsAggregator()
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_integration = NewsSentimentIntegration(config_data)
    
    try:
        # Test 1: Fetch news from all sources
        print("\n📰 Step 1: Fetching news from all sources...")
        news_items = await news_aggregator.fetch_all_news(test_symbols)
        print(f"✅ Fetched {len(news_items)} news items")
        
        if news_items:
            print(f"📄 Sample headlines:")
            for i, item in enumerate(news_items[:3]):
                print(f"  {i+1}. {item.title[:80]}... ({item.source})")
        
        # Test 2: Analyze sentiment
        print(f"\n🎭 Step 2: Analyzing sentiment...")
        symbol_sentiments = await sentiment_analyzer.analyze_news_sentiment(news_items, test_symbols)
        
        for symbol in test_symbols:
            if symbol in symbol_sentiments:
                data = symbol_sentiments[symbol]
                print(f"  {symbol}: Score={data['overall_score']:.3f}, "
                      f"Confidence={data['confidence']:.2f}, "
                      f"News={data['total_count']}")
        
        # Test 3: Get trading signals
        print(f"\n📈 Step 3: Getting trading signals...")
        trading_signals = await sentiment_integration.get_sentiment_signals(test_symbols)
        
        for symbol in test_symbols:
            if symbol in trading_signals:
                data = trading_signals[symbol]
                print(f"  {symbol}: {data['recommendation']} "
                      f"(Contribution: {data['sentiment_contribution']:.3f})")
        
        # Test 4: Market overview
        print(f"\n🌍 Step 4: Market sentiment overview...")
        market_overview = await sentiment_integration.get_market_sentiment_overview(test_symbols)
        print(f"  Market Mood: {market_overview['market_mood']}")
        print(f"  Market Score: {market_overview['market_sentiment_score']:.3f}")
        print(f"  Total News: {market_overview['total_news_analyzed']}")
        
        # Test 5: Signal integration
        print(f"\n🔧 Step 5: Testing signal integration...")
        for symbol in test_symbols:
            if symbol in trading_signals:
                sentiment_data = trading_signals[symbol]
                
                # Test buy signal adjustment
                original_buy_score = 2.0
                adjusted_buy, reasoning = sentiment_integration.apply_sentiment_to_technical_signal(
                    original_buy_score, sentiment_data, "BUY"
                )
                print(f"  {symbol} BUY: {original_buy_score:.2f} → {adjusted_buy:.2f}")
                print(f"    Reasoning: {reasoning}")
        
        print(f"\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_sentiment_integration())
