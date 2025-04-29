import asyncio
import logging
import re
from datetime import datetime
from typing import Dict

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler, MessageHandler, filters
from telegram.constants import ParseMode
from services.pattern_recognition import TalibPatternRecognition

logger = logging.getLogger(__name__)

class TelegramBot:
    def __init__(self, config: dict, db_manager, stock_collector):
        self.token = config.get('api_key')
        self.chat_id = config.get('chat_id')
        self.db_manager = db_manager
        self.stock_collector = stock_collector
        self.application = None
        self.pattern_recognizer = TalibPatternRecognition()
        self.sentiment_tracker=None

        # Timeframe configurations
        self.timeframe_weights = {
            'long_term': 0.5,
            'medium_term': 0.3,
            'short_term': 0.2
        }

        # Confidence thresholds
        self.confidence_thresholds = {
            'very_high': 0.9,
            'high': 0.7,
            'medium-high': 0.6,
            'medium': 0.5,
            'low': 0.3
        }

    async def send_news_notification(self, news):
        """Send news summary notification to the Telegram chat."""
        try:
            formatted_message,rating = self._format_news_summary(news)
            if rating in range(3,7):
                disable_notification = True
            else:
                disable_notification= False
            await self.application.bot.send_message(
                chat_id=self.chat_id,
                text=formatted_message,
                parse_mode=ParseMode.MARKDOWN_V2,
                disable_notification=disable_notification
            )
        except Exception as e:
            logger.error(f"Error sending news notification: {e}")

    async def send_pattern_notification(self, message: str):
        """Send pattern detection notification to the Telegram chat."""
        try:
            await self.application.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception as e:
            logger.error(f"Error sending pattern notification: {e}")

    def _format_news_summary(self, news) -> str:
        """Format news summary for Telegram."""
        summary = news.summary
        title = news.article.title
        article_url = news.article.url

        # Extract information
        sentiment_category = self._extract_sentiment_category(summary)
        rating = self._extract_rating(summary)
        reasoning = self._extract_reasoning(summary)
        implications = self._extract_market_implications(summary)
        insights = self._extract_actionable_insights(summary)

        # Escape for Markdown V2
        title_escaped = self._escape_markdown(title)

        formatted_message = (
            f"📊 *FINANCIAL NEWS ALERT* 📊\n\n"
            f"📰 *{title_escaped}* 📰\n\n"
            f"*SENTIMENT:* {self._escape_markdown(sentiment_category)}\n"
            f"*Rating:* {self._escape_markdown(rating)}/10\n"
            f"*Reasoning:* {self._escape_markdown(reasoning)}\n\n"
            f"*KEY MARKET IMPLICATIONS:*\n{self._format_bullet_list(implications)}\n\n"
            f"*ACTIONABLE INSIGHTS:*\n{self._format_insights_list(insights)}\n\n"
        )

        if article_url:
            escaped_url = self._escape_markdown(article_url)
            formatted_message += f"🔗 [Read Full Article]({escaped_url})"

        return (formatted_message,rating)

    def _extract_sentiment_category(self, summary):
        """Extract sentiment category from summary."""
        match = re.search(r'\*\*Sentiment Category:\*\*\s*([^()\n]+)(?:\([\d-]+/10\))?', summary)
        if not match:
            match = re.search(r'Sentiment Category:\s*([^()\n]+)(?:\([\d-]+/10\))?', summary, re.IGNORECASE)
        return match.group(1).strip() if match else "Neutral"

    def _extract_rating(self, summary):
        """Extract numerical rating from summary."""
        match = re.search(r'\*\*Numerical Rating:\*\*\s*(\d+)', summary)
        if not match:
            match = re.search(r'Numerical Rating:\s*(\d+)', summary, re.IGNORECASE)
        if not match:
            match = re.search(r'Sentiment Category:.*?\((\d+)(?:-\d+)?/10\)', summary, re.IGNORECASE)

        # Bug fix: Return default value if match is None
        return match.group(1).strip() if match else "5"

    def _extract_reasoning(self, summary):
        """Extract reasoning from summary."""
        match = re.search(r'\*\*Reasoning:\*\*\s*(.*?)(?=\d+\.\s+\*\*KEY|\*\*KEY|KEY)', summary, re.DOTALL)
        if not match:
            match = re.search(r'Reasoning:\s*(.*?)(?=\d+\.\s+\*\*KEY|\*\*KEY|KEY)', summary, re.DOTALL | re.IGNORECASE)

        if match:
            reasoning = match.group(1).strip()
            if len(reasoning) > 200:
                last_period = reasoning[:197].rfind('.')
                if last_period > 150:
                    reasoning = reasoning[:last_period+1]
                else:
                    reasoning = reasoning[:197] + "..."
            return reasoning
        return "See full analysis for details"

    def _extract_market_implications(self, summary):
        """Extract market implications from summary."""
        implications = []
        section_match = re.search(r'(?:\d+\.\s+)?\*\*KEY MARKET IMPLICATIONS:\*\*(.*?)(?=\d+\.\s+\*\*ACTIONABLE|\*\*ACTIONABLE|ACTIONABLE)',
                                  summary, re.DOTALL)
        if not section_match:
            section_match = re.search(r'(?:\d+\.\s+)?KEY MARKET IMPLICATIONS:(.*?)(?=\d+\.\s+\*\*ACTIONABLE|\*\*ACTIONABLE|ACTIONABLE)',
                                      summary, re.DOTALL | re.IGNORECASE)

        if section_match:
            section = section_match.group(1)
            point_matches = re.findall(r'\*\*Point\s+\d+:\*\*\s*(.*?)(?=\s+[-•]\s+\*\*Impact|\s+\*\*Impact|$)',
                                       section, re.DOTALL | re.IGNORECASE)
            if not point_matches:
                point_matches = re.findall(r'Point\s+\d+:\s*(.*?)(?=\s+[-•]\s+Impact|\s+Impact|$)',
                                           section, re.DOTALL | re.IGNORECASE)
            if not point_matches:
                bullet_matches = re.findall(r'[-•]\s+\*\*([^:*]+)(?:\*\*)?:(.*?)(?=[-•]|$)', section, re.DOTALL)
                point_matches = [f"{header}: {content.strip()}" for header, content in bullet_matches]
            if not point_matches:
                point_matches = re.findall(r'[-•]\s+(.*?)(?=[-•]|$)', section, re.DOTALL)
            implications = [p.strip() for p in point_matches if p.strip()]

        return implications[:5] if implications else ["Market impact analysis available in full report"]

    def _extract_actionable_insights(self, summary):
        """Extract actionable insights from summary."""
        insights = []
        section_match = re.search(r'(?:\d+\.\s+)?\*\*ACTIONABLE INSIGHTS:\*\*(.*?)(?=Reference link:|$)',
                                  summary, re.DOTALL)
        if not section_match:
            section_match = re.search(r'(?:\d+\.\s+)?ACTIONABLE INSIGHTS:(.*?)(?=Reference link:|$)',
                                      summary, re.DOTALL | re.IGNORECASE)

        if section_match:
            section = section_match.group(1)
            strategy_matches = re.findall(r'\*\*Investment Strategy\s+\d+:\*\*\s*(.*?)(?=\*\*Investment Strategy|\*\*Confidence|$)',
                                          section, re.DOTALL)
            if not strategy_matches:
                strategy_matches = re.findall(r'Investment Strategy\s+\d+:\s*(.*?)(?=Investment Strategy|Confidence|$)',
                                              section, re.DOTALL | re.IGNORECASE)
            if not strategy_matches:
                strategy_matches = re.findall(r'[-•]\s+(.*?)(?=[-•]|$)', section, re.DOTALL)
            insights = [s.strip() for s in strategy_matches if s.strip()]

        return insights[:3] if insights else ["Actionable insights available in full report"]

    def _escape_markdown(self, text):
        """
        Helper function to properly escape MarkdownV2 special characters.
        Must escape: _ * [ ] ( ) ~ ` > # + - = | { } . !
        """
        if not text:
            return ""

        # Bug fix: Handle numeric types
        if isinstance(text, (int, float)):
            text = str(text)

        escape_chars = r'_*[]()~`>#+-=|{}.!'
        return ''.join(f'\\{c}' if c in escape_chars else c for c in text)

    def _format_bullet_list(self, items):
        """Format a list of items as bullet points with proper Markdown escaping"""
        if not items or (len(items) == 1 and not items[0].strip()):
            return "• See full analysis for details"

        result = ""
        for item in items:
            escaped_item = self._escape_markdown(item)
            result += f"• *{escaped_item}*\n"
        return result

    def _format_insights_list(self, items):
        """Format a list of actionable insights with emoji and proper Markdown escaping"""
        if not items or (len(items) == 1 and not items[0].strip()):
            return "💡 See full analysis for details"

        result = ""
        for item in items:
            escaped_item = self._escape_markdown(item)
            result += f"💡 *{escaped_item}*\n"
        return result

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        message = (
            "📊 *Financial Monitor Bot* 📊\n\n"
            "Available commands:\n"
            "/watchlist - Show current watchlist\n"
            "/add <ticker> - Add stock to watchlist\n"
            "/remove <ticker> - Remove stock from watchlist\n"
            "/price <ticker> - Get current price (multi-timeframe)\n"
            "/history <ticker> - Get price history\n"
            "/pattern <ticker> - Analyze patterns (multi-timeframe)\n"
            "/sentiment - View overall market sentiment\n"
            "/tickersentiment <ticker> - Get sentiment for a specific ticker\n"
            "/analyze <ticker> - Comprehensive technical + sentiment analysis\n"
            "/latest - Get latest news summaries\n"
            "/stats - Get system statistics\n"
            "/help - Show this help message"
        )
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

    async def watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced watchlist command with key insights for each ticker."""
        watchlist = self.db_manager.get_active_watchlist()

        if not watchlist:
            await update.message.reply_text("Your watchlist is empty. Add stocks with /add TICKER")
            return

        message = "📋 *WATCHLIST SUMMARY*\n\n"

        # Get multi-timeframe prices and technical data for all tickers
        all_prices = self.stock_collector.get_latest_prices()

        # Get sentiment data if available
        sentiment_data = {}
        if hasattr(self, 'sentiment_tracker'):
            try:
                for ticker in watchlist:
                    ticker_sentiment = self.sentiment_tracker.get_ticker_sentiment(ticker)
                    if ticker_sentiment:
                        sentiment_data[ticker] = ticker_sentiment
            except Exception as e:
                logger.error(f"Error getting sentiment data: {e}")

        # Process each ticker
        watchlist_items = []

        for ticker in watchlist:
            item = {"ticker": ticker}

            # Get price data
            if ticker in all_prices:
                ticker_data = all_prices[ticker]

                # Find the most recent price
                current_price = None
                for tf in ['short_term', 'medium_term', 'long_term', 'very_short_term']:
                    if tf in ticker_data and ticker_data[tf].get('price') is not None:
                        current_price = ticker_data[tf].get('price')

                        # Try to get day change if available
                        if 'open' in ticker_data[tf] and ticker_data[tf]['open'] is not None:
                            open_price = ticker_data[tf]['open']
                            if open_price > 0:
                                day_change = ((current_price / open_price) - 1) * 100
                                item['day_change'] = day_change
                        break

                item['price'] = current_price

            # Get technical signals
            tech_signals = {"bullish": 0, "bearish": 0, "neutral": 0}

            try:
                # Get multi-timeframe data
                all_data = self.stock_collector.get_multi_timeframe_data(ticker)

                for timeframe, data in all_data.items():
                    if data.empty:
                        continue

                    # Calculate indicators
                    indicators = self.stock_collector.calculate_technical_indicators(data)

                    if not indicators:
                        continue

                    # RSI
                    if 'rsi' in indicators:
                        rsi = indicators['rsi']
                        if isinstance(rsi, (int, float)):
                            if rsi < 30:
                                tech_signals['bullish'] += 1
                            elif rsi > 70:
                                tech_signals['bearish'] += 1
                            else:
                                tech_signals['neutral'] += 1

                    # MACD
                    if all(k in indicators for k in ['macd', 'macd_signal', 'macd_hist']):
                        hist = indicators['macd_hist']
                        if isinstance(hist, (int, float)):
                            if hist > 0:
                                tech_signals['bullish'] += 1
                            else:
                                tech_signals['bearish'] += 1

                    # MA Trend
                    if 'ma_trend' in indicators:
                        ma_trend = indicators['ma_trend']
                        if ma_trend == 'bullish':
                            tech_signals['bullish'] += 1
                        elif ma_trend == 'bearish':
                            tech_signals['bearish'] += 1
                        else:
                            tech_signals['neutral'] += 1
            except Exception as e:
                logger.error(f"Error calculating technical signals for {ticker}: {e}")

            item['tech_signals'] = tech_signals

            # Set technical bias
            if tech_signals['bullish'] > tech_signals['bearish'] * 1.5:
                item['tech_bias'] = "BULLISH"
            elif tech_signals['bearish'] > tech_signals['bullish'] * 1.5:
                item['tech_bias'] = "BEARISH"
            elif tech_signals['bullish'] > tech_signals['bearish']:
                item['tech_bias'] = "SLIGHTLY BULLISH"
            elif tech_signals['bearish'] > tech_signals['bullish']:
                item['tech_bias'] = "SLIGHTLY BEARISH"
            else:
                item['tech_bias'] = "NEUTRAL"

            # Get sentiment data
            if ticker in sentiment_data:
                item['sentiment'] = sentiment_data[ticker]['value']
                item['sentiment_status'] = sentiment_data[ticker]['status']

            # Get pattern signals
            pattern_signals = {"bullish": 0, "bearish": 0, "neutral": 0}

            try:
                # Check for patterns in the shortest timeframe available
                for timeframe in ['short_term', 'medium_term', 'long_term']:
                    if timeframe in all_data and not all_data[timeframe].empty:
                        patterns = self.pattern_recognizer.detect_patterns(all_data[timeframe], lookback_periods=3)

                        if patterns:
                            for pattern_name, occurrences in patterns.items():
                                if occurrences:
                                    for occurrence in occurrences:
                                        signal = occurrence['signal']
                                        if signal > 0:
                                            pattern_signals['bullish'] += 1
                                        elif signal < 0:
                                            pattern_signals['bearish'] += 1
                                        else:
                                            pattern_signals['neutral'] += 1
                        break
            except Exception as e:
                logger.error(f"Error detecting patterns for {ticker}: {e}")

            item['pattern_signals'] = pattern_signals

            # Generate overall signal
            bullish_count = 0
            bearish_count = 0

            # Count technical bias
            if 'tech_bias' in item:
                if "BULLISH" in item['tech_bias']:
                    bullish_count += 1
                elif "BEARISH" in item['tech_bias']:
                    bearish_count += 1

            # Count pattern signals
            if pattern_signals['bullish'] > pattern_signals['bearish']:
                bullish_count += 1
            elif pattern_signals['bearish'] > pattern_signals['bullish']:
                bearish_count += 1

            # Count sentiment
            if 'sentiment' in item:
                if item['sentiment'] >= 6.5:
                    bullish_count += 1
                elif item['sentiment'] <= 4.0:
                    bearish_count += 1

            # Set overall signal
            if bullish_count >= 2 and bearish_count == 0:
                item['signal'] = "BUY"
            elif bullish_count > bearish_count:
                item['signal'] = "BULLISH"
            elif bearish_count >= 2 and bullish_count == 0:
                item['signal'] = "SELL"
            elif bearish_count > bullish_count:
                item['signal'] = "BEARISH"
            else:
                item['signal'] = "NEUTRAL"

            watchlist_items.append(item)

        # Sort watchlist items by signal priority: BUY, BULLISH, SELL, BEARISH, NEUTRAL
        signal_priority = {"BUY": 0, "BULLISH": 1, "SELL": 2, "BEARISH": 3, "NEUTRAL": 4}
        watchlist_items.sort(key=lambda x: signal_priority.get(x.get('signal', "NEUTRAL"), 5))

        # Format watchlist table
        for item in watchlist_items:
            ticker = item['ticker']

            # Format price and change
            price_str = f"${item['price']:.2f}" if 'price' in item and item['price'] is not None else "N/A"

            # Add change with emoji if available
            if 'day_change' in item:
                change = item['day_change']
                change_emoji = "🔼" if change > 0 else "🔽" if change < 0 else "➖"
                price_str += f" {change_emoji} {change:.2f}%"

            # Format signal with emoji
            signal = item.get('signal', "NEUTRAL")
            if signal == "BUY":
                signal_emoji = "🟢"
            elif signal == "BULLISH":
                signal_emoji = "🟢"
            elif signal == "SELL":
                signal_emoji = "🔴"
            elif signal == "BEARISH":
                signal_emoji = "🔴"
            else:
                signal_emoji = "⚪"

            # Add sentiment emoji if available
            sentiment_emoji = ""
            if 'sentiment' in item:
                sentiment = item['sentiment']
                if sentiment >= 7.0:
                    sentiment_emoji = "😀"
                elif sentiment >= 6.0:
                    sentiment_emoji = "🙂"
                elif sentiment <= 3.0:
                    sentiment_emoji = "😞"
                elif sentiment <= 4.0:
                    sentiment_emoji = "🙁"
                else:
                    sentiment_emoji = "😐"

            # Format insights
            insights = []

            # Technical indicators insight
            if 'tech_bias' in item:
                insights.append(f"Tech: {item['tech_bias'].lower()}")

            # Pattern insight
            pattern_signals = item.get('pattern_signals', {})
            if pattern_signals.get('bullish', 0) > 0 or pattern_signals.get('bearish', 0) > 0:
                bullish = pattern_signals.get('bullish', 0)
                bearish = pattern_signals.get('bearish', 0)
                if bullish > bearish:
                    insights.append(f"{bullish} bullish patterns")
                elif bearish > bullish:
                    insights.append(f"{bearish} bearish patterns")

            # Sentiment insight if available
            if 'sentiment_status' in item:
                insights.append(f"Sentiment: {item['sentiment_status'].lower()}")

            # Format final insights text
            insights_text = ", ".join(insights) if insights else "No insights available"

            # Add to message
            message += f"{signal_emoji} *{self.escape_markdown(ticker)}*  {self.escape_markdown(price_str)} {sentiment_emoji}\n"
            message += f"  {self.escape_markdown(insights_text)}\n\n"

        # Add command suggestions
        message += "*Commands:*\n"
        message += "• For detailed insights: `/analyze TICKER`\n"
        message += "• For pattern analysis: `/pattern TICKER`\n"
        message += "• For price details: `/price TICKER`\n"
        message += "• For sentiment analysis: `/tickersentiment TICKER`\n\n"

        # Create inline keyboard for easy management
        keyboard = [
            [InlineKeyboardButton(f"🔍 Analyze {ticker}", callback_data=f"analyze_{ticker}"),
             InlineKeyboardButton(f"❌ Remove {ticker}", callback_data=f"remove_{ticker}")]
            for ticker in watchlist
        ]

        # Add a "Refresh" button
        keyboard.append([InlineKeyboardButton("🔄 Refresh Watchlist", callback_data="refresh_watchlist")])

        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=reply_markup)

    async def add_stock(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /add <ticker> command."""
        if not context.args:
            await update.message.reply_text("Please provide a ticker symbol: /add AAPL")
            return

        ticker = context.args[0].upper()
        self.stock_collector.add_stock(ticker)

        await update.message.reply_text(f"✅ Added {ticker} to watchlist")

    async def remove_stock(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /remove <ticker> command."""
        if not context.args:
            await update.message.reply_text("Please provide a ticker symbol: /remove AAPL")
            return

        ticker = context.args[0].upper()
        if self.db_manager.remove_from_watchlist(ticker):
            self.stock_collector.remove_stock(ticker)
            await update.message.reply_text(f"❌ Removed {ticker} from watchlist")
        else:
            await update.message.reply_text(f"{ticker} not found in watchlist")

    async def price(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced price analysis with multi-timeframe data and technical context."""
        if not context.args:
            await update.message.reply_text("Please provide a ticker symbol: /price AAPL")
            return
        ticker = context.args[0].upper()

        # Get multi-timeframe data
        all_prices = self.stock_collector.get_latest_prices()
        all_data = self.stock_collector.get_multi_timeframe_data(ticker)

        if ticker not in all_prices or not all_prices[ticker]:
            await update.message.reply_text(f"No price data available for {ticker}\\. You might need to add it to your watchlist first\\.")
            return

        ticker_data = all_prices[ticker]
        message = f"💰 *{self.escape_markdown(ticker)} Price Analysis*\n\n"

        # Get current price for reference
        current_price = None
        if 'short_term' in ticker_data:
            current_price = ticker_data['short_term'].get('price', None)

        if current_price:
            message += f"*Current Price:* ${self.escape_markdown(f'{current_price:.2f}')}\n\n"

        # Add historical context - calculate changes across timeframes
        message += "*Price Changes:*\n"

        timeframe_names = {
            'long_term': '1h',
            'medium_term': '15m',
            'short_term': '5m',
            'very_short_term': '2m'
        }

        # Calculate and show price changes
        for tf_name, tf_display in timeframe_names.items():
            if tf_name in all_data and not all_data[tf_name].empty:
                df = all_data[tf_name]
                if len(df) > 1:
                    current = df['close'].iloc[-1]

                    # Calculate changes for different lookback periods
                    changes = {}
                    lookbacks = {
                        '1 period': 1,
                        '5 periods': min(5, len(df) - 1),
                        '20 periods': min(20, len(df) - 1)
                    }

                    for period_name, periods in lookbacks.items():
                        if len(df) > periods:
                            previous = df['close'].iloc[-periods-1]
                            pct_change = ((current / previous) - 1) * 100
                            changes[period_name] = pct_change

                    # Only show this timeframe if we have change data
                    if changes:
                        message += f"*{self.escape_markdown(tf_display)}:* "

                        for period, change in changes.items():
                            # Format with arrow and color indicator
                            if change > 0:
                                arrow = "🔼"
                            elif change < 0:
                                arrow = "🔽"
                            else:
                                arrow = "➖"

                            message += f"{arrow}{self.escape_markdown(f'{change:.2f}%')} \\({period}\\) "

                        message += "\n"

        message += "\n"

        # Add volume analysis
        message += "*Volume Analysis:*\n"
        for tf_name, tf_display in timeframe_names.items():
            if tf_name in all_data and not all_data[tf_name].empty:
                df = all_data[tf_name]
                if len(df) > 20 and 'volume' in df.columns:
                    current_vol = df['volume'].iloc[-1]
                    avg_vol = df['volume'].iloc[-20:].mean()

                    if avg_vol > 0:
                        vol_ratio = current_vol / avg_vol

                        if vol_ratio > 1.5:
                            vol_desc = "Very High"
                            vol_emoji = "📈📈"
                        elif vol_ratio > 1.2:
                            vol_desc = "Above Average"
                            vol_emoji = "📈"
                        elif vol_ratio < 0.8:
                            vol_desc = "Below Average"
                            vol_emoji = "📉"
                        elif vol_ratio < 0.5:
                            vol_desc = "Very Low"
                            vol_emoji = "📉📉"
                        else:
                            vol_desc = "Average"
                            vol_emoji = "➖"

                        message += f"*{self.escape_markdown(tf_display)}:* {vol_emoji} {self.escape_markdown(vol_desc)} \\({self.escape_markdown(f'{vol_ratio:.2f}x')} avg\\)\n"

        message += "\n"

        # Add key price levels (support/resistance)
        message += "*Key Price Levels:*\n"

        # Use the longest timeframe for identifying support/resistance levels
        primary_tf = 'long_term'
        if primary_tf in all_data and not all_data[primary_tf].empty:
            df = all_data[primary_tf]

            if len(df) >= 30:
                # Simple support/resistance detection using recent peaks and troughs
                # This is a simplified version - the real implementation would be more complex

                # Get recent high/low
                recent_high = df['high'].iloc[-20:].max()
                recent_low = df['low'].iloc[-20:].min()

                # Last close
                last_close = df['close'].iloc[-1]

                # Additional levels using SMA
                sma_20 = df['close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else None
                sma_50 = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else None

                message += f"Resistance: ${self.escape_markdown(f'{recent_high:.2f}')}\n"
                message += f"Support: ${self.escape_markdown(f'{recent_low:.2f}')}\n"

                if sma_20 is not None:
                    message += f"SMA 20: ${self.escape_markdown(f'{sma_20:.2f}')}"
                    # Show position relative to SMA
                    if last_close > sma_20:
                        message += " \\(Price Above ↗️\\)"
                    else:
                        message += " \\(Price Below ↘️\\)"
                    message += "\n"

                if sma_50 is not None:
                    message += f"SMA 50: ${self.escape_markdown(f'{sma_50:.2f}')}"
                    # Show position relative to SMA
                    if last_close > sma_50:
                        message += " \\(Price Above ↗️\\)"
                    else:
                        message += " \\(Price Below ↘️\\)"
                    message += "\n"

        message += "\n"

        # Add detailed technical indicators
        message += "*Technical Indicators:*\n"

        # Get technical indicators for each timeframe
        try:
            summary = self.stock_collector.get_summary(ticker)
            if 'timeframes' in summary:
                for timeframe, tf_data in summary['timeframes'].items():
                    if timeframe in timeframe_names and 'indicators' in tf_data:
                        tf_display = timeframe_names.get(timeframe)
                        indicators = tf_data['indicators']

                        # Only show timeframes with actual indicators
                        has_indicators = False
                        for ind in ['rsi', 'macd', 'stoch_k', 'bb_pct']:
                            if ind in indicators:
                                has_indicators = True
                                break

                        if not has_indicators:
                            continue

                        message += f"*{self.escape_markdown(tf_display)}:*\n"

                        # RSI with interpretation
                        if 'rsi' in indicators:
                            rsi = indicators['rsi']
                            if isinstance(rsi, (int, float)):
                                rsi_str = f"{rsi:.1f}"

                                # Add interpretation
                                if rsi > 70:
                                    rsi_str += " (Overbought 🔴)"
                                elif rsi < 30:
                                    rsi_str += " (Oversold 🟢)"
                                elif rsi > 60:
                                    rsi_str += " (Bullish 🟢)"
                                elif rsi < 40:
                                    rsi_str += " (Bearish 🔴)"

                                message += f"RSI: {self.escape_markdown(rsi_str)}\n"

                        # MACD
                        if all(k in indicators for k in ['macd', 'macd_signal', 'macd_hist']):
                            macd = indicators['macd']
                            signal = indicators['macd_signal']
                            hist = indicators['macd_hist']

                            if all(isinstance(x, (int, float)) for x in [macd, signal, hist]):
                                # Determine if bullish or bearish
                                if hist > 0 and hist > indicators.get('macd_hist_prev', 0):
                                    macd_str = f"MACD: {hist:.3f} (Bullish Momentum 🟢)"
                                elif hist < 0 and hist < indicators.get('macd_hist_prev', 0):
                                    macd_str = f"MACD: {hist:.3f} (Bearish Momentum 🔴)"
                                elif hist > 0:
                                    macd_str = f"MACD: {hist:.3f} (Positive 🟢)"
                                elif hist < 0:
                                    macd_str = f"MACD: {hist:.3f} (Negative 🔴)"
                                else:
                                    macd_str = f"MACD: {hist:.3f} (Neutral ⚪)"

                                message += f"{self.escape_markdown(macd_str)}\n"

                        # Bollinger Bands position
                        if 'bb_pct' in indicators:
                            bb_pct = indicators['bb_pct']
                            if isinstance(bb_pct, (int, float)):
                                bb_str = f"{bb_pct:.2f}"

                                # Add interpretation
                                if bb_pct > 0.8:
                                    bb_str += " (Near Upper Band 🔴)"
                                elif bb_pct < 0.2:
                                    bb_str += " (Near Lower Band 🟢)"
                                else:
                                    bb_str += " (Middle Range ⚪)"

                                message += f"BBands: {self.escape_markdown(bb_str)}\n"

                        # Stochastic oscillator
                        if all(k in indicators for k in ['stoch_k', 'stoch_d']):
                            k = indicators['stoch_k']
                            d = indicators['stoch_d']

                            if all(isinstance(x, (int, float)) for x in [k, d]):
                                if k > 80 and d > 80:
                                    stoch_str = f"Stoch: {k:.1f}/{d:.1f} (Overbought 🔴)"
                                elif k < 20 and d < 20:
                                    stoch_str = f"Stoch: {k:.1f}/{d:.1f} (Oversold 🟢)"
                                elif k > d:
                                    stoch_str = f"Stoch: {k:.1f}/{d:.1f} (Bullish Crossover 🟢)"
                                elif k < d:
                                    stoch_str = f"Stoch: {k:.1f}/{d:.1f} (Bearish Crossover 🔴)"
                                else:
                                    stoch_str = f"Stoch: {k:.1f}/{d:.1f} (Neutral ⚪)"

                                message += f"{self.escape_markdown(stoch_str)}\n"

                        message += "\n"
        except Exception as e:
            logger.error(f"Error getting technical indicators: {e}")
            message += "Error retrieving technical indicators\n"

        # Add market context
        try:
            # This would pull current market data from a market sentiment service
            if hasattr(self, 'sentiment_tracker'):
                market_sentiment = self.sentiment_tracker.get_current_sentiment()
                if market_sentiment:
                    message += "*Market Context:*\n"
                    value = market_sentiment['value']
                    message += f"Market Sentiment: {self.escape_markdown(market_sentiment['status'])} \\({self.escape_markdown(f'{value:.1f}/10')}\\)\n"
        except Exception as e:
            logger.error(f"Error getting market context: {e}")

        # Add volume profile information if available
        # (This would require additional implementation for volume profile analysis)

        # Add disclaimer
        message += "\n⚠️ *Disclaimer:* This is technical analysis based on historical data and indicators\\. Past performance is not indicative of future results\\."

        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN_V2)



    async def history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /history <ticker> command."""
        if not context.args:
            await update.message.reply_text("Please provide a ticker symbol: /history AAPL")
            return

        ticker = context.args[0].upper()
        history = self.db_manager.get_stock_history(ticker, limit=10)

        if not history:
            await update.message.reply_text(f"No historical data available for {ticker}")
            return

        message = f"📈 *{ticker} Price History*\n\n"
        for data in history:
            message += f"{data.timestamp.strftime('%Y-%m-%d %H:%M')} - ${data.close:.2f}\n"

        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
    async def check_pattern(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced pattern analysis with multi-timeframe insights and actionable signals."""
        if not context.args:
            await update.message.reply_text("Please provide a ticker symbol: /pattern AAPL")
            return
        ticker = context.args[0].upper()

        # Get multi-timeframe data
        all_data = self.stock_collector.get_multi_timeframe_data(ticker)

        if not all_data or all(data.empty for data in all_data.values()):
            await update.message.reply_text(f"No data available for {ticker}. Make sure it's in your watchlist.")
            return

        message = f"📊 *Advanced Pattern Analysis for {self.escape_markdown(ticker)}*\n\n"

        # Create a more sophisticated pattern analyzer using our enhanced class
        from services.pattern_recognition import TalibPatternRecognition
        enhanced_recognizer = TalibPatternRecognition()

        # Initialize the enhanced pattern recognition with our base recognizer
        # This would be better if we had the enhanced class available directly
        from pattern_recognition import TalibPatternRecognition as EnhancedPatternRecognizer
        enhanced_analyzer = EnhancedPatternRecognizer(enhanced_recognizer)

        # Use the advanced analyzer to get comprehensive insights
        analysis_results = enhanced_analyzer.analyze_patterns(ticker, all_data)

        # Add current price information
        try:
            latest_prices = self.stock_collector.get_latest_prices()
            if ticker in latest_prices and 'short_term' in latest_prices[ticker]:
                current_price = latest_prices[ticker]['short_term']['price']
                message += f"💰 *Current Price:* ${self.escape_markdown(f'{current_price:.2f}')}\n\n"
        except Exception as e:
            logger.error(f"Error getting current price: {e}")

        # Add trading signals section
        trading_signals = analysis_results.get("trading_signals", {})
        signal = trading_signals.get("signal", "NEUTRAL")
        confidence = trading_signals.get("confidence", "medium")

        message += "🎯 *Trading Signal:*\n"
        message += f"Action: *{self.escape_markdown(signal)}*\n"
        message += f"Confidence: *{self.escape_markdown(confidence.upper())}*\n\n"

        # Add multi-timeframe confirmation details
        confirmation = analysis_results.get("multi_timeframe_confirmation", {})
        primary_bias = confirmation.get("primary_bias", "neutral")
        alignment_score = confirmation.get("alignment_score", 0)
        aligned_timeframes = confirmation.get("aligned_timeframes", [])

        message += "⏱️ *Multi\\-Timeframe Confirmation:*\n"
        message += f"Bias: *{self.escape_markdown(primary_bias.upper())}*\n"
        message += f"Alignment Score: *{self.escape_markdown(f'{alignment_score:.1f}%')}*\n"

        if aligned_timeframes:
            message += f"Aligned Timeframes: *{self.escape_markdown(', '.join(aligned_timeframes))}*\n\n"
        else:
            message += "Aligned Timeframes: *None*\n\n"

        # Add pattern clusters information
        clusters = analysis_results.get("pattern_clusters", {})
        cluster_bias = clusters.get("bias", "neutral")

        message += "🔍 *Pattern Clusters:*\n"
        message += f"Bias: *{self.escape_markdown(cluster_bias.upper())}*\n"

        # Add bullish clusters
        bullish_clusters = clusters.get("bullish", [])
        if bullish_clusters:
            message += "Bullish Patterns:\n"
            for cluster in bullish_clusters[:2]:  # Show top 2 clusters
                cluster_type = cluster.get("cluster_type", "unknown")
                patterns = [p["pattern"] for p in cluster.get("patterns", [])]
                if patterns:
                    message += f"\\- {self.escape_markdown(cluster_type)}: {self.escape_markdown(', '.join(patterns[:3]))}\n"

        # Add bearish clusters
        bearish_clusters = clusters.get("bearish", [])
        if bearish_clusters:
            message += "Bearish Patterns:\n"
            for cluster in bearish_clusters[:2]:  # Show top 2 clusters
                cluster_type = cluster.get("cluster_type", "unknown")
                patterns = [p["pattern"] for p in cluster.get("patterns", [])]
                if patterns:
                    message += f"\\- {self.escape_markdown(cluster_type)}: {self.escape_markdown(', '.join(patterns[:3]))}\n"

        message += "\n"

        # Add key insights from the analysis
        insights = analysis_results.get("insights", {})

        message += "💡 *Key Insights:*\n"

        summary = insights.get("summary", "")
        if summary:
            message += f"{self.escape_markdown(summary)}\n\n"

        key_points = insights.get("key_points", [])
        if key_points:
            message += "*Key Points:*\n"
            for point in key_points[:3]:  # Top 3 key points
                message += f"\\- {self.escape_markdown(point)}\n"
            message += "\n"

        # Add action items for the trader
        action_items = insights.get("action_items", [])
        if action_items:
            message += "*Recommended Actions:*\n"
            for item in action_items:
                message += f"\\- {self.escape_markdown(item)}\n"
            message += "\n"

        # Add risk factors to be aware of
        risk_factors = insights.get("risk_factors", [])
        if risk_factors:
            message += "*Risk Factors:*\n"
            for risk in risk_factors:
                message += f"\\- {self.escape_markdown(risk)}\n"
            message += "\n"

        # Add entry and exit strategy if signal is not neutral
        if signal != "NEUTRAL":
            entry_strategy = insights.get("entry_strategy", {})
            exit_strategy = insights.get("exit_strategy", {})
            position_sizing = insights.get("position_sizing", {})

            message += "*Trading Strategy:*\n"

            if entry_strategy:
                message += f"Entry: {self.escape_markdown(entry_strategy.get('strategy', 'N/A'))}\n"

            if exit_strategy:
                message += f"Exit: {self.escape_markdown(exit_strategy.get('take_profit', 'N/A'))}\n"
                message += f"Stop Loss: {self.escape_markdown(exit_strategy.get('stop_loss', 'N/A'))}\n"

            if position_sizing:
                message += f"Position Size: {self.escape_markdown(position_sizing.get('recommendation', 'N/A'))}\n"

        # Add disclaimer
        message += "\n⚠️ *Disclaimer:* This is algorithmic analysis based on candlestick patterns and technical indicators\\. Always conduct your own research and consider your risk tolerance before trading\\."

        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN_V2)

    # Add this helper function to escape Markdown characters
    def escape_markdown(self, text):
        """
        Escapes MarkdownV2 special characters as per Telegram's official documentation.
        The characters to escape are: _ * [ ] ( ) ~ ` > # + - = | { } . !
        """
        if not isinstance(text, str):
            text = str(text)

        # Use a set for faster lookup
        escape_chars = {'_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!'}
        return ''.join(f'\\{c}' if c in escape_chars else c for c in text)

    def _combine_timeframe_signals(self, signals):
        """Combine signals from all timeframes to generate a comprehensive trading decision."""
        combined_strength = 0
        action_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0, 'WATCH': 0}
        for timeframe, signal_list in signals.items():
            weight = self.timeframe_weights.get(timeframe, 0.33)

            for signal_item in signal_list:
                signal = signal_item['signal'] if isinstance(signal_item, dict) else signal_item
                if signal and signal['action']:
                    confidence_value = self.confidence_thresholds.get(signal['confidence'], 0.5)

                    if signal['action'] == 'BUY':
                        combined_strength += weight * confidence_value
                        action_scores['BUY'] += weight * confidence_value
                    elif signal['action'] == 'SELL':
                        combined_strength -= weight * confidence_value
                        action_scores['SELL'] += weight * confidence_value
                    elif signal['action'] == 'HOLD':
                        action_scores['HOLD'] += weight * confidence_value
                    else:  # WATCH
                        action_scores['WATCH'] += weight * confidence_value

        # Determine final action
        if combined_strength > 0.4:  # Changed from 0.6
            action = 'BUY'
            confidence = 'high'
        elif combined_strength < -0.4:  # Changed from -0.6
            action = 'SELL'
            confidence = 'high'
        else:
            # Find the action with highest score
            action = max(action_scores, key=action_scores.get)
            if action_scores[action] > 0.3:  # Changed from 0.5
                confidence = 'medium-high'
            else:
                confidence = 'medium'

        return action, confidence

    async def latest_news(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /latest command."""
        articles = self.db_manager.get_recent_articles(limit=5)

        if not articles:
            await update.message.reply_text("No recent news articles found.")
            return

        message = "📰 *Latest News Summaries*\n\n"
        for article in articles:
            message += f"• [{article.title}]({article.url})\n"
            message += f"  Sentiment: {article.sentiment_category} ({article.sentiment_rating}/10)\n\n"

        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN_V2)

    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command."""
        session = self.db_manager.get_session()
        try:
            # Import models here to avoid circular imports
            from database.models import NewsArticle, WatchlistItem, StockData

            # Get counts
            article_count = session.query(NewsArticle).count()
            watchlist_count = session.query(WatchlistItem).filter_by(active=True).count()
            stock_data_count = session.query(StockData).count()

            message = (
                "📊 *System Statistics*\n\n"
                f"📰 News Articles: {article_count}\n"
                f"📈 Watchlist Items: {watchlist_count}\n"
                f"💹 Stock Data Points: {stock_data_count}\n"
            )

            await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
        finally:
            session.close()

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline button callbacks."""
        query = update.callback_query
        await query.answer()

        if query.data.startswith("remove_"):
            ticker = query.data[7:]
            if self.db_manager.remove_from_watchlist(ticker):
                self.stock_collector.remove_stock(ticker)
                await query.edit_message_text(f"❌ Removed {ticker} from watchlist")
            else:
                await query.edit_message_text(f"Failed to remove {ticker}")

            self.application = Application.builder().token(self.token).build()

            # Add error handler
    async def error_handler(self,update, context):
        logger.error(f"Update {update} caused error: {context.error}", exc_info=context.error)

    async def log_update(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Log all updates to help with debugging."""
        logger.info(f"Received update: {update}")
        # Continue processing with next handler
        return None




    async def run_async(self):
        """Run the bot in an existing event loop."""
        try:
            # Build the application
            self.application = Application.builder().token(self.token).build()

            # Add handlers
            self.application.add_handler(CommandHandler("start", self.start))
            self.application.add_handler(CommandHandler("help", self.start))
            self.application.add_handler(CommandHandler("watchlist", self.watchlist))
            self.application.add_handler(CommandHandler("add", self.add_stock))
            self.application.add_handler(CommandHandler("remove", self.remove_stock))
            self.application.add_handler(CommandHandler("price", self.price))
            self.application.add_handler(CommandHandler("history", self.history))
            self.application.add_handler(CommandHandler("pattern", self.check_pattern))
            self.application.add_handler(CommandHandler("latest", self.latest_news))
            self.application.add_handler(CommandHandler("stats", self.stats))
            self.application.add_handler(CommandHandler("sentiment", self.sentiment))
            self.application.add_handler(CommandHandler("tickersentiment", self.ticker_sentiment))
            self.application.add_handler(CommandHandler("analyze", self.analyze))
            self.application.add_handler(CallbackQueryHandler(self.button_callback))

            # Add a simple message handler for debugging
            from telegram.ext import MessageHandler, filters
            async def echo(update, context):
                logger.info(f"Echo received: {update.message.text}")
                await update.message.reply_text(f"You said: {update.message.text}")
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

            # Initialize and start the application
            logger.info("Initializing Telegram bot application")
            await self.application.initialize()
            logger.info("Starting Telegram bot application")
            await self.application.start()

            # Start polling with proper configuration for continuous updates
            logger.info("Starting polling")
            await self.application.updater.start_polling(
                drop_pending_updates=False,
                allowed_updates=None,  # Allow all update types
                timeout=30,  # Longer timeout for better reliability
                poll_interval=0.5  # Poll every 0.5 seconds
            )
            logger.info("Polling started successfully")

            # Keep the bot running
            while True:
                # This keeps the task alive but doesn't block other async operations
                await asyncio.sleep(10)
                logger.info("Bot is still running...")

        except Exception as e:
            logger.error(f"Error in bot async run: {e}", exc_info=True)
            raise
        finally:
            # Clean shutdown
            logger.info("Starting bot shutdown sequence")
            try:
                if hasattr(self, 'application') and self.application:
                    if hasattr(self.application, 'updater') and self.application.updater:
                        logger.info("Stopping updater...")
                        await self.application.updater.stop()
                    logger.info("Stopping application...")
                    await self.application.stop()
                    logger.info("Shutting down application...")
                    await self.application.shutdown()
                    logger.info("Bot shutdown complete")
            except Exception as e:
                logger.error(f"Error during bot shutdown: {e}", exc_info=True)


    async def sentiment(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /sentiment command."""
        # Check if sentiment tracker exists
        if not hasattr(self, 'sentiment_tracker'):
            # Initialize sentiment tracker if not already done
            try:
                from services.sentiment_tracker import SentimentTracker
                self.sentiment_tracker = SentimentTracker(self.db_manager)
                logger.info("Sentiment tracker initialized")
            except Exception as e:
                await update.message.reply_text("Sentiment tracking service is not available.")
                logger.error(f"Error initializing sentiment tracker: {e}")
                return

        # Get current sentiment
        sentiment_data = self.sentiment_tracker.get_current_sentiment()

        # Create sentiment meter visualization
        message = self._format_sentiment_meter(sentiment_data)

        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

    async def ticker_sentiment(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /tickersentiment <ticker> command."""
        if not context.args:
            await update.message.reply_text("Please provide a ticker symbol: /tickersentiment AAPL")
            return

        ticker = context.args[0].upper()

        # Check if sentiment tracker exists
        if not hasattr(self, 'sentiment_tracker'):
            # Initialize sentiment tracker if not already done
            try:
                from services.sentiment_tracker import SentimentTracker
                self.sentiment_tracker = SentimentTracker(self.db_manager)
                logger.info("Sentiment tracker initialized")
            except Exception as e:
                await update.message.reply_text("Sentiment tracking service is not available.")
                logger.error(f"Error initializing sentiment tracker: {e}")
                return

        # Get ticker-specific sentiment
        sentiment_data = self.sentiment_tracker.get_ticker_sentiment(ticker)

        # Create sentiment meter visualization
        message = self._format_ticker_sentiment_meter(ticker, sentiment_data)

        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN_V2)
    async def analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced comprehensive analysis combining technical, pattern, and sentiment data."""
        if not context.args:
            await update.message.reply_text("Please provide a ticker symbol: /analyze AAPL")
            return

        ticker = context.args[0].upper()

        # First check if the ticker is valid
        validity_check = self.stock_collector.check_ticker_validity(ticker)
        if not validity_check.get('valid', False):
            await update.message.reply_text(f"Invalid ticker symbol: {ticker}. {validity_check.get('reason', '')}")
            return

        # Begin building comprehensive analysis message
        message = f"🔬 *COMPREHENSIVE ANALYSIS: {self.escape_markdown(ticker)}* 🔬\n\n"

        # Get company info
        company_info = self.stock_collector.get_company_info(ticker)

        # SECTION 1: COMPANY OVERVIEW
        if company_info and not company_info.get('error'):
            message += "*COMPANY OVERVIEW*\n"
            message += f"Name: *{self.escape_markdown(company_info.get('name', ticker))}*\n"
            message += f"Sector: {self.escape_markdown(company_info.get('sector', 'N/A'))}\n"
            message += f"Industry: {self.escape_markdown(company_info.get('industry', 'N/A'))}\n"

            # Add key metrics if available
            if 'market_cap' in company_info:
                market_cap = company_info['market_cap']
                if isinstance(market_cap, (int, float)) and market_cap > 0:
                    # Format market cap in billions/millions
                    if market_cap >= 1_000_000_000:
                        formatted_mcap = f"${market_cap/1_000_000_000:.2f}B"
                    else:
                        formatted_mcap = f"${market_cap/1_000_000:.2f}M"
                    message += f"Market Cap: {self.escape_markdown(formatted_mcap)}\n"

            if 'pe_ratio' in company_info and company_info['pe_ratio'] != 'N/A':
                message += f"P/E Ratio: {self.escape_markdown(str(company_info['pe_ratio']))}\n"

            if 'beta' in company_info and company_info['beta'] != 'N/A':
                message += f"Beta: {self.escape_markdown(str(company_info['beta']))}\n"

            message += "\n"

        # SECTION 2: CURRENT PRICE DATA
        message += "*CURRENT PRICE*\n"

        # Get multi-timeframe prices
        latest_prices = self.stock_collector.get_latest_prices()
        current_price = None
        daily_change = None

        if ticker in latest_prices:
            ticker_data = latest_prices[ticker]

            # Get the most recent price across timeframes
            if 'short_term' in ticker_data:
                current_price = ticker_data['short_term'].get('price')

                # Try to calculate daily change if we have open price
                open_price = ticker_data['short_term'].get('open')
                if current_price is not None and open_price is not None and open_price > 0:
                    daily_change = ((current_price / open_price) - 1) * 100

            # If we couldn't get price from short-term, try other timeframes
            if current_price is None:
                for tf in ['medium_term', 'long_term', 'very_short_term']:
                    if tf in ticker_data and ticker_data[tf].get('price') is not None:
                        current_price = ticker_data[tf].get('price')
                        break

        if current_price is not None:
            message += f"Price: *${self.escape_markdown(f'{current_price:.2f}')}*"

            # Add daily change if available
            if daily_change is not None:
                change_emoji = "🔼" if daily_change > 0 else "🔽" if daily_change < 0 else "➖"
                message += f" {change_emoji} {self.escape_markdown(f'{daily_change:.2f}%')}\n"
            else:
                message += "\n"

            # Add 52-week range if available
            if company_info and not company_info.get('error'):
                high_52w = company_info.get('fifty_two_week_high')
                low_52w = company_info.get('fifty_two_week_low')

                if high_52w not in (None, 'N/A') and low_52w not in (None, 'N/A'):
                    message += f"52\\-Week Range: ${self.escape_markdown(f'{low_52w:.2f}')} \\- ${self.escape_markdown(f'{high_52w:.2f}')}\n"

                    # Calculate where current price is in the 52-week range (0-100%)
                    if high_52w > low_52w and current_price is not None:
                        pct_of_range = (current_price - low_52w) / (high_52w - low_52w) * 100
                        message += f"Current price is at {self.escape_markdown(f'{pct_of_range:.1f}%')} of 52\\-week range\n"
        else:
            message += "Price data not available\n"

        message += "\n"

        # SECTION 3: TECHNICAL ANALYSIS
        message += "*TECHNICAL ANALYSIS*\n"

        # Get multi-timeframe data for technical analysis
        all_data = self.stock_collector.get_multi_timeframe_data(ticker)

        # Define user-friendly timeframe labels
        timeframe_labels = {
            'very_short_term': '2min',
            'short_term': '5min',
            'medium_term': '15min',
            'long_term': '1hour'
        }

        # Calculate indicators for multiple timeframes
        tech_signals = {}

        for timeframe, data in all_data.items():
            if data.empty:
                continue

            # Calculate key indicators
            indicators = self.stock_collector.calculate_technical_indicators(data)
            if not indicators:
                continue

            tf_label = timeframe_labels.get(timeframe, timeframe)
            tech_signals[tf_label] = {
                'bullish': 0,
                'bearish': 0,
                'neutral': 0,
                'indicators': {}
            }

            # Analyze RSI
            if 'rsi' in indicators:
                rsi = indicators['rsi']
                if isinstance(rsi, (int, float)):
                    tech_signals[tf_label]['indicators']['rsi'] = rsi
                    if rsi < 30:
                        tech_signals[tf_label]['bullish'] += 1
                        tech_signals[tf_label]['indicators']['rsi_signal'] = 'bullish'
                    elif rsi > 70:
                        tech_signals[tf_label]['bearish'] += 1
                        tech_signals[tf_label]['indicators']['rsi_signal'] = 'bearish'
                    else:
                        tech_signals[tf_label]['neutral'] += 1
                        tech_signals[tf_label]['indicators']['rsi_signal'] = 'neutral'

            # Analyze MACD
            if all(k in indicators for k in ['macd', 'macd_signal', 'macd_hist']):
                hist = indicators['macd_hist']
                if isinstance(hist, (int, float)):
                    tech_signals[tf_label]['indicators']['macd_hist'] = hist
                    if hist > 0:
                        tech_signals[tf_label]['bullish'] += 1
                        tech_signals[tf_label]['indicators']['macd_signal'] = 'bullish'
                    else:
                        tech_signals[tf_label]['bearish'] += 1
                        tech_signals[tf_label]['indicators']['macd_signal'] = 'bearish'

            # Analyze Moving Averages
            if 'ma_trend' in indicators:
                ma_trend = indicators['ma_trend']
                tech_signals[tf_label]['indicators']['ma_trend'] = ma_trend
                if ma_trend == 'bullish':
                    tech_signals[tf_label]['bullish'] += 1
                elif ma_trend == 'bearish':
                    tech_signals[tf_label]['bearish'] += 1
                else:
                    tech_signals[tf_label]['neutral'] += 1

        # Calculate overall technical bias
        overall_bullish = sum(tf['bullish'] for tf in tech_signals.values())
        overall_bearish = sum(tf['bearish'] for tf in tech_signals.values())
        overall_neutral = sum(tf['neutral'] for tf in tech_signals.values())

        # Determine technical trend
        tech_trend = "NEUTRAL"
        if overall_bullish > overall_bearish * 1.5:
            tech_trend = "BULLISH"
        elif overall_bearish > overall_bullish * 1.5:
            tech_trend = "BEARISH"
        elif overall_bullish > overall_bearish:
            tech_trend = "MODERATELY BULLISH"
        elif overall_bearish > overall_bullish:
            tech_trend = "MODERATELY BEARISH"

        # Display technical analysis summary
        message += f"Overall Trend: *{self.escape_markdown(tech_trend)}*\n"
        message += f"Bullish Signals: {self.escape_markdown(str(overall_bullish))}\n"
        message += f"Bearish Signals: {self.escape_markdown(str(overall_bearish))}\n\n"

        # Display key technical indicators by timeframe
        message += "Key Indicators:\n"
        for tf_label, signals in tech_signals.items():
            indicators = signals['indicators']
            if indicators:
                # Only show timeframes with actual indicators
                message += f"• {self.escape_markdown(tf_label)}: "

                # Show bias first
                if signals['bullish'] > signals['bearish']:
                    message += "Bullish 📈 "
                elif signals['bearish'] > signals['bullish']:
                    message += "Bearish 📉 "
                else:
                    message += "Neutral ↔️ "

                # Add RSI if available
                if 'rsi' in indicators:
                    rsi = indicators['rsi']
                    rsi_signal = indicators.get('rsi_signal', 'neutral')

                    # Format RSI with signal
                    if rsi_signal == 'bullish':
                        message += f"RSI: {self.escape_markdown(f'{rsi:.1f}')} ✅ "
                    elif rsi_signal == 'bearish':
                        message += f"RSI: {self.escape_markdown(f'{rsi:.1f}')} ❌ "
                    else:
                        message += f"RSI: {self.escape_markdown(f'{rsi:.1f}')} ↔️ "

                # Add MA trend if available
                if 'ma_trend' in indicators:
                    ma_trend = indicators['ma_trend']
                    if ma_trend == 'bullish':
                        message += "MA: Above ✅ "
                    elif ma_trend == 'bearish':
                        message += "MA: Below ❌ "

                message += "\n"

        message += "\n"

        # SECTION 4: PATTERN ANALYSIS
        message += "*PATTERN ANALYSIS*\n"

        # Detect patterns across timeframes
        patterns_by_timeframe = {}
        combined_patterns = {}

        for timeframe, data in all_data.items():
            if data.empty:
                continue

            tf_label = timeframe_labels.get(timeframe, timeframe)
            patterns = self.pattern_recognizer.detect_patterns(data, lookback_periods=5)

            if patterns:
                patterns_by_timeframe[tf_label] = []

                for pattern_name, occurrences in patterns.items():
                    if occurrences:
                        # Get the most recent occurrence
                        latest_occurrence = max(occurrences, key=lambda x: x['timestamp'])
                        signal_value = latest_occurrence['signal']
                        signal_type = "bullish" if signal_value > 0 else "bearish" if signal_value < 0 else "neutral"

                        # Add to timeframe patterns
                        patterns_by_timeframe[tf_label].append({
                            'pattern': pattern_name,
                            'signal': signal_type
                        })

                        # Add to combined patterns for overall bias
                        if pattern_name not in combined_patterns:
                            combined_patterns[pattern_name] = {'bullish': 0, 'bearish': 0, 'neutral': 0}

                        combined_patterns[pattern_name][signal_type] += 1

        # Calculate pattern bias
        pattern_bullish = sum(p['bullish'] for p in combined_patterns.values())
        pattern_bearish = sum(p['bearish'] for p in combined_patterns.values())

        # Determine pattern trend
        pattern_trend = "NEUTRAL"
        if pattern_bullish > pattern_bearish * 1.5:
            pattern_trend = "BULLISH"
        elif pattern_bearish > pattern_bullish * 1.5:
            pattern_trend = "BEARISH"
        elif pattern_bullish > pattern_bearish:
            pattern_trend = "MODERATELY BULLISH"
        elif pattern_bearish > pattern_bullish:
            pattern_trend = "MODERATELY BEARISH"

        # Display pattern analysis summary
        message += f"Pattern Bias: *{self.escape_markdown(pattern_trend)}*\n"

        # Show patterns by timeframe
        if patterns_by_timeframe:
            message += "Detected Patterns:\n"
            for tf_label, patterns in patterns_by_timeframe.items():
                if patterns:
                    message += f"• {self.escape_markdown(tf_label)}: "

                    # Count bullish vs bearish patterns
                    tf_bullish = sum(1 for p in patterns if p['signal'] == 'bullish')
                    tf_bearish = sum(1 for p in patterns if p['signal'] == 'bearish')

                    # Show counts and top patterns
                    if tf_bullish > tf_bearish:
                        message += f"{tf_bullish} bullish, {tf_bearish} bearish \\- "
                    elif tf_bearish > tf_bullish:
                        message += f"{tf_bullish} bullish, {tf_bearish} bearish \\- "

                    # Show up to 2 patterns
                    pattern_texts = []
                    for pattern in patterns[:2]:
                        pattern_text = pattern['pattern']
                        signal = pattern['signal']

                        # Add emoji based on signal
                        if signal == 'bullish':
                            pattern_text += " 📈"
                        elif signal == 'bearish':
                            pattern_text += " 📉"

                        pattern_texts.append(pattern_text)

                    message += self.escape_markdown(", ".join(pattern_texts))

                    if len(patterns) > 2:
                        message += self.escape_markdown(f" (+{len(patterns)-2} more)")

                    message += "\n"
        else:
            message += "No significant patterns detected\n"

        message += "\n"

        # SECTION 5: SENTIMENT ANALYSIS
        message += "*SENTIMENT ANALYSIS*\n"

        # Check if sentiment tracker is available
        sentiment_data = None
        if hasattr(self, 'sentiment_tracker'):
            try:
                sentiment_data = self.sentiment_tracker.get_ticker_sentiment(ticker)
            except Exception as e:
                logger.error(f"Error getting sentiment data: {e}")

        if sentiment_data:
            value = sentiment_data['value']
            status = sentiment_data['status']
            article_count = sentiment_data['article_count']

            # Display sentiment summary
            message += f"Sentiment: *{self.escape_markdown(status)}*\n"
            message += f"Rating: {self.escape_markdown(f'{value:.1f}/10')}\n"
            message += f"Based on {self.escape_markdown(str(article_count))} recent articles\n\n"

            # Add sentiment meter
            meter_length = 10
            filled_segments = min(int(value * meter_length / 10), meter_length)

            # Define meter character based on sentiment
            if value >= 6.5:  # Positive
                meter_char = "🟢"
            elif value >= 5.5:  # Slightly positive
                meter_char = "🟡"
            elif value >= 4.5:  # Neutral
                meter_char = "⚪"
            elif value >= 3.5:  # Slightly negative
                meter_char = "🟠"
            else:  # Negative
                meter_char = "🔴"

            meter = meter_char * filled_segments + "⚫" * (meter_length - filled_segments)
            message += f"{meter}\n\n"

            # Try to get recent news headlines
            recent_articles = None

            if recent_articles:
                message += "Recent Headlines:\n"
                for article in recent_articles:
                    headline = self._escape_markdown(article.title[:50] + "..." if len(article.title) > 50 else article.title)
                    article_sentiment = self._sentiment_to_emoji(article.sentiment_rating)
                    message += f"{article_sentiment} {headline}\n"
        else:
            message += "Sentiment data not available\n"

        message += "\n"

        # SECTION 6: TRADING SIGNAL
        message += "*TRADING SIGNAL*\n"

        # Combine technical, pattern, and sentiment analysis
        signal = "NEUTRAL"
        confidence = "LOW"

        # Get individual signals
        tech_signal = tech_trend
        pattern_signal = pattern_trend

        # Define sentiment signal
        sentiment_signal = "NEUTRAL"
        if sentiment_data:
            value = sentiment_data['value']
            if value >= 7.0:
                sentiment_signal = "BULLISH"
            elif value >= 6.0:
                sentiment_signal = "MODERATELY BULLISH"
            elif value <= 3.0:
                sentiment_signal = "BEARISH"
            elif value <= 4.0:
                sentiment_signal = "MODERATELY BEARISH"

        # Count aligned signals
        bullish_signals = 0
        bearish_signals = 0

        # Technical signal
        if "BULLISH" in tech_signal:
            bullish_signals += 1
        elif "BEARISH" in tech_signal:
            bearish_signals += 1

        # Pattern signal
        if "BULLISH" in pattern_signal:
            bullish_signals += 1
        elif "BEARISH" in pattern_signal:
            bearish_signals += 1

        # Sentiment signal
        if "BULLISH" in sentiment_signal:
            bullish_signals += 1
        elif "BEARISH" in sentiment_signal:
            bearish_signals += 1

        # Determine aligned signal
        if bullish_signals >= 2 and bearish_signals == 0:
            signal = "BUY"
            confidence = "HIGH"
        elif bullish_signals >= 2:
            signal = "BUY"
            confidence = "MEDIUM"
        elif bearish_signals >= 2 and bullish_signals == 0:
            signal = "SELL"
            confidence = "HIGH"
        elif bearish_signals >= 2:
            signal = "SELL"
            confidence = "MEDIUM"
        elif bullish_signals > bearish_signals:
            signal = "WATCH (Bullish Bias)"
            confidence = "LOW"
        elif bearish_signals > bullish_signals:
            signal = "WATCH (Bearish Bias)"
            confidence = "LOW"
        else:
            signal = "NEUTRAL"
            confidence = "LOW"

        # Display signal
        message += f"Signal: *{self.escape_markdown(signal)}*\n"
        message += f"Confidence: *{self.escape_markdown(confidence)}*\n\n"

        # Show reasoning for the signal
        message += "Signal Basis:\n"
        message += f"• Technical Analysis: {self.escape_markdown(tech_signal)}\n"
        message += f"• Pattern Analysis: {self.escape_markdown(pattern_signal)}\n"
        message += f"• Sentiment Analysis: {self.escape_markdown(sentiment_signal)}\n\n"

        # SECTION 7: TRADE SETUP (if signal is BUY or SELL)
        if signal in ["BUY", "SELL"]:
            message += "*TRADE SETUP*\n"

            # Calculate suggested entry, target, and stop levels
            if current_price is not None:
                # Get ATR if available for stop calculation
                atr = None
                for tf_name in ['medium_term', 'short_term']:
                    if tf_name in all_data and not all_data[tf_name].empty:
                        indicators = self.stock_collector.calculate_technical_indicators(all_data[tf_name])
                        if 'atr' in indicators:
                            atr = indicators['atr']
                            break

                # If no ATR, use a percentage-based approach
                if signal == "BUY":
                    entry = current_price

                    # Calculate stop and target based on ATR or percentage
                    if atr is not None:
                        stop = entry - (atr * 2)
                        target1 = entry + (atr * 2)
                        target2 = entry + (atr * 4)
                    else:
                        stop = entry * 0.97  # 3% stop
                        target1 = entry * 1.03  # 3% first target
                        target2 = entry * 1.06  # 6% second target

                    # Format message
                    message += f"Entry: ${self.escape_markdown(f'{entry:.2f}')}\n"
                    message += f"Stop Loss: ${self.escape_markdown(f'{stop:.2f}')} \\({self.escape_markdown(f'{((stop/entry)-1)*100:.1f}%')}\\)\n"
                    message += f"Target 1: ${self.escape_markdown(f'{target1:.2f}')} \\({self.escape_markdown(f'{((target1/entry)-1)*100:.1f}%')}\\)\n"
                    message += f"Target 2: ${self.escape_markdown(f'{target2:.2f}')} \\({self.escape_markdown(f'{((target2/entry)-1)*100:.1f}%')}\\)\n"

                    # Calculate risk-reward ratio
                    rr1 = abs((target1 - entry) / (entry - stop))
                    message += f"Risk/Reward Ratio: 1:{self.escape_markdown(f'{rr1:.1f}')}\n"

                elif signal == "SELL":
                    entry = current_price

                    # Calculate stop and target based on ATR or percentage
                    if atr is not None:
                        stop = entry + (atr * 2)
                        target1 = entry - (atr * 2)
                        target2 = entry - (atr * 4)
                    else:
                        stop = entry * 1.03  # 3% stop for short
                        target1 = entry * 0.97  # 3% first target
                        target2 = entry * 0.94  # 6% second target

                    # Format message
                    message += f"Entry: ${self.escape_markdown(f'{entry:.2f}')}\n"
                    message += f"Stop Loss: ${self.escape_markdown(f'{stop:.2f}')} \\({self.escape_markdown(f'{((stop/entry)-1)*100:.1f}%')}\\)\n"
                    message += f"Target 1: ${self.escape_markdown(f'{target1:.2f}')} \\({self.escape_markdown(f'{((target1/entry)-1)*100:.1f}%')}\\)\n"
                    message += f"Target 2: ${self.escape_markdown(f'{target2:.2f}')} \\({self.escape_markdown(f'{((target2/entry)-1)*100:.1f}%')}\\)\n"

                    # Calculate risk-reward ratio
                    rr1 = abs((entry - target1) / (stop - entry))
                    message += f"Risk/Reward Ratio: 1:{self.escape_markdown(f'{rr1:.1f}')}\n"

                # Position sizing suggestion
                message += "\nSuggested Position Sizing:\n"

                if confidence == "HIGH":
                    message += "• Full position size \\(confidence is high\\)\n"
                elif confidence == "MEDIUM":
                    message += "• 1/2 to 2/3 position size \\(moderate confidence\\)\n"
                else:
                    message += "• 1/4 to 1/3 position size \\(speculative trade\\)\n"

                # Add scaling suggestion
                message += "• Consider scaling in/out at key levels\n"

            message += "\n"

        # SECTION 8: KEY EVENTS
        message += "*KEY EVENTS & CATALYSTS*\n"

        # Get earnings dates
        earnings_dates = self.stock_collector.get_earnings_dates(ticker)
        if earnings_dates and len(earnings_dates) > 0:
            next_earnings = earnings_dates[0]
            message += f"Next Earnings: {self.escape_markdown(next_earnings['date'])}\n"

            if 'estimate' in next_earnings and next_earnings['estimate'] != 'N/A':
                message += f"EPS Estimate: {self.escape_markdown(str(next_earnings['estimate']))}\n"
        else:
            message += "No upcoming earnings dates found\n"

        message += "\n"

        # SECTION 9: MARKET CONTEXT
        message += "*MARKET CONTEXT*\n"

        # Get overall market sentiment if available
        if hasattr(self, 'sentiment_tracker'):
            try:
                market_sentiment = self.sentiment_tracker.get_current_sentiment()
                if market_sentiment:
                    message += f"Market Sentiment: {self.escape_markdown(market_sentiment['status'])} "
                    value = market_sentiment['value']
                    message += f"\\({self.escape_markdown(f'{value:.1f}/10')}\\)\n"

                    # Add alignment with market sentiment
                    if "BULLISH" in signal and "BULLISH" in market_sentiment['status'].upper():
                        message += "✅ Trade aligned with overall market sentiment\n"
                    elif "BEARISH" in signal and "BEARISH" in market_sentiment['status'].upper():
                        message += "✅ Trade aligned with overall market sentiment\n"
                    elif "BULLISH" in signal and "BEARISH" in market_sentiment['status'].upper():
                        message += "⚠️ Trade against overall market sentiment - use caution\n"
                    elif "BEARISH" in signal and "BULLISH" in market_sentiment['status'].upper():
                        message += "⚠️ Trade against overall market sentiment - use caution\n"
            except Exception as e:
                logger.error(f"Error getting market sentiment: {e}")

        # Get sector performance (this would need to be implemented)
        # For now, we'll just add a placeholder
        message += "Sector Performance: Not available in this version\n\n"

        # SECTION 10: CONCLUSION AND DISCLAIMER
        message += "*CONCLUSION*\n"

        # Generate conclusion based on all the analysis
        if signal == "BUY":
            if confidence == "HIGH":
                message += "Strong buy signal with high confidence\\. Multiple indicators aligned across technical, pattern, and sentiment analysis\n"
            else:
                message += "Bullish bias with moderate confidence\\. Monitor price action and consider scaling in\n"
        elif signal == "SELL":
            if confidence == "HIGH":
                message += "Strong sell signal with high confidence\\. Multiple indicators aligned across technical, pattern, and sentiment analysis\n"
            else:
                message += "Bearish bias with moderate confidence\\. Monitor price action and consider scaling in\n"
        elif "Bullish" in signal:
            message += "Early bullish bias developing\\. More confirmation needed before taking a position\n"
        elif "Bearish" in signal:
            message += "Early bearish bias developing. More confirmation needed before taking a position\n"
        else:
            message += "No clear directional bias at this time\\. Wait for better setup\n"

        # Add disclaimer
        message += "\n⚠️ *DISCLAIMER:* This analysis is generated algorithmically and should not be considered financial advice\\. Always conduct your own research and consider your risk tolerance before trading\\."

        # Send the message in smaller chunks if too long
        if len(message) > 4000:
            # Split the message into parts
            parts = []
            current_part = ""

            for line in message.split('\n'):
                if len(current_part) + len(line) + 1 > 4000:
                    parts.append(current_part)
                    current_part = line + '\n'
                else:
                    current_part += line + '\n'

            if current_part:
                parts.append(current_part)

            # Send parts
            for i, part in enumerate(parts):
                part_header = f"ANALYSIS PART {i+1}/{len(parts)}\n\n" if i > 0 else ""
                await update.message.reply_text(part_header + part, parse_mode=ParseMode.MARKDOWN_V2)
        else:
            await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN_V2)

    def _format_sentiment_meter(self, sentiment_data: Dict) -> str:
        """Format sentiment meter for display."""
        value = sentiment_data['value']
        status = sentiment_data['status']
        description = sentiment_data['description']
        article_count = sentiment_data['article_count']

        # Create meter (10 segments)
        meter_length = 10
        filled_segments = min(int(value * meter_length / 10), meter_length)

        # Define meter character based on sentiment
        if value >= 6.5:  # Positive
            meter_char = "🟢"
        elif value >= 5.5:  # Slightly positive
            meter_char = "🟡"
        elif value >= 4.5:  # Neutral
            meter_char = "⚪"
        elif value >= 3.5:  # Slightly negative
            meter_char = "🟠"
        else:  # Negative
            meter_char = "🔴"

        meter = meter_char * filled_segments + "⚫" * (meter_length - filled_segments)

        # Create time-based trend indicator if available
        trend_chart = ""
        if all(sentiment_data.get(x) is not None for x in ['morning_sentiment', 'midday_sentiment', 'afternoon_sentiment']):
            morning = sentiment_data['morning_sentiment']
            midday = sentiment_data['midday_sentiment']
            afternoon = sentiment_data['afternoon_sentiment']

            trend_chart = "\n\n*Intraday Trend:*\n"
            trend_chart += f"Morning: {self._sentiment_to_emoji(morning)} ({morning:.1f}/10)\n"
            trend_chart += f"Midday: {self._sentiment_to_emoji(midday)} ({midday:.1f}/10)\n"
            trend_chart += f"Afternoon: {self._sentiment_to_emoji(afternoon)} ({afternoon:.1f}/10)"

        # Add affected tickers if available
        tickers_info = ""
        if sentiment_data.get('affected_tickers'):
            tickers = sentiment_data['affected_tickers'].split(',')
            if tickers:
                tickers_info = "\n\n*Top Mentioned Tickers:*\n"
                tickers_info += ", ".join(tickers)

        message = (
            f"📊 *Market Sentiment Meter* 📊\n\n"
            f"Status: *{status}*\n"
            f"Value: *{value:.1f}/10*\n"
            f"Articles: {article_count}\n\n"
            f"{meter}\n\n"
            f"{description}{trend_chart}{tickers_info}\n\n"
            f"_Updated: {datetime.now().strftime('%H:%M:%S')}_"
        )

        return message

    def _format_ticker_sentiment_meter(self, ticker: str, sentiment_data: Dict) -> str:
        """Format ticker-specific sentiment meter."""
        value = sentiment_data['value']
        status = sentiment_data['status']
        description = sentiment_data['description']
        article_count = sentiment_data['article_count']

        # Create meter (10 segments)
        meter_length = 10
        filled_segments = min(int(value * meter_length / 10), meter_length)

        # Define meter character based on sentiment
        if value >= 6.5:  # Positive
            meter_char = "🟢"
        elif value >= 5.5:  # Slightly positive
            meter_char = "🟡"
        elif value >= 4.5:  # Neutral
            meter_char = "⚪"
        elif value >= 3.5:  # Slightly negative
            meter_char = "🟠"
        else:  # Negative
            meter_char = "🔴"

        meter = meter_char * filled_segments + "⚫" * (meter_length - filled_segments)

        # Add trading signals advice
        trading_advice = self._generate_sentiment_trading_advice(ticker, value)

        # Escape special characters for Markdown V2
        ticker_escaped = self._escape_markdown(ticker)
        status_escaped = self._escape_markdown(status)
        value_escaped = self._escape_markdown(f"{value:.1f}")
        article_count_escaped = self._escape_markdown(str(article_count))
        description_escaped = self._escape_markdown(description)
        trading_advice_escaped = self._escape_markdown(trading_advice)

        message = (
            f"📈 *{ticker_escaped} Sentiment Meter* 📈\n\n"
            f"Status: *{status_escaped}*\n"
            f"Value: *{value_escaped}/10*\n"
            f"Articles: {article_count_escaped}\n\n"
            f"{meter}\n\n"
            f"{description_escaped}\n\n"
            f"{trading_advice_escaped}\n\n"
            f"_Updated: {datetime.now().strftime('%H:%M:%S')}_"
        )

        return message

    def _sentiment_to_emoji(self, value: float) -> str:
        """Convert sentiment value to appropriate emoji."""
        if value >= 8.0:
            return "🟢🟢"  # Very Positive
        elif value >= 7.0:
            return "🟢"  # Positive
        elif value >= 6.0:
            return "🟡"  # Slightly Positive
        elif value >= 5.0:
            return "⚪"  # Neutral
        elif value >= 4.0:
            return "🟠"  # Slightly Negative
        elif value >= 3.0:
            return "🔴"  # Negative
        else:
            return "🔴🔴"  # Very Negative

    def _generate_sentiment_trading_advice(self, ticker: str, sentiment_value: float) -> str:
        """Generate trading advice based on sentiment."""
        # Get current technical signals for ticker
        technical_summary = None
        try:
            technical_summary = self.stock_collector.get_summary(ticker)
        except Exception as e:
            logger.error(f"Error getting technical summary: {e}")

        # No technical data available
        if not technical_summary or 'timeframes' not in technical_summary:
            # Pure sentiment-based advice
            if sentiment_value >= 8.0:
                return "💡 Trading Signal: Strong bullish sentiment indicates potential buying opportunity."
            elif sentiment_value >= 6.5:
                return "💡 Trading Signal: Positive sentiment suggests considering long positions with proper risk management."
            elif sentiment_value <= 3.0:
                return "💡 Trading Signal: Strong bearish sentiment indicates caution or potential shorting opportunity."
            elif sentiment_value <= 4.5:
                return "💡 Trading Signal: Negative sentiment suggests reducing exposure or considering short positions."
            else:
                return "💡 Trading Signal: Neutral sentiment suggests waiting for clearer directional signals."

        # We have both sentiment and technical data - combine them
        technical_bullish = 0
        technical_bearish = 0

        # Count bullish/bearish signals from technical indicators
        for timeframe, tf_data in technical_summary['timeframes'].items():
            if 'indicators' in tf_data:
                indicators = tf_data['indicators']

                # RSI
                if 'rsi' in indicators:
                    rsi = indicators['rsi']
                    if isinstance(rsi, (int, float)):
                        if rsi < 30:
                            technical_bullish += 1  # Oversold
                        elif rsi > 70:
                            technical_bearish += 1  # Overbought

                # MA Trend
                if 'ma_trend' in indicators:
                    ma_trend = indicators['ma_trend']
                    if ma_trend == 'bullish':
                        technical_bullish += 1
                    elif ma_trend == 'bearish':
                        technical_bearish += 1

        # Combine technical and sentiment signals
        is_technical_bullish = technical_bullish > technical_bearish
        is_technical_bearish = technical_bearish > technical_bullish
        is_sentiment_bullish = sentiment_value >= 6.5
        is_sentiment_bearish = sentiment_value <= 4.5

        # Strong signal when technical and sentiment align
        if is_technical_bullish and is_sentiment_bullish:
            return "💡 Trading Signal: Strong buy - bullish technical indicators with positive sentiment."
        elif is_technical_bearish and is_sentiment_bearish:
            return "💡 Trading Signal: Strong sell - bearish technical indicators with negative sentiment."

        # Mixed signals
        if is_technical_bullish and is_sentiment_bearish:
            return "💡 Trading Signal: Mixed signals - bullish technicals but bearish sentiment. Consider waiting for alignment."
        elif is_technical_bearish and is_sentiment_bullish:
            return "💡 Trading Signal: Mixed signals - bearish technicals but bullish sentiment. Monitor for trend change."

        # Weak signals
        if is_sentiment_bullish:
            return "💡 Trading Signal: Moderately bullish - positive sentiment with neutral technical indicators."
        elif is_sentiment_bearish:
            return "💡 Trading Signal: Moderately bearish - negative sentiment with neutral technical indicators."

        return "💡 Trading Signal: Neutral - no clear directional bias from either sentiment or technical indicators."