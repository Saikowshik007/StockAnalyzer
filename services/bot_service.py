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
        """Handle /watchlist command."""
        watchlist = self.db_manager.get_active_watchlist()

        if not watchlist:
            await update.message.reply_text("Your watchlist is empty.")
            return

        message = "📋 *Current Watchlist*\n\n"
        for ticker in watchlist:
            # Get multi-timeframe prices
            multi_prices = self.stock_collector.get_latest_prices()
            ticker_data = multi_prices.get(ticker, {})

            if 'short_term' in ticker_data:
                latest_price = ticker_data['short_term'].get('price', 'N/A')
            else:
                latest_price = 'N/A'

            message += f"• {ticker}: ${latest_price}\n"

        # Create inline keyboard for easy management
        keyboard = [
            [InlineKeyboardButton(f"Remove {ticker}", callback_data=f"remove_{ticker}")]
            for ticker in watchlist
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)

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
        """Handle /price <ticker> command with multi-timeframe data."""
        if not context.args:
            await update.message.reply_text("Please provide a ticker symbol: /price AAPL")
            return

        ticker = context.args[0].upper()

        # Get multi-timeframe data
        all_prices = self.stock_collector.get_latest_prices()

        if ticker not in all_prices or not all_prices[ticker]:
            await update.message.reply_text(f"No price data available for {ticker}. You might need to add it to your watchlist first.")
            return

        ticker_data = all_prices[ticker]
        message = f"💰 *{ticker} Multi-Timeframe Analysis*\n\n"

        logger.info(ticker_data)

        # Format each timeframe
        timeframe_names = {
            'long_term': '📅 Hourly',
            'medium_term': '🕐 15 minute',
            'short_term': '⏱️ 5 minute',
            "very_short_term": '⏱️ 2 minute'
        }

        for timeframe, price_info in ticker_data.items():
            display_name = timeframe_names.get(timeframe, timeframe)
            message += f"{display_name}:\n"
            message += f"  Price: ${price_info['price']:.2f}\n"
            message += f"  Open: {price_info['open']:,}\n"
            message += f"  High: {price_info['high']:,}\n"
            message += f"  Low: {price_info['low']:,}\n"
            message += f"  Volume: {price_info['volume']:,}\n"
            message += f"  Time: {price_info['datetime'].strftime('%Y-%m-%d %H:%M')}\n\n"

        # Get technical indicators for each timeframe
        try:
            summary = self.stock_collector.get_summary(ticker)
            if 'timeframes' in summary:
                message += "*Technical Indicators:*\n\n"
                for timeframe, tf_data in summary['timeframes'].items():
                    if 'indicators' in tf_data:
                        display_name = timeframe_names.get(timeframe, timeframe)
                        indicators = tf_data['indicators']
                        message += f"{display_name}:\n"

                        # Handle RSI - safely format only if it's a number
                        rsi = indicators.get('rsi')
                        if rsi is not None and isinstance(rsi, (int, float)):
                            message += f"  RSI: {rsi:.1f}\n"
                        else:
                            message += f"  RSI: {rsi}\n"

                        # Handle MA Trend
                        ma_trend = indicators.get('ma_trend', 'N/A')
                        message += f"  MA Trend: {ma_trend.upper() if isinstance(ma_trend, str) else ma_trend}\n"

                        # Handle Volume Ratio - safely format only if it's a number
                        vol_ratio = indicators.get('volume_ratio')
                        if vol_ratio is not None and isinstance(vol_ratio, (int, float)):
                            message += f"  Volume Ratio: {vol_ratio:.2f}x\n\n"
                        else:
                            message += f"  Volume Ratio: {vol_ratio}\n\n"
        except Exception as e:
            logger.error(f"Error getting technical indicators: {e}")

        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

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
        """Handle /pattern <ticker> command with multi-timeframe analysis."""
        if not context.args:
            await update.message.reply_text("Please provide a ticker symbol: /pattern AAPL")
            return
        ticker = context.args[0].upper()

        # Get multi-timeframe data
        all_data = self.stock_collector.get_multi_timeframe_data(ticker)

        if not all_data:
            await update.message.reply_text(f"No data available for {ticker} Make sure it's in your watchlist")
            return

        message = f"📊 *Multi Timeframe Pattern Analysis for {ticker}*\n\n"
        combined_signals = {}
        pattern_counts = {'long_term': 0, 'medium_term': 0, 'short_term': 0}

        timeframe_names = {
            'long_term': '📅 Hourly',
            'medium_term': '🕐 15 minute',
            'short_term': '⏱️ 5 minute',
            "very_short_term": '⏱️ 2 minute'
        }

        # Analyze patterns for each timeframe
        for timeframe, data in all_data.items():
            if data.empty or timeframe == 'very_short_term':
                continue

            patterns = self.pattern_recognizer.detect_patterns(data,lookback_periods=10)

            if patterns:
                current_price = data['close'].iloc[-1]
                indicators = self.stock_collector.calculate_technical_indicators(data)
                pattern_counts[timeframe] = len(patterns)

                # Process patterns for combined analysis
                for pattern_name, occurrences in patterns.items():
                    if occurrences:
                        latest_occurrence = max(occurrences, key=lambda x: x['timestamp'])
                        signal = self.pattern_recognizer.get_trading_signal(
                            pattern_name,
                            latest_occurrence['signal'],
                            current_price,
                            atr=indicators.get('atr'),
                            volume_ratio=indicators.get('volume_ratio', 1.0),
                            additional_indicators=indicators
                        )

                        # Store for combined analysis
                        if timeframe not in combined_signals:
                            combined_signals[timeframe] = []
                        combined_signals[timeframe].append({
                            'pattern': pattern_name,
                            'signal': signal
                        })

        # Create a summary section for patterns found
        if any(pattern_counts.values()):
            message += "📈 *Patterns Detected:*\n"
            for timeframe, count in pattern_counts.items():
                if count > 0:
                    display_name = timeframe_names.get(timeframe, timeframe)
                    message += f"{display_name}: {count} patterns\n"
            message += "\n"

        # Combine signals from all timeframes
        if combined_signals:
            combined_action, combined_confidence = self._combine_timeframe_signals(combined_signals)

            message += "📊 *Combined Signal:*\n"
            message += f"Action: {combined_action}\n"
            message += f"Confidence: {self.escape_markdown(combined_confidence.upper())}\n\n"

            # Less strict conditions for showing strong signal
            if combined_action in ['BUY', 'SELL']:
                message += "✅ *Trading Signal Detected*\n"

                # Show the patterns found across timeframes
                for timeframe, signals in combined_signals.items():
                    if signals:
                        display_name = timeframe_names.get(timeframe, timeframe)
                        for signal_data in signals[:1]:  # Show just the first pattern for each timeframe
                            pattern = signal_data.get('pattern', 'Unknown pattern')
                            action = signal_data.get('signal', {}).get('action', 'UNKNOWN')
                            message += f"{display_name}: {pattern} ({action})\n"
            else:
                message += "⚠️ No clear directional signal at this time\n"
        else:
            message += "No significant patterns detected in any timeframe\n"

        # Add current price
        try:
            latest_prices = self.stock_collector.get_latest_prices()
            if ticker in latest_prices and 'very_short_term' in latest_prices[ticker]:
                current_price = latest_prices[ticker]['very_short_term']['price']
                # Escape special characters for MarkdownV2
                message += f"\n💰 Current Price: ${self.escape_markdown(f'{current_price:.2f}')}"
        except Exception as e:
            logger.error(f"Error getting current price: {e}")

        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN_V2)

    # Add this helper function to escape Markdown characters
    def escape_markdown(self,text):
        """
        Helper function to escape MarkdownV2 special characters.
        Must escape: _ * [ ] ( ) ~ ` > # + - = | { } . !
        """
        escape_chars = r'_*[]()~`>#+-=|{}.!'
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
        """Handle /analyze <ticker> command - combining technical and sentiment analysis."""
        if not context.args:
            await update.message.reply_text("Please provide a ticker symbol: /analyze AAPL")
            return

        ticker = context.args[0].upper()

        message = f"🔍 *Comprehensive Analysis for {ticker}* 🔍\n\n"

        # 1. Get sentiment data
        if not hasattr(self, 'sentiment_tracker'):
            try:
                from services.sentiment_tracker import SentimentTracker
                self.sentiment_tracker = SentimentTracker(self.db_manager)
            except Exception as e:
                logger.error(f"Error initializing sentiment tracker: {e}")

        if hasattr(self, 'sentiment_tracker'):
            try:
                ticker_sentiment = self.sentiment_tracker.get_ticker_sentiment(ticker)
                market_sentiment = self.sentiment_tracker.get_current_sentiment()

                message += "📊 *Sentiment Analysis*\n"
                message += f"• {ticker} Sentiment: {ticker_sentiment['status']} ({ticker_sentiment['value']:.1f}/10)\n"
                message += f"• Market Sentiment: {market_sentiment['status']} ({market_sentiment['value']:.1f}/10)\n"

                # Add articles count
                if ticker_sentiment['article_count'] > 0:
                    message += f"• Based on {ticker_sentiment['article_count']} articles about {ticker} today\n"

                message += "\n"
            except Exception as e:
                logger.error(f"Error getting sentiment data: {e}")
                message += "📊 *Sentiment Analysis*\n• Error retrieving sentiment data\n\n"

        # 2. Get technical analysis
        try:
            # Get multi-timeframe data
            all_data = self.stock_collector.get_multi_timeframe_data(ticker)

            message += "📈 *Technical Analysis*\n"

            # Get latest price
            latest_prices = self.stock_collector.get_latest_prices()
            current_price = None
            if ticker in latest_prices and 'short_term' in latest_prices[ticker]:
                current_price = latest_prices[ticker]['short_term']['price']
                message += f"• Current Price: ${current_price:.2f}\n"

            # Calculate indicators for each timeframe
            overall_bullish = 0
            overall_bearish = 0
            timeframe_names = {
                'long_term': '📅 Hourly',
                'medium_term': '🕐 15 minute',
                'short_term': '⏱️ 5 minute',
                'very_short_term': '⏱️ 2 minute'
            }

            for timeframe, data in all_data.items():
                if data.empty or timeframe == 'very_short_term':
                    continue

                # Calculate indicators
                indicators = self.stock_collector.calculate_technical_indicators(data)

                # Detect patterns
                patterns = self.pattern_recognizer.detect_patterns(data, lookback_periods=5)

                # Determine signal for this timeframe
                bullish_signals = 0
                bearish_signals = 0

                # Process indicators
                if 'rsi' in indicators:
                    rsi = indicators['rsi']
                    if isinstance(rsi, (int, float)):
                        if rsi < 30:
                            bullish_signals += 1  # Oversold
                        elif rsi > 70:
                            bearish_signals += 1  # Overbought

                if 'ma_trend' in indicators:
                    ma_trend = indicators['ma_trend']
                    if ma_trend == 'bullish':
                        bullish_signals += 1
                    elif ma_trend == 'bearish':
                        bearish_signals += 1

                # Process patterns
                pattern_count = sum(len(occurrences) for occurrences in patterns.values())

                # Add to overall count
                overall_bullish += bullish_signals
                overall_bearish += bearish_signals

                # Only add timeframe details if significant signals exist
                if bullish_signals > 0 or bearish_signals > 0 or pattern_count > 0:
                    tf_display = timeframe_names.get(timeframe, timeframe)

                    # Get the direction for this timeframe
                    if bullish_signals > bearish_signals:
                        tf_direction = "🟢 Bullish"
                    elif bearish_signals > bullish_signals:
                        tf_direction = "🔴 Bearish"
                    else:
                        tf_direction = "⚪ Neutral"

                    message += f"• {tf_display}: {tf_direction}"

                    # Add pattern info if available
                    if pattern_count > 0:
                        message += f" ({pattern_count} patterns detected)"

                    message += "\n"

            message += "\n"
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            message += "📈 *Technical Analysis*\n• Error retrieving technical data\n\n"

        # 3. Generate combined trading signal
        try:
            message += "🎯 *Combined Trading Signal*\n"

            # Default neutral if no data
            signal = "NEUTRAL"
            confidence = "low"

            # If we have both sentiment and technical data
            have_sentiment = hasattr(self, 'sentiment_tracker')
            have_technical = overall_bullish > 0 or overall_bearish > 0

            if have_sentiment and have_technical:
                # Get sentiment values
                ticker_sentiment_value = ticker_sentiment['value']
                market_sentiment_value = market_sentiment['value']

                # Calculate weighted sentiment (70% ticker, 30% market)
                weighted_sentiment = 0.7 * ticker_sentiment_value + 0.3 * market_sentiment_value

                # Determine if sentiment is bullish/bearish/neutral
                sentiment_signal = "NEUTRAL"
                if weighted_sentiment >= 7.0:
                    sentiment_signal = "BULLISH"
                elif weighted_sentiment <= 4.0:
                    sentiment_signal = "BEARISH"

                # Determine if technical is bullish/bearish/neutral
                technical_signal = "NEUTRAL"
                if overall_bullish > overall_bearish * 1.5:
                    technical_signal = "BULLISH"
                elif overall_bearish > overall_bullish * 1.5:
                    technical_signal = "BEARISH"

                # Combine signals
                if sentiment_signal == technical_signal:
                    # Strong signal when they align
                    signal = sentiment_signal
                    confidence = "high"
                elif sentiment_signal != "NEUTRAL" and technical_signal != "NEUTRAL":
                    # Conflicting signals
                    signal = "MIXED"
                    confidence = "low"
                elif sentiment_signal != "NEUTRAL":
                    # Sentiment-driven signal
                    signal = sentiment_signal
                    confidence = "medium"
                elif technical_signal != "NEUTRAL":
                    # Technically-driven signal
                    signal = technical_signal
                    confidence = "medium"
            elif have_sentiment:
                ticker_sentiment_value = ticker_sentiment['value']
                # Only sentiment data available
                if ticker_sentiment_value >= 7.0:
                    signal = "BULLISH"
                    confidence = "medium"
                elif ticker_sentiment_value <= 4.0:
                    signal = "BEARISH"
                    confidence = "medium"
            elif have_technical:
                # Only technical data available
                if overall_bullish > overall_bearish * 1.5:
                    signal = "BULLISH"
                    confidence = "medium"
                elif overall_bearish > overall_bullish * 1.5:
                    signal = "BEARISH"
                    confidence = "medium"

            # Format confidence
            confidence_display = "⭐⭐⭐" if confidence == "high" else "⭐⭐" if confidence == "medium" else "⭐"

            # Display signal
            message += f"• Signal: *{signal}*\n"
            message += f"• Confidence: {confidence_display}\n\n"

            # Add action recommendations
            if signal == "BULLISH" and confidence in ["medium", "high"]:
                message += "💡 *Recommended Action*: Consider buying or increasing long positions.\n"

                # Add stop loss and target recommendations if price available
                if current_price:
                    stop_loss = current_price * 0.97  # 3% stop loss
                    target = current_price * 1.06    # 6% target
                    message += f"• Entry: ${current_price:.2f}\n"
                    message += f"• Stop Loss: ${stop_loss:.2f}\n"
                    message += f"• Target: ${target:.2f}\n"
                    message += f"• Risk/Reward: 1:2\n"

            elif signal == "BEARISH" and confidence in ["medium", "high"]:
                message += "💡 *Recommended Action*: Consider reducing exposure or short positions.\n"

                # Add stop loss and target recommendations if price available
                if current_price:
                    stop_loss = current_price * 1.03  # 3% stop loss for short
                    target = current_price * 0.94    # 6% target for short
                    message += f"• Entry: ${current_price:.2f}\n"
                    message += f"• Stop Loss: ${stop_loss:.2f}\n"
                    message += f"• Target: ${target:.2f}\n"
                    message += f"• Risk/Reward: 1:2\n"

            elif signal == "MIXED":
                message += "💡 *Recommended Action*: Wait for clearer signals before taking a position.\n"
                message += "• Monitor for alignment between sentiment and technical indicators.\n"

            else:
                message += "💡 *Recommended Action*: No clear directional bias. Monitor for new signals.\n"

        except Exception as e:
            logger.error(f"Error generating combined signal: {e}")
            message += "🎯 *Combined Trading Signal*\n• Error generating signal\n"

        # Add disclaimer
        message += "\n⚠️ This is algorithmic analysis and not financial advice. Always do your own research."

        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

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