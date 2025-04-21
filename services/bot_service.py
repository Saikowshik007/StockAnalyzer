# services/bot_service.py
import asyncio
import logging
import re

import pandas as pd
from pyexpat.errors import messages
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler
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
            formatted_message = self._format_news_summary(news)
            await self.application.bot.send_message(
                chat_id=self.chat_id,
                text=formatted_message,
                parse_mode=ParseMode.MARKDOWN_V2
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

        return formatted_message

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
        """Escape special characters for Telegram's Markdown V2 format"""
        if not text:
            return ""

        escape_chars = '_*[]()~`>#+-=|{}.!'
        escaped_text = ""
        for char in text:
            if char in escape_chars:
                escaped_text += f"\\{char}"
            else:
                escaped_text += char

        return escaped_text

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
        self.db_manager.add_to_watchlist(ticker)
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

            patterns = self.pattern_recognizer.detect_patterns(data,lookback_periods=3)

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
            message += f"Confidence: {combined_confidence.upper()}\n\n"

            # Add key pattern details only for strong signals
            if combined_action in ['BUY', 'SELL'] and combined_confidence in ['high', 'very_high']:
                message += "✅ *Strong Signal Alert*\n"
                message += "Multiple timeframes are aligned for this trade opportunity\n\n"

                # Show only the most relevant pattern for each timeframe
                for timeframe, signals in combined_signals.items():
                    if signals:
                        # Get the strongest signal
                        strongest_signal = max(signals, key=lambda x: self.confidence_thresholds.get(x['signal']['confidence'], 0))

                        display_name = timeframe_names.get(timeframe, timeframe)
                        message += f"{display_name}: {strongest_signal['pattern']} ({strongest_signal['signal']['action']})\n"
            else:
                message += "⚠️ Weak or conflicting signals across timeframes\n"
        else:
            message += "No significant patterns detected in any timeframe\n"

        # Add current price
        try:
            latest_prices = self.stock_collector.get_latest_prices()
            if ticker in latest_prices and 'short_term' in latest_prices[ticker]:
                current_price = latest_prices[ticker]['short_term']['price']
                # Escape special characters for MarkdownV2
                message += f"\n💰 Current Price: ${escape_markdown(f'{current_price:.2f}')}"
        except Exception as e:
            logger.error(f"Error getting current price: {e}")

        print(message)
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN_V2)

    # Add this helper function to escape Markdown characters
    def escape_markdown(text):
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
        if combined_strength > 0.6:
            action = 'BUY'
            confidence = 'high'
        elif combined_strength < -0.6:
            action = 'SELL'
            confidence = 'high'
        else:
            # Find the action with highest score
            action = max(action_scores, key=action_scores.get)
            if action_scores[action] > 0.5:
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

        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

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

    async def run_async(self):
        """Run the bot in an existing event loop."""
        try:
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
            self.application.add_handler(CallbackQueryHandler(self.button_callback))

            # Initialize the application without running the loop
            await self.application.initialize()
            await self.application.start()

            # Start the updater
            await self.application.updater.start_polling()

            # Keep running indefinitely
            try:
                while True:
                    await asyncio.sleep(3600)  # Sleep for an hour
            except asyncio.CancelledError:
                logger.info("Bot task cancelled, shutting down...")
                raise
            finally:
                # Clean shutdown
                await self.application.updater.stop()
                await self.application.stop()
                await self.application.shutdown()

        except Exception as e:
            logger.error(f"Error in bot async run: {e}")
            # Re-raise to let the main loop handle it
            raise