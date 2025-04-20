# services/bot_service.py
import asyncio
import logging
import re
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
            "/price <ticker> - Get current price\n"
            "/history <ticker> - Get price history\n"
            "/pattern <ticker> - Analyze technical patterns\n"
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
            latest_price = self.stock_collector.get_latest_prices().get(ticker, {}).get('price', 'N/A')
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
        """Handle /price <ticker> command."""
        if not context.args:
            await update.message.reply_text("Please provide a ticker symbol: /price AAPL")
            return

        ticker = context.args[0].upper()
        prices = self.stock_collector.get_latest_prices()

        if ticker in prices:
            price_info = prices[ticker]
            message = (
                f"💰 *{ticker} Current Price*\n\n"
                f"Price: ${price_info['price']:.2f}\n"
                f"Volume: {price_info['volume']:,}\n"
                f"Time: {price_info['datetime'].strftime('%Y-%m-%d %H:%M:%S')}"
            )
        else:
            message = f"No price data available for {ticker}. You might need to add it to your watchlist first."

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
        """Handle /pattern <ticker> command to check for patterns using TA-Lib."""
        if not context.args:
            await update.message.reply_text("Please provide a ticker symbol: /pattern AAPL")
            return

        ticker = context.args[0].upper()

        # Get stock data from the collector
        data = self.stock_collector.get_data(ticker)

        if data.empty:
            await update.message.reply_text(f"No data available for {ticker}. Make sure it's in your watchlist.")
            return

        # Analyze patterns using TA-Lib
        detected_patterns = self.pattern_recognizer.detect_patterns(data)

        if not detected_patterns:
            await update.message.reply_text(f"No candlestick patterns detected for {ticker}.")
            return

        # Format pattern information
        current_price = data['close'].iloc[-1]
        message = f"📊 *Candlestick Pattern Analysis for {ticker}*\n\n"
        message += f"Current Price: ${current_price:.2f}\n\n"

        for pattern_name, occurrences in detected_patterns.items():
            # Only show most recent occurrence of each pattern
            latest_occurrence = max(occurrences, key=lambda x: x['timestamp'])
            signal = self.pattern_recognizer.get_trading_signal(
                pattern_name,
                latest_occurrence['signal'],
                current_price
            )

            message += f"🎯 *{pattern_name}*\n"
            message += f"Time: {latest_occurrence['timestamp'].strftime('%Y-%m-%d %H:%M')}\n"
            message += f"Signal: {latest_occurrence['signal']}\n"

            if signal['action']:
                message += f"Action: *{signal['action']}*\n"
                message += f"Reason: {signal['reason']}\n"
                message += f"Confidence: {signal['confidence'].upper()}\n"
                if signal['action'] in ['BUY', 'SELL']:
                    message += f"Entry: ${signal['entry_price']:.2f}, SL: ${signal['stop_loss']:.2f}, TP: ${signal['take_profit']:.2f}\n"
            else:
                message += "No actionable signal currently\n"

            message += "\n"

        # Limit message length to avoid Telegram API errors
        if len(message) > 4000:
            message = message[:3900] + "\n\n... (Message truncated)"

        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

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
        """Run the bot asynchronously."""
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

        # Start the bot polling
        await self.application.run_polling()
