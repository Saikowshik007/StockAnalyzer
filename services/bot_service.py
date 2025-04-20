import asyncio
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler
from telegram.constants import ParseMode
from typing import List
from services.pattern_recognition import TalibPatternRecognition

logger = logging.getLogger(__name__)

class TelegramBot:
    def __init__(self, config: dict, db_manager, stock_collector, notifier):
        self.token = config.get('api_key')
        self.chat_id = config.get('chat_id')
        self.db_manager = db_manager
        self.stock_collector = stock_collector
        self.notifier = notifier
        self.application = None
        self.pattern_recognizer = TalibPatternRecognition()

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
    pattern_recognizer = TalibPatternRecognition()
    detected_patterns = pattern_recognizer.detect_patterns(data)

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
        signal = pattern_recognizer.get_trading_signal(
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

    async def send_pattern_message(self, message: str):
        """Send pattern detection notification to the Telegram chat."""
        try:
            if self.application:
                await self.application.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode=ParseMode.MARKDOWN
                )
            else:
                logger.error("Bot application not initialized")
        except Exception as e:
            logger.error(f"Error sending pattern notification: {e}")

    def run(self):
        """Start the bot with proper asyncio setup."""
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

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

            # Start the bot in this thread's event loop
            loop.run_until_complete(self.application.run_polling())
        finally:
            try:
                loop.close()
            except:
                pass