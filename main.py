import asyncio
import logging
import os
import threading
import time
import traceback

from collectors.news_collector import NewsMonitor
from collectors.stock_collector import YahooMultiStockCollector
from database.db_manager import DatabaseManager
from services.bot_service import TelegramBot
from services.pattern_monitor import TalibPatternMonitor
from services.sentiment_tracker import SentimentTracker
from utils.config_loader import ConfigLoader


def setup_logging():
    """Setup logging configuration."""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'application.log')),
            logging.StreamHandler()
        ]
    )


logger = logging.getLogger(__name__)


class FinancialMonitorApp:
    def __init__(self):
        # Load configuration
        self.config = ConfigLoader()
        # Initialize components
        db_config = self.config.get('database', {})
        self.db_manager = DatabaseManager(db_config)

        # Update stock collector to use Yahoo Finance via yfinance
        self.stock_collector = YahooMultiStockCollector(self.db_manager)

        # Set timeframes from config or use defaults
        timeframes = self.config.get('timeframes', {
            'long_term': {'interval': '1d', 'weight': 0.5},
            'medium_term': {'interval': '1h', 'weight': 0.3},
            'short_term': {'interval': '15m', 'weight': 0.2},
            'very_short_term': {'interval': '5m', 'weight': 0.2}
        })
        self.stock_collector.set_timeframes(timeframes)

        # Initialize sentiment tracker
        self.sentiment_tracker = SentimentTracker(self.db_manager)

        telegram_config = self.config.get('telegram', {})
        self.bot = TelegramBot(
            telegram_config,
            self.db_manager,
            self.stock_collector
        )

        # Make sentiment tracker available to the bot
        self.bot.sentiment_tracker = self.sentiment_tracker

        # Now initialize components that use the bot for notifications
        self.news_monitor = NewsMonitor(self.config, self.db_manager, self.bot)
        self.pattern_monitor = TalibPatternMonitor(
            self.config,
            self.db_manager,
            self.stock_collector,
            self.bot
        )

        # Set up confidence thresholds if configured
        confidence_thresholds = self.config.get('confidence_thresholds', {})
        if confidence_thresholds:
            self.pattern_monitor.confidence_thresholds = confidence_thresholds

        # Initialize default watchlist
        self._initialize_watchlist()
        self.shutdown_event = threading.Event()
        self.tasks = []

        # Create event loops for async threads
        self.news_loop = None
        self.bot_loop = None

    def _initialize_watchlist(self):
        """Initialize the watchlist with default tickers from config."""
        default_watchlist = self.config.get('stock_collector.default_watchlist', [])
        for ticker in default_watchlist:
            self.db_manager.add_to_watchlist(ticker)
            self.stock_collector.add_stock(ticker)

    def run_stock_data_saver(self):
        """Periodically save stock data to database for all timeframes."""
        while not self.shutdown_event.is_set():
            try:
                watchlist = self.db_manager.get_active_watchlist()
                for ticker in watchlist:
                    # Get data for all timeframes
                    all_data = self.stock_collector.get_multi_timeframe_data(ticker)

                    for timeframe, data in all_data.items():
                        if not data.empty:
                            latest = data.iloc[-1]
                            # Save with timeframe information
                            self.db_manager.save_stock_data({
                                'ticker': ticker,
                                'timestamp': latest['datetime'],
                                'open': latest['open'],
                                'high': latest['high'],
                                'low': latest['low'],
                                'close': latest['close'],
                                'volume': latest['volume']
                            })

                            # Also calculate and save technical indicators
                            indicators = self.stock_collector.calculate_technical_indicators(data)
                            if indicators:
                                self.db_manager.save_technical_indicators(
                                    ticker,
                                    timeframe,
                                    latest['datetime'],
                                    indicators
                                )

                time.sleep(60)  # Save every minute
            except Exception as e:
                logger.error(f"Error saving stock data: {e}")
                time.sleep(60)

    def run_pattern_monitor(self):
        """Run the pattern monitoring loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self.pattern_monitor.monitor_patterns())
        except Exception as e:
            logger.error(f"Error in pattern monitor: {e}")
            logger.error(traceback.format_exc())

    def run_sentiment_tracker(self):
        """Run periodic sentiment updates."""
        while not self.shutdown_event.is_set():
            try:
                self.sentiment_tracker.update_daily_sentiment()
                logger.info("Sentiment data updated")
                time.sleep(900)  # Update every 15 minutes
            except Exception as e:
                logger.error(f"Error updating sentiment data: {e}")
                time.sleep(300)  # Wait 5 minutes and try again

    def run_data_refresher(self):
        """Periodically refresh stock data for all watchlist items."""
        while not self.shutdown_event.is_set():
            try:
                # Refresh all stock data
                watchlist = self.db_manager.get_active_watchlist()
                logger.info(f"Refreshing data for {len(watchlist)} stocks in watchlist")

                for ticker in watchlist:
                    try:
                        self.stock_collector.refresh_data(ticker)
                        # Small delay between refreshes to avoid potential rate limits
                        time.sleep(0.5)
                    except Exception as e:
                        logger.error(f"Error refreshing data for {ticker}: {e}")

                logger.info("Data refresh completed")
                time.sleep(300)  # Refresh every 5 minutes
            except Exception as e:
                logger.error(f"Error in data refresher: {e}")
                time.sleep(60)  # Wait 1 minute and try again

    def run_news_monitor(self):
        """Run the news monitoring loop in a dedicated thread."""
        self.news_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.news_loop)

        try:
            logger.info("Starting news monitor...")
            self.news_loop.run_until_complete(self.news_monitor.monitor())
        except Exception as e:
            logger.error(f"Error in news monitor thread: {e}")
            logger.error(traceback.format_exc())
        finally:
            self.news_loop.close()
            logger.info("News monitor thread stopped")

    def run_telegram_bot(self):
        """Run the Telegram bot in a dedicated thread."""
        self.bot_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.bot_loop)

        try:
            logger.info("Starting Telegram bot...")
            self.bot_loop.run_until_complete(self.bot.run_async())
        except Exception as e:
            logger.error(f"Error in Telegram bot thread: {e}")
            logger.error(traceback.format_exc())
        finally:
            self.bot_loop.close()
            logger.info("Telegram bot thread stopped")

    async def run(self):
        """Main application runner."""
        try:
            # Create necessary directories
            os.makedirs('logs', exist_ok=True)
            os.makedirs('config', exist_ok=True)
            os.makedirs('backups', exist_ok=True)

            # Start backup manager
            self.db_manager.backup_database()

            # Initially refresh stock data for the watchlist
            logger.info("Initializing stock data...")
            watchlist = self.db_manager.get_active_watchlist()
            for ticker in watchlist:
                self.stock_collector.refresh_data(ticker)
            logger.info("Stock data initialized")

            # Start stock data saver in a separate thread
            logger.info("Starting stock data saver thread...")
            saver_thread = threading.Thread(target=self.run_stock_data_saver, daemon=True)
            saver_thread.start()

            # Start pattern monitor in a separate thread
            logger.info("Starting pattern monitor thread...")
            pattern_thread = threading.Thread(target=self.run_pattern_monitor, daemon=True)
            pattern_thread.start()

            # Start sentiment tracker in a separate thread
            logger.info("Starting sentiment tracker thread...")
            sentiment_thread = threading.Thread(target=self.run_sentiment_tracker, daemon=True)
            sentiment_thread.start()

            # Start data refresher in a separate thread
            logger.info("Starting data refresher thread...")
            refresher_thread = threading.Thread(target=self.run_data_refresher, daemon=True)
            refresher_thread.start()

            # Start news monitor in a separate thread
            logger.info("Starting news monitor thread...")
            news_thread = threading.Thread(target=self.run_news_monitor, daemon=True)
            news_thread.start()

            # Start Telegram bot in a separate thread
            logger.info("Starting Telegram bot thread...")
            bot_thread = threading.Thread(target=self.run_telegram_bot, daemon=True)
            bot_thread.start()

            # Main application loop - keep the main thread alive
            logger.info("All components started successfully!")
            while not self.shutdown_event.is_set():
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("Tasks cancelled during shutdown")
        except Exception as e:
            logger.error(f"Application error: {e}")
            logger.error(traceback.format_exc())
            # Re-raise the exception to ensure it's properly handled
            raise
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Cleanup resources properly."""
        logger.info("Starting cleanup...")

        # Signal components to stop
        self.shutdown_event.set()
        if hasattr(self, 'news_monitor'):
            self.news_monitor.stop()

        # Close stock collector resources
        if hasattr(self, 'stock_collector'):
            logger.info("Closing stock collector...")
            self.stock_collector.close()
            logger.info("Stock collector closed")

        # Give a moment for threads to process the shutdown event
        await asyncio.sleep(2)

        # Cancel all running tasks
        if self.news_loop and self.news_loop.is_running():
            for task in asyncio.all_tasks(self.news_loop):
                task.cancel()

        if self.bot_loop and self.bot_loop.is_running():
            for task in asyncio.all_tasks(self.bot_loop):
                task.cancel()

        # Give threads time to shutdown
        await asyncio.sleep(2)

        # Final backup
        self.db_manager.backup_database()
        logger.info("Cleanup completed")


def main():
    """Entry point for the application."""
    setup_logging()
    app = FinancialMonitorApp()

    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    main()