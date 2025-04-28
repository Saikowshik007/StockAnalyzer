import asyncio
import logging
import os
import threading
import time

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

        # Update stock collector to use Yahoo Finance (no API key needed)
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

    def _initialize_watchlist(self):
        """Initialize the watchlist with default tickers from config."""
        default_watchlist = self.config.get('stock_collector.default_watchlist', [])
        for ticker in default_watchlist:
            self.db_manager.add_to_watchlist(ticker)
            self.stock_collector.add_stock(ticker)


    def run_stock_collector(self):
        """Run the stock collector."""
        try:
            # Run in a completely separate thread with clear isolation
            collector_thread = threading.Thread(
                target=self._start_stock_collector_with_new_loop,
                daemon=True
            )
            collector_thread.start()

            # Wait for connection to be established (with timeout)
            start_time = time.time()
            while not self.stock_collector.connected and time.time() - start_time < 30:
                time.sleep(0.1)

            if not self.stock_collector.connected:
                logger.warning("WebSocket connection not established after timeout. Bot may have issues.")
            else:
                logger.info("Stock collector WebSocket connected successfully")
        except Exception as e:
            logger.error(f"Error running stock collector: {e}")

    def _start_stock_collector_with_new_loop(self):
        """Start stock collector with a dedicated event loop in a separate thread."""
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Connect to Yahoo WebSocket
            self.stock_collector.connect_websocket()
            # Run the loop until interrupted
            loop.run_forever()
        except Exception as e:
            logger.error(f"Error in stock collector thread: {e}")
        finally:
            loop.close()

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
                                'timeframe': timeframe,
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

    async def run(self):
        """Main application runner."""
        try:
            # Create necessary directories
            os.makedirs('logs', exist_ok=True)
            os.makedirs('config', exist_ok=True)
            os.makedirs('backups', exist_ok=True)

            # Start backup manager
            self.db_manager.backup_database()

            # Connect to Yahoo WebSocket first
            self.run_stock_collector()
            logger.info("Yahoo Finance WebSocket connection initialized")

            # Start stock data saver in a separate thread
            saver_thread = threading.Thread(target=self.run_stock_data_saver, daemon=True)
            saver_thread.start()

            # Start pattern monitor in a separate thread
            pattern_thread = threading.Thread(target=self.run_pattern_monitor, daemon=True)
            pattern_thread.start()

            # Start sentiment tracker in a separate thread
            sentiment_thread = threading.Thread(target=self.run_sentiment_tracker, daemon=True)
            sentiment_thread.start()

            # Create tasks for async components
            news_task = asyncio.create_task(self.news_monitor.monitor())
            bot_task = asyncio.create_task(self.bot.run_async())
            self.tasks = [news_task, bot_task]

            # Wait for tasks to complete
            await asyncio.gather(*self.tasks)

        except asyncio.CancelledError:
            logger.info("Tasks cancelled during shutdown")
        except Exception as e:
            logger.error(f"Application error: {e}")
            # Re-raise the exception to ensure it's properly handled
            raise
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Cleanup resources properly."""
        logger.info("Starting cleanup...")

        # Signal components to stop
        self.shutdown_event.set()
        self.news_monitor.stop()

        # Close Yahoo WebSocket connection first
        if hasattr(self, 'stock_collector'):
            logger.info("Closing Yahoo Finance WebSocket connection...")
            self.stock_collector.close()
            logger.info("Yahoo Finance WebSocket connection closed")

        # Give the WebSocket time to fully disconnect before handling tasks
        await asyncio.sleep(1)

        # Cancel all running tasks
        for task in self.tasks:
            if not task.done():
                logger.info(f"Cancelling task: {task}")
                task.cancel()

        # Wait for tasks to be cancelled
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

        # Give threads time to shutdown
        await asyncio.sleep(1)

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
        import traceback
        traceback.print_exc()
    finally:
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    main()