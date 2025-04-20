# main.py
import asyncio
import signal
import sys
import logging
import threading
from datetime import datetime
import os
import time

from utils.config_loader import ConfigLoader
from database.db_manager import DatabaseManager
from collectors.stock_collector import MultiStockCollector
from collectors.news_collector import NewsMonitor
from services.bot_service import TelegramBot
from services.pattern_monitor import TalibPatternMonitor

# Configure logging
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

        self.stock_collector = MultiStockCollector(
            window_size=self.config.get('stock_collector.window_size', 30),
            interval_seconds=self.config.get('stock_collector.interval_seconds', 60)
        )

        telegram_config = self.config.get('telegram', {})
        self.bot = TelegramBot(
            telegram_config,
            self.db_manager,
            self.stock_collector
        )

        # Now initialize components that use the bot for notifications
        self.news_monitor = NewsMonitor(self.config, self.db_manager, self.bot)
        self.pattern_monitor = TalibPatternMonitor(
            self.config,
            self.db_manager,
            self.stock_collector,
            self.bot
        )

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
            self.stock_collector.start()
        except Exception as e:
            logger.error(f"Error running stock collector: {e}")

    def run_stock_data_saver(self):
        """Periodically save stock data to database."""
        while not self.shutdown_event.is_set():
            try:
                watchlist = self.db_manager.get_active_watchlist()
                for ticker in watchlist:
                    data = self.stock_collector.get_data(ticker)
                    if not data.empty:
                        latest = data.iloc[-1]
                        self.db_manager.save_stock_data({
                            'ticker': ticker,
                            'timestamp': latest['datetime'],
                            'open': latest['open'],
                            'high': latest['high'],
                            'low': latest['low'],
                            'close': latest['close'],
                            'volume': latest['volume']
                        })
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

    async def run(self):
        """Main application runner."""
        try:
            # Create necessary directories
            os.makedirs('logs', exist_ok=True)
            os.makedirs('config', exist_ok=True)
            os.makedirs('backups', exist_ok=True)

            # Start backup manager
            self.db_manager.backup_database()

            # Start stock collector in a separate thread
            stock_thread = threading.Thread(target=self.run_stock_collector, daemon=True)
            stock_thread.start()

            # Start stock data saver in a separate thread
            saver_thread = threading.Thread(target=self.run_stock_data_saver, daemon=True)
            saver_thread.start()

            # Start pattern monitor in a separate thread
            pattern_thread = threading.Thread(target=self.run_pattern_monitor, daemon=True)
            pattern_thread.start()

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
        self.stock_collector.stop()

        # Cancel all running tasks
        for task in self.tasks:
            if not task.done():
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