import asyncio
import sys
import logging
import threading
from datetime import datetime
import os
import time

from services.pattern_monitor import PatternMonitor
from utils.config_loader import ConfigLoader
from database.db_manager import DatabaseManager
from collectors.stock_collector import MultiStockCollector
from collectors.news_collector import NewsMonitor
from services.bot_service import TelegramBot

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

        self.news_monitor = NewsMonitor(self.config, self.db_manager)

        telegram_config = self.config.get('telegram', {})
        self.bot = TelegramBot(
            telegram_config,
            self.db_manager,
            self.stock_collector,
            self.news_monitor.notifier
        )
        self.pattern_monitor = PatternMonitor(
            self.config,
            self.db_manager,
            self.stock_collector,
            self.bot
        )

        # Initialize default watchlist
        self._initialize_watchlist()

    def _initialize_watchlist(self):
        """Initialize the watchlist with default tickers from config."""
        default_watchlist = self.config.get('stock_collector.default_watchlist', [])
        for ticker in default_watchlist:
            self.db_manager.add_to_watchlist(ticker)
            self.stock_collector.add_stock(ticker)

    def run_bot(self):
        """Run the Telegram bot in a separate thread."""
        try:
            self.bot.run()
        except Exception as e:
            logger.error(f"Error running bot: {e}")

    def run_stock_collector(self):
        """Run the stock collector."""
        try:
            self.stock_collector.start()
        except Exception as e:
            logger.error(f"Error running stock collector: {e}")

    def run_pattern_monitor(self):
        """Run the pattern monitoring loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self.pattern_monitor.monitor_patterns())
        except Exception as e:
            logger.error(f"Error in pattern monitor: {e}")

    def run_stock_data_saver(self):
        """Periodically save stock data to database."""
        while True:
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

    async def run(self):
        """Main application runner."""
        try:
            # Create necessary directories
            os.makedirs('logs', exist_ok=True)
            os.makedirs('config', exist_ok=True)
            os.makedirs('backups', exist_ok=True)

            # Start backup manager
            self.db_manager.backup_database()

            # Start the bot in a separate thread
            bot_thread = threading.Thread(target=self.run_bot)
            bot_thread.daemon = True
            bot_thread.start()

            # Start stock collector in a separate thread
            stock_thread = threading.Thread(target=self.run_stock_collector)
            stock_thread.daemon = True
            stock_thread.start()

            # Start stock data saver in a separate thread
            saver_thread = threading.Thread(target=self.run_stock_data_saver)
            saver_thread.daemon = True
            saver_thread.start()

            pattern_thread = threading.Thread(target=self.run_pattern_monitor)
            pattern_thread.daemon = True
            pattern_thread.start()

            # Run news monitor in the main async loop
            await self.news_monitor.monitor()

        except KeyboardInterrupt:
            logger.info("Application stopped by user")
        except Exception as e:
            logger.error(f"Application error: {e}")
        finally:
            self.stock_collector.stop()
            self.db_manager.backup_database()

def main():
    """Entry point for the application."""
    if sys.platform == 'win32':
        # For Windows, use a different event loop policy
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    setup_logging()

    try:
        app = FinancialMonitorApp()
        asyncio.run(app.run())
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

if __name__ == "__main__":
    main()