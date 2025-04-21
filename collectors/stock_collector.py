import logging
import yfinance as yf
import pandas as pd
import datetime
import time
import threading
from cachetools import TTLCache
logger = logging.getLogger(__name__)
class MultiStockCollector:
    def __init__(self, window_size=30, interval_seconds=60):
        self.window_minutes = window_size
        self.interval = interval_seconds
        self.watchlist = set()

        # Initialize TTLCache with time-to-live set to window size in seconds
        self.data_cache = TTLCache(maxsize=10000, ttl=window_size * 60)
        self.is_running = False
        self.collector_thread = None
        self.lock = threading.Lock()

    def _get_cache_key(self, ticker_symbol, timestamp):
        """Create a unique cache key for each ticker and timestamp."""
        return f"{ticker_symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

    def add_stock(self, ticker_symbol):
        """Add a stock to the watchlist."""
        with self.lock:
            if ticker_symbol not in self.watchlist:
                self.watchlist.add(ticker_symbol)
                logger.info(f"Added {ticker_symbol} to watchlist")
                return True
            else:
                logger.info(f"{ticker_symbol} already in watchlist")
                return False

    def remove_stock(self, ticker_symbol):
        """Remove a stock from the watchlist."""
        with self.lock:
            if ticker_symbol in self.watchlist:
                self.watchlist.remove(ticker_symbol)
                # Remove all cached entries for this ticker
                keys_to_remove = [key for key in self.data_cache.keys() if key.startswith(f"{ticker_symbol}_")]
                for key in keys_to_remove:
                    self.data_cache.pop(key, None)
                logger.info(f"Removed {ticker_symbol} from watchlist")
                return True
            else:
                logger.info(f"{ticker_symbol} not in watchlist")
                return False

    def get_watchlist(self):
        """Return current watchlist."""
        return list(self.watchlist)

    def fetch_current_data(self, ticker_symbol):
        """Get the most recent stock data using yfinance."""
        try:
            stock = yf.Ticker(ticker_symbol)
            # Fetch the last few minutes of data to catch any we might have missed
            hist = stock.history(period='1d', interval='1m', prepost=True)

            if not hist.empty:
                # Get the most recent data points from the last few minutes
                # This helps if the collection loop was delayed
                now = pd.Timestamp.now(tz='UTC')
                two_minutes_ago = now - pd.Timedelta(minutes=2)

                recent_data = hist[hist.index >= two_minutes_ago]

                if not recent_data.empty:
                    data_list = []
                    for timestamp, row in recent_data.iterrows():
                        data_list.append({
                            'datetime': timestamp,
                            'open': row['Open'],
                            'high': row['High'],
                            'low': row['Low'],
                            'close': row['Close'],
                            'volume': row['Volume']
                        })
                    return data_list
        except Exception as e:
            logger.error(f"Error fetching data for {ticker_symbol}: {e}")
        return []

    def update_cache(self, ticker_symbol):
        """Update cache with new data points."""
        new_data_list = self.fetch_current_data(ticker_symbol)

        if new_data_list:
            with self.lock:
                for new_data in new_data_list:
                    if new_data['datetime'].tzinfo is None:
                        new_data['datetime'] = new_data['datetime'].tz_localize('UTC')

                    # Create cache key and store data
                    cache_key = self._get_cache_key(ticker_symbol, new_data['datetime'])

                    # Only add if not already in cache to avoid duplicates
                    if cache_key not in self.data_cache:
                        self.data_cache[cache_key] = new_data

    def collector_loop(self):
        """Main collection loop running in background."""
        while self.is_running:
            watchlist_copy = list(self.watchlist)

            for ticker in watchlist_copy:
                if ticker in self.watchlist:
                    self.update_cache(ticker)

            time.sleep(self.interval)

    def start(self):
        """Start the data collection process."""
        if not self.is_running:
            self.is_running = True
            self.collector_thread = threading.Thread(target=self.collector_loop)
            self.collector_thread.daemon = True
            self.collector_thread.start()
            logger.info(f"Started collecting data every {self.interval} seconds")

    def stop(self):
        """Stop the data collection process."""
        self.is_running = False
        if self.collector_thread:
            self.collector_thread.join()
        logger.info("Stopped collecting data")

    def get_data(self, ticker_symbol):
        """Return a DataFrame of the current cached data for a specific ticker."""
        data_entries = []
        with self.lock:
            # Extract all entries for this ticker from the cache
            for key, value in self.data_cache.items():
                if key.startswith(f"{ticker_symbol}_"):
                    data_entries.append(value)

        if not data_entries:
            return pd.DataFrame()

        # Sort by datetime and create DataFrame
        data_entries.sort(key=lambda x: x['datetime'])
        return pd.DataFrame(data_entries)

    def get_summary(self, ticker_symbol):
        """Get summary statistics of cached data for a specific ticker."""
        data = self.get_data(ticker_symbol)

        if data.empty:
            return f"No data available for {ticker_symbol}"

        return {
            'ticker': ticker_symbol,
            'entries': len(data),
            'window_minutes': self.window_minutes,
            'start_time': data['datetime'].min(),
            'end_time': data['datetime'].max(),
            'avg_close': data['close'].mean(),
            'min_close': data['close'].min(),
            'max_close': data['close'].max(),
            'total_volume': data['volume'].sum()
        }

    def get_all_summaries(self):
        """Get summaries for all stocks in watchlist."""
        summaries = {}
        for ticker in self.watchlist:
            summaries[ticker] = self.get_summary(ticker)
        return summaries

    def save_to_csv(self, ticker_symbol=None, filename=None):
        """Save cached data to CSV file."""
        if ticker_symbol is None:
            # Save all stocks data
            for ticker in self.watchlist:
                self._save_single_csv(ticker)
        else:
            # Save specific stock data
            self._save_single_csv(ticker_symbol, filename)

    def _save_single_csv(self, ticker_symbol, filename=None):
        """Helper method to save a single stock's data to CSV."""
        data = self.get_data(ticker_symbol)

        if filename is None:
            filename = f"{ticker_symbol}_cache_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        if not data.empty:
            data.to_csv(filename, index=False)
            logger.info(f"Data saved to {filename}")
        else:
            logger.info(f"No data to save for {ticker_symbol}")

    def get_latest_prices(self):
        """Get the latest price for each stock in watchlist."""
        latest_prices = {}
        for ticker in self.watchlist:
            data = self.get_data(ticker)
            if not data.empty:
                latest_row = data.iloc[-1]
                latest_prices[ticker] = {
                    'datetime': latest_row['datetime'],
                    'price': latest_row['close'],
                    'volume': latest_row['volume']
                }
        return latest_prices