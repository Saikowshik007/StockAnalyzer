import logging

import yfinance as yf
import pandas as pd
import datetime
import time
import threading
logger = logging.getLogger(__name__)

class MultiStockCollector:
    def __init__(self, window_size=30, interval_seconds=60):
        """
        Initialize a multi-stock data collector with rolling time window.

        Args:
            window_size (int): Window size in minutes
            interval_seconds (int): Data collection interval in seconds
        """
        self.window_minutes = window_size
        self.interval = interval_seconds
        self.watchlist = set()
        self.data_caches = {}
        self.is_running = False
        self.collector_thread = None
        self.lock = threading.Lock()

    def add_stock(self, ticker_symbol):
        """Add a stock to the watchlist."""
        with self.lock:
            if ticker_symbol not in self.watchlist:
                self.watchlist.add(ticker_symbol)
                self.data_caches[f"data_{ticker_symbol}"] = pd.DataFrame(
                    columns=['datetime', 'open', 'high', 'low', 'close', 'volume']
                )
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
                del self.data_caches[f"data_{ticker_symbol}"]
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
            hist = stock.history(period='1d', interval='1m')

            if not hist.empty:
                latest = hist.iloc[-1]
                return {
                    'datetime': hist.index[-1],
                    'open': latest['Open'],
                    'high': latest['High'],
                    'low': latest['Low'],
                    'close': latest['Close'],
                    'volume': latest['Volume']
                }
        except Exception as e:
            logger.error(f"Error fetching data for {ticker_symbol}: {e}")
        return None

    def update_cache(self, ticker_symbol):
        """Update cache with new data and remove expired entries for a specific ticker."""
        new_data = self.fetch_current_data(ticker_symbol)

        if new_data:
            if new_data['datetime'].tzinfo is None:
                new_data['datetime'] = new_data['datetime'].tz_localize('UTC')

            cache_key = f"data_{ticker_symbol}"
            new_row = pd.DataFrame([new_data])

            with self.lock:
                if not self.data_caches[cache_key].empty:
                    self.data_caches[cache_key] = pd.concat(
                        [self.data_caches[cache_key], new_row],
                        ignore_index=True
                    )
                else:
                    self.data_caches[cache_key] = new_row

                current_time = pd.Timestamp.now('UTC')
                cutoff_time = current_time - pd.Timedelta(minutes=self.window_minutes)
                self.data_caches[cache_key] = self.data_caches[cache_key][
                    self.data_caches[cache_key]['datetime'] > cutoff_time
                    ]

    def collector_loop(self):
        """Main collection loop running in background."""
        while self.is_running:
            watchlist_copy = list(self.watchlist)  # Create a copy to avoid modification during iteration

            for ticker in watchlist_copy:
                if ticker in self.watchlist:  # Double-check the ticker is still in watchlist
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
        """Return a copy of the current cached data for a specific ticker."""
        cache_key = f"data_{ticker_symbol}"
        with self.lock:
            if cache_key in self.data_caches:
                return self.data_caches[cache_key].copy()
        return pd.DataFrame()

    def get_summary(self, ticker_symbol):
        """Get summary statistics of cached data for a specific ticker."""
        cache_key = f"data_{ticker_symbol}"
        with self.lock:
            if cache_key not in self.data_caches or self.data_caches[cache_key].empty:
                return f"No data available for {ticker_symbol}"

            data = self.data_caches[cache_key]
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
        cache_key = f"data_{ticker_symbol}"

        if filename is None:
            filename = f"{ticker_symbol}_cache_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        with self.lock:
            if cache_key in self.data_caches and not self.data_caches[cache_key].empty:
                self.data_caches[cache_key].to_csv(filename, index=False)
                logger.info(f"Data saved to {filename}")
            else:
                logger.info(f"No data to save for {ticker_symbol}")

    def get_latest_prices(self):
        """Get the latest price for each stock in watchlist."""
        latest_prices = {}
        with self.lock:
            for ticker in self.watchlist:
                cache_key = f"data_{ticker}"
                if cache_key in self.data_caches and not self.data_caches[cache_key].empty:
                    latest_row = self.data_caches[cache_key].iloc[-1]
                    latest_prices[ticker] = {
                        'datetime': latest_row['datetime'],
                        'price': latest_row['close'],
                        'volume': latest_row['volume']
                    }
        return latest_prices
