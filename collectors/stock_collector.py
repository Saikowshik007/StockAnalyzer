# collectors/yahoo_stock_collector.py
import logging
import pandas as pd
import threading
import talib
import time
import json
from datetime import datetime, timedelta
import pytz
from queue import Queue
import asyncio
import yfinance as yf  # For historical data
from yliveticker import YLiveTicker  # For real-time data

logger = logging.getLogger(__name__)
pd.set_option('future.no_silent_downcasting', True)

class YahooMultiStockCollector:
    def __init__(self, db_manager=None):
        self.watchlist = set()
        self.watchlist.add("SPY")
        self.db_manager = db_manager
        self.lock = threading.Lock()

        # Data storage for each ticker and timeframe
        self.stock_data = {}

        # Real-time latest quote storage
        self.real_time_quotes = {}

        # Tick data storage (for real-time intraday bars)
        self.tick_data = {}

        # Cache for API responses
        self.api_cache = {}
        self.cache_expiry = {}

        # Track connection status
        self.connected = False
        self.live_ticker_running = False

        # Message queue for processing
        self.message_queue = Queue()

        # Default time intervals for multi-timeframe analysis
        self.timeframes = {
            'very_short_term': {'interval': '2m', 'weight': 0.2},
            'short_term': {'interval': '5m', 'weight': 0.2},
            'medium_term': {'interval': '15m', 'weight': 0.3},
            'long_term': {'interval': '1h', 'weight': 0.5}
        }

        self.interval_to_minutes = {
            '1m': 1,
            '2m': 2,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '60m': 60,
            '1h': 60,
            '1d': 1440,
            '5d': 7200,
            '1wk': 10080,
            '1mo': 43200  # Approximate
        }

        # Changed base timeframe from 1m to 5m to avoid API limits
        self.base_timeframe = '5m'

        # Load existing watchlist from DB if available
        if self.db_manager:
            self._load_watchlist_from_db()

        # Start processing thread for any background tasks
        self.processing_thread = threading.Thread(target=self._process_tasks)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        # Start cache management thread
        self.cache_thread = threading.Thread(target=self._manage_cache)
        self.cache_thread.daemon = True
        self.cache_thread.start()

        # Start live ticker thread
        self.live_ticker_thread = threading.Thread(target=self._start_live_ticker)
        self.live_ticker_thread.daemon = True
        self.live_ticker_thread.start()

        # Start real-time bar aggregation thread
        self.bar_aggregation_thread = threading.Thread(target=self._aggregate_real_time_bars)
        self.bar_aggregation_thread.daemon = True
        self.bar_aggregation_thread.start()

        # Initial data load for watchlist
        for ticker in self.watchlist:
            self._initialize_ticker_data(ticker)

    def _load_watchlist_from_db(self):
        """Load watchlist from database."""
        if self.db_manager:
            db_watchlist = self.db_manager.get_active_watchlist()
            with self.lock:
                self.watchlist = set(db_watchlist)
                logger.info(f"Loaded watchlist from database: {list(self.watchlist)}")

    def _process_tasks(self):
        """Process background tasks."""
        while True:
            try:
                # Process any tasks in the queue if needed
                if not self.message_queue.empty():
                    task = self.message_queue.get()
                    try:
                        if task.get('type') == 'refresh_data':
                            ticker = task.get('ticker')
                            self._initialize_ticker_data(ticker)
                    except Exception as e:
                        logger.error(f"Error processing task: {e}")
            except Exception as e:
                logger.error(f"Error in task processing: {e}")

            # Small sleep to prevent CPU hogging
            time.sleep(0.1)

    def _manage_cache(self):
        """Background thread to manage cache expiry."""
        while True:
            try:
                current_time = datetime.now()
                expired_keys = []

                # Find expired cache entries
                with self.lock:
                    for key, expiry_time in self.cache_expiry.items():
                        if current_time > expiry_time:
                            expired_keys.append(key)

                    # Remove expired entries
                    for key in expired_keys:
                        if key in self.api_cache:
                            del self.api_cache[key]
                        if key in self.cache_expiry:
                            del self.cache_expiry[key]

            except Exception as e:
                logger.error(f"Error in cache management: {e}")

            # Check every 10 seconds
            time.sleep(10)

    def _start_live_ticker(self):
        """Start the YLiveTicker real-time data feed."""
        while True:
            try:
                logger.info("Starting YLiveTicker connection...")
                self.live_ticker_running = True

                # Define callback for real-time data
                def on_new_msg(ws, data):
                    try:
                        # Extract relevant data from the ticker update
                        ticker_symbol = data.get('id', '').split('.')[0]

                        # Skip if not in our watchlist
                        if ticker_symbol not in self.watchlist:
                            return

                        # Extract price data
                        price_data = data.get('price', {})
                        if not price_data:
                            return

                        timestamp = datetime.now(pytz.UTC)

                        # Store the real-time quote
                        with self.lock:
                            self.real_time_quotes[ticker_symbol] = {
                                'datetime': timestamp,
                                'price': price_data.get('regularMarketPrice', None),
                                'volume': price_data.get('regularMarketVolume', 0),
                                'high': price_data.get('regularMarketDayHigh', None),
                                'low': price_data.get('regularMarketDayLow', None),
                                'open': price_data.get('regularMarketOpen', None),
                                'timestamp': timestamp
                            }

                            # Also store in tick data for aggregation
                            if ticker_symbol not in self.tick_data:
                                self.tick_data[ticker_symbol] = []

                            # Add tick data (limited to last 10,000 ticks to avoid memory issues)
                            tick = {
                                'datetime': timestamp,
                                'price': price_data.get('regularMarketPrice', None),
                                'volume': price_data.get('regularMarketVolume', 0) / 100,  # Approximate tick volume
                            }
                            self.tick_data[ticker_symbol].append(tick)

                            # Limit size of tick data
                            if len(self.tick_data[ticker_symbol]) > 10000:
                                self.tick_data[ticker_symbol] = self.tick_data[ticker_symbol][-10000:]

                        # Debug log for first few updates
                        if len(self.tick_data.get(ticker_symbol, [])) < 5:
                            logger.debug(f"Real-time update for {ticker_symbol}: {price_data.get('regularMarketPrice')}")

                    except Exception as e:
                        logger.error(f"Error processing ticker data: {e}")

                # Start the YLiveTicker with our watchlist
                # Using the correct parameter name 'on_ticker' instead of 'on_ticker_data'
                YLiveTicker(on_ticker=on_new_msg, ticker_names=list(self.watchlist))

                # YLiveTicker runs in its own event loop, so this code is only reached if it stops
                logger.warning("YLiveTicker connection closed. Reconnecting in 5 seconds...")
                self.live_ticker_running = False
                time.sleep(5)

            except Exception as e:
                logger.error(f"Error in YLiveTicker connection: {e}")
                self.live_ticker_running = False
                time.sleep(5)  # Wait before retry

    def _aggregate_real_time_bars(self):
        """Aggregate tick data into real-time bars of various timeframes."""
        while True:
            try:
                current_time = datetime.now(pytz.UTC)

                # For each ticker in our watchlist with tick data
                for ticker in list(self.watchlist):
                    if ticker not in self.tick_data or not self.tick_data[ticker]:
                        continue
                    ticks = self.tick_data[ticker]

                    # Skip if we don't have enough ticks
                    if len(ticks) < 2:
                        continue

                    # Aggregate for each timeframe we care about
                    for tf_name, tf_config in self.timeframes.items():
                        interval = tf_config['interval']
                        minutes = self.interval_to_minutes.get(interval, 5)

                        # Determine the start of the current bar period
                        bar_start = current_time.replace(
                            second=0, microsecond=0,
                            minute=(current_time.minute // minutes) * minutes
                        )

                        # Get ticks that belong to the current bar
                        current_bar_ticks = [
                            t for t in ticks
                            if t['datetime'] >= bar_start - timedelta(minutes=minutes) and
                               t['datetime'] < bar_start
                        ]

                        # Skip if we don't have any ticks for this bar
                        if not current_bar_ticks:
                            continue

                        # Create OHLCV bar
                        prices = [t['price'] for t in current_bar_ticks if t['price'] is not None]
                        volumes = [t['volume'] for t in current_bar_ticks if 'volume' in t]

                        if not prices:
                            continue

                        bar = {
                            'datetime': bar_start - timedelta(minutes=minutes),
                            'open': prices[0],
                            'high': max(prices),
                            'low': min(prices),
                            'close': prices[-1],
                            'volume': sum(volumes)
                        }

                        # Update our data structures
                        if ticker not in self.stock_data:
                            self.stock_data[ticker] = {}

                        if tf_name not in self.stock_data[ticker]:
                            self.stock_data[ticker][tf_name] = pd.DataFrame(columns=[
                                'datetime', 'open', 'high', 'low', 'close', 'volume'
                            ])

                        # Check if we already have a bar for this timeframe and datetime
                        df = self.stock_data[ticker][tf_name]
                        existing_bar = df[df['datetime'] == bar['datetime']]

                        if not existing_bar.empty:
                            # Update existing bar
                            idx = existing_bar.index[0]
                            df.at[idx, 'high'] = max(df.at[idx, 'high'], bar['high'])
                            df.at[idx, 'low'] = min(df.at[idx, 'low'], bar['low'])
                            df.at[idx, 'close'] = bar['close']
                            df.at[idx, 'volume'] += bar['volume']
                        else:
                            # Add new bar
                            new_row = pd.DataFrame([bar])
                            self.stock_data[ticker][tf_name] = pd.concat([df, new_row], ignore_index=True)

                        # Sort and limit size
                        self.stock_data[ticker][tf_name] = self.stock_data[ticker][tf_name].sort_values('datetime')
                        if len(self.stock_data[ticker][tf_name]) > 10000:
                            self.stock_data[ticker][tf_name] = self.stock_data[ticker][tf_name].tail(10000)

            except Exception as e:
                logger.error(f"Error in real-time bar aggregation: {e}")

            # Run this process every second
            time.sleep(1)

    def _update_ticker_data(self, ticker, bar_data):
        """Update internal data structure with new bar data and derive higher timeframes."""
        with self.lock:
            if ticker not in self.stock_data:
                self.stock_data[ticker] = {}

            # For base timeframe data, just append
            if self.base_timeframe not in self.stock_data[ticker]:
                self.stock_data[ticker][self.base_timeframe] = pd.DataFrame(columns=[
                    'datetime', 'open', 'high', 'low', 'close', 'volume'
                ])

            # Add new data point
            df = self.stock_data[ticker][self.base_timeframe]
            new_row = pd.DataFrame([bar_data])

            # Check if this bar is already in our data
            if not df.empty and df['datetime'].max() >= bar_data['datetime']:
                # Update existing bar if needed
                idx = df[df['datetime'] == bar_data['datetime']].index
                if not idx.empty:
                    # Update the high, low, close, and volume
                    df.at[idx[0], 'high'] = max(df.at[idx[0], 'high'], bar_data['high'])
                    df.at[idx[0], 'low'] = min(df.at[idx[0], 'low'], bar_data['low'])
                    df.at[idx[0], 'close'] = bar_data['close']
                    df.at[idx[0], 'volume'] += bar_data['volume']
                    self.stock_data[ticker][self.base_timeframe] = df
                return

            # Append the new row
            self.stock_data[ticker][self.base_timeframe] = pd.concat([df, new_row], ignore_index=True)

            # Limit the size of stored data
            if len(self.stock_data[ticker][self.base_timeframe]) > 10000:  # Increased for better aggregation
                self.stock_data[ticker][self.base_timeframe] = self.stock_data[ticker][self.base_timeframe].tail(10000)

            # Sort by datetime to ensure correct order
            self.stock_data[ticker][self.base_timeframe] = self.stock_data[ticker][self.base_timeframe].sort_values('datetime')

            # Now update all derived timeframes
            self._update_derived_timeframes(ticker)

    def _update_derived_timeframes(self, ticker):
        """Derive higher timeframes from base timeframe data."""
        if ticker not in self.stock_data or self.base_timeframe not in self.stock_data[ticker]:
            return

        base_data = self.stock_data[ticker][self.base_timeframe]
        if base_data.empty:
            return

        # Process each timeframe
        for tf_name, tf_config in self.timeframes.items():
            interval = tf_config['interval']

            # Skip base timeframe since it's already updated
            if interval == self.base_timeframe:
                continue

            # For lower timeframes than base, fetch directly instead of deriving
            if self.interval_to_minutes.get(interval, 0) < self.interval_to_minutes.get(self.base_timeframe, 0):
                # Only fetch if we need this timeframe
                continue

            # Derive higher timeframe data
            derived_data = self._aggregate_timeframe(base_data, interval)

            # Store derived data
            self.stock_data[ticker][tf_name] = derived_data

    def _aggregate_timeframe(self, base_data, target_interval):
        """Aggregate lower timeframe data to higher timeframe."""
        if base_data.empty:
            return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])

        # Get minutes for target interval
        minutes = self.interval_to_minutes.get(target_interval, 1)

        # Create copy to avoid modifying original
        data = base_data.copy()

        # Ensure datetime is properly set
        data['datetime'] = pd.to_datetime(data['datetime'])

        # Round datetime to the target interval
        data['interval_timestamp'] = data['datetime'].apply(
            lambda x: x.replace(
                minute=(x.minute // minutes) * minutes,
                second=0,
                microsecond=0
            )
        )

        # Group by the interval timestamp and aggregate
        aggregated = data.groupby('interval_timestamp').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()

        # Rename column back to datetime
        aggregated = aggregated.rename(columns={'interval_timestamp': 'datetime'})

        return aggregated

    def add_stock(self, ticker_symbol):
        """Add a stock to the watchlist and persist to database."""
        # Validate ticker symbol before adding
        if not self._validate_ticker(ticker_symbol):
            logger.warning(f"Invalid ticker symbol: {ticker_symbol}")
            return False

        if ticker_symbol not in self.watchlist:
            self.watchlist.add(ticker_symbol)
            # Also add to database if db_manager is available
            if self.db_manager:
                self.db_manager.add_to_watchlist(ticker_symbol)
            # Initialize historical data for this ticker
            self._initialize_ticker_data(ticker_symbol)

            # Restart YLiveTicker to include new symbol
            self._restart_live_ticker()

            logger.info(f"Added {ticker_symbol} to watchlist")
            return True
        else:
            logger.info(f"{ticker_symbol} already in watchlist")
            return False

    def _restart_live_ticker(self):
        """Restart the YLiveTicker connection with updated watchlist."""
        # This will be picked up by the live_ticker_thread's reconnection logic
        self.live_ticker_running = False
        logger.info("Signaled YLiveTicker to restart with updated watchlist")

    def _validate_ticker(self, ticker_symbol):
        """Validate if ticker symbol exists in Yahoo Finance."""
        try:
            ticker_obj = yf.Ticker(ticker_symbol)
            # Try to get just a single data point
            data = ticker_obj.history(period="1d")
            if data.empty:
                return False
            return True
        except Exception as e:
            logger.error(f"Error validating ticker {ticker_symbol}: {e}")
            return False

    def _initialize_ticker_data(self, ticker_symbol):
        """Initialize historical data for a ticker symbol using yfinance."""
        # Determine how much data we need based on the longest timeframe
        max_minutes = max(self.interval_to_minutes.get(tf_config['interval'], 1)
                          for tf_config in self.timeframes.values())

        # Calculate how many days of data we need
        bars_needed = max_minutes * 100  # Get enough bars for indicators
        days_back = max(1, int(bars_needed / (6.5 * 60)))  # Trading hours per day

        # Limit days_back to 7 days to avoid API limits
        days_back = min(7, days_back)

        try:
            # Use yfinance to get historical data with chunking for 1m data
            if self.base_timeframe == '1m':
                # For 1m data, we need to chunk the requests to avoid API limits
                data = self._fetch_chunked_historical_data(
                    ticker_symbol,
                    days_back=days_back,
                    interval=self.base_timeframe
                )
            else:
                # For other timeframes, we can use a single request
                data = self._fetch_yfinance_historical_data(
                    ticker_symbol,
                    period=f"{days_back+1}d",  # Add extra day for safety
                    interval=self.base_timeframe
                )

            if data is not None and not data.empty:
                # Store in our data structure
                with self.lock:
                    if ticker_symbol not in self.stock_data:
                        self.stock_data[ticker_symbol] = {}

                    # Store base timeframe data
                    self.stock_data[ticker_symbol][self.base_timeframe] = data

                    # Derive all other timeframes
                    self._update_derived_timeframes(ticker_symbol)
            else:
                logger.warning(f"No data available for {ticker_symbol}")

        except Exception as e:
            logger.error(f"Error initializing data for {ticker_symbol}: {e}")

    def _fetch_chunked_historical_data(self, ticker, days_back=7, interval="1m"):
        """Fetch historical data in chunks to avoid API limits."""
        all_data = []

        # For 1m data, Yahoo only allows 7 days at a time
        # We'll fetch in chunks of 1 day
        end_date = datetime.now()

        for i in range(min(7, days_back)):
            start_date = end_date - timedelta(days=1)

            try:
                # Format dates for yfinance
                start_str = start_date.strftime('%Y-%m-%d')
                end_str = end_date.strftime('%Y-%m-%d')

                # Get data for this chunk
                ticker_obj = yf.Ticker(ticker)
                chunk_data = ticker_obj.history(
                    start=start_str,
                    end=end_str,
                    interval=interval
                )

                if not chunk_data.empty:
                    # Reset index to get Date as a column
                    chunk_data = chunk_data.reset_index()
                    all_data.append(chunk_data)

            except Exception as e:
                logger.error(f"Error fetching chunk for {ticker} ({start_str} to {end_str}): {e}")

            # Move to previous day
            end_date = start_date

        # Combine all chunks
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)

            # Format dataframe to match our expected format
            combined_df = combined_df.rename(columns={
                'Datetime': 'datetime',
                'Date': 'datetime',  # Handle both column names
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Convert datetime to UTC for consistency
            combined_df['datetime'] = pd.to_datetime(combined_df['datetime']).dt.tz_convert('UTC')

            # Select only the columns we need
            combined_df = combined_df[['datetime', 'open', 'high', 'low', 'close', 'volume']]

            # Remove duplicates and sort
            combined_df = combined_df.drop_duplicates(subset=['datetime']).sort_values('datetime')

            return combined_df
        else:
            return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])

    def _fetch_yfinance_historical_data(self, ticker, period="5d", interval="5m"):
        """Fetch historical data using yfinance library."""
        # Create cache key
        cache_key = f"{ticker}_{interval}_{period}"

        # Return cached response if available
        with self.lock:
            if cache_key in self.api_cache:
                logger.info(f"Using cached data for {cache_key}")
                return self.api_cache[cache_key].copy()

        try:
            # Convert interval for yfinance if needed
            yf_interval = interval
            if interval == '60m':
                yf_interval = '1h'

            # Get historical data using yfinance
            ticker_obj = yf.Ticker(ticker)

            # For shorter timeframes, use a shorter period to avoid limits
            adjusted_period = period
            if interval in ['1m', '2m']:
                # For 1m and 2m data, use max 7d as per Yahoo limits
                adjusted_period = "7d"

            df = ticker_obj.history(period=adjusted_period, interval=yf_interval)

            # Format dataframe to match our expected format
            if not df.empty:
                # Reset index to get Date as a column
                df = df.reset_index()

                # Rename columns to match our format
                df = df.rename(columns={
                    'Datetime': 'datetime',
                    'Date': 'datetime',  # Handle both column names
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })

                # Convert datetime to UTC for consistency
                df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_convert('UTC')

                # Select only the columns we need
                columns_to_select = ['datetime', 'open', 'high', 'low', 'close', 'volume']
                df = df[[col for col in columns_to_select if col in df.columns]]

                # Add any missing columns with default values
                for col in columns_to_select:
                    if col not in df.columns:
                        if col == 'datetime':
                            df[col] = pd.to_datetime('now', utc=True)
                        elif col == 'volume':
                            df[col] = 0
                        else:
                            df[col] = 0.0

                # Cache the result
                with self.lock:
                    self.api_cache[cache_key] = df.copy()
                    # Set expiry time (30 minutes)
                    self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=30)

                return df
            else:
                logger.warning(f"Empty dataframe returned for {ticker}")
                return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])

        except Exception as e:
            logger.error(f"Error fetching yfinance data for {ticker}: {e}")
            # Check if it's a delisted error
            if "possibly delisted" in str(e):
                logger.warning(f"Stock {ticker} appears to be delisted or invalid")

            return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])

    def remove_stock(self, ticker_symbol):
        """Remove a stock from the watchlist and update database."""
        with self.lock:
            if ticker_symbol in self.watchlist:
                self.watchlist.remove(ticker_symbol)

                # Also remove from database if db_manager is available
                if self.db_manager:
                    self.db_manager.remove_from_watchlist(ticker_symbol)

                # Remove from our data storage
                if ticker_symbol in self.stock_data:
                    del self.stock_data[ticker_symbol]

                # Remove from real-time quotes
                if ticker_symbol in self.real_time_quotes:
                    del self.real_time_quotes[ticker_symbol]

                # Remove from tick data
                if ticker_symbol in self.tick_data:
                    del self.tick_data[ticker_symbol]

                # Signal to restart YLiveTicker
                self._restart_live_ticker()

                logger.info(f"Removed {ticker_symbol} from watchlist")
                return True
            else:
                logger.info(f"{ticker_symbol} not in watchlist")
                return False

    def get_watchlist(self):
        """Return current watchlist."""
        return list(self.watchlist)

    def get_data(self, ticker_symbol, interval='5m'):
        """
        Get data for a specific ticker and interval.
        For real-time data, it prefers the latest real-time bars when available,
        then falls back to historical data from yfinance.
        """
        # First check if we have real-time data for this ticker
        with self.lock:
            # Check if we have real-time quotes
            has_real_time = ticker_symbol in self.real_time_quotes

            # Check if the requested interval matches one of our timeframes
            matching_tf = None
            for tf_name, tf_config in self.timeframes.items():
                if tf_config['interval'] == interval:
                    matching_tf = tf_name
                    break

            # If we have matching real-time data from our aggregated bars
            if matching_tf and ticker_symbol in self.stock_data and matching_tf in self.stock_data[ticker_symbol]:
                # Use our real-time aggregated data
                return self.stock_data[ticker_symbol][matching_tf].copy()

        # If we don't have real-time data, fall back to yfinance historical data
        # Validate interval
        if interval not in self.interval_to_minutes:
            logger.warning(f"Invalid interval: {interval}. Using {self.base_timeframe} instead.")
            interval = self.base_timeframe

        with self.lock:
            # Find the timeframe that matches this interval
            matching_tf = None
            for tf_name, tf_config in self.timeframes.items():
                if tf_config['interval'] == interval:
                    matching_tf = tf_name
                    break

            # Check if we have the exact timeframe data already
            if matching_tf and ticker_symbol in self.stock_data and matching_tf in self.stock_data[ticker_symbol]:
                return self.stock_data[ticker_symbol][matching_tf].copy()

            # If requesting base timeframe data directly
            if interval == self.base_timeframe and ticker_symbol in self.stock_data and interval in self.stock_data[ticker_symbol]:
                return self.stock_data[ticker_symbol][interval].copy()

            # For a timeframe that's higher than our base, derive it
            if self.interval_to_minutes.get(interval, 0) >= self.interval_to_minutes.get(self.base_timeframe, 0):
                if ticker_symbol in self.stock_data and self.base_timeframe in self.stock_data[ticker_symbol]:
                    base_data = self.stock_data[ticker_symbol][self.base_timeframe]
                    if not base_data.empty:
                        derived_data = self._aggregate_timeframe(base_data, interval)

                        # Cache this derived data for future use
                        if matching_tf:
                            self.stock_data[ticker_symbol][matching_tf] = derived_data

                        return derived_data

        # If we need to fetch the data directly (either lower timeframe or missing data)
        # Determine appropriate period based on interval
        period = "5d"  # Default period
        if interval in ['1d', '5d', '1wk', '1mo']:
            period = "60d"  # Get more history for longer intervals
        elif interval in ['1m', '2m']:
            period = "7d"  # Respect Yahoo's limit for 1m data

        # For 1m data, use chunked approach to avoid limits
        if interval == '1m':
            return self._fetch_chunked_historical_data(
                ticker_symbol,
                days_back=7,  # Yahoo's limit
                interval=interval
            )
        else:
            return self._fetch_yfinance_historical_data(
                ticker_symbol,
                period=period,
                interval=interval
            )

    def get_multi_timeframe_data(self, ticker_symbol):
        """Get data for all timeframes."""
        data = {}
        for tf_name, tf_config in self.timeframes.items():
            # Extract interval from the configuration dictionary
            interval = tf_config['interval'] if isinstance(tf_config, dict) else tf_config
            timeframe_data = self.get_data(ticker_symbol, interval=interval)
            if timeframe_data is not None and not timeframe_data.empty:
                data[tf_name] = timeframe_data
            else:
                data[tf_name] = pd.DataFrame()  # Return empty DataFrame for missing data

        return data

    def calculate_technical_indicators(self, data):
        """Calculate technical indicators for the data."""
        if data.empty:
            return {}

        try:
            indicators = {}

            # Ensure we have enough data points for indicators
            if len(data) < 20:
                logger.warning(f"Insufficient data points ({len(data)}) for technical indicators")
                return {}

            # RSI
            if len(data) >= 14:
                indicators['rsi'] = talib.RSI(data['close'], timeperiod=14).iloc[-1]

            # MACD
            if len(data) >= 26:
                macd, signal, hist = talib.MACD(data['close'])
                indicators['macd'] = macd.iloc[-1]
                indicators['macd_signal'] = signal.iloc[-1]
                indicators['macd_hist'] = hist.iloc[-1]

            # Moving Averages
            if len(data) >= 20:
                sma_20 = talib.SMA(data['close'], timeperiod=20).iloc[-1]
                indicators['sma_20'] = sma_20

            if len(data) >= 50:
                sma_50 = talib.SMA(data['close'], timeperiod=50).iloc[-1]
                indicators['sma_50'] = sma_50

                # Moving Average trend
                if 'sma_20' in indicators and 'sma_50' in indicators:
                    if indicators['sma_20'] > indicators['sma_50']:
                        indicators['ma_trend'] = 'bullish'
                    elif indicators['sma_20'] < indicators['sma_50']:
                        indicators['ma_trend'] = 'bearish'
                    else:
                        indicators['ma_trend'] = 'neutral'

            # ATR for volatility
            if len(data) >= 14:
                indicators['atr'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14).iloc[-1]

            # Volume ratio
            if len(data) >= 10:
                recent_volume = data['volume'].tail(5).mean()
                prior_volume = data['volume'].tail(10).head(5).mean()
                indicators['volume_ratio'] = recent_volume / prior_volume if prior_volume > 0 else 1.0

            # Bollinger Bands
            if len(data) >= 20:
                upper, middle, lower = talib.BBANDS(data['close'], timeperiod=20)
                indicators['bb_upper'] = upper.iloc[-1]
                indicators['bb_middle'] = middle.iloc[-1]
                indicators['bb_lower'] = lower.iloc[-1]

                # Calculate % distance from price to bands
                current_price = data['close'].iloc[-1]
                indicators['bb_pct'] = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]) if (upper.iloc[-1] - lower.iloc[-1]) > 0 else 0.5

            # Stochastic Oscillator
            if len(data) >= 14:
                slowk, slowd = talib.STOCH(data['high'], data['low'], data['close'],
                                           fastk_period=14, slowk_period=3, slowd_period=3)
                indicators['stoch_k'] = slowk.iloc[-1]
                indicators['stoch_d'] = slowd.iloc[-1]

            return indicators
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}

    def get_summary(self, ticker_symbol):
        """Get enhanced summary statistics for a specific ticker."""
        all_data = self.get_multi_timeframe_data(ticker_symbol)

        summaries = {}
        for timeframe, data in all_data.items():
            if not data.empty:
                indicators = self.calculate_technical_indicators(data)
                summaries[timeframe] = {
                    'entries': len(data),
                    'start_time': data['datetime'].min(),
                    'end_time': data['datetime'].max(),
                    'avg_close': data['close'].mean(),
                    'min_close': data['close'].min(),
                    'max_close': data['close'].max(),
                    'total_volume': data['volume'].sum(),
                    'avg_volume': data['volume'].mean(),
                    'zero_volume_count': len(data[data['volume'] == 0]),
                    'indicators': indicators
                }

        # Also include the real-time data if available
        with self.lock:
            if ticker_symbol in self.real_time_quotes:
                rt_data = self.real_time_quotes[ticker_symbol]
                summaries['real_time'] = {
                    'last_price': rt_data.get('price'),
                    'last_update': rt_data.get('datetime'),
                    'day_high': rt_data.get('high'),
                    'day_low': rt_data.get('low'),
                    'volume': rt_data.get('volume')
                }

        return {
            'ticker': ticker_symbol,
            'timeframes': summaries
        }

    def get_all_summaries(self):
        """Get summaries for all stocks in watchlist."""
        summaries = {}
        for ticker in self.watchlist:
            summaries[ticker] = self.get_summary(ticker)
        return summaries

    def get_latest_prices(self):
        """Get the latest price for each stock in watchlist across all timeframes."""
        latest_prices = {}

        for ticker in self.watchlist:
            # First check if we have real-time data
            with self.lock:
                if ticker in self.real_time_quotes:
                    rt_data = self.real_time_quotes[ticker]
                    latest_prices[ticker] = {
                        'real_time': {
                            'datetime': rt_data.get('datetime'),
                            'price': rt_data.get('price'),
                            'open': rt_data.get('open'),
                            'high': rt_data.get('high'),
                            'low': rt_data.get('low'),
                            'volume': rt_data.get('volume')
                        }
                    }

                    # Also include data for all timeframes
                    for tf_name, tf_config in self.timeframes.items():
                        if ticker in self.stock_data and tf_name in self.stock_data[ticker]:
                            df = self.stock_data[ticker][tf_name]
                            if not df.empty:
                                latest_row = df.iloc[-1]
                                latest_prices[ticker][tf_name] = {
                                    'datetime': latest_row['datetime'],
                                    'open': latest_row['open'],
                                    'high': latest_row['high'],
                                    'low': latest_row['low'],
                                    'price': latest_row['close'],
                                    'volume': latest_row['volume']
                                }
                else:
                    # Fallback to yfinance if no real-time data
                    try:
                        ticker_obj = yf.Ticker(ticker)
                        live_data = ticker_obj.history(period="1d", interval="1m").tail(1)

                        if not live_data.empty:
                            # Format the live data
                            latest_row = live_data.iloc[0]
                            live_price = {
                                'datetime': live_data.index[0],
                                'open': latest_row['Open'],
                                'high': latest_row['High'],
                                'low': latest_row['Low'],
                                'price': latest_row['Close'],
                                'volume': latest_row['Volume']
                            }

                            latest_prices[ticker] = {'yfinance_live': live_price}

                            # Also get data for all timeframes
                            all_data = self.get_multi_timeframe_data(ticker)
                            for timeframe, data in all_data.items():
                                if not data.empty:
                                    tf_latest_row = data.iloc[-1]
                                    latest_prices[ticker][timeframe] = {
                                        'datetime': tf_latest_row['datetime'],
                                        'open': tf_latest_row['open'],
                                        'high': tf_latest_row['high'],
                                        'low': tf_latest_row['low'],
                                        'price': tf_latest_row['close'],
                                        'volume': tf_latest_row['volume']
                                    }
                    except Exception as e:
                        logger.error(f"Error getting latest price for {ticker}: {e}")
                        # Fallback to our stored data
                        all_data = self.get_multi_timeframe_data(ticker)
                        ticker_prices = {}
                        for timeframe, data in all_data.items():
                            if not data.empty:
                                latest_row = data.iloc[-1]
                                ticker_prices[timeframe] = {
                                    'datetime': latest_row['datetime'],
                                    'open': latest_row['open'],
                                    'high': latest_row['high'],
                                    'low': latest_row['low'],
                                    'price': latest_row['close'],
                                    'volume': latest_row['volume']
                                }

                        if ticker_prices:
                            latest_prices[ticker] = ticker_prices

        return latest_prices

    def set_timeframes(self, timeframes):
        """Set custom timeframes configuration."""
        self.timeframes = timeframes
        logger.info(f"Timeframes updated: {timeframes}")

        # Re-derive timeframe data for all tickers
        for ticker in self.watchlist:
            if ticker in self.stock_data and self.base_timeframe in self.stock_data[ticker]:
                self._update_derived_timeframes(ticker)
            else:
                # If we don't have base timeframe data, need to initialize
                self._initialize_ticker_data(ticker)

    def refresh_data(self, ticker_symbol=None):
        """
        Refresh data for a specific ticker or all tickers in watchlist.
        Uses efficient caching when possible.
        """
        tickers_to_refresh = [ticker_symbol] if ticker_symbol else list(self.watchlist)

        for ticker in tickers_to_refresh:
            try:
                # Only initialize if we don't have any data or data is stale
                should_refresh = False

                with self.lock:
                    if ticker not in self.stock_data or self.base_timeframe not in self.stock_data[ticker]:
                        should_refresh = True
                    else:
                        # Check if data is stale (more than 10 minutes old)
                        base_data = self.stock_data[ticker][self.base_timeframe]
                        if not base_data.empty:
                            latest_time = base_data['datetime'].max()
                            if datetime.now(pytz.UTC) - latest_time > timedelta(minutes=10):
                                should_refresh = True

                if should_refresh:
                    logger.info(f"Refreshing data for {ticker}")
                    self._initialize_ticker_data(ticker)

            except Exception as e:
                logger.error(f"Error refreshing data for {ticker}: {e}")

    async def refresh_all_data_async(self):
        """Asynchronously refresh data for all tickers in the watchlist."""
        tickers = list(self.watchlist)

        # Add delay between ticker refreshes to avoid overwhelming the API
        delay_seconds = 1

        for ticker in tickers:
            try:
                # Check if we should refresh
                should_refresh = False

                with self.lock:
                    if ticker not in self.stock_data or self.base_timeframe not in self.stock_data[ticker]:
                        should_refresh = True
                    else:
                        # Check if data is stale (more than 10 minutes old)
                        base_data = self.stock_data[ticker][self.base_timeframe]
                        if not base_data.empty:
                            latest_time = base_data['datetime'].max()
                            if datetime.now(pytz.UTC) - latest_time > timedelta(minutes=10):
                                should_refresh = True

                if should_refresh:
                    logger.info(f"Refreshing data for {ticker}")
                    # Add refresh task to queue
                    self.message_queue.put({'type': 'refresh_data', 'ticker': ticker})

                    # Sleep briefly between refreshes to avoid rate limits
                    await asyncio.sleep(delay_seconds)

            except Exception as e:
                logger.error(f"Error scheduling refresh for {ticker}: {e}")

    def get_company_info(self, ticker_symbol):
        """Get company information using yfinance."""
        try:
            ticker_obj = yf.Ticker(ticker_symbol)
            info = ticker_obj.info

            # Handle case where info may be empty or None
            if not info:
                return {"error": "No company information available"}

            # Extract relevant company information
            company_info = {
                'name': info.get('shortName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'dividend_yield': info.get('dividendYield', 'N/A'),
                'beta': info.get('beta', 'N/A'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
                'avg_volume': info.get('averageVolume', 'N/A'),
                'description': info.get('longBusinessSummary', 'N/A')
            }

            return company_info

        except Exception as e:
            logger.error(f"Error getting company info for {ticker_symbol}: {e}")
            return {"error": f"Unable to retrieve company information: {str(e)}"}

    def get_earnings_dates(self, ticker_symbol):
        """Get upcoming earnings dates using yfinance."""
        try:
            ticker_obj = yf.Ticker(ticker_symbol)
            earnings_dates = ticker_obj.earnings_dates

            # Handle case where earnings_dates may be None
            if earnings_dates is None:
                return []

            if not earnings_dates.empty:
                # Format the earnings dates
                earnings_list = []
                for date, row in earnings_dates.iterrows():
                    earnings_list.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'estimate': row.get('EPS Estimate', 'N/A'),
                        'actual': row.get('Reported EPS', 'N/A'),
                        'surprise': row.get('Surprise(%)', 'N/A')
                    })
                return earnings_list
            else:
                return []

        except Exception as e:
            logger.error(f"Error getting earnings dates for {ticker_symbol}: {e}")
            return []

    def check_ticker_validity(self, ticker_symbol):
        """Check if a ticker symbol is valid and active."""
        try:
            ticker_obj = yf.Ticker(ticker_symbol)
            info = ticker_obj.info

            # Check if we got valid info back
            if info and 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                return {
                    'valid': True,
                    'name': info.get('shortName', ticker_symbol),
                    'price': info.get('regularMarketPrice', 'N/A')
                }
            else:
                # Try to get just recent history as another validation method
                history = ticker_obj.history(period="1d")
                if history.empty:
                    return {
                        'valid': False,
                        'reason': 'No price data available'
                    }
                else:
                    return {
                        'valid': True,
                        'name': ticker_symbol,
                        'price': history['Close'].iloc[-1] if not history.empty else 'N/A'
                    }

        except Exception as e:
            logger.error(f"Error checking ticker validity for {ticker_symbol}: {e}")
            return {
                'valid': False,
                'reason': str(e)
            }

    def get_available_timeframes(self):
        """Return the available timeframes configuration."""
        return {name: config['interval'] for name, config in self.timeframes.items()}

    def clear_cache(self):
        """Clear the API response cache."""
        with self.lock:
            self.api_cache.clear()
            self.cache_expiry.clear()
        logger.info("API cache cleared")
        return True

    def is_live_ticker_running(self):
        """Check if the live ticker is running."""
        return self.live_ticker_running

    def get_real_time_stats(self):
        """Get statistics about real-time data collection."""
        stats = {
            'live_ticker_running': self.live_ticker_running,
            'tickers_with_real_time_data': 0,
            'total_ticks_received': 0,
            'latest_tick_times': {}
        }

        with self.lock:
            # Count tickers with real-time data
            tickers_with_data = 0
            total_ticks = 0

            for ticker, ticks in self.tick_data.items():
                if ticks:
                    tickers_with_data += 1
                    total_ticks += len(ticks)

                    # Get the latest tick time
                    if ticks:
                        latest_tick = ticks[-1]
                        stats['latest_tick_times'][ticker] = latest_tick.get('datetime')

            stats['tickers_with_real_time_data'] = tickers_with_data
            stats['total_ticks_received'] = total_ticks

        return stats

    def close(self):
        """Clean up resources."""
        logger.info("Shutting down YahooMultiStockCollector...")

        # Flag threads to stop
        self.live_ticker_running = False

        # Wait for threads to finish
        if self.cache_thread and self.cache_thread.is_alive():
            self.cache_thread.join(timeout=1)

        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1)

        if self.live_ticker_thread and self.live_ticker_thread.is_alive():
            self.live_ticker_thread.join(timeout=1)

        if self.bar_aggregation_thread and self.bar_aggregation_thread.is_alive():
            self.bar_aggregation_thread.join(timeout=1)

        logger.info("YahooMultiStockCollector closed successfully")