# collectors/yahoo_stock_collector.py
import logging
import pandas as pd
import threading
import talib
import websocket
import json
import time
from datetime import datetime, timedelta
import requests
import pytz
from queue import Queue
import asyncio

logger = logging.getLogger(__name__)
pd.set_option('future.no_silent_downcasting', True)

class YahooMultiStockCollector:
    def __init__(self, db_manager=None):
        self.base_url = "https://query1.finance.yahoo.com"
        self.ws_url = "wss://streamer.finance.yahoo.com"  # Yahoo Finance WebSocket URL
        self.watchlist = set()
        self.db_manager = db_manager
        self.lock = threading.Lock()

        # Data storage for each ticker and timeframe
        self.stock_data = {}

        # Cache for API responses
        self.api_cache = {}
        self.cache_expiry = {}

        # Track connection status
        self.connected = False

        # Rate limiting tracking
        self.request_count = 0
        self.request_reset_time = datetime.now() + timedelta(minutes=1)
        self.max_requests_per_minute = 5  # Conservative default

        # Websocket connection
        self.ws = None
        self.ws_connected = False
        self.message_queue = Queue()
        self.ws_thread = None
        self.socket_id = None  # Yahoo specific socket ID
        self.crumb = None  # Yahoo crumb for authentication

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

        # Base timeframe for data collection (collect once and derive others)
        self.base_timeframe = '1m'

        # Load existing watchlist from DB if available
        if self.db_manager:
            self._load_watchlist_from_db()

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_messages)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        # Start cache management thread
        self.cache_thread = threading.Thread(target=self._manage_cache)
        self.cache_thread.daemon = True
        self.cache_thread.start()

        # Connection monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_connection)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def _load_watchlist_from_db(self):
        """Load watchlist from database."""
        if self.db_manager:
            db_watchlist = self.db_manager.get_active_watchlist()
            with self.lock:
                self.watchlist = set(db_watchlist)
                logger.info(f"Loaded watchlist from database: {list(self.watchlist)}")

    def _get_yahoo_crumb(self):
        """Get Yahoo Finance crumb for authentication."""
        try:
            session = requests.Session()
            response = session.get("https://finance.yahoo.com")

            if response.status_code == 200:
                # Extract crumb from the response - this is simplified
                # In reality, you might need to parse the HTML to extract the crumb
                # or use a specific endpoint to get it
                text = response.text
                start_index = text.find('"crumb":"') + 9
                if start_index > 9:  # Found
                    end_index = text.find('"', start_index)
                    self.crumb = text[start_index:end_index]
                    logger.info(f"Got Yahoo crumb: {self.crumb}")
                    return self.crumb

            logger.error(f"Failed to get Yahoo crumb, status code: {response.status_code}")
            return None

        except Exception as e:
            logger.error(f"Error getting Yahoo crumb: {e}")
            return None

    def connect_websocket(self):
        """Connect to Yahoo Finance websocket with proper authentication."""
        # Make sure we have authentication
        if not hasattr(self, 'cookie') or not self.crumb:
            if not self._get_yahoo_auth():
                logger.error("Failed to authenticate with Yahoo Finance")
                return

        def on_open(ws):
            logger.info("WebSocket connection opened")
            self.ws_connected = True
            self.connected = True

            # Generate a unique socket ID
            self.socket_id = f"websocket_{int(time.time() * 1000)}"

            # Subscribe to watchlist tickers
            # self._subscribe_watchlist()

        def on_message(ws, message):
            # Add message to queue for processing
            self.message_queue.put(message)

        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
            self.ws_connected = False
            self.connected = False

        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
            self.ws_connected = False
            self.connected = False

        # Create websocket connection to Yahoo Finance
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
            "Cookie": f"{self.cookie.name}={self.cookie.value}"
        }

        # Add crumb to URL if available
        ws_url = f"{self.ws_url}"
        if self.crumb:
            ws_url += f"?crumb={self.crumb}"

        self.ws = websocket.WebSocketApp(
            ws_url,
            header=headers,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        # Start websocket in a separate thread
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

    def _monitor_connection(self):
        """Monitor and maintain websocket connection."""
        reconnect_delay = 5  # seconds
        max_reconnect_delay = 300  # 5 minutes

        while True:
            if not self.ws_connected:
                logger.info(f"Connection lost or not established. Reconnecting in {reconnect_delay} seconds...")
                time.sleep(reconnect_delay)
                try:
                    self.connect_websocket()
                    # Reset the reconnect delay upon successful connection
                    reconnect_delay = 5
                except Exception as e:
                    logger.error(f"Failed to reconnect: {e}")
                    # Exponential backoff, max 5 minutes
                    reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
            else:
                # Check if we need to resubscribe
                if self.ws_connected and not self._check_subscriptions():
                    logger.info("Resubscribing to watchlist tickers")
                    self._subscribe_watchlist()

                # Ping to keep connection alive
                try:
                    if self.ws_connected:
                        self.ws.send('ping')
                except Exception as e:
                    logger.error(f"Error sending ping: {e}")
                    self.ws_connected = False

                # Wait before next check
                time.sleep(30)

    def _check_subscriptions(self):
        """Check if we have active subscriptions."""
        # Simplified check - in a real system you might want to track subscription status
        return len(self.watchlist) > 0 and self.ws_connected

    def _process_messages(self):
        """Process messages from the websocket queue."""
        while True:
            try:
                if not self.message_queue.empty():
                    message = self.message_queue.get()
                    try:
                        # Yahoo sends data in different format than Polygon
                        # Parse and process accordingly
                        if message == 'ping':
                            # Respond to ping with pong
                            if self.ws_connected:
                                self.ws.send('pong')
                            continue

                        data = json.loads(message)
                        self._process_yahoo_message(data)

                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in message: {message}")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
            except Exception as e:
                logger.error(f"Error in message processing: {e}")

            # Small sleep to prevent CPU hogging
            time.sleep(0.01)

    def _process_yahoo_message(self, data):
        """Process Yahoo Finance WebSocket message."""
        try:
            # Yahoo messages have different structure than Polygon
            # Example format (simplified):
            # {'data': [{'id': 'AAPL', 'price': 200.50, 'time': timestamp, 'volume': 1000}]}

            if 'data' in data:
                for item in data['data']:
                    ticker = item.get('id')
                    if not ticker:
                        continue

                    timestamp_ms = item.get('time')
                    if not timestamp_ms:
                        continue

                    timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=pytz.UTC)

                    # Yahoo might send price updates that don't contain OHLC
                    # For real-time ticks, you might need to convert to OHLC yourself
                    # This is a simplified approach
                    price = item.get('price')
                    volume = item.get('volume', 0)

                    if price is None:
                        continue

                    # Create a simplified bar data point
                    bar_data = {
                        'datetime': timestamp,
                        'open': price,
                        'high': price,
                        'low': price,
                        'close': price,
                        'volume': volume
                    }

                    # Update our internal data structure
                    self._update_ticker_data(ticker, bar_data)

        except Exception as e:
            logger.error(f"Error processing Yahoo message: {e}, data: {data}")

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

                # Reset rate limit counter if needed
                if current_time > self.request_reset_time:
                    self.request_count = 0
                    self.request_reset_time = current_time + timedelta(minutes=1)

            except Exception as e:
                logger.error(f"Error in cache management: {e}")

            # Check every 10 seconds
            time.sleep(10)

    def _update_ticker_data(self, ticker, bar_data):
        """Update internal data structure with new bar data and derive higher timeframes."""
        with self.lock:
            if ticker not in self.stock_data:
                self.stock_data[ticker] = {}

            # For 1-minute data (base timeframe), just append
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

    def _check_rate_limit(self):
        """Check if we're approaching rate limit and back off if needed."""
        with self.lock:
            current_time = datetime.now()

            # Reset counter if needed
            if current_time > self.request_reset_time:
                self.request_count = 0
                self.request_reset_time = current_time + timedelta(minutes=1)

            # Check if we're at rate limit
            if self.request_count >= self.max_requests_per_minute:
                # Calculate sleep time until reset
                sleep_time = (self.request_reset_time - current_time).total_seconds()
                logger.warning(f"Rate limit approaching. Sleeping for {sleep_time} seconds")
                return sleep_time

            # Increment counter
            self.request_count += 1
            return 0

    def _subscribe_watchlist(self):
        """Subscribe to all tickers in watchlist."""
        if not self.ws_connected or not self.watchlist:
            logger.warning("Cannot subscribe: not connected or empty watchlist")
            return

        try:
            # Get list of tickers to subscribe to
            tickers = list(self.watchlist)

            # Create simple subscription message - no need for "ticker/" prefix
            subscribe_msg = {
                "subscribe": tickers
            }

            logger.info(f"Subscribing to {len(tickers)} ticker channels")
            self.ws.send(json.dumps(subscribe_msg))

        except Exception as e:
            logger.error(f"Error subscribing to watchlist: {e}")

    def add_stock(self, ticker_symbol):
        """Add a stock to the watchlist and persist to database."""
        with self.lock:
            if ticker_symbol not in self.watchlist:
                self.watchlist.add(ticker_symbol)
                # Also add to database if db_manager is available
                if self.db_manager:
                    self.db_manager.add_to_watchlist(ticker_symbol)
                # Subscribe to the new ticker if websocket is connected
                if self.ws_connected:
                    subscribe_msg = {
                        "subscribe": [ticker_symbol]
                    }

                    self.ws.send(json.dumps(subscribe_msg))
                # Initialize historical data for this ticker
                self._initialize_ticker_data(ticker_symbol)
                logger.info(f"Added {ticker_symbol} to watchlist")
                return True
            else:
                logger.info(f"{ticker_symbol} already in watchlist")
                return False

    def _initialize_ticker_data(self, ticker_symbol):
        """Initialize historical data for a ticker symbol using Yahoo Finance API."""
        # Determine how much data we need based on the longest timeframe
        max_minutes = max(self.interval_to_minutes.get(tf_config['interval'], 1)
                          for tf_config in self.timeframes.values())

        # Calculate how many days of 1-minute data we need
        bars_needed = max_minutes * 100  # Get enough bars for indicators
        days_back = max(1, int(bars_needed / (6.5 * 60)))  # Trading hours per day

        try:
            # Fetch 1-minute data (our base timeframe) once
            data = self._fetch_yahoo_historical_data(
                ticker_symbol,
                self.base_timeframe,
                days_back
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

        except Exception as e:
            logger.error(f"Error initializing data for {ticker_symbol}: {e}")

    def _fetch_yahoo_historical_data(self, ticker, interval, days_back):
        """Fetch historical data from Yahoo Finance API using proper authentication."""
        # Convert interval to Yahoo format if needed
        yahoo_interval = self._convert_to_yahoo_interval(interval)

        # Create cache key
        cache_key = f"{ticker}_{yahoo_interval}_{days_back}"

        # Return cached response if available
        with self.lock:
            if cache_key in self.api_cache:
                logger.info(f"Using cached data for {cache_key}")
                return self.api_cache[cache_key].copy()

        # Check rate limit before proceeding
        sleep_time = self._check_rate_limit()
        if sleep_time > 0:
            time.sleep(sleep_time)

        # Make sure we have authentication
        if not hasattr(self, 'cookie') or not self.crumb:
            if not self._get_yahoo_auth():
                logger.error("Failed to authenticate with Yahoo Finance")
                return None

        # Set date range
        end_ts = int(time.time())
        start_ts = end_ts - (days_back * 24 * 60 * 60)

        # Yahoo Finance API endpoint
        url = f"{self.base_url}/v7/finance/download/{ticker}"

        params = {
            "period1": start_ts,
            "period2": end_ts,
            "interval": yahoo_interval,
            "events": "history",
            "crumb": self.crumb
        }

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
            }

            response = requests.get(
                url,
                params=params,
                headers=headers,
                cookies={self.cookie.name: self.cookie.value}
            )

            if response.status_code != 200:
                logger.error(f"Error fetching Yahoo data: {response.status_code} - {response.text}")

                # If authentication failed, try to refresh auth
                if response.status_code in [401, 403, 429]:
                    logger.info("Attempting to refresh authentication...")
                    if self._get_yahoo_auth():
                        # Retry with new auth data
                        return self._fetch_yahoo_historical_data(ticker, interval, days_back)

                return None

            # Process CSV response
            content = response.text
            if ',' in content and 'Date' in content:
                # Parse CSV data
                lines = content.strip().split('\n')
                headers = lines[0].split(',')

                data_dict = {
                    "datetime": [],
                    "open": [],
                    "high": [],
                    "low": [],
                    "close": [],
                    "volume": []
                }

                for line in lines[1:]:
                    values = line.split(',')
                    if len(values) >= 6:
                        date_str = values[0]
                        try:
                            # Parse date
                            date = datetime.strptime(date_str, '%Y-%m-%d')
                            # Add data to dict
                            data_dict["datetime"].append(date)
                            data_dict["open"].append(float(values[1]) if values[1] != 'null' else None)
                            data_dict["high"].append(float(values[2]) if values[2] != 'null' else None)
                            data_dict["low"].append(float(values[3]) if values[3] != 'null' else None)
                            data_dict["close"].append(float(values[4]) if values[4] != 'null' else None)
                            data_dict["volume"].append(int(values[6]) if values[6] != 'null' and len(values) > 6 else 0)
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Error parsing line: {line}, error: {e}")

                df = pd.DataFrame(data_dict)
                df = df.dropna()

                # Cache the result
                if not df.empty:
                    with self.lock:
                        self.api_cache[cache_key] = df.copy()
                        # Set expiry time (30 minutes)
                        self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=30)

                return df

            logger.error("Unexpected response format from Yahoo Finance")
            return None

        except Exception as e:
            logger.error(f"Error fetching Yahoo data for {ticker}: {e}")
            return None

    def _convert_to_yahoo_interval(self, interval):
        """Convert our interval format to Yahoo Finance format."""
        # Our format: 1m, 2m, 5m, 15m, 30m, 60m, 1h, 1d, 5d, 1wk, 1mo
        # Yahoo format: 1m, 2m, 5m, 15m, 30m, 60m, 1h, 1d, 5d, 1wk, 1mo

        # Map common intervals
        mapping = {
            '1m': '1m',
            '2m': '2m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '60m': '60m',
            '1h': '1h',
            '1d': '1d',
            '5d': '5d',
            '1wk': '1wk',
            '1mo': '1mo'
        }

        return mapping.get(interval, '1m')  # Default to 1m if not found

    def remove_stock(self, ticker_symbol):
        """Remove a stock from the watchlist and update database."""
        with self.lock:
            if ticker_symbol in self.watchlist:
                self.watchlist.remove(ticker_symbol)

                # Also remove from database if db_manager is available
                if self.db_manager:
                    self.db_manager.remove_from_watchlist(ticker_symbol)

                # Unsubscribe from the ticker if websocket is connected
                if self.ws_connected:
                    unsubscribe_msg = {
                        "unsubscribe": [{
                            "subscribe": f"ticker/{ticker_symbol}"
                        }],
                        "socket_id": self.socket_id
                    }

                    if self.crumb:
                        unsubscribe_msg["crumb"] = self.crumb

                    self.ws.send(json.dumps(unsubscribe_msg))

                # Remove from our data storage
                if ticker_symbol in self.stock_data:
                    del self.stock_data[ticker_symbol]

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
        Retrieves from cache/memory first, then derives from base data if possible,
        finally falls back to API only when necessary.
        """
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

            # If we have base timeframe data, derive the requested interval
            if ticker_symbol in self.stock_data and self.base_timeframe in self.stock_data[ticker_symbol]:
                base_data = self.stock_data[ticker_symbol][self.base_timeframe]
                if not base_data.empty:
                    derived_data = self._aggregate_timeframe(base_data, interval)

                    # Cache this derived data for future use
                    if matching_tf:
                        self.stock_data[ticker_symbol][matching_tf] = derived_data

                    return derived_data

        # If we don't have the data locally, fetch it from the API as a last resort
        return self._fetch_yahoo_historical_data(
            ticker_symbol,
            interval,
            5  # Default to 5 days for ad-hoc requests
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
        """Asynchronously refresh data for all tickers in the watchlist with throttling."""
        tickers = list(self.watchlist)

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
                    # Create a task to run the initialization in background
                    asyncio.create_task(self._initialize_ticker_data_async(ticker))

                    # Sleep briefly between refreshes to avoid rate limits
                    await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error refreshing data for {ticker}: {e}")

    async def _initialize_ticker_data_async(self, ticker_symbol):
        """Async version of _initialize_ticker_data that runs in a thread pool."""
        loop = asyncio.get_event_loop()
        # Run the blocking _initialize_ticker_data in a thread pool
        await loop.run_in_executor(None, self._initialize_ticker_data, ticker_symbol)

    def close(self):
        """Close the websocket connection and clean up resources."""
        if self.ws:
            self.ws.close()
            logger.info("Websocket connection closed")

        # Wait for threads to finish
        if self.cache_thread and self.cache_thread.is_alive():
            self.cache_thread.join(timeout=1)

        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1)

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1)

        logger.info("YahooMultiStockCollector closed successfully")


    def _get_yahoo_auth(self):
        """Get Yahoo Finance cookie and crumb for authentication using a simpler approach."""
        try:
            # Setup headers with user agent
            user_agent_key = "User-Agent"
            user_agent_value = "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
            headers = {user_agent_key: user_agent_value}

            # Step 1: Get the cookie
            response = requests.get(
                "https://fc.yahoo.com", headers=headers, allow_redirects=True
            )

            if not response.cookies:
                logger.error("Failed to obtain Yahoo auth cookie")
                return False

            cookie = list(response.cookies)[0]

            # Step 2: Use the cookie to get the crumb
            crumb_response = requests.get(
                "https://query1.finance.yahoo.com/v1/test/getcrumb",
                headers=headers,
                cookies={cookie.name: cookie.value},
                allow_redirects=True,
            )

            if crumb_response.status_code != 200:
                logger.error(f"Failed to get crumb, status code: {crumb_response.status_code}")
                return False

            crumb = crumb_response.text

            if not crumb:
                logger.error("Empty crumb returned from Yahoo")
                return False

            # Store auth data for later use
            self.cookie = cookie
            self.crumb = crumb
            logger.info(f"Successfully obtained Yahoo auth: cookie={cookie.name}, crumb={crumb}")
            return True

        except Exception as e:
            logger.error(f"Error getting Yahoo authentication: {e}")
            return False