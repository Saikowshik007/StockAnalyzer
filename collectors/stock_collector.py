import logging
import pandas as pd
import threading
import talib
import time
import json
from datetime import datetime, timedelta
import pytz
from queue import Queue
from typing import List, Dict, Optional
import asyncio
from polygon import WebSocketClient, RESTClient
from polygon.websocket.models import WebSocketMessage, EquityAgg, EquityTrade, EquityQuote
logger = logging.getLogger(__name__)
pd.set_option('future.no_silent_downcasting', True)
class PolygonMultiStockCollector:
    def __init__(self, api_key: str, db_manager=None):
        """
        Initialize Polygon Multi-Timeframe Stock Collector.
            Args:
                api_key: Polygon.io API key
                db_manager: Optional database manager for persistence
            """
        self.api_key = api_key
        self.watchlist = set()
        self.db_manager = db_manager
        self.lock = threading.Lock()

        # Initialize REST and WebSocket clients
        self.rest_client = RESTClient(api_key)
        self.ws_client = None
        self.ws_connected = False
        self.ws_thread = None

        # Data storage for each ticker and timeframe
        self.stock_data = {}

        # Latest trades and quotes
        self.latest_trades = {}
        self.latest_quotes = {}

        # API cache
        self.api_cache = {}
        self.cache_expiry = {}

        # Rate limiting tracking
        self.request_count = 0
        self.request_reset_time = datetime.now() + timedelta(minutes=1)
        self.max_requests_per_minute = 5  # Conservative default

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

        # Flag for clean shutdown
        self.running = True

        # Load existing watchlist from DB if available
        if self.db_manager:
            self._load_watchlist_from_db()

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

    def connect_websocket(self):
        """Connect to Polygon.io websocket."""
        if self.ws_client:
            logger.info("WebSocket already connected")
            return

        try:
            # Create websocket client using official SDK
            self.ws_client = WebSocketClient(
                api_key=self.api_key,
                feed="sip",  # Use SIP feed for higher quality data
                market="stocks"
            )

            # Start the websocket thread
            self.ws_thread = threading.Thread(
                target=self.ws_client.run,
                args=(self._handle_ws_message,),
                daemon=True
            )
            self.ws_thread.start()

            # Mark as connected (we assume it will connect successfully)
            self.ws_connected = True
            logger.info("WebSocket connection started")

            # Wait a moment for connection to establish before subscribing
            time.sleep(2)

            # Subscribe to the watchlist
            self._subscribe_watchlist()

        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {e}")
            self.ws_connected = False

    def _monitor_connection(self):
        """Monitor and maintain websocket connection."""
        reconnect_delay = 5  # seconds
        max_reconnect_delay = 300  # 5 minutes

        while self.running:
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

                # Wait before next check
                time.sleep(30)

    def _check_subscriptions(self):
        """Check if we have active subscriptions."""
        # This is a simplified check based on watchlist size
        return len(self.watchlist) > 0 and self.ws_connected

    def _handle_ws_message(self, msgs: List[WebSocketMessage]):
        """Process messages from the WebSocket."""
        try:
            for msg in msgs:
                if isinstance(msg, EquityAgg):
                    self._process_agg(msg)
                elif isinstance(msg, EquityTrade):
                    self._process_trade(msg)
                elif isinstance(msg, EquityQuote):
                    self._process_quote(msg)
        except Exception as e:
            logger.error(f"Error processing websocket message: {e}")

    def _process_agg(self, agg: EquityAgg):
        """Process a single aggregate bar message."""
        ticker = agg.symbol

        # Skip if not in our watchlist
        if ticker not in self.watchlist:
            return

        # Convert to our internal format
        timestamp = datetime.fromtimestamp(agg.start_timestamp / 1000.0, tz=pytz.UTC)

        bar_data = {
            'datetime': timestamp,
            'open': agg.open,
            'high': agg.high,
            'low': agg.low,
            'close': agg.close,
            'volume': agg.volume
        }

        # Update our internal data structure
        self._update_ticker_data(ticker, bar_data)

    def _process_trade(self, trade: EquityTrade):
        """Process a single trade message."""
        ticker = trade.symbol

        # Skip if not in our watchlist
        if ticker not in self.watchlist:
            return

        # Update latest trade information
        timestamp = datetime.fromtimestamp(trade.timestamp / 1000.0, tz=pytz.UTC)

        with self.lock:
            self.latest_trades[ticker] = {
                'price': trade.price,
                'size': trade.size,
                'timestamp': timestamp,
                'exchange': trade.exchange,
                'conditions': trade.conditions
            }

    def _process_quote(self, quote: EquityQuote):
        """Process a single quote message."""
        ticker = quote.symbol

        # Skip if not in our watchlist
        if ticker not in self.watchlist:
            return

        # Update latest quote information
        timestamp = datetime.fromtimestamp(quote.timestamp / 1000.0, tz=pytz.UTC)

        with self.lock:
            self.latest_quotes[ticker] = {
                'bid_price': quote.bid_price,
                'bid_size': quote.bid_size,
                'ask_price': quote.ask_price,
                'ask_size': quote.ask_size,
                'timestamp': timestamp,
                'exchange': quote.bid_exchange  # or ask_exchange
            }

    def _manage_cache(self):
        """Background thread to manage cache expiry."""
        while self.running:
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
        """Update internal data structure with new bar data."""
        with self.lock:
            if ticker not in self.stock_data:
                self.stock_data[ticker] = {}

            # Update base timeframe data
            if self.base_timeframe not in self.stock_data[ticker]:
                self.stock_data[ticker][self.base_timeframe] = pd.DataFrame(columns=[
                    'datetime', 'open', 'high', 'low', 'close', 'volume'
                ])

            # Add new data point to base timeframe
            df = self.stock_data[ticker][self.base_timeframe]

            # Check if this bar is already in our data
            existing_bar = df[df['datetime'] == bar_data['datetime']]
            if len(existing_bar) == 0:
                # Add new bar
                new_row = pd.DataFrame([bar_data])
                self.stock_data[ticker][self.base_timeframe] = pd.concat([df, new_row], ignore_index=True)
            else:
                # Update existing bar
                idx = df[df['datetime'] == bar_data['datetime']].index[0]
                df.at[idx, 'high'] = max(df.at[idx, 'high'], bar_data['high'])
                df.at[idx, 'low'] = min(df.at[idx, 'low'], bar_data['low'])
                df.at[idx, 'close'] = bar_data['close']
                df.at[idx, 'volume'] += bar_data['volume']
                self.stock_data[ticker][self.base_timeframe] = df

            # Limit the size of stored data
            if len(self.stock_data[ticker][self.base_timeframe]) > 10000:
                self.stock_data[ticker][self.base_timeframe] = self.stock_data[ticker][self.base_timeframe].tail(10000)

            # Sort by datetime to ensure correct order
            self.stock_data[ticker][self.base_timeframe] = self.stock_data[ticker][self.base_timeframe].sort_values('datetime')

            # Update derived timeframes
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

    def _subscribe_watchlist(self):
        """Subscribe to all tickers in watchlist."""
        if not self.ws_connected or not self.watchlist:
            logger.warning("Cannot subscribe: not connected or empty watchlist")
            return

        # Create subscription channels
        channels = []

        for ticker in self.watchlist:
            # Subscribe to aggregates (A), trades (T), and quotes (Q)
            channels.extend([
                f"A.{ticker}",
                f"T.{ticker}",
                f"Q.{ticker}"
            ])

        if channels:
            try:
                self.ws_client.subscribe(*channels)
                logger.info(f"Subscribed to {len(channels)} channels for {len(self.watchlist)} tickers")
            except Exception as e:
                logger.error(f"Error subscribing to channels: {e}")

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
                    self.ws_client.subscribe(
                        f"A.{ticker_symbol}",
                        f"T.{ticker_symbol}",
                        f"Q.{ticker_symbol}"
                    )

                # Initialize historical data for this ticker
                self._initialize_ticker_data(ticker_symbol)

                logger.info(f"Added {ticker_symbol} to watchlist")
                return True
            else:
                logger.info(f"{ticker_symbol} already in watchlist")
                return False

    def _initialize_ticker_data(self, ticker_symbol):
        """Initialize historical data for a ticker symbol."""
        # Determine how much data we need based on the longest timeframe
        max_minutes = max(self.interval_to_minutes.get(tf_config['interval'], 1)
                          for tf_config in self.timeframes.values())

        # Calculate how many days of base timeframe data we need
        bars_needed = max_minutes * 100  # Get enough bars for indicators
        days_back = max(1, int(bars_needed / (6.5 * 60)))  # Trading hours per day

        try:
            # Fetch base timeframe data once
            data = self._fetch_historical_data_cached(
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

    def _fetch_historical_data_cached(self, ticker, interval, days_back):
        """Fetch historical data with caching to reduce API calls."""
        # Convert interval to polygon format
        polygon_multiplier, polygon_timespan = self._convert_interval_format(interval)

        # Create cache key
        cache_key = f"{ticker}_{polygon_multiplier}_{polygon_timespan}_{days_back}"

        # Return cached response if available
        with self.lock:
            if cache_key in self.api_cache:
                logger.info(f"Using cached data for {cache_key}")
                return self.api_cache[cache_key].copy()

        # Check rate limit before proceeding
        sleep_time = self._check_rate_limit()
        if sleep_time > 0:
            time.sleep(sleep_time)

        # Set date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Format dates for API
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        try:
            # Use the RESTClient from official SDK
            aggs = self.rest_client.get_aggs(
                ticker=ticker,
                multiplier=polygon_multiplier,
                timespan=polygon_timespan,
                from_=start_str,
                to=end_str,
                limit=50000
            )

            # Convert to our standard DataFrame format
            if aggs:
                df = pd.DataFrame([{
                    'datetime': datetime.fromtimestamp(a.timestamp / 1000.0, tz=pytz.UTC),
                    'open': a.open,
                    'high': a.high,
                    'low': a.low,
                    'close': a.close,
                    'volume': a.volume
                } for a in aggs])

                # Cache the result
                if not df.empty:
                    with self.lock:
                        self.api_cache[cache_key] = df.copy()
                        # Set expiry time (30 minutes)
                        self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=30)

                return df

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {e}")
            return pd.DataFrame()

    def _convert_interval_format(self, yf_interval):
        """Convert yfinance interval format to polygon.io format."""
        # yfinance: 1m, 2m, 5m, 15m, 30m, 60m, 1h, 1d, 5d, 1wk, 1mo
        # polygon: multiplier/timespan (minute, hour, day, week, month, quarter, year)

        if yf_interval == '1m':
            return 1, 'minute'
        elif yf_interval == '2m':
            return 2, 'minute'
        elif yf_interval == '5m':
            return 5, 'minute'
        elif yf_interval == '15m':
            return 15, 'minute'
        elif yf_interval == '30m':
            return 30, 'minute'
        elif yf_interval == '60m' or yf_interval == '1h':
            return 1, 'hour'
        elif yf_interval == '1d':
            return 1, 'day'
        elif yf_interval == '5d':
            return 5, 'day'
        elif yf_interval == '1wk':
            return 1, 'week'
        elif yf_interval == '1mo':
            return 1, 'month'
        else:
            # Default
            return 1, 'minute'

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
                    self.ws_client.unsubscribe(
                        f"A.{ticker_symbol}",
                        f"T.{ticker_symbol}",
                        f"Q.{ticker_symbol}"
                    )

                # Remove from our data storage
                if ticker_symbol in self.stock_data:
                    del self.stock_data[ticker_symbol]

                # Remove from latest trades and quotes
                if ticker_symbol in self.latest_trades:
                    del self.latest_trades[ticker_symbol]
                if ticker_symbol in self.latest_quotes:
                    del self.latest_quotes[ticker_symbol]

                logger.info(f"Removed {ticker_symbol} from watchlist")
                return True
            else:
                logger.info(f"{ticker_symbol} not in watchlist")
                return False

    def get_watchlist(self):
        """Return current watchlist."""
        return list(self.watchlist)

    def get_latest_trade(self, ticker_symbol):
        """Get latest trade for a specific ticker."""
        with self.lock:
            if ticker_symbol in self.latest_trades:
                return self.latest_trades[ticker_symbol].copy()
            return None

    def get_latest_quote(self, ticker_symbol):
        """Get latest quote for a specific ticker."""
        with self.lock:
            if ticker_symbol in self.latest_quotes:
                return self.latest_quotes[ticker_symbol].copy()
            return None

    def get_data(self, ticker_symbol, interval='5m'):
        """
        Get data for a specific ticker and interval.
        Retrieves from cache/memory first, then falls back to API when necessary.
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

        # If we don't have the data locally, fetch it from the API
        return self._fetch_historical_data_cached(
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
            data[tf_name] = timeframe_data

        return data

    def calculate_technical_indicators(self, data):
        """Calculate technical indicators for the data."""
        if data is None or data.empty:
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

        # Add latest trade and quote information
        latest_trade = self.get_latest_trade(ticker_symbol)
        latest_quote = self.get_latest_quote(ticker_symbol)

        return {
            'ticker': ticker_symbol,
            'timeframes': summaries,
            'latest_trade': latest_trade,
            'latest_quote': latest_quote
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

            # Add the latest trade if available
            latest_trade = self.get_latest_trade(ticker)
            if latest_trade:
                ticker_prices['realtime'] = {
                    'datetime': latest_trade['timestamp'],
                    'price': latest_trade['price'],
                    'size': latest_trade['size']
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
        self.running = False

        if self.ws_client:
            self.ws_client.close()
            self.ws_connected = False
            logger.info("WebSocket connection closed")

        # Wait for threads to finish
        if hasattr(self, 'cache_thread') and self.cache_thread and self.cache_thread.is_alive():
            self.cache_thread.join(timeout=1)

        if hasattr(self, 'monitor_thread') and self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1)

        logger.info("PolygonMultiStockCollector closed successfully")

