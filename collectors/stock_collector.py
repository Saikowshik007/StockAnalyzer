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

logger = logging.getLogger(__name__)
pd.set_option('future.no_silent_downcasting', True)

class PolygonMultiStockCollector:
    def __init__(self, api_key, db_manager=None):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.watchlist = set()
        self.db_manager = db_manager
        self.lock = threading.Lock()

        # Data storage for each ticker and timeframe
        self.stock_data = {}

        # Websocket connection
        self.ws = None
        self.ws_connected = False
        self.message_queue = Queue()
        self.ws_thread = None

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

        # Load existing watchlist from DB if available
        if self.db_manager:
            self._load_watchlist_from_db()

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_messages)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def _load_watchlist_from_db(self):
        """Load watchlist from database."""
        if self.db_manager:
            db_watchlist = self.db_manager.get_active_watchlist()
            with self.lock:
                self.watchlist = set(db_watchlist)
                logger.info(f"Loaded watchlist from database: {list(self.watchlist)}")

    def connect_websocket(self):
        """Connect to Polygon.io websocket."""
        def on_open(ws):
            logger.info("Websocket connection opened")
            self.ws_connected = True

            # Authenticate
            auth_msg = {
                "action": "auth",
                "params": self.api_key
            }
            ws.send(json.dumps(auth_msg))

            # Subscribe to current watchlist
            self._subscribe_watchlist()

        def on_message(ws, message):
            # Add message to queue for processing
            self.message_queue.put(message)

        def on_error(ws, error):
            logger.error(f"Websocket error: {error}")
            self.ws_connected = False

        def on_close(ws, close_status_code, close_msg):
            logger.info(f"Websocket connection closed: {close_status_code} - {close_msg}")
            self.ws_connected = False
            # Attempt to reconnect after a delay
            time.sleep(5)
            self.connect_websocket()

        # Create websocket connection
        self.ws = websocket.WebSocketApp(
            f"wss://socket.polygon.io/stocks",
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        # Start websocket in a separate thread
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

    def _process_messages(self):
        """Process messages from the websocket queue."""
        while True:
            try:
                if not self.message_queue.empty():
                    message = self.message_queue.get()
                    data = json.loads(message)

                    # Handle different message types
                    if isinstance(data, list):
                        for event in data:
                            self._process_single_event(event)
                    elif isinstance(data, dict):
                        if data.get("status") == "connected":
                            logger.info("Connected to Polygon.io websocket")
                        elif data.get("status") == "auth_success":
                            logger.info("Successfully authenticated with Polygon.io")
                        elif data.get("status") == "success" and data.get("message") == "subscribed":
                            logger.info(f"Successfully subscribed to: {data.get('subscriptions', [])}")
                        else:
                            self._process_single_event(data)
            except Exception as e:
                logger.error(f"Error processing websocket message: {e}")

            # Small sleep to prevent CPU hogging
            time.sleep(0.01)

    def _process_single_event(self, event):
        """Process a single event from Polygon websocket."""
        if event.get("ev") == "AM":  # Aggregate minute data
            # Process minute bar
            ticker = event.get("sym")
            timestamp = datetime.fromtimestamp(event.get("s") / 1000.0, tz=pytz.UTC)

            bar_data = {
                'datetime': timestamp,
                'open': event.get("o"),
                'high': event.get("h"),
                'low': event.get("l"),
                'close': event.get("c"),
                'volume': event.get("v")
            }

            # Update our internal data structure
            self._update_ticker_data(ticker, bar_data)

        elif event.get("ev") == "T":  # Trade event
            # Process trade data if needed
            pass

    def _update_ticker_data(self, ticker, bar_data):
        """Update internal data structure with new bar data."""
        with self.lock:
            if ticker not in self.stock_data:
                self.stock_data[ticker] = {}

            # Determine which timeframes this bar belongs to
            for tf_name, tf_config in self.timeframes.items():
                interval = tf_config['interval']

                # Initialize dataframe if it doesn't exist
                if tf_name not in self.stock_data[ticker]:
                    self.stock_data[ticker][tf_name] = pd.DataFrame(columns=[
                        'datetime', 'open', 'high', 'low', 'close', 'volume'
                    ])

                # Get the current minute data
                df = self.stock_data[ticker][tf_name]

                # For 1-minute data, just append
                if interval == '1m':
                    new_row = pd.DataFrame([bar_data])
                    self.stock_data[ticker][tf_name] = pd.concat([df, new_row], ignore_index=True)

                # For other intervals, we need to aggregate
                else:
                    minutes = self.interval_to_minutes.get(interval, 1)
                    timestamp = bar_data['datetime']

                    # Calculate the interval start time (round down to nearest interval)
                    interval_start = timestamp.replace(
                        minute=(timestamp.minute // minutes) * minutes,
                        second=0,
                        microsecond=0
                    )

                    # Check if we already have a bar for this interval
                    existing_bar = df[df['datetime'] == interval_start]

                    if existing_bar.empty:
                        # Create new bar
                        new_bar = {
                            'datetime': interval_start,
                            'open': bar_data['open'],
                            'high': bar_data['high'],
                            'low': bar_data['low'],
                            'close': bar_data['close'],
                            'volume': bar_data['volume']
                        }
                        new_row = pd.DataFrame([new_bar])
                        self.stock_data[ticker][tf_name] = pd.concat([df, new_row], ignore_index=True)
                    else:
                        # Update existing bar
                        idx = df[df['datetime'] == interval_start].index[0]
                        df.at[idx, 'high'] = max(df.at[idx, 'high'], bar_data['high'])
                        df.at[idx, 'low'] = min(df.at[idx, 'low'], bar_data['low'])
                        df.at[idx, 'close'] = bar_data['close']
                        df.at[idx, 'volume'] += bar_data['volume']
                        self.stock_data[ticker][tf_name] = df

                # Limit the size of stored data (keep last 1000 bars)
                if len(self.stock_data[ticker][tf_name]) > 1000:
                    self.stock_data[ticker][tf_name] = self.stock_data[ticker][tf_name].tail(1000)

    def _subscribe_watchlist(self):
        """Subscribe to all tickers in watchlist."""
        if not self.ws_connected or not self.watchlist:
            return

        # Create subscription message for minute aggregates
        ticker_channels = [f"AM.{ticker}" for ticker in self.watchlist]

        subscribe_msg = {
            "action": "subscribe",
            "params": ",".join(ticker_channels)
        }

        self.ws.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to {len(ticker_channels)} ticker channels")

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
                        "action": "subscribe",
                        "params": f"AM.{ticker_symbol}"
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
        """Initialize historical data for a ticker symbol."""
        # For each timeframe, fetch historical data
        for tf_name, tf_config in self.timeframes.items():
            interval = tf_config['interval']

            # Convert interval format from yfinance to polygon.io format
            polygon_multiplier, polygon_timespan = self._convert_interval_format(interval)

            # Determine how far back to look based on interval
            minutes = self.interval_to_minutes.get(interval, 1)
            days_back = max(1, int(minutes * 1000 / 1440))  # At least 1 day, more for longer intervals

            # Fetch historical data from Polygon REST API
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            try:
                data = self._fetch_historical_aggs(
                    ticker_symbol,
                    polygon_multiplier,
                    polygon_timespan,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )

                if data is not None and not data.empty:
                    # Store in our data structure
                    with self.lock:
                        if ticker_symbol not in self.stock_data:
                            self.stock_data[ticker_symbol] = {}
                        self.stock_data[ticker_symbol][tf_name] = data
            except Exception as e:
                logger.error(f"Error initializing data for {ticker_symbol} ({interval}): {e}")

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

    def _fetch_historical_aggs(self, ticker, multiplier, timespan, start_date, end_date):
        """Fetch historical aggregates from Polygon REST API."""
        endpoint = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
        params = {
            "apiKey": self.api_key,
            "sort": "asc",
            "limit": 50000
        }

        url = f"{self.base_url}{endpoint}"

        try:
            response = requests.get(url, params=params)

            if response.status_code != 200:
                logger.error(f"Error fetching historical data: {response.status_code} - {response.text}")
                return None

            data = response.json()

            if data["status"] != "OK" or "results" not in data:
                logger.error("No results found or API error")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data["results"])

            # Rename columns to match our expected format
            df = df.rename(columns={
                "v": "volume",
                "o": "open",
                "c": "close",
                "h": "high",
                "l": "low",
                "t": "timestamp"
            })

            # Convert timestamp from milliseconds to datetime
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

            # Select only the columns we need
            df = df[["datetime", "open", "high", "low", "close", "volume"]]

            return df

        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {e}")
            return None

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
                        "action": "unsubscribe",
                        "params": f"AM.{ticker_symbol}"
                    }
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
        Falls back to fetching from REST API if not available locally.
        """
        with self.lock:
            # Find the timeframe that matches this interval
            matching_tf = None
            for tf_name, tf_config in self.timeframes.items():
                if tf_config['interval'] == interval:
                    matching_tf = tf_name
                    break

            if matching_tf and ticker_symbol in self.stock_data and matching_tf in self.stock_data[ticker_symbol]:
                return self.stock_data[ticker_symbol][matching_tf].copy()

        # If we don't have the data locally, try to fetch it
        polygon_multiplier, polygon_timespan = self._convert_interval_format(interval)

        # Determine how far back to look based on interval
        minutes = self.interval_to_minutes.get(interval, 1)
        days_back = max(1, int(minutes * 1000 / 1440))  # At least 1 day, more for longer intervals

        # Fetch historical data from Polygon REST API
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        return self._fetch_historical_aggs(
            ticker_symbol,
            polygon_multiplier,
            polygon_timespan,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
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
                indicators['atr'] = talib.ATR(data['high'], data['low'], data['close']).iloc[-1]

            # Volume ratio
            if len(data) >= 10:
                recent_volume = data['volume'].tail(5).mean()
                prior_volume = data['volume'].tail(10).head(5).mean()
                indicators['volume_ratio'] = recent_volume / prior_volume if prior_volume > 0 else 1.0

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

        # Re-initialize data for all tickers with new timeframes
        for ticker in self.watchlist:
            self._initialize_ticker_data(ticker)

    def close(self):
        """Close the websocket connection."""
        if self.ws:
            self.ws.close()
            logger.info("Websocket connection closed")