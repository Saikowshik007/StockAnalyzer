import logging
import yfinance as yf
import pandas as pd
import threading
import talib
logger = logging.getLogger(__name__)
pd.set_option('future.no_silent_downcasting', True)
class MultiStockCollector:
    def __init__(self,db_manager):

        self.watchlist = set()
        self.db_manager = db_manager
        self.lock = threading.Lock()
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
        if self.db_manager:
            self._load_watchlist_from_db()

    def _load_watchlist_from_db(self):
        """Load watchlist from database."""
        if self.db_manager:
            db_watchlist = self.db_manager.get_active_watchlist()
            with self.lock:
                self.watchlist = set(db_watchlist)
                logger.info(f"Loaded watchlist from database: {list(self.watchlist)}")

    def add_stock(self, ticker_symbol):
        """Add a stock to the watchlist and persist to database."""
        with self.lock:
            if ticker_symbol not in self.watchlist:
                self.watchlist.add(ticker_symbol)
                # Also add to database if db_manager is available
                if self.db_manager:
                    self.db_manager.add_to_watchlist(ticker_symbol)
                logger.info(f"Added {ticker_symbol} to watchlist")
                return True
            else:
                logger.info(f"{ticker_symbol} already in watchlist")
                return False

    def remove_stock(self, ticker_symbol):
        """Remove a stock from the watchlist and update database."""
        with self.lock:
            if ticker_symbol in self.watchlist:
                self.watchlist.remove(ticker_symbol)
                # Also remove from database if db_manager is available
                if self.db_manager:
                    self.db_manager.remove_from_watchlist(ticker_symbol)
                logger.info(f"Removed {ticker_symbol} from watchlist")
                return True
            else:
                logger.info(f"{ticker_symbol} not in watchlist")
                return False

    def get_watchlist(self):
        """Return current watchlist."""
        return list(self.watchlist)

    def get_data(self, ticker_symbol, window_minutes=None, interval='5m', period='1d', start=None, end=None):
        """
        Fetch recent data for a specific ticker using yfinance.
        Enhanced to support different intervals for multi-timeframe analysis.
        Parameters:
        - start: datetime object or string (YYYY-MM-DD HH:MM:SS)
        - end: datetime object or string (YYYY-MM-DD HH:MM:SS)
        """

        try:
            stock = yf.Ticker(ticker_symbol)

            if start and end:
                data = stock.history(start=start, end=end, interval=interval, prepost=True)
            else:
                # Determine period based on interval

                if interval == '1h' or interval == '60m':
                    period = '5d'
                else:
                    period = '1d'


                # Fetch data with prepost included to get more complete data
                data = stock.history(period=period, interval=interval, prepost=True)

            # Reset index and convert to the same format
            if not data.empty:
                data = data.reset_index()
                data = data.rename(columns={
                    'Datetime': 'datetime',
                    'Date': 'datetime',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })

                # Ensure datetime column is timezone-aware
                if 'datetime' in data.columns and data['datetime'].dt.tz is None:
                    data['datetime'] = data['datetime'].dt.tz_localize('UTC')

                # Fix zero volume issues
                data = self._fix_zero_volume(data, interval)

            return data
        except Exception as e:
            logger.error(f"Error fetching data for {ticker_symbol}: {e}")
            return pd.DataFrame()

    def _fix_zero_volume(self, data, interval):
        """Fix zero volume issues in the data."""
        if data.empty:
            return data

        # Check for zero volume rows
        zero_volume_rows = data[data['volume'] == 0]

        if not zero_volume_rows.empty:
            # logger.warning(f"Found {len(zero_volume_rows)} rows with zero volume")

            # For intraday data, try to fill zero volumes with interpolation
            if interval in ['2m', '5m','15m', '30m', '60m', '1h']:
                # First try forward fill, then backward fill for remaining
                data['volume'] = data['volume'].replace(0, pd.NA)
                data['volume'] = data['volume'].ffill().bfill().infer_objects(copy=False)

                # If still zeros, use average of nearby non-zero volumes
                if (data['volume'] == 0).any():
                    data['volume'] = data['volume'].replace(0, pd.NA)
                    data['volume'] = data['volume'].interpolate(method='linear', limit_direction='both')

                # As last resort, use a minimum threshold based on daily average
                daily_avg = data['volume'].mean()
                if daily_avg > 0:
                    min_volume = max(100, int(daily_avg * 0.01))  # At least 1% of average or 100
                    data.loc[data['volume'] < min_volume, 'volume'] = min_volume
        return data

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
                        'open' : latest_row['open'],
                        'high' : latest_row['high'],
                        'low' : latest_row['low'],
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