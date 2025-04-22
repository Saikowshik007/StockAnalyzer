import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional
import logging
logger = logging.getLogger(__name__)
class TalibPatternRecognition:
    def __init__(self):
        self.pattern_functions = {
            'Two Crows': talib.CDL2CROWS,
            'Three Black Crows': talib.CDL3BLACKCROWS,
            'Three Inside Up/Down': talib.CDL3INSIDE,
            'Three Line Strike': talib.CDL3LINESTRIKE,
            'Three Outside Up/Down': talib.CDL3OUTSIDE,
            'Three Stars In The South': talib.CDL3STARSINSOUTH,
            'Three Advancing White Soldiers': talib.CDL3WHITESOLDIERS,
            'Abandoned Baby': talib.CDLABANDONEDBABY,
            'Advance Block': talib.CDLADVANCEBLOCK,
            'Belt-hold': talib.CDLBELTHOLD,
            'Breakaway': talib.CDLBREAKAWAY,
            'Closing Marubozu': talib.CDLCLOSINGMARUBOZU,
            'Concealing Baby Swallow': talib.CDLCONCEALBABYSWALL,
            'Counterattack': talib.CDLCOUNTERATTACK,
            'Dark Cloud Cover': talib.CDLDARKCLOUDCOVER,
            'Doji': talib.CDLDOJI,
            'Doji Star': talib.CDLDOJISTAR,
            'Dragonfly Doji': talib.CDLDRAGONFLYDOJI,
            'Engulfing Pattern': talib.CDLENGULFING,
            'Evening Doji Star': talib.CDLEVENINGDOJISTAR,
            'Evening Star': talib.CDLEVENINGSTAR,
            'Up/Down-gap side-by-side white lines': talib.CDLGAPSIDESIDEWHITE,
            'Gravestone Doji': talib.CDLGRAVESTONEDOJI,
            'Hammer': talib.CDLHAMMER,
            'Hanging Man': talib.CDLHANGINGMAN,
            'Harami Pattern': talib.CDLHARAMI,
            'Harami Cross Pattern': talib.CDLHARAMICROSS,
            'High-Wave Candle': talib.CDLHIGHWAVE,
            'Hikkake Pattern': talib.CDLHIKKAKE,
            'Modified Hikkake Pattern': talib.CDLHIKKAKEMOD,
            'Homing Pigeon': talib.CDLHOMINGPIGEON,
            'Identical Three Crows': talib.CDLIDENTICAL3CROWS,
            'In-Neck Pattern': talib.CDLINNECK,
            'Inverted Hammer': talib.CDLINVERTEDHAMMER,
            'Kicking': talib.CDLKICKING,
            'Kicking by Length': talib.CDLKICKINGBYLENGTH,
            'Ladder Bottom': talib.CDLLADDERBOTTOM,
            'Long Legged Doji': talib.CDLLONGLEGGEDDOJI,
            'Long Line Candle': talib.CDLLONGLINE,
            'Marubozu': talib.CDLMARUBOZU,
            'Matching Low': talib.CDLMATCHINGLOW,
            'Mat Hold': talib.CDLMATHOLD,
            'Morning Doji Star': talib.CDLMORNINGDOJISTAR,
            'Morning Star': talib.CDLMORNINGSTAR,
            'On-Neck Pattern': talib.CDLONNECK,
            'Piercing Line': talib.CDLPIERCING,
            'Rickshaw Man': talib.CDLRICKSHAWMAN,
            'Rising/Falling Three Methods': talib.CDLRISEFALL3METHODS,
            'Separating Lines': talib.CDLSEPARATINGLINES,
            'Shooting Star': talib.CDLSHOOTINGSTAR,
            'Short Line Candle': talib.CDLSHORTLINE,
            'Spinning Top': talib.CDLSPINNINGTOP,
            'Stalled Pattern': talib.CDLSTALLEDPATTERN,
            'Stick Sandwich': talib.CDLSTICKSANDWICH,
            'Takuri': talib.CDLTAKURI,
            'Tasuki Gap': talib.CDLTASUKIGAP,
            'Thrusting Pattern': talib.CDLTHRUSTING,
            'Tristar Pattern': talib.CDLTRISTAR,
            'Unique 3 River': talib.CDLUNIQUE3RIVER,
            'Upside Gap Two Crows': talib.CDLUPSIDEGAP2CROWS,
            'Upside/Downside Gap Three Methods': talib.CDLXSIDEGAP3METHODS
        }
        # Priority-based pattern categorization
        self.pattern_priorities = {
            1: ['Engulfing Pattern', 'Morning Star', 'Evening Star', 'Abandoned Baby',
                'Three Advancing White Soldiers', 'Three Black Crows'],
            2: ['Hammer', 'Shooting Star', 'Hanging Man', 'Inverted Hammer',
                'Dark Cloud Cover', 'Piercing Line'],
            3: ['Mat Hold', 'Three Line Strike', 'Tasuki Gap', 'Separating Lines'],
            4: ['Doji', 'Spinning Top', 'Harami Pattern', 'Harami Cross Pattern'],
            5: ['High-Wave Candle', 'Long Legged Doji', 'Rickshaw Man', 'Marubozu']
        }

    def detect_patterns(self, data: pd.DataFrame, lookback_periods: int = 10) -> Dict[str, List]:
        """
        Detect all candlestick patterns in the data.

        Parameters:
        - data: DataFrame containing OHLC data
        - lookback_periods: Number of recent candles to consider for pattern detection (default: 10)
        """
        if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            logger.error("Data must contain 'open', 'high', 'low', 'close' columns")
            return {}

        # Get only the recent data based on lookback_periods
        # Make sure we have enough data for TA-Lib functions
        min_periods = max(lookback_periods, 5)  # Some patterns need more historical data
        if len(data) > min_periods:
            # Keep more data for calculation but focus on recent periods for detection
            recent_data = data.tail(min_periods).copy()
        else:
            recent_data = data.copy()

        detected_patterns = {}

        for pattern_name, pattern_func in self.pattern_functions.items():
            try:
                # Convert data to numpy arrays
                open_prices = recent_data['open'].values.astype(float)
                high_prices = recent_data['high'].values.astype(float)
                low_prices = recent_data['low'].values.astype(float)
                close_prices = recent_data['close'].values.astype(float)

                # Detect pattern
                result = pattern_func(open_prices, high_prices, low_prices, close_prices)

                # Find where pattern is detected (non-zero values)
                # But only consider the last 'lookback_periods' candles for actual signals
                pattern_indices = np.where(result[-lookback_periods:] != 0)[0]

                # Adjust indices to refer to the correct position in the result array
                pattern_indices = pattern_indices + (len(result) - lookback_periods)

                if len(pattern_indices) > 0:
                    detected_patterns[pattern_name] = [
                        {
                            'index': int(idx),
                            'timestamp': recent_data.index[idx - (len(result) - len(recent_data))],
                            'signal': int(result[idx]),  # 100, -100, etc.
                            'price': float(close_prices[idx - (len(result) - len(recent_data))]),
                            'priority': self._get_pattern_priority(pattern_name)
                        }
                        for idx in pattern_indices
                    ]
            except Exception as e:
                logger.error(f"Error detecting {pattern_name}: {e}")

        return detected_patterns

    def _get_pattern_priority(self, pattern_name: str) -> int:
        """Get the priority of a pattern based on its reliability."""
        for priority, patterns in self.pattern_priorities.items():
            if pattern_name in patterns:
                return priority
        return 6  # Default priority for unlisted patterns

    def get_trading_signal(self, pattern_name: str, signal_value: int, current_price: float,
                           atr: Optional[float] = None, volume_ratio: float = 1.0,
                           additional_indicators: Dict = None) -> Dict:
        """Generate enhanced trading signal based on pattern and additional context."""
        signal = {
            'action': None,
            'reason': '',
            'confidence': 'medium',
            'entry_price': current_price,
            'stop_loss': None,
            'take_profit': None,
            'pattern_type': None,
            'risk_reward_ratio': None,
            'additional_context': {}
        }

        # Categorize pattern
        pattern_type = self._categorize_pattern(pattern_name)
        signal['pattern_type'] = pattern_type

        # Calculate adaptive stop loss and take profit based on ATR
        if atr:
            stop_distance = atr * 2  # 2 ATR for stop loss
            profit_distance = atr * 3  # 3 ATR for take profit
        else:
            # Default to percentage-based
            stop_distance = current_price * 0.02
            profit_distance = current_price * 0.05

        # Adjust confidence based on signal strength, volume, and additional indicators
        confidence_level = self._calculate_enhanced_confidence(
            signal_value, volume_ratio, pattern_name, additional_indicators
        )
        signal['confidence'] = confidence_level

        # Add additional context
        if additional_indicators:
            signal['additional_context'] = additional_indicators

        # Generate trading signal
        if signal_value > 0:  # Bullish signal
            if pattern_type == 'bullish_reversal' or pattern_type == 'bullish_continuation':
                signal['action'] = 'BUY'
                signal['reason'] = f'Bullish {pattern_name} pattern detected'
                signal['stop_loss'] = current_price - stop_distance
                signal['take_profit'] = current_price + profit_distance
            elif pattern_type == 'neutral':
                signal['action'] = 'HOLD'
                signal['reason'] = f'Neutral pattern {pattern_name} - monitor for confirmation'
            else:
                signal['action'] = 'WATCH'
                signal['reason'] = f'Conflicting signal for {pattern_name}'

        elif signal_value < 0:  # Bearish signal
            if pattern_type == 'bearish_reversal' or pattern_type == 'bearish_continuation':
                signal['action'] = 'SELL'
                signal['reason'] = f'Bearish {pattern_name} pattern detected'
                signal['stop_loss'] = current_price + stop_distance
                signal['take_profit'] = current_price - profit_distance
            elif pattern_type == 'neutral':
                signal['action'] = 'HOLD'
                signal['reason'] = f'Neutral pattern {pattern_name} - monitor for confirmation'
            else:
                signal['action'] = 'WATCH'
                signal['reason'] = f'Conflicting signal for {pattern_name}'

        # Calculate risk-reward ratio
        if signal['stop_loss'] and signal['take_profit']:
            risk = abs(current_price - signal['stop_loss'])
            reward = abs(signal['take_profit'] - current_price)
            if risk > 0:
                signal['risk_reward_ratio'] = reward / risk

        return signal

    def _categorize_pattern(self, pattern_name: str) -> str:
        """Categorize the pattern type more accurately."""
        # Bullish reversal patterns
        if pattern_name in ['Morning Star', 'Morning Doji Star', 'Hammer',
                            'Inverted Hammer', 'Piercing Line',
                            'Three Advancing White Soldiers', 'Ladder Bottom',
                            'Concealing Baby Swallow', 'Unique 3 River']:
            return 'bullish_reversal'

        # Special cases for patterns that can be both bullish and bearish
        if pattern_name == 'Engulfing Pattern':
            return 'both_reversal'
        if pattern_name == 'Abandoned Baby':
            return 'both_reversal'

        # Bearish reversal patterns
        if pattern_name in ['Evening Star', 'Evening Doji Star', 'Hanging Man',
                            'Shooting Star', 'Dark Cloud Cover', 'Three Black Crows',
                            'Two Crows', 'Upside Gap Two Crows', 'Advance Block']:
            return 'bearish_reversal'

        # Bullish continuation patterns
        if pattern_name in ['Mat Hold', 'Three Line Strike', 'Tasuki Gap',
                            'Separating Lines', 'Kicking', 'Kicking by Length']:
            return 'bullish_continuation'

        # Special case for Rising/Falling Three Methods
        if pattern_name == 'Rising/Falling Three Methods':
            return 'both_continuation'

        # Bearish continuation patterns
        if pattern_name in ['Falling Three Methods', 'In-Neck Pattern',
                            'On-Neck Pattern', 'Thrusting Pattern']:
            return 'bearish_continuation'

        # Neutral patterns (indecision)
        if pattern_name in ['Doji', 'Spinning Top', 'Harami Pattern',
                            'Harami Cross Pattern', 'Long Legged Doji',
                            'Gravestone Doji', 'Dragonfly Doji', 'High-Wave Candle',
                            'Rickshaw Man']:
            return 'neutral'

        # Trend patterns
        if pattern_name in ['Marubozu', 'Long Line Candle', 'Belt-hold',
                            'Short Line Candle', 'Closing Marubozu']:
            return 'trend'

        # Undefined patterns
        return 'unknown'

    def _calculate_enhanced_confidence(self, signal_value: int, volume_ratio: float,
                                       pattern_name: str, additional_indicators: Dict = None) -> str:
        """Calculate enhanced confidence level based on multiple factors."""
        abs_signal = abs(signal_value)
        pattern_priority = self._get_pattern_priority(pattern_name)

        # Base confidence from signal strength
        if abs_signal >= 100:
            base_confidence = 0.8
        elif abs_signal >= 50:
            base_confidence = 0.6
        else:
            base_confidence = 0.4

        # Adjust for pattern priority
        priority_multiplier = 1.2 - (pattern_priority - 1) * 0.1  # Range: 1.2 to 0.7
        base_confidence *= priority_multiplier

        # Adjust for volume
        if volume_ratio > 1.5:
            base_confidence *= 1.2
        elif volume_ratio > 1.0:
            base_confidence *= 1.1
        elif volume_ratio < 0.8:
            base_confidence *= 0.9

        # Adjust for additional indicators if provided
        if additional_indicators:
            indicator_score = self._evaluate_indicators(additional_indicators)
            base_confidence *= indicator_score

        # Convert to confidence level
        if base_confidence > 0.85:
            return 'very_high'
        elif base_confidence > 0.7:
            return 'high'
        elif base_confidence > 0.55:
            return 'medium_high'
        elif base_confidence > 0.4:
            return 'medium'
        else:
            return 'low'

    def _evaluate_indicators(self, indicators: Dict) -> float:
        """Evaluate additional indicators to adjust confidence."""
        score = 1.0

        # RSI confirmation
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if rsi < 30 or rsi > 70:  # Oversold or overbought
                score *= 1.2
            elif 40 < rsi < 60:  # Neutral
                score *= 0.9

        # MACD confirmation
        if 'macd' in indicators:
            macd_signal = indicators.get('macd_signal', 0)
            if macd_signal > 0:  # Bullish
                score *= 1.1
            elif macd_signal < 0:  # Bearish
                score *= 1.1

        # Moving average confirmation
        if 'ma_trend' in indicators:
            if indicators['ma_trend'] == 'bullish':
                score *= 1.15
            elif indicators['ma_trend'] == 'bearish':
                score *= 1.15
            else:
                score *= 0.9

        return min(score, 1.5)  # Cap the maximum adjustment