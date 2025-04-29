import numpy as np
import pandas as pd
import talib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from io import BytesIO
import base64
from typing import Dict, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TalibPatternRecognition:
    """Enhanced pattern recognition class that builds on the TalibPatternRecognition class."""

    def __init__(self, pattern_recognizer=None):
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
        """Initialize the enhanced pattern recognition with an optional pattern recognizer."""
        self.pattern_recognizer = pattern_recognizer
        self.pattern_history = {}  # To track pattern success/failure rate

        # Define pattern success metrics thresholds
        self.success_thresholds = {
            'bullish_reversal': 0.02,  # 2% move up to consider success
            'bearish_reversal': 0.02,  # 2% move down to consider success
            'bullish_continuation': 0.015,  # 1.5% move up
            'bearish_continuation': 0.015,  # 1.5% move down
            'neutral': 0.01,  # 1% move either way
        }

        # Number of candles to look ahead for pattern validation
        self.validation_periods = {
            'bullish_reversal': 3,
            'bearish_reversal': 3,
            'bullish_continuation': 5,
            'bearish_continuation': 5,
            'neutral': 2,
        }

        # Pattern cluster definitions (patterns that reinforce each other)
        self.pattern_clusters = {
            'strong_reversal_bottom': [
                'Morning Star', 'Hammer', 'Piercing Line', 'Bullish Engulfing',
                'Three White Soldiers', 'Bullish Harami', 'Dragonfly Doji'
            ],
            'strong_reversal_top': [
                'Evening Star', 'Shooting Star', 'Dark Cloud Cover', 'Bearish Engulfing',
                'Three Black Crows', 'Bearish Harami', 'Gravestone Doji'
            ],
            'momentum_continuation': [
                'Three White Soldiers', 'Marubozu', 'Gap Up', 'Mat Hold',
                'Three Line Strike', 'Tasuki Gap'
            ],
            'exhaustion_warning': [
                'Doji', 'Spinning Top', 'High Wave', 'Long Legged Doji',
                'Gravestone Doji', 'Dragonfly Doji'
            ]
        }

        # Market context factors
        self.market_context_factors = [
            'trend_direction',
            'trend_strength',
            'volume_confirmation',
            'volatility_state',
            'support_resistance_proximity',
            'multi_timeframe_alignment',
            'previous_pattern_success'
        ]

        # Define better naming map for patterns
        self.pattern_display_names = {
            'CDL2CROWS': 'Two Crows',
            'CDL3BLACKCROWS': 'Three Black Crows',
            'CDL3INSIDE': 'Three Inside Up/Down',
            'CDL3LINESTRIKE': 'Three Line Strike',
            'CDL3OUTSIDE': 'Three Outside Up/Down',
            'CDL3STARSINSOUTH': 'Three Stars In The South',
            'CDL3WHITESOLDIERS': 'Three Advancing White Soldiers',
            'CDLABANDONEDBABY': 'Abandoned Baby',
            'CDLADVANCEBLOCK': 'Advance Block',
            'CDLBELTHOLD': 'Belt-hold',
            'CDLBREAKAWAY': 'Breakaway',
            'CDLCLOSINGMARUBOZU': 'Closing Marubozu',
            'CDLCONCEALBABYSWALL': 'Concealing Baby Swallow',
            'CDLCOUNTERATTACK': 'Counterattack',
            'CDLDARKCLOUDCOVER': 'Dark Cloud Cover',
            'CDLDOJI': 'Doji',
            'CDLDOJISTAR': 'Doji Star',
            'CDLDRAGONFLYDOJI': 'Dragonfly Doji',
            'CDLENGULFING': 'Engulfing Pattern',
            'CDLEVENINGDOJISTAR': 'Evening Doji Star',
            'CDLEVENINGSTAR': 'Evening Star',
            'CDLGAPSIDESIDEWHITE': 'Up/Down-gap side-by-side white lines',
            'CDLGRAVESTONEDOJI': 'Gravestone Doji',
            'CDLHAMMER': 'Hammer',
            'CDLHANGINGMAN': 'Hanging Man',
            'CDLHARAMI': 'Harami Pattern',
            'CDLHARAMICROSS': 'Harami Cross Pattern',
            'CDLHIGHWAVE': 'High-Wave Candle',
            'CDLHIKKAKE': 'Hikkake Pattern',
            'CDLHIKKAKEMOD': 'Modified Hikkake Pattern',
            'CDLHOMINGPIGEON': 'Homing Pigeon',
            'CDLIDENTICAL3CROWS': 'Identical Three Crows',
            'CDLINNECK': 'In-Neck Pattern',
            'CDLINVERTEDHAMMER': 'Inverted Hammer',
            'CDLKICKING': 'Kicking',
            'CDLKICKINGBYLENGTH': 'Kicking by Length',
            'CDLLADDERBOTTOM': 'Ladder Bottom',
            'CDLLONGLEGGEDDOJI': 'Long Legged Doji',
            'CDLLONGLINE': 'Long Line Candle',
            'CDLMARUBOZU': 'Marubozu',
            'CDLMATCHINGLOW': 'Matching Low',
            'CDLMATHOLD': 'Mat Hold',
            'CDLMORNINGDOJISTAR': 'Morning Doji Star',
            'CDLMORNINGSTAR': 'Morning Star',
            'CDLONNECK': 'On-Neck Pattern',
            'CDLPIERCING': 'Piercing Line',
            'CDLRICKSHAWMAN': 'Rickshaw Man',
            'CDLRISEFALL3METHODS': 'Rising/Falling Three Methods',
            'CDLSEPARATINGLINES': 'Separating Lines',
            'CDLSHOOTINGSTAR': 'Shooting Star',
            'CDLSHORTLINE': 'Short Line Candle',
            'CDLSPINNINGTOP': 'Spinning Top',
            'CDLSTALLEDPATTERN': 'Stalled Pattern',
            'CDLSTICKSANDWICH': 'Stick Sandwich',
            'CDLTAKURI': 'Takuri',
            'CDLTASUKIGAP': 'Tasuki Gap',
            'CDLTHRUSTING': 'Thrusting Pattern',
            'CDLTRISTAR': 'Tristar Pattern',
            'CDLUNIQUE3RIVER': 'Unique 3 River',
            'CDLUPSIDEGAP2CROWS': 'Upside Gap Two Crows',
            'CDLXSIDEGAP3METHODS': 'Upside/Downside Gap Three Methods',
        }

    def analyze_patterns(self, ticker: str, multi_timeframe_data: Dict[str, pd.DataFrame], lookback_periods: int = 10):
        """
        Analyze patterns across multiple timeframes and provide enhanced insights.

        Parameters:
        - ticker: The ticker symbol
        - multi_timeframe_data: Dictionary of dataframes for different timeframes
        - lookback_periods: Number of periods to look back for pattern detection

        Returns:
        - Dictionary containing pattern analysis results
        """
        if not self.pattern_recognizer:
            logger.error("Pattern recognizer not initialized")
            return {"error": "Pattern recognizer not initialized"}

        analysis_results = {
            "ticker": ticker,
            "timeframes": {},
            "multi_timeframe_confirmation": {},
            "pattern_clusters": {},
            "trading_signals": {},
            "insights": {}
        }

        # Pattern counts across timeframes for cluster analysis
        all_patterns = {}

        # Process each timeframe
        for timeframe, data in multi_timeframe_data.items():
            if data.empty:
                continue

            # Detect patterns using the base recognizer
            patterns = self.pattern_recognizer.detect_patterns(data, lookback_periods=lookback_periods)

            # Calculate technical indicators for context
            indicators = self._calculate_indicators(data)

            # Get pattern success rates
            pattern_stats = self._get_pattern_stats(patterns, data)

            # Store patterns found in this timeframe
            timeframe_patterns = []
            if patterns:
                for pattern_name, occurrences in patterns.items():
                    if occurrences:
                        # Track all patterns for later cluster analysis
                        if pattern_name not in all_patterns:
                            all_patterns[pattern_name] = []

                        # Add this occurrence with timeframe info
                        for occurrence in occurrences:
                            all_patterns[pattern_name].append({
                                "timeframe": timeframe,
                                "index": occurrence["index"],
                                "signal": occurrence["signal"],
                                "price": occurrence["price"],
                                "timestamp": occurrence["timestamp"]
                            })

                        # Get enhanced signal with market context
                        latest_occurrence = max(occurrences, key=lambda x: x["timestamp"])
                        enhanced_signal = self._get_enhanced_signal(
                            pattern_name,
                            latest_occurrence["signal"],
                            data,
                            indicators,
                            timeframe,
                            pattern_stats.get(pattern_name, {})
                        )

                        # Add to timeframe patterns
                        timeframe_patterns.append({
                            "pattern": pattern_name,
                            "signal": enhanced_signal["action"],
                            "confidence": enhanced_signal["confidence"],
                            "reason": enhanced_signal["reason"],
                            "last_occurrence": latest_occurrence["timestamp"],
                            "success_rate": pattern_stats.get(pattern_name, {}).get("success_rate", "N/A"),
                            "context": enhanced_signal["context"]
                        })

            # Add to results
            analysis_results["timeframes"][timeframe] = {
                "patterns": timeframe_patterns,
                "indicators": indicators,
                "pattern_count": len(timeframe_patterns)
            }

        # Identify pattern clusters
        clusters = self._identify_pattern_clusters(all_patterns)
        analysis_results["pattern_clusters"] = clusters

        # Generate multi-timeframe confirmation analysis
        confirmation = self._analyze_multi_timeframe_confirmation(analysis_results["timeframes"])
        analysis_results["multi_timeframe_confirmation"] = confirmation

        # Generate final trading signals
        trading_signals = self._generate_trading_signals(
            analysis_results["timeframes"],
            analysis_results["pattern_clusters"],
            analysis_results["multi_timeframe_confirmation"]
        )
        analysis_results["trading_signals"] = trading_signals

        # Generate actionable insights
        insights = self._generate_insights(
            ticker,
            analysis_results["timeframes"],
            analysis_results["pattern_clusters"],
            trading_signals
        )
        analysis_results["insights"] = insights

        return analysis_results

    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate technical indicators for context analysis."""
        global sma20
        indicators = {}

        if data.empty or len(data) < 20:
            return indicators

        try:
            # Calculate RSI
            if len(data) >= 14:
                rsi = talib.RSI(data['close'].values, timeperiod=14)
                indicators['rsi'] = rsi[-1]

                # RSI trend
                indicators['rsi_trend'] = 'bullish' if rsi[-1] < 30 else 'bearish' if rsi[-1] > 70 else 'neutral'

            # Calculate MACD
            if len(data) >= 26:
                macd, signal, hist = talib.MACD(
                    data['close'].values,
                    fastperiod=12,
                    slowperiod=26,
                    signalperiod=9
                )
                indicators['macd'] = macd[-1]
                indicators['macd_signal'] = signal[-1]
                indicators['macd_hist'] = hist[-1]

                # MACD trend
                indicators['macd_trend'] = 'bullish' if hist[-1] > 0 and hist[-1] > hist[-2] else \
                    'bearish' if hist[-1] < 0 and hist[-1] < hist[-2] else 'neutral'

            # Calculate moving averages
            if len(data) >= 20:
                sma20 = talib.SMA(data['close'].values, timeperiod=20)
                indicators['sma20'] = sma20[-1]

                # Price relative to SMA
                indicators['price_to_sma20'] = data['close'].iloc[-1] / sma20[-1] - 1

            if len(data) >= 50:
                sma50 = talib.SMA(data['close'].values, timeperiod=50)
                indicators['sma50'] = sma50[-1]

                # SMA trend
                indicators['ma_trend'] = 'bullish' if sma20[-1] > sma50[-1] else \
                    'bearish' if sma20[-1] < sma50[-1] else 'neutral'

                # Price relative to SMA50
                indicators['price_to_sma50'] = data['close'].iloc[-1] / sma50[-1] - 1

            # Calculate ATR for volatility
            if len(data) >= 14:
                atr = talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=14)
                indicators['atr'] = atr[-1]
                indicators['atr_percent'] = atr[-1] / data['close'].iloc[-1] * 100

                # Volatility state
                avg_atr_pct = np.mean(atr[-10:]) / np.mean(data['close'].iloc[-10:]) * 100
                indicators['volatility'] = 'high' if indicators['atr_percent'] > avg_atr_pct * 1.2 else \
                    'low' if indicators['atr_percent'] < avg_atr_pct * 0.8 else 'normal'

            # Calculate volume metrics
            if 'volume' in data.columns and len(data) >= 20:
                indicators['volume'] = data['volume'].iloc[-1]
                indicators['volume_sma20'] = data['volume'].rolling(window=20).mean().iloc[-1]
                indicators['volume_ratio'] = data['volume'].iloc[-1] / indicators['volume_sma20']

                # Volume trend
                vol_change = data['volume'].pct_change().iloc[-5:].mean() * 100
                indicators['volume_trend'] = 'increasing' if vol_change > 5 else \
                    'decreasing' if vol_change < -5 else 'stable'

            # Trend detection
            if len(data) >= 20:
                # Simple trend detection
                price_change = (data['close'].iloc[-1] / data['close'].iloc[-20] - 1) * 100
                indicators['trend_20_period'] = price_change
                indicators['trend_direction'] = 'uptrend' if price_change > 5 else \
                    'downtrend' if price_change < -5 else 'sideways'

                # Trend strength
                if abs(price_change) > 15:
                    indicators['trend_strength'] = 'strong'
                elif abs(price_change) > 8:
                    indicators['trend_strength'] = 'moderate'
                else:
                    indicators['trend_strength'] = 'weak'

            # Bollinger Bands
            if len(data) >= 20:
                upperband, middleband, lowerband = talib.BBANDS(
                    data['close'].values,
                    timeperiod=20,
                    nbdevup=2,
                    nbdevdn=2,
                    matype=0
                )
                indicators['bb_upper'] = upperband[-1]
                indicators['bb_middle'] = middleband[-1]
                indicators['bb_lower'] = lowerband[-1]

                # BB width (volatility indicator)
                indicators['bb_width'] = (upperband[-1] - lowerband[-1]) / middleband[-1]

                # Position within BBs
                close = data['close'].iloc[-1]
                bb_position = (close - lowerband[-1]) / (upperband[-1] - lowerband[-1])
                indicators['bb_position'] = bb_position

                if bb_position > 0.8:
                    indicators['bb_signal'] = 'overbought'
                elif bb_position < 0.2:
                    indicators['bb_signal'] = 'oversold'
                else:
                    indicators['bb_signal'] = 'neutral'

            # Support/Resistance proximity
            sr_levels = self._identify_support_resistance(data)
            indicators['support_levels'] = sr_levels['support']
            indicators['resistance_levels'] = sr_levels['resistance']

            # Check proximity to S/R levels
            current_price = data['close'].iloc[-1]
            indicators['sr_proximity'] = self._check_sr_proximity(current_price, sr_levels)

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")

        return indicators

    def _identify_support_resistance(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """Identify key support and resistance levels."""
        if data.empty or len(data) < 30:
            return {"support": [], "resistance": []}

        try:
            # Use pivot points for S/R identification
            pivot_high = talib.MAX(data['high'].values, timeperiod=10)
            pivot_low = talib.MIN(data['low'].values, timeperiod=10)

            # Find local maxima
            resistance_points = []
            for i in range(10, len(pivot_high) - 10):
                if pivot_high[i] == data['high'].iloc[i] and \
                        pivot_high[i] > pivot_high[i-5:i].max() and \
                        pivot_high[i] > pivot_high[i+1:i+6].max():
                    resistance_points.append(pivot_high[i])

            # Find local minima
            support_points = []
            for i in range(10, len(pivot_low) - 10):
                if pivot_low[i] == data['low'].iloc[i] and \
                        pivot_low[i] < pivot_low[i-5:i].min() and \
                        pivot_low[i] < pivot_low[i+1:i+6].min():
                    support_points.append(pivot_low[i])

            # Group nearby levels (within 0.5% of each other)
            grouped_resistance = self._group_nearby_levels(resistance_points, 0.005)
            grouped_support = self._group_nearby_levels(support_points, 0.005)

            # Sort levels by strength (frequency)
            resistance_levels = sorted(grouped_resistance.items(), key=lambda x: x[1], reverse=True)
            support_levels = sorted(grouped_support.items(), key=lambda x: x[1], reverse=True)

            # Return top levels
            return {
                "support": [level for level, _ in support_levels[:5]],
                "resistance": [level for level, _ in resistance_levels[:5]]
            }

        except Exception as e:
            logger.error(f"Error identifying support/resistance: {e}")
            return {"support": [], "resistance": []}

    def _group_nearby_levels(self, levels: List[float], threshold: float) -> Dict[float, int]:
        """Group price levels that are within threshold% of each other."""
        if not levels:
            return {}

        grouped = {}

        for level in sorted(levels):
            # Check if this level is close to any existing group
            found_group = False
            for group_level in list(grouped.keys()):
                if abs(level / group_level - 1) < threshold:
                    # Update group with average and increment count
                    new_level = (group_level * grouped[group_level] + level) / (grouped[group_level] + 1)
                    grouped[new_level] = grouped.pop(group_level) + 1
                    found_group = True
                    break

            # If not close to any group, create new group
            if not found_group:
                grouped[level] = 1

        return grouped

    def _check_sr_proximity(self, price: float, sr_levels: Dict[str, List[float]]) -> Dict:
        """Check if current price is near support or resistance levels."""
        result = {"near_support": False, "near_resistance": False}

        # Check proximity (within 1%)
        threshold = price * 0.01

        for support in sr_levels["support"]:
            if abs(price - support) < threshold:
                result["near_support"] = True
                result["support_level"] = support
                break

        for resistance in sr_levels["resistance"]:
            if abs(resistance - price) < threshold:
                result["near_resistance"] = True
                result["resistance_level"] = resistance
                break

        return result

    def _get_pattern_stats(self, patterns: Dict, data: pd.DataFrame) -> Dict:
        """Get historical success/failure statistics for detected patterns."""
        pattern_stats = {}

        if not patterns or data.empty:
            return pattern_stats

        for pattern_name, occurrences in patterns.items():
            if not occurrences:
                continue

            # Get pattern type
            pattern_type = "neutral"
            if hasattr(self.pattern_recognizer, '_categorize_pattern'):
                pattern_type = self.pattern_recognizer._categorize_pattern(pattern_name)

            success_rate = self._calculate_pattern_success_rate(pattern_name, pattern_type)

            # Store stats
            pattern_stats[pattern_name] = {
                "pattern_type": pattern_type,
                "success_rate": success_rate,
                "occurrences": len(occurrences)
            }

        return pattern_stats

    def _calculate_pattern_success_rate(self, pattern_name: str, pattern_type: str) -> float:
        """Calculate success rate for a pattern based on historical data."""
        # Get from pattern history or use default values
        if pattern_name in self.pattern_history:
            successes = self.pattern_history[pattern_name].get("successes", 0)
            failures = self.pattern_history[pattern_name].get("failures", 0)
            total = successes + failures

            if total > 0:
                return round(successes / total * 100, 1)

        # Default success rates by pattern type if no history
        default_rates = {
            "bullish_reversal": 65,
            "bearish_reversal": 63,
            "bullish_continuation": 58,
            "bearish_continuation": 56,
            "neutral": 50,
            "both_reversal": 60,
            "both_continuation": 55,
            "trend": 52,
            "unknown": 50
        }

        return default_rates.get(pattern_type, 50)

    def _get_enhanced_signal(self, pattern_name: str, signal_value: int,
                             data: pd.DataFrame, indicators: Dict,
                             timeframe: str, pattern_stats: Dict) -> Dict:
        """Generate enhanced trading signal with market context."""
        # Start with a basic signal
        pattern_type = pattern_stats.get("pattern_type", "unknown")
        success_rate = pattern_stats.get("success_rate", 50)

        # Determine preliminary action
        action = "NEUTRAL"
        if signal_value > 0:  # Bullish signal
            if pattern_type in ["bullish_reversal", "bullish_continuation"]:
                action = "BUY"
            elif pattern_type in ["both_reversal", "both_continuation"]:
                action = "BUY"  # Default to the signal direction
        elif signal_value < 0:  # Bearish signal
            if pattern_type in ["bearish_reversal", "bearish_continuation"]:
                action = "SELL"
            elif pattern_type in ["both_reversal", "both_continuation"]:
                action = "SELL"  # Default to the signal direction

        # Set initial confidence based on success rate
        confidence = "medium"
        if success_rate >= 70:
            confidence = "very_high"
        elif success_rate >= 60:
            confidence = "high"
        elif success_rate >= 50:
            confidence = "medium"
        else:
            confidence = "low"

        # Market context factors to consider for signal enhancement
        context = {}

        # 1. Trend alignment
        if "trend_direction" in indicators:
            context["trend_direction"] = indicators["trend_direction"]

            # Adjust confidence based on trend alignment
            if action == "BUY" and indicators["trend_direction"] == "uptrend":
                confidence = self._upgrade_confidence(confidence)
            elif action == "SELL" and indicators["trend_direction"] == "downtrend":
                confidence = self._upgrade_confidence(confidence)
            elif action == "BUY" and indicators["trend_direction"] == "downtrend":
                confidence = self._downgrade_confidence(confidence)
            elif action == "SELL" and indicators["trend_direction"] == "uptrend":
                confidence = self._downgrade_confidence(confidence)

        # 2. Trend strength
        if "trend_strength" in indicators:
            context["trend_strength"] = indicators["trend_strength"]

            # Strong trends increase confidence in trend-aligned signals
            if indicators["trend_strength"] == "strong":
                if (action == "BUY" and indicators.get("trend_direction") == "uptrend") or \
                        (action == "SELL" and indicators.get("trend_direction") == "downtrend"):
                    confidence = self._upgrade_confidence(confidence)

        # 3. Volume confirmation
        if "volume_ratio" in indicators:
            context["volume_confirmation"] = indicators["volume_ratio"] > 1.2

            # High volume increases confidence
            if indicators["volume_ratio"] > 1.5:
                confidence = self._upgrade_confidence(confidence)
            elif indicators["volume_ratio"] < 0.8:
                confidence = self._downgrade_confidence(confidence)

        # 4. Oscillator confirmation
        if "rsi_trend" in indicators:
            context["rsi_confirmation"] = indicators["rsi_trend"]

            # RSI oversold/overbought confirms reversal patterns
            if pattern_type == "bullish_reversal" and indicators["rsi_trend"] == "bullish":
                confidence = self._upgrade_confidence(confidence)
            elif pattern_type == "bearish_reversal" and indicators["rsi_trend"] == "bearish":
                confidence = self._upgrade_confidence(confidence)

        # 5. Support/Resistance proximity
        if "sr_proximity" in indicators:
            context["sr_proximity"] = indicators["sr_proximity"]

            # Patterns near S/R levels have higher importance
            if action == "BUY" and indicators["sr_proximity"].get("near_support", False):
                confidence = self._upgrade_confidence(confidence)
                context["key_level"] = indicators["sr_proximity"].get("support_level")
            elif action == "SELL" and indicators["sr_proximity"].get("near_resistance", False):
                confidence = self._upgrade_confidence(confidence)
                context["key_level"] = indicators["sr_proximity"].get("resistance_level")

        # 6. Volatility context
        if "volatility" in indicators:
            context["volatility"] = indicators["volatility"]

            # Adjust stop distance based on volatility
            if "atr" in indicators:
                context["suggested_stop_distance"] = indicators["atr"] * (
                    2.5 if indicators["volatility"] == "high" else
                    2.0 if indicators["volatility"] == "normal" else 1.5
                )

        # Generate reason based on context
        reason = f"{pattern_name} detected"

        if action != "NEUTRAL":
            reason += f" with {success_rate}% historical success rate"

            if context.get("trend_direction"):
                reason += f" in {context['trend_direction']} market"

            if context.get("sr_proximity", {}).get("near_support") and action == "BUY":
                reason += " near support level"
            elif context.get("sr_proximity", {}).get("near_resistance") and action == "SELL":
                reason += " near resistance level"

            if context.get("volume_confirmation"):
                reason += " with above-average volume"

        # Calculate target price
        target_price = None
        stop_price = None

        if data is not None and not data.empty and action != "NEUTRAL":
            current_price = data["close"].iloc[-1]

            # Use ATR for stop/target if available, otherwise use percentage
            if "atr" in indicators:
                atr = indicators["atr"]
                if action == "BUY":
                    stop_price = current_price - (atr * 2)
                    target_price = current_price + (atr * 3)
                else:  # SELL
                    stop_price = current_price + (atr * 2)
                    target_price = current_price - (atr * 3)
            else:
                # Default percentage-based
                if action == "BUY":
                    stop_price = current_price * 0.97  # 3% stop
                    target_price = current_price * 1.06  # 6% target
                else:  # SELL
                    stop_price = current_price * 1.03  # 3% stop
                    target_price = current_price * 0.94  # 6% target

        # Return enhanced signal
        return {
            "action": action,
            "confidence": confidence,
            "reason": reason,
            "target_price": target_price,
            "stop_price": stop_price,
            "pattern_type": pattern_type,
            "success_rate": success_rate,
            "timeframe": timeframe,
            "context": context
        }

    def _upgrade_confidence(self, current_confidence: str) -> str:
        """Upgrade confidence level by one step."""
        confidence_levels = ["low", "medium", "medium_high", "high", "very_high"]
        try:
            current_index = confidence_levels.index(current_confidence)
            if current_index < len(confidence_levels) - 1:
                return confidence_levels[current_index + 1]
        except ValueError:
            pass
        return current_confidence

    def _downgrade_confidence(self, current_confidence: str) -> str:
        """Downgrade confidence level by one step."""
        confidence_levels = ["low", "medium", "medium_high", "high", "very_high"]
        try:
            current_index = confidence_levels.index(current_confidence)
            if current_index > 0:
                return confidence_levels[current_index - 1]
        except ValueError:
            pass
        return current_confidence

    def _identify_pattern_clusters(self, all_patterns: Dict) -> Dict:
        """Identify pattern clusters across timeframes."""
        clusters = {
            "bullish": [],
            "bearish": [],
            "consolidation": [],
            "reversal_signals": []
        }

        # Check for pattern groups
        for cluster_name, cluster_patterns in self.pattern_clusters.items():
            # Count patterns in this cluster
            patterns_in_cluster = []
            for pattern in cluster_patterns:
                if pattern in all_patterns and all_patterns[pattern]:
                    patterns_in_cluster.append({
                        "pattern": pattern,
                        "occurrences": len(all_patterns[pattern]),
                        "timeframes": set(occ["timeframe"] for occ in all_patterns[pattern])
                    })

            # If we have multiple patterns in the cluster
            if len(patterns_in_cluster) >= 2:
                # Determine the cluster type
                if cluster_name == "strong_reversal_bottom":
                    clusters["bullish"].append({
                        "cluster_type": "reversal_bottom",
                        "patterns": patterns_in_cluster,
                        "strength": len(patterns_in_cluster) * 20  # 20% per pattern
                    })
                    clusters["reversal_signals"].append({
                        "direction": "bullish",
                        "cluster_type": "reversal_bottom",
                        "patterns": [p["pattern"] for p in patterns_in_cluster],
                        "strength": min(100, len(patterns_in_cluster) * 20)
                    })
                elif cluster_name == "strong_reversal_top":
                    clusters["bearish"].append({
                        "cluster_type": "reversal_top",
                        "patterns": patterns_in_cluster,
                        "strength": len(patterns_in_cluster) * 20
                    })
                    clusters["reversal_signals"].append({
                        "direction": "bearish",
                        "cluster_type": "reversal_top",
                        "patterns": [p["pattern"] for p in patterns_in_cluster],
                        "strength": min(100, len(patterns_in_cluster) * 20)
                    })
                elif cluster_name == "momentum_continuation":
                    # Determine direction based on first pattern's signal
                    first_pattern = patterns_in_cluster[0]["pattern"]
                    if first_pattern in all_patterns and all_patterns[first_pattern]:
                        signal = all_patterns[first_pattern][0]["signal"]
                        if signal > 0:  # bullish
                            clusters["bullish"].append({
                                "cluster_type": "momentum_continuation",
                                "patterns": patterns_in_cluster,
                                "strength": len(patterns_in_cluster) * 15
                            })
                        elif signal < 0:  # bearish
                            clusters["bearish"].append({
                                "cluster_type": "momentum_continuation",
                                "patterns": patterns_in_cluster,
                                "strength": len(patterns_in_cluster) * 15
                            })
                elif cluster_name == "exhaustion_warning":
                    clusters["consolidation"].append({
                        "cluster_type": "exhaustion",
                        "patterns": patterns_in_cluster,
                        "strength": len(patterns_in_cluster) * 15
                    })

        # Calculate overall strength for each direction
        clusters["bullish_strength"] = sum(c["strength"] for c in clusters["bullish"])
        clusters["bearish_strength"] = sum(c["strength"] for c in clusters["bearish"])
        clusters["consolidation_strength"] = sum(c["strength"] for c in clusters["consolidation"])

        # Determine overall bias
        if clusters["bullish_strength"] > clusters["bearish_strength"] * 1.5:
            clusters["bias"] = "strongly_bullish"
        elif clusters["bullish_strength"] > clusters["bearish_strength"]:
            clusters["bias"] = "moderately_bullish"
        elif clusters["bearish_strength"] > clusters["bullish_strength"] * 1.5:
            clusters["bias"] = "strongly_bearish"
        elif clusters["bearish_strength"] > clusters["bullish_strength"]:
            clusters["bias"] = "moderately_bearish"
        elif clusters["consolidation_strength"] > max(clusters["bullish_strength"], clusters["bearish_strength"]):
            clusters["bias"] = "consolidation"
        else:
            clusters["bias"] = "neutral"

        return clusters

    def _analyze_multi_timeframe_confirmation(self, timeframe_data: Dict) -> Dict:
        """Analyze pattern confirmation across multiple timeframes."""
        confirmation = {
            "confirmed_patterns": [],
            "conflicting_patterns": [],
            "alignment_score": 0,
            "aligned_timeframes": [],
            "primary_bias": "neutral"
        }

        # Extract signals from each timeframe
        timeframe_signals = {}
        for timeframe, data in timeframe_data.items():
            if "patterns" in data and data["patterns"]:
                signals = {"BUY": 0, "SELL": 0, "NEUTRAL": 0, "WATCH": 0}
                confidence_sum = {"BUY": 0, "SELL": 0, "NEUTRAL": 0, "WATCH": 0}

                for pattern in data["patterns"]:
                    signals[pattern["signal"]] += 1

                    # Add confidence weight
                    conf_value = {
                        "very_high": 1.0,
                        "high": 0.8,
                        "medium_high": 0.6,
                        "medium": 0.5,
                        "low": 0.3
                    }.get(pattern["confidence"], 0.5)

                    confidence_sum[pattern["signal"]] += conf_value

                # Determine timeframe bias
                if signals["BUY"] > signals["SELL"]:
                    bias = "bullish"
                    bias_strength = confidence_sum["BUY"]
                elif signals["SELL"] > signals["BUY"]:
                    bias = "bearish"
                    bias_strength = confidence_sum["SELL"]
                else:
                    bias = "neutral"
                    bias_strength = max(confidence_sum["BUY"], confidence_sum["SELL"])

                timeframe_signals[timeframe] = {
                    "bias": bias,
                    "strength": bias_strength,
                    "signals": signals,
                    "confidence_sum": confidence_sum
                }

        # No signals found
        if not timeframe_signals:
            return confirmation

        # Identify aligned timeframes
        timeframe_weights = {
            "long_term": 0.5,
            "medium_term": 0.3,
            "short_term": 0.2,
            "very_short_term": 0.1
        }

        bullish_score = 0
        bearish_score = 0
        bullish_timeframes = []
        bearish_timeframes = []

        for timeframe, signal_data in timeframe_signals.items():
            weight = timeframe_weights.get(timeframe, 0.25)

            if signal_data["bias"] == "bullish":
                bullish_score += signal_data["strength"] * weight
                bullish_timeframes.append(timeframe)
            elif signal_data["bias"] == "bearish":
                bearish_score += signal_data["strength"] * weight
                bearish_timeframes.append(timeframe)

        # Determine aligned timeframes and primary bias
        if bullish_score > bearish_score:
            confirmation["primary_bias"] = "bullish"
            confirmation["aligned_timeframes"] = bullish_timeframes
            confirmation["alignment_score"] = min(100, bullish_score * 100)
        elif bearish_score > bullish_score:
            confirmation["primary_bias"] = "bearish"
            confirmation["aligned_timeframes"] = bearish_timeframes
            confirmation["alignment_score"] = min(100, bearish_score * 100)
        else:
            confirmation["primary_bias"] = "neutral"
            confirmation["aligned_timeframes"] = []
            confirmation["alignment_score"] = 0

        # Find confirmed patterns (appear in multiple timeframes)
        pattern_counts = {}
        for timeframe, data in timeframe_data.items():
            if "patterns" in data:
                for pattern in data["patterns"]:
                    pattern_name = pattern["pattern"]
                    if pattern_name not in pattern_counts:
                        pattern_counts[pattern_name] = {"count": 0, "timeframes": [], "signals": []}

                    pattern_counts[pattern_name]["count"] += 1
                    pattern_counts[pattern_name]["timeframes"].append(timeframe)
                    pattern_counts[pattern_name]["signals"].append(pattern["signal"])

        # Patterns appearing in multiple timeframes are confirmed
        for pattern_name, data in pattern_counts.items():
            if data["count"] >= 2:
                # Check if signals are aligned
                signals = set(data["signals"])
                if len(signals) == 1:  # All signals are the same
                    confirmation["confirmed_patterns"].append({
                        "pattern": pattern_name,
                        "timeframes": data["timeframes"],
                        "signal": data["signals"][0]
                    })
                else:
                    confirmation["conflicting_patterns"].append({
                        "pattern": pattern_name,
                        "timeframes": data["timeframes"],
                        "signals": data["signals"]
                    })

        return confirmation

    def _generate_trading_signals(self, timeframe_data: Dict, pattern_clusters: Dict,
                                  confirmation: Dict) -> Dict:
        """Generate final trading signals based on all analyses."""
        # Determine overall signal
        signal = "NEUTRAL"
        confidence = "medium"

        # Start with multi-timeframe confirmation
        if confirmation["primary_bias"] == "bullish" and confirmation["alignment_score"] > 60:
            signal = "BUY"
            confidence = "high" if confirmation["alignment_score"] > 80 else "medium_high"
        elif confirmation["primary_bias"] == "bearish" and confirmation["alignment_score"] > 60:
            signal = "SELL"
            confidence = "high" if confirmation["alignment_score"] > 80 else "medium_high"

        # Strengthen signal if pattern clusters agree
        if signal == "BUY" and pattern_clusters["bias"] in ["strongly_bullish", "moderately_bullish"]:
            confidence = self._upgrade_confidence(confidence)
        elif signal == "SELL" and pattern_clusters["bias"] in ["strongly_bearish", "moderately_bearish"]:
            confidence = self._upgrade_confidence(confidence)

        # Weaken signal if pattern clusters disagree
        if signal == "BUY" and pattern_clusters["bias"] in ["strongly_bearish", "moderately_bearish"]:
            confidence = self._downgrade_confidence(confidence)
        elif signal == "SELL" and pattern_clusters["bias"] in ["strongly_bullish", "moderately_bullish"]:
            confidence = self._downgrade_confidence(confidence)

        # If no clear signal from confirmation, use pattern clusters
        if signal == "NEUTRAL":
            if pattern_clusters["bias"] == "strongly_bullish":
                signal = "BUY"
                confidence = "medium_high"
            elif pattern_clusters["bias"] == "moderately_bullish":
                signal = "BUY"
                confidence = "medium"
            elif pattern_clusters["bias"] == "strongly_bearish":
                signal = "SELL"
                confidence = "medium_high"
            elif pattern_clusters["bias"] == "moderately_bearish":
                signal = "SELL"
                confidence = "medium"

        # Get supporting patterns
        supporting_patterns = []
        if signal == "BUY":
            for timeframe, data in timeframe_data.items():
                if "patterns" in data:
                    for pattern in data["patterns"]:
                        if pattern["signal"] == "BUY":
                            supporting_patterns.append({
                                "pattern": pattern["pattern"],
                                "timeframe": timeframe,
                                "confidence": pattern["confidence"]
                            })
        elif signal == "SELL":
            for timeframe, data in timeframe_data.items():
                if "patterns" in data:
                    for pattern in data["patterns"]:
                        if pattern["signal"] == "SELL":
                            supporting_patterns.append({
                                "pattern": pattern["pattern"],
                                "timeframe": timeframe,
                                "confidence": pattern["confidence"]
                            })

        # Return trading signal
        return {
            "signal": signal,
            "confidence": confidence,
            "supporting_patterns": supporting_patterns,
            "confirmation_score": confirmation["alignment_score"],
            "cluster_bias": pattern_clusters["bias"],
            "timeframe_alignment": confirmation["aligned_timeframes"]
        }

    def _generate_insights(self, ticker: str, timeframe_data: Dict,
                           pattern_clusters: Dict, trading_signals: Dict) -> Dict:
        """Generate actionable insights based on pattern analysis."""
        insights = {
            "summary": "",
            "key_points": [],
            "action_items": [],
            "risk_factors": [],
            "entry_strategy": {},
            "exit_strategy": {},
            "position_sizing": {}
        }

        # Generate summary
        signal = trading_signals["signal"]
        confidence = trading_signals["confidence"]

        if signal == "BUY":
            insights["summary"] = f"Bullish pattern formation detected for {ticker} with {confidence} confidence."

            # Key points for bullish outlook
            insights["key_points"] = [
                f"Multiple timeframe confirmation: {', '.join(trading_signals['timeframe_alignment'])}",
                f"Pattern cluster analysis shows {pattern_clusters['bias']} bias",
            ]

            # Add supporting patterns
            if trading_signals["supporting_patterns"]:
                top_patterns = sorted(trading_signals["supporting_patterns"],
                                      key=lambda x: {"very_high": 5, "high": 4, "medium_high": 3, "medium": 2, "low": 1}
                                      .get(x["confidence"], 0), reverse=True)[:3]

                pattern_points = []
                for p in top_patterns:
                    pattern_points.append(f"{p['pattern']} detected in {p['timeframe']} timeframe")

                insights["key_points"].extend(pattern_points)

            # Action items
            insights["action_items"] = [
                "Consider opening long position with appropriate risk management",
                f"Look for entry confirmation such as higher low formation or breakout above resistance"
            ]

            # Risk factors
            insights["risk_factors"] = [
                "Watch for invalidation signals such as breakdown below recent support",
                "Monitor for potential divergence in indicators"
            ]

            # Entry strategy
            insights["entry_strategy"] = {
                "strategy": "Look for low-risk entry on pullbacks or consolidation breakout",
                "confirmation": "Wait for higher low formation or volume confirmation"
            }

            # Exit strategy
            insights["exit_strategy"] = {
                "take_profit": "Consider scaling out at resistance levels or using trailing stop",
                "stop_loss": "Place stop below recent swing low or support level"
            }

        elif signal == "SELL":
            insights["summary"] = f"Bearish pattern formation detected for {ticker} with {confidence} confidence."

            # Key points for bearish outlook
            insights["key_points"] = [
                f"Multiple timeframe confirmation: {', '.join(trading_signals['timeframe_alignment'])}",
                f"Pattern cluster analysis shows {pattern_clusters['bias']} bias",
            ]

            # Add supporting patterns
            if trading_signals["supporting_patterns"]:
                top_patterns = sorted(trading_signals["supporting_patterns"],
                                      key=lambda x: {"very_high": 5, "high": 4, "medium_high": 3, "medium": 2, "low": 1}
                                      .get(x["confidence"], 0), reverse=True)[:3]

                pattern_points = []
                for p in top_patterns:
                    pattern_points.append(f"{p['pattern']} detected in {p['timeframe']} timeframe")

                insights["key_points"].extend(pattern_points)

            # Action items
            insights["action_items"] = [
                "Consider reducing long exposure or opening short position",
                "Look for entry confirmation such as lower high formation or breakdown below support"
            ]

            # Risk factors
            insights["risk_factors"] = [
                "Watch for invalidation signals such as break above recent resistance",
                "Be aware of potential short squeeze in heavily shorted stocks"
            ]

            # Entry strategy
            insights["entry_strategy"] = {
                "strategy": "Look for bounces to resistance or breakdown confirmation",
                "confirmation": "Wait for lower high formation or volume confirmation on breakdown"
            }

            # Exit strategy
            insights["exit_strategy"] = {
                "take_profit": "Consider scaling out at support levels or using trailing stop",
                "stop_loss": "Place stop above recent swing high or resistance level"
            }

        else:  # NEUTRAL
            insights["summary"] = f"No clear directional signal detected for {ticker}."

            insights["key_points"] = [
                "Conflicting patterns across timeframes",
                f"Pattern cluster analysis shows {pattern_clusters['bias']} bias but lacks confirmation",
                "Consider waiting for clearer signals before taking position"
            ]

            insights["action_items"] = [
                "Monitor for pattern development and breakout direction",
                "Look for volume confirmation before taking position"
            ]

            insights["risk_factors"] = [
                "Choppy price action may lead to false signals",
                "Low volume may indicate lack of conviction"
            ]

        # Position sizing based on confidence
        position_size_map = {
            "very_high": "75-100% of max position size",
            "high": "50-75% of max position size",
            "medium_high": "35-50% of max position size",
            "medium": "25-35% of max position size",
            "low": "10-25% of max position size"
        }

        insights["position_sizing"] = {
            "recommendation": position_size_map.get(confidence, "25% of max position size"),
            "confidence_level": confidence,
            "scaling": "Consider scaling in on confirmation, scaling out at targets"
        }

        return insights

    def generate_pattern_visualization(self, ticker: str, data: pd.DataFrame, patterns: Dict,
                                       num_candles: int = 40) -> str:
        """Generate a visualization of candlestick patterns."""
        try:
            # Use the most recent candles
            plot_data = data.tail(num_candles).copy()

            if plot_data.empty:
                return None

            # Create figure and axes
            fig, ax = plt.subplots(figsize=(12, 6))

            # Format dates
            plot_data.reset_index(inplace=True)
            dates = mdates.date2num(plot_data['datetime'].dt.to_pydatetime())

            # Create OHLC data for candlestick plot
            ohlc = np.column_stack((dates, plot_data['open'], plot_data['high'],
                                    plot_data['low'], plot_data['close']))

            # Plot candlesticks
            from mplfinance.original_flavor import candlestick_ohlc
            candlestick_ohlc(ax, ohlc, width=0.6, colorup='g', colordown='r')

            # Format x-axis as dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.xticks(rotation=45)

            # Add volume as bar chart at the bottom
            if 'volume' in plot_data.columns:
                volume_ax = ax.twinx()
                volume_ax.bar(dates, plot_data['volume'], width=0.3, alpha=0.3, color='blue')
                volume_ax.set_ylim(0, plot_data['volume'].max() * 3)
                volume_ax.set_ylabel('Volume')
                volume_ax.grid(False)

            # Add pattern markers
            marked_patterns = []

            for pattern_name, occurrences in patterns.items():
                for occurrence in occurrences:
                    # Find index in plot data
                    if isinstance(occurrence['timestamp'], pd.Timestamp):
                        pattern_time = occurrence['timestamp']
                        idx = plot_data[plot_data['datetime'] == pattern_time].index

                        if len(idx) > 0:
                            idx = idx[0]

                            # Determine pattern properties
                            signal = occurrence['signal']
                            color = 'g' if signal > 0 else 'r' if signal < 0 else 'blue'

                            # Add marker
                            if signal > 0:  # Bullish
                                ax.annotate('▲', xy=(dates[idx], plot_data['low'].iloc[idx]),
                                            xytext=(0, -20), textcoords='offset points',
                                            va='top', ha='center', color=color, fontsize=14)
                            elif signal < 0:  # Bearish
                                ax.annotate('▼', xy=(dates[idx], plot_data['high'].iloc[idx]),
                                            xytext=(0, 20), textcoords='offset points',
                                            va='bottom', ha='center', color=color, fontsize=14)

                            # Add to marked patterns for legend
                            marked_patterns.append((pattern_name, color))

            # Add legend for patterns
            if marked_patterns:
                from matplotlib.lines import Line2D

                # Remove duplicates while preserving order
                seen = set()
                unique_patterns = [(p, c) for p, c in marked_patterns if not (p in seen or seen.add(p))]

                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                           label=pattern, markersize=8)
                    for pattern, color in unique_patterns
                ]

                ax.legend(handles=legend_elements, loc='upper left')

            # Add title and labels
            plt.title(f'{ticker} Candlestick Chart with Pattern Recognition')
            ax.set_ylabel('Price')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save figure to bytes
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close(fig)
            buf.seek(0)

            # Convert to base64 for embedding
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{img_str}"

        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return None

    def update_pattern_history(self, pattern_name: str, outcome: bool):
        """Update pattern success/failure history."""
        if pattern_name not in self.pattern_history:
            self.pattern_history[pattern_name] = {
                "successes": 0,
                "failures": 0,
                "last_update": datetime.now()
            }

        if outcome:
            self.pattern_history[pattern_name]["successes"] += 1
        else:
            self.pattern_history[pattern_name]["failures"] += 1

        self.pattern_history[pattern_name]["last_update"] = datetime.now()

    def validate_historical_patterns(self, ticker: str, historical_data: pd.DataFrame):
        """Validate pattern success/failure on historical data."""
        if not self.pattern_recognizer or historical_data.empty:
            return

        try:
            # Process data in windows
            window_size = 100  # Process 100 candles at a time

            for i in range(0, len(historical_data) - window_size, 20):  # Step by 20 candles
                window = historical_data.iloc[i:i+window_size].copy()

                # Detect patterns in this window
                patterns = self.pattern_recognizer.detect_patterns(window, lookback_periods=30)

                # For each pattern, check future performance
                for pattern_name, occurrences in patterns.items():
                    for occurrence in occurrences:
                        # Get pattern type
                        pattern_type = self.pattern_recognizer._categorize_pattern(pattern_name)

                        # Determine success threshold and validation period
                        success_threshold = self.success_thresholds.get(pattern_type, 0.02)
                        validation_period = self.validation_periods.get(pattern_type, 3)

                        # Get index in window
                        idx = occurrence["index"]
                        if idx >= len(window) - validation_period:
                            continue  # Skip if not enough forward data

                        # Get price at pattern and validation price
                        pattern_price = occurrence["price"]
                        future_idx = min(idx + validation_period, len(window) - 1)
                        future_price = window.iloc[future_idx]["close"]

                        # Determine outcome
                        if pattern_type.startswith("bullish"):
                            # For bullish patterns, price should go up
                            success = (future_price / pattern_price - 1) >= success_threshold
                        elif pattern_type.startswith("bearish"):
                            # For bearish patterns, price should go down
                            success = (pattern_price / future_price - 1) >= success_threshold
                        else:
                            # For neutral patterns, any significant move counts
                            success = abs(future_price / pattern_price - 1) >= success_threshold

                        # Update pattern history
                        self.update_pattern_history(pattern_name, success)

        except Exception as e:
            logger.error(f"Error validating historical patterns: {e}")

    def get_pattern_history_stats(self):
        """Get statistics about pattern success/failure history."""
        if not self.pattern_history:
            return {"message": "No pattern history available"}

        stats = {}

        for pattern_name, history in self.pattern_history.items():
            successes = history.get("successes", 0)
            failures = history.get("failures", 0)
            total = successes + failures

            if total > 0:
                success_rate = round(successes / total * 100, 1)

                stats[pattern_name] = {
                    "success_rate": success_rate,
                    "total_occurrences": total,
                    "successes": successes,
                    "failures": failures,
                    "last_update": history.get("last_update")
                }

        return stats
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
