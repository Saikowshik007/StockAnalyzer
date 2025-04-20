import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class TalibPatternRecognition:
    def __init__(self):
        # Mapping of pattern functions to their names and significance
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

        # Pattern type to signal interpretation
        self.pattern_types = {
            'bullish': ['Morning Star', 'Morning Doji Star', 'Three White Soldiers', 'Hammer',
                        'Inverted Hammer', 'Piercing Line', 'Bullish Engulfing'],
            'bearish': ['Evening Star', 'Evening Doji Star', 'Three Black Crows', 'Hanging Man',
                        'Shooting Star', 'Dark Cloud Cover', 'Bearish Engulfing'],
            'reversal': ['Doji', 'Spinning Top', 'Harami', 'Harami Cross'],
            'continuation': ['Rising Three Methods', 'Falling Three Methods']
        }

    def detect_patterns(self, data: pd.DataFrame) -> Dict[str, List]:
        """Detect all candlestick patterns in the data."""
        if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            logger.error("Data must contain 'open', 'high', 'low', 'close' columns")
            return {}

        detected_patterns = {}

        for pattern_name, pattern_func in self.pattern_functions.items():
            try:
                # Convert data to numpy arrays
                open_prices = data['open'].values.astype(float)
                high_prices = data['high'].values.astype(float)
                low_prices = data['low'].values.astype(float)
                close_prices = data['close'].values.astype(float)

                # Detect pattern
                result = pattern_func(open_prices, high_prices, low_prices, close_prices)

                # Find where pattern is detected (non-zero values)
                pattern_indices = np.where(result != 0)[0]

                if len(pattern_indices) > 0:
                    detected_patterns[pattern_name] = [
                        {
                            'index': int(idx),
                            'timestamp': data.index[idx],
                            'signal': int(result[idx]),  # 100, -100, etc.
                            'price': float(close_prices[idx])
                        }
                        for idx in pattern_indices
                    ]
            except Exception as e:
                logger.error(f"Error detecting {pattern_name}: {e}")

        return detected_patterns

    def get_trading_signal(self, pattern_name: str, signal_value: int, current_price: float) -> Dict:
        """Generate trading signal based on pattern and signal value."""
        signal = {
            'action': None,
            'reason': '',
            'confidence': 'medium',
            'entry_price': current_price,
            'stop_loss': None,
            'take_profit': None
        }

        # Interpret signal
        if signal_value > 0:  # Bullish signal
            if pattern_name in self.pattern_types['bullish']:
                signal['action'] = 'BUY'
                signal['reason'] = f'Bullish {pattern_name} pattern detected'
                signal['confidence'] = 'high' if signal_value >= 100 else 'medium'
                signal['stop_loss'] = current_price * 0.98  # 2% stop loss
                signal['take_profit'] = current_price * 1.05  # 5% take profit
            elif pattern_name in self.pattern_types['reversal']:
                signal['action'] = 'WATCH'
                signal['reason'] = f'Potential reversal pattern: {pattern_name}'
                signal['confidence'] = 'low'

        elif signal_value < 0:  # Bearish signal
            if pattern_name in self.pattern_types['bearish']:
                signal['action'] = 'SELL'
                signal['reason'] = f'Bearish {pattern_name} pattern detected'
                signal['confidence'] = 'high' if signal_value <= -100 else 'medium'
                signal['stop_loss'] = current_price * 1.02  # 2% stop loss
                signal['take_profit'] = current_price * 0.95  # 5% take profit
            elif pattern_name in self.pattern_types['reversal']:
                signal['action'] = 'WATCH'
                signal['reason'] = f'Potential reversal pattern: {pattern_name}'
                signal['confidence'] = 'low'

        return signal