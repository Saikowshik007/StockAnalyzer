# services/pattern_monitor.py
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
from services.pattern_recognition import TalibPatternRecognition
logger = logging.getLogger(__name__)
class TalibPatternMonitor:
    def __init__(self, config=None, db_manager=None, stock_collector=None, bot_notifier=None):
        """Initialize pattern monitor with optional dependencies."""
        self.config = config
        self.db_manager = db_manager
        self.stock_collector = stock_collector
        self.bot_notifier = bot_notifier
        self.pattern_recognizer = TalibPatternRecognition()
        self.last_notification = {}
        # Get configuration values or use defaults
        if config:
            self.monitoring_interval = config.get('pattern_monitor.interval_seconds', 300)
            self.notification_cooldown = config.get('pattern_monitor.notification_cooldown', 3600)
        else:
            self.monitoring_interval = 300
            self.notification_cooldown = 3600

        # Multi-timeframe settings
        self.timeframe_weights = {
            'long_term': 0.5,    # Daily - trend confirmation
            'medium_term': 0.3,  # Hourly - main signals
            'short_term': 0.2    # 15-minute - entry timing
        }

        # Confidence thresholds for notifications
        self.confidence_thresholds = {
            'very_high': 0.9,
            'high': 0.7,
            'medium_high': 0.6,
            'medium': 0.5,
            'low': 0.3
        }

    async def monitor_patterns(self):
        """Continuously monitor for patterns with intelligent filtering and market context."""
        while True:
            try:
                if not self.db_manager or not self.stock_collector or not self.bot_notifier:
                    logger.error("Dependencies not set for pattern monitor")
                    await asyncio.sleep(60)
                    continue

                watchlist = self.db_manager.get_active_watchlist()

                # Track market conditions for context (placeholder)
                # market_context = self.get_market_context()

                # Track notification statistics for adaptive cooldown
                signals_found = 0
                high_confidence_signals = 0

                # Process each ticker
                for ticker in watchlist:
                    # Get the multi-timeframe signals
                    combined_signal = await self.analyze_multiple_timeframes(ticker)

                    if combined_signal:
                        signals_found += 1
                        if combined_signal['confidence'] in ['high', 'very_high']:
                            high_confidence_signals += 1

                        # Determine if the signal is strong enough to notify
                        if self.should_notify_strong_signal(ticker, combined_signal):
                            message = self.format_actionable_notification(ticker, combined_signal)
                            await self.bot_notifier.send_pattern_notification(message)
                            self.last_notification[ticker] = datetime.now()

                # Adapt monitoring interval based on market activity
                self._adapt_monitoring_settings(signals_found, high_confidence_signals)

                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in pattern monitoring: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(60)

    def _adapt_monitoring_settings(self, signals_found, high_confidence_signals):
        """Adaptively change monitoring interval and notification thresholds
        based on market activity."""
        # If lots of signals are being found, could mean high volatility
        # In that case, we might want to be more selective or check more frequently
        if high_confidence_signals > 10:
            # Lots of high confidence signals - market might be very active
            # Reduce interval to check more frequently
            new_interval = max(60, self.monitoring_interval // 2)
            if new_interval != self.monitoring_interval:
                logger.info(f"High market activity detected, adjusting interval to {new_interval}s")
                self.monitoring_interval = new_interval
        elif signals_found < 2 and self.monitoring_interval < 300:
            # Very few signals - market might be quiet
            # Increase interval to save resources
            new_interval = min(600, self.monitoring_interval * 2)
            if new_interval != self.monitoring_interval:
                logger.info(f"Low market activity detected, adjusting interval to {new_interval}s")
                self.monitoring_interval = new_interval

    async def analyze_multiple_timeframes(self, ticker: str) -> Dict:
        """Enhanced multi-timeframe analysis with smarter integration of indicators."""
        all_timeframe_data = self.stock_collector.get_multi_timeframe_data(ticker)
        signals = {}
        indicators_by_timeframe = {}

        # First pass: collect all indicators and patterns
        for timeframe, data in all_timeframe_data.items():
            if not data.empty:
                # Get technical indicators first
                indicators = self.stock_collector.calculate_technical_indicators(data)
                indicators_by_timeframe[timeframe] = indicators

                # Then detect patterns with context from indicators
                patterns = self.pattern_recognizer.detect_patterns(data,lookback_periods=5)

                if patterns:
                    # Process patterns with indicator context
                    signal = self.evaluate_patterns(patterns, data, indicators)
                    if signal:
                        # Add additional context to the signal
                        signal['additional_context'] = {
                            'rsi': indicators.get('rsi', 50),
                            'volume_ratio': indicators.get('volume_ratio', 1.0),
                            'ma_trend': indicators.get('ma_trend', 'neutral'),
                            'macd_hist': indicators.get('macd_hist', 0)
                        }
                        signals[timeframe] = signal

        # Second pass: adjust signals based on overall context
        self._adjust_signals_with_context(signals, indicators_by_timeframe)

        # Combine signals from all timeframes
        if signals:
            return self.generate_combined_signal(signals)
        return {}

    def _adjust_signals_with_context(self, signals, indicators_by_timeframe):
        """Adjust signal confidence based on cross-timeframe context."""
        if not signals or len(signals) < 2:
            return

        # Check for alignment between higher and lower timeframes
        if 'long_term' in signals and 'medium_term' in signals:
            long_action = signals['long_term']['action']
            medium_action = signals['medium_term']['action']

            # If long-term trend matches medium-term signal, boost confidence
            if long_action == medium_action:
                if signals['medium_term']['confidence'] == 'medium':
                    signals['medium_term']['confidence'] = 'medium_high'
                elif signals['medium_term']['confidence'] == 'medium_high':
                    signals['medium_term']['confidence'] = 'high'
            # If they conflict, reduce confidence
            elif long_action != 'WATCH' and medium_action != 'WATCH' and long_action != medium_action:
                if signals['medium_term']['confidence'] == 'high':
                    signals['medium_term']['confidence'] = 'medium_high'
                elif signals['medium_term']['confidence'] == 'medium_high':
                    signals['medium_term']['confidence'] = 'medium'

        # Check if short-term signal is against both higher timeframes
        if 'short_term' in signals and 'medium_term' in signals and 'long_term' in signals:
            short_action = signals['short_term']['action']
            medium_action = signals['medium_term']['action']
            long_action = signals['long_term']['action']

            if medium_action == long_action and short_action != medium_action:
                # Short-term signal going against both higher timeframes
                # This could be a false signal or a reversal starting
                # Reduce confidence unless volume is very high
                short_vol_ratio = indicators_by_timeframe.get('short_term', {}).get('volume_ratio', 1.0)
                if short_vol_ratio < 2.0:  # Not enough volume to confirm reversal
                    if signals['short_term']['confidence'] == 'high':
                        signals['short_term']['confidence'] = 'medium_high'
                    elif signals['short_term']['confidence'] == 'medium_high':
                        signals['short_term']['confidence'] = 'medium'

    def evaluate_patterns(self, patterns: Dict, data: pd.DataFrame, indicators: Dict) -> Dict:
        """Evaluate patterns and determine the strongest signal for a timeframe."""
        if not patterns:
            return None

        current_price = data['close'].iloc[-1]
        strongest_signal = None
        highest_confidence = 0

        for pattern_name, occurrences in patterns.items():
            for occurrence in occurrences:
                # Check if occurrence is at the end of the data (recent)
                # Fix the error by checking the type of occurrence['index']
                if isinstance(occurrence['index'], int):
                    # If index is an integer, compare directly with data length
                    is_recent = occurrence['index'] >= len(data) - 5
                else:
                    # If index is another type (might be from old code), handle accordingly
                    try:
                        is_recent = len(occurrence['index']) > len(data) - 5
                    except:
                        logger.warning(f"Unexpected index type: {type(occurrence['index'])}")
                        is_recent = False

                if is_recent:  # Process only recent patterns
                    signal = self.pattern_recognizer.get_trading_signal(
                        pattern_name,
                        occurrence['signal'],
                        current_price,
                        atr=indicators.get('atr'),
                        volume_ratio=indicators.get('volume_ratio', 1.0),
                        additional_indicators=indicators
                    )

                    # Compare confidence levels
                    confidence_value = self.confidence_thresholds.get(signal['confidence'], 0.5)
                    if confidence_value > highest_confidence:
                        strongest_signal = signal
                        strongest_signal['pattern_name'] = pattern_name
                        strongest_signal['detection_time'] = occurrence['timestamp']
                        highest_confidence = confidence_value

        return strongest_signal

    def generate_combined_signal(self, signals: Dict) -> Dict:
        """Combine signals from all timeframes to generate a comprehensive trading decision."""
        if not signals:
            return None

        combined_strength = 0
        action_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0, 'WATCH': 0}

        for timeframe, signal in signals.items():
            if signal and signal['action']:
                weight = self.timeframe_weights.get(timeframe, 0.33)
                confidence_value = self.confidence_thresholds.get(signal['confidence'], 0.5)

                if signal['action'] == 'BUY':
                    combined_strength += weight * confidence_value
                    action_scores['BUY'] += weight * confidence_value
                elif signal['action'] == 'SELL':
                    combined_strength -= weight * confidence_value
                    action_scores['SELL'] += weight * confidence_value
                elif signal['action'] == 'HOLD':
                    action_scores['HOLD'] += weight * confidence_value
                else:  # WATCH
                    action_scores['WATCH'] += weight * confidence_value

        # Determine final action
        if combined_strength > 0.6:
            action = 'BUY'
            confidence = 'high'
        elif combined_strength < -0.6:
            action = 'SELL'
            confidence = 'high'
        else:
            # Find the action with highest score
            action = max(action_scores, key=action_scores.get)
            if action_scores[action] > 0.5:
                confidence = 'medium_high'
            else:
                confidence = 'medium'

        return {
            'action': action,
            'strength': combined_strength,
            'confidence': confidence,
            'timeframe_signals': signals,
            'action_scores': action_scores
        }

    def should_notify(self, ticker: str) -> bool:
        """Check if we should send a notification for this ticker."""
        if ticker not in self.last_notification:
            return True

        time_since_last = datetime.now() - self.last_notification[ticker]
        return time_since_last.total_seconds() > self.notification_cooldown

    def should_notify_strong_signal(self, ticker: str, signal: Dict) -> bool:
        """Determine if signal is strong enough to warrant notification with enhanced intelligence."""
        if not signal or not signal['action']:
            return False

        # Skip HOLD or WATCH signals unless they're very high confidence
        if signal['action'] in ['HOLD', 'WATCH'] and signal['confidence'] not in ['very_high']:
            return False

        # First check - timeframe alignment
        timeframe_signals = signal['timeframe_signals']
        aligned_signal = self.check_signal_alignment(timeframe_signals)
        if not aligned_signal:
            # Signals across timeframes conflict significantly
            return False

        # Check signal confidence against threshold with adaptive criteria
        if signal['confidence'] in ['high', 'very_high']:
            # Check volume confirmation
            if 'short_term' in timeframe_signals and timeframe_signals['short_term'].get('additional_context', {}).get('volume_ratio', 1.0) < 1.2:
                # Low volume relative to average, reduce signal importance
                if signal['confidence'] != 'very_high':
                    return False

            # Check if signal matches overall market trend (if we had market data)
            # This would be a good addition in the future

            return self.should_notify(ticker)

        elif signal['confidence'] == 'medium_high' and signal['action'] in ['BUY', 'SELL']:
            # For medium-high confidence, need stronger confirmation

            # Check if short and medium timeframes align
            if 'short_term' in timeframe_signals and 'medium_term' in timeframe_signals:
                short_action = timeframe_signals['short_term']['action']
                medium_action = timeframe_signals['medium_term']['action']

                if short_action == medium_action == signal['action']:
                    # Strong alignment between timeframes

                    # Check technical indicators
                    if 'short_term' in timeframe_signals and 'additional_context' in timeframe_signals['short_term']:
                        context = timeframe_signals['short_term']['additional_context']

                        # Check RSI for confirmation
                        if signal['action'] == 'BUY' and context.get('rsi', 50) < 30:
                            # Oversold condition supports buy signal
                            return self.should_notify(ticker)
                        elif signal['action'] == 'SELL' and context.get('rsi', 50) > 70:
                            # Overbought condition supports sell signal
                            return self.should_notify(ticker)

                        # Check MA trend alignment
                        if signal['action'] == 'BUY' and context.get('ma_trend') == 'bullish':
                            return self.should_notify(ticker)
                        elif signal['action'] == 'SELL' and context.get('ma_trend') == 'bearish':
                            return self.should_notify(ticker)

        # More conservative on medium confidence signals
        elif signal['confidence'] == 'medium':
            # Only notify on medium confidence if all timeframes align perfectly
            # AND other conditions are met (like high volume, strong MA trend, etc.)
            all_actions = [s['action'] for s in timeframe_signals.values() if s.get('action')]
            if all(action == signal['action'] for action in all_actions) and len(all_actions) >= 3:
                # All timeframes showing same signal

                # Additional confirmation from indicators
                context = next((s.get('additional_context', {}) for s in timeframe_signals.values()
                                if 'additional_context' in s), {})

                # Strong volume confirmation
                if context.get('volume_ratio', 1.0) > 2.0:
                    # Very high volume
                    return self.should_notify(ticker)

        return False

    def check_signal_alignment(self, timeframe_signals: Dict) -> bool:
        """Check if signals across timeframes are aligned or conflicting."""
        if not timeframe_signals or len(timeframe_signals) < 2:
            return True  # Not enough data to determine conflict

        actions = {}
        # Count occurrences of each action
        for tf, signal in timeframe_signals.items():
            if 'action' in signal:
                action = signal['action']
                actions[action] = actions.get(action, 0) + 1

        # Check if we have conflicting signals (Buy vs Sell)
        if 'BUY' in actions and 'SELL' in actions:
            return False  # Direct conflict

        # Check if majority of timeframes agree
        total_signals = sum(actions.values())
        highest_count = max(actions.values()) if actions else 0

        # If more than 66% of signals agree, consider it aligned
        return highest_count / total_signals >= 0.66 if total_signals > 0 else True

    def get_market_context(self, ticker: str) -> Dict:
        """Get overall market context to confirm signals.
        This is a placeholder for future implementation."""
        # This could fetch S&P 500 or sector-specific data
        # to check if the stock's movement aligns with broader market
        return {
            'market_trend': 'bullish',  # or bearish, neutral
            'sector_trend': 'bullish',  # or bearish, neutral
            'market_volatility': 'low',  # or high, medium
        }

    def format_actionable_notification(self, ticker: str, combined_signal: Dict) -> str:
        """Format a highly actionable notification with clear trading plan."""
        current_time = datetime.now()

        # Determine confidence emoji
        confidence_emoji = "🟢" if combined_signal['confidence'] in ['high', 'very_high'] else \
            "🟡" if combined_signal['confidence'] == 'medium_high' else "🟠"

        # Action emoji
        action_emoji = "🟩" if combined_signal['action'] == 'BUY' else \
            "🟥" if combined_signal['action'] == 'SELL' else "⬜"

        message = (
            f"{action_emoji} *{ticker}* {confidence_emoji}\n\n"
            f"📊 *{combined_signal['action']}* (Confidence: {combined_signal['confidence'].upper()})\n"
        )

        # Get price data from short-term timeframe
        short_term_data = None
        if 'short_term' in combined_signal['timeframe_signals']:
            short_term_data = combined_signal['timeframe_signals']['short_term']

            # Add entry, stop loss and take profit
            if short_term_data.get('entry_price') and combined_signal['action'] in ['BUY', 'SELL']:
                entry = short_term_data['entry_price']
                message += f"💰 Entry: ${entry:.2f}\n"

                if short_term_data.get('stop_loss') and short_term_data.get('take_profit'):
                    sl = short_term_data['stop_loss']
                    tp = short_term_data['take_profit']
                    risk = entry - sl if combined_signal['action'] == 'BUY' else sl - entry
                    reward = tp - entry if combined_signal['action'] == 'BUY' else entry - tp

                    # Calculate risk percentage
                    risk_pct = (risk / entry) * 100

                    message += f"🛑 Stop Loss: ${sl:.2f} ({risk_pct:.1f}%)\n"
                    message += f"🎯 Take Profit: ${tp:.2f}\n"

                    if short_term_data.get('risk_reward_ratio'):
                        message += f"⚖️ R/R Ratio: 1:{short_term_data['risk_reward_ratio']:.1f}\n"

        # Key reason for the signal
        message += f"\n📝 *Signal Based On:*\n"

        # Add primary pattern
        if 'pattern_name' in next(iter(combined_signal['timeframe_signals'].values()), {}):
            primary_pattern = next(iter(combined_signal['timeframe_signals'].values()))['pattern_name']
            message += f"• Pattern: {primary_pattern}\n"

        # Add timeframe confluence
        aligned_timeframes = []
        for tf, signal in combined_signal['timeframe_signals'].items():
            if signal['action'] == combined_signal['action']:
                aligned_timeframes.append(self._format_timeframe_name(tf))

        if aligned_timeframes:
            message += f"• Aligned Timeframes: {', '.join(aligned_timeframes)}\n"

        # Add key technical indicators
        if short_term_data and 'additional_context' in short_term_data:
            context = short_term_data['additional_context']

            if 'rsi' in context:
                rsi_value = context['rsi']
                rsi_status = "Oversold" if rsi_value < 30 else "Overbought" if rsi_value > 70 else "Neutral"
                message += f"• RSI: {rsi_value:.0f} ({rsi_status})\n"

            if 'ma_trend' in context:
                message += f"• MA Trend: {context['ma_trend'].upper()}\n"

            if 'volume_ratio' in context and context['volume_ratio'] > 1.5:
                message += f"• Volume: {context['volume_ratio']:.1f}x average\n"

        # Add useful notes based on action type
        message += f"\n📋 *Trading Notes:*\n"
        if combined_signal['action'] == 'BUY':
            message += "• Consider scaling in rather than full position\n"
            message += "• Monitor for continuation patterns\n"
        elif combined_signal['action'] == 'SELL':
            message += "• Check for overall market weakness\n"
            message += "• Beware of potential short squeezes\n"

        message += (
            f"\n⏰ {current_time.strftime('%H:%M:%S')}\n"
            f"\n⚠️ Trading involves risk. This is not financial advice."
        )

        return message

    def _format_timeframe_name(self, timeframe: str) -> str:
        """Convert timeframe key to display name."""
        timeframe_names = {
            'long_term': '📅 Daily',
            'medium_term': '🕐 Hourly',
            'short_term': '⏱️ 15-minute'
        }
        return timeframe_names.get(timeframe, timeframe)

    async def process_detected_patterns(self, ticker: str, patterns: Dict, data: pd.DataFrame):
        """Process detected patterns and send notifications (legacy method)."""
        current_price = data['close'].iloc[-1]

        for pattern_name, occurrences in patterns.items():
            for occurrence in occurrences:
                # Handle both datetime and timestamp formats
                pattern_time = occurrence['timestamp']
                if isinstance(pattern_time, int) or isinstance(pattern_time, float):
                    pattern_time = datetime.fromtimestamp(pattern_time)

                # Only process recent patterns
                if (datetime.now() - pattern_time) < timedelta(hours=24):
                    signal = self.pattern_recognizer.get_trading_signal(
                        pattern_name,
                        occurrence['signal'],
                        current_price
                    )

                    if signal['action'] and self.should_notify(ticker):
                        message = self._format_pattern_notification(
                            ticker, pattern_name, occurrence, signal, current_price
                        )
                        await self.bot_notifier.send_pattern_notification(message)
                        self.last_notification[ticker] = datetime.now()

    def _format_pattern_notification(self, ticker: str, pattern_name: str,
                                     occurrence: Dict, signal: Dict, current_price: float) -> str:
        """Format the notification message (legacy method)."""
        # Handle both datetime and timestamp formats for formatting
        pattern_time = occurrence['timestamp']
        if isinstance(pattern_time, int) or isinstance(pattern_time, float):
            pattern_time = datetime.fromtimestamp(pattern_time)

        message = (
            f"🎯 *PATTERN ALERT: {ticker}*\n\n"
            f"🔍 Pattern: *{pattern_name}*\n"
            f"💰 Current Price: ${current_price:.2f}\n"
            f"📈 Signal Strength: {occurrence['signal']}\n"
            f"📊 Action: *{signal['action']}*\n"
            f"📝 Reason: {signal['reason']}\n"
            f"⚖️ Confidence: {signal['confidence'].upper()}\n\n"
        )

        if signal['action'] in ['BUY', 'SELL']:
            message += (
                f"📊 *Trade Details:*\n"
                f"• Entry Price: ${signal['entry_price']:.2f}\n"
                f"• Stop Loss: ${signal['stop_loss']:.2f}\n"
                f"• Take Profit: ${signal['take_profit']:.2f}\n\n"
            )

        message += (
            f"⏰ Pattern Detected: {pattern_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"⚠️ Trading involves risks. This is not financial advice."
        )

        return message