import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd

from services.pattern_recognition import TalibPatternRecognition

logger = logging.getLogger(__name__)

class TalibPatternMonitor:
    def __init__(self, config, db_manager, stock_collector, bot_notifier):
        self.config = config
        self.db_manager = db_manager
        self.stock_collector = stock_collector
        self.bot_notifier = bot_notifier
        self.pattern_recognizer = TalibPatternRecognition()
        self.last_notification = {}
        self.monitoring_interval = config.get('pattern_monitor.interval_seconds', 300)
        self.notification_cooldown = config.get('pattern_monitor.notification_cooldown', 3600)

    async def monitor_patterns(self):
        """Continuously monitor for patterns in watchlist stocks."""
        while True:
            try:
                watchlist = self.db_manager.get_active_watchlist()

                for ticker in watchlist:
                    data = self.stock_collector.get_data(ticker)
                    if not data.empty:
                        detected_patterns = self.pattern_recognizer.detect_patterns(data)

                        if detected_patterns:
                            await self.process_detected_patterns(ticker, detected_patterns, data)

                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in pattern monitoring: {e}")
                await asyncio.sleep(60)

    async def process_detected_patterns(self, ticker: str, patterns: Dict, data: pd.DataFrame):
        """Process detected patterns and send notifications."""
        current_price = data['close'].iloc[-1]

        for pattern_name, occurrences in patterns.items():
            for occurrence in occurrences:
                # Only process recent patterns
                if (datetime.now() - occurrence['timestamp']) < timedelta(hours=24):
                    signal = self.pattern_recognizer.get_trading_signal(
                        pattern_name,
                        occurrence['signal'],
                        current_price
                    )

                    if signal['action'] and self.should_notify(ticker):
                        await self.send_pattern_notification(
                            ticker,
                            pattern_name,
                            occurrence,
                            signal,
                            current_price
                        )
                        self.last_notification[ticker] = datetime.now()

    def should_notify(self, ticker: str) -> bool:
        """Check if we should send a notification for this ticker."""
        if ticker not in self.last_notification:
            return True

        time_since_last = datetime.now() - self.last_notification[ticker]
        return time_since_last.total_seconds() > self.notification_cooldown

    async def send_pattern_notification(self, ticker: str, pattern_name: str,
                                        occurrence: Dict, signal: Dict, current_price: float):
        """Send Telegram notification about the pattern and trading signal."""
        message = self._format_pattern_notification(ticker, pattern_name, occurrence, signal, current_price)
        await self.bot_notifier.send_pattern_message(message)

    def _format_pattern_notification(self, ticker: str, pattern_name: str,
                                     occurrence: Dict, signal: Dict, current_price: float) -> str:
        """Format the notification message."""
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
            f"⏰ Pattern Detected: {occurrence['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"⚠️ Trading involves risks. This is not financial advice."
        )

        return message