# services/sentiment_tracker.py
import logging
from datetime import datetime, time, timedelta
from sqlalchemy import func
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class SentimentTracker:
    """Tracks sentiment over time periods and provides aggregated market metrics."""

    def __init__(self, db_manager):
        """Initialize the sentiment tracker."""
        self.db_manager = db_manager
        self.today_sentiment = None

    def update_daily_sentiment(self) -> Optional[Dict]:
        """Calculate and update sentiment for the current day."""
        session = self.db_manager.get_session()
        try:
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

            # Check if record exists for today
            from database.models import DailySentiment
            sentiment_record = session.query(DailySentiment).filter(
                func.date(DailySentiment.date) == func.date(today)
            ).first()

            if not sentiment_record:
                sentiment_record = DailySentiment(date=today, average_sentiment=5.0)
                session.add(sentiment_record)

            # Get all articles from today
            from database.models import NewsArticle
            today_articles = session.query(NewsArticle).filter(
                func.date(NewsArticle.collected_at) == func.date(today),
                NewsArticle.sentiment_rating.isnot(None)
            ).all()

            if today_articles:
                # Calculate average sentiment
                total_sentiment = sum(article.sentiment_rating for article in today_articles)
                avg_sentiment = total_sentiment / len(today_articles)

                # Update record
                sentiment_record.average_sentiment = avg_sentiment
                sentiment_record.article_count = len(today_articles)
                sentiment_record.min_sentiment = min(article.sentiment_rating for article in today_articles)
                sentiment_record.max_sentiment = max(article.sentiment_rating for article in today_articles)

                # Calculate time-based sentiment
                morning_articles = [a for a in today_articles if a.collected_at.time() < time(11, 0)]
                midday_articles = [a for a in today_articles if time(11, 0) <= a.collected_at.time() < time(13, 0)]
                afternoon_articles = [a for a in today_articles if a.collected_at.time() >= time(13, 0)]

                if morning_articles:
                    sentiment_record.morning_sentiment = sum(a.sentiment_rating for a in morning_articles) / len(morning_articles)

                if midday_articles:
                    sentiment_record.midday_sentiment = sum(a.sentiment_rating for a in midday_articles) / len(midday_articles)

                if afternoon_articles:
                    sentiment_record.afternoon_sentiment = sum(a.sentiment_rating for a in afternoon_articles) / len(afternoon_articles)

                # Track affected tickers - find most mentioned tickers in articles
                ticker_mentions = {}
                for article in today_articles:
                    if article.title:
                        # Extract potential ticker symbols (uppercase words 1-5 chars)
                        import re
                        tickers = re.findall(r'\b[A-Z]{1,5}\b', article.title)
                        for ticker in tickers:
                            if ticker in ['A', 'I', 'FOR', 'AT', 'ON', 'THE', 'IN', 'BY']:
                                continue  # Skip common words
                            ticker_mentions[ticker] = ticker_mentions.get(ticker, 0) + 1

                # Get top mentioned tickers
                if ticker_mentions:
                    top_tickers = sorted(ticker_mentions.items(), key=lambda x: x[1], reverse=True)[:5]
                    sentiment_record.affected_tickers = ','.join([t[0] for t in top_tickers])

                # Store updated record
                session.commit()

                # Cache today's sentiment
                self.today_sentiment = sentiment_record

                return {
                    'average_sentiment': sentiment_record.average_sentiment,
                    'article_count': sentiment_record.article_count,
                    'morning_sentiment': sentiment_record.morning_sentiment,
                    'midday_sentiment': sentiment_record.midday_sentiment,
                    'afternoon_sentiment': sentiment_record.afternoon_sentiment,
                    'affected_tickers': sentiment_record.affected_tickers
                }
            else:
                logger.info("No articles with sentiment ratings found for today")
                session.commit()  # Still save the empty record
                return None

        except Exception as e:
            logger.error(f"Error updating daily sentiment: {e}")
            session.rollback()
            return None
        finally:
            session.close()

    def get_sentiment_trend(self, days: int = 7) -> List:
        """Get sentiment trend for the last N days."""
        session = self.db_manager.get_session()
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            from database.models import DailySentiment
            sentiment_records = session.query(DailySentiment).filter(
                DailySentiment.date >= start_date,
                DailySentiment.date <= end_date
            ).order_by(DailySentiment.date).all()

            return sentiment_records
        except Exception as e:
            logger.error(f"Error getting sentiment trend: {e}")
            return []
        finally:
            session.close()

    def get_current_sentiment(self) -> Dict:
        """Get the current sentiment state including trends."""
        # Update daily sentiment first
        today_sentiment_data = self.update_daily_sentiment()

        if not today_sentiment_data:
            return {
                'status': 'No data',
                'value': 5.0,
                'trend': 'neutral',
                'article_count': 0,
                'description': 'No sentiment data available for today'
            }

        # Get recent trend
        trend_data = self.get_sentiment_trend(3)

        # Determine trend direction
        trend = 'neutral'
        if len(trend_data) >= 2:
            prev_sentiment = trend_data[-2].average_sentiment if len(trend_data) > 1 else None
            if prev_sentiment:
                if today_sentiment_data['average_sentiment'] > prev_sentiment + 0.5:
                    trend = 'improving'
                elif today_sentiment_data['average_sentiment'] < prev_sentiment - 0.5:
                    trend = 'deteriorating'

        # Generate status based on sentiment value
        status = self._sentiment_value_to_status(today_sentiment_data['average_sentiment'])

        # Generate description
        description = self._generate_sentiment_description(today_sentiment_data, trend)

        return {
            'status': status,
            'value': today_sentiment_data['average_sentiment'],
            'trend': trend,
            'article_count': today_sentiment_data['article_count'],
            'morning_sentiment': today_sentiment_data.get('morning_sentiment'),
            'midday_sentiment': today_sentiment_data.get('midday_sentiment'),
            'afternoon_sentiment': today_sentiment_data.get('afternoon_sentiment'),
            'affected_tickers': today_sentiment_data.get('affected_tickers'),
            'description': description
        }

    def _sentiment_value_to_status(self, value: float) -> str:
        """Convert numerical sentiment to status string."""
        if value >= 8.0:
            return 'Very Positive'
        elif value >= 7.0:
            return 'Positive'
        elif value >= 6.0:
            return 'Slightly Positive'
        elif value > 4.0:
            return 'Neutral'
        elif value > 3.0:
            return 'Slightly Negative'
        elif value > 2.0:
            return 'Negative'
        else:
            return 'Very Negative'

    def _generate_sentiment_description(self, sentiment_data: Dict, trend: str) -> str:
        """Generate a description of the sentiment."""
        status = self._sentiment_value_to_status(sentiment_data['average_sentiment'])

        # Base description
        description = f"Market sentiment is {status.lower()} today based on {sentiment_data['article_count']} articles."

        # Add trend information
        if trend == 'improving':
            description += " Sentiment has been improving compared to previous days."
        elif trend == 'deteriorating':
            description += " Sentiment has been deteriorating compared to previous days."

        # Add time-based information if available
        time_descriptions = []
        if sentiment_data.get('morning_sentiment') is not None:
            morning_status = self._sentiment_value_to_status(sentiment_data['morning_sentiment'])
            time_descriptions.append(f"morning ({morning_status.lower()})")

        if sentiment_data.get('midday_sentiment') is not None:
            midday_status = self._sentiment_value_to_status(sentiment_data['midday_sentiment'])
            time_descriptions.append(f"midday ({midday_status.lower()})")

        if sentiment_data.get('afternoon_sentiment') is not None:
            afternoon_status = self._sentiment_value_to_status(sentiment_data['afternoon_sentiment'])
            time_descriptions.append(f"afternoon ({afternoon_status.lower()})")

        if time_descriptions:
            description += f" Sentiment during {', '.join(time_descriptions)}."

        return description

    def get_ticker_sentiment(self, ticker: str) -> Dict:
        """Get sentiment specific to a ticker symbol based on relevant articles."""
        session = self.db_manager.get_session()
        try:
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

            # Search for ticker in title or content
            from database.models import NewsArticle
            ticker_articles = session.query(NewsArticle).filter(
                func.date(NewsArticle.collected_at) >= func.date(today - timedelta(days=1)),
                NewsArticle.sentiment_rating.isnot(None),
                (NewsArticle.title.ilike(f'%{ticker}%') | NewsArticle.content.ilike(f'%{ticker}%'))
            ).all()

            if not ticker_articles:
                return {
                    'status': 'No data',
                    'value': 5.0,
                    'article_count': 0,
                    'description': f'No sentiment data available for {ticker} today'
                }

            # Calculate average sentiment
            total_sentiment = sum(article.sentiment_rating for article in ticker_articles)
            avg_sentiment = total_sentiment / len(ticker_articles)

            # Generate status based on sentiment value
            status = self._sentiment_value_to_status(avg_sentiment)

            return {
                'status': status,
                'value': avg_sentiment,
                'article_count': len(ticker_articles),
                'description': f"{ticker} sentiment is {status.lower()} based on {len(ticker_articles)} articles today."
            }

        except Exception as e:
            logger.error(f"Error getting ticker sentiment: {e}")
            return {
                'status': 'Error',
                'value': 5.0,
                'article_count': 0,
                'description': f'Error retrieving sentiment data for {ticker}'
            }
        finally:
            session.close()