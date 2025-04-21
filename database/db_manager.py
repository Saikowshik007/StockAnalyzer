import os
import shutil
from datetime import datetime
from typing import List, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from .models import Base, NewsArticle, MarketImplication, ActionableInsight, StockData, WatchlistItem
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, config: dict = None):
        if config is None:
            config = {}

        self.db_type = config.get('type', 'sqlite')
        self.db_name = config.get('name', 'financial_monitor.db')
        self.backup_enabled = config.get('backup_enabled', True)
        self.backup_path = config.get('backup_path', './backups')

        try:
            # Create database file if it doesn't exist
            os.makedirs(os.path.dirname(self.db_name) or '.', exist_ok=True)

            self.engine = create_engine(f'sqlite:///{self.db_name}')
            Base.metadata.create_all(self.engine)
            self.SessionLocal = sessionmaker(bind=self.engine)

            logger.info(f"Database initialized successfully at {self.db_name}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    def backup_database(self):
        """Create a backup of the database."""
        if not self.backup_enabled:
            return

        try:
            if not os.path.exists(self.backup_path):
                os.makedirs(self.backup_path)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = os.path.join(self.backup_path, f'backup_{timestamp}.db')

            if os.path.exists(self.db_name):
                shutil.copy2(self.db_name, backup_file)
                logger.info(f"Database backed up to {backup_file}")
            else:
                logger.warning(f"Database file {self.db_name} not found for backup")
        except Exception as e:
            logger.error(f"Error backing up database: {e}")

    def save_news_article(self, article_data: dict) -> NewsArticle:
        """Save a news article to the database."""
        session = self.get_session()
        try:
            article = NewsArticle(**article_data)
            session.add(article)
            session.commit()
            return article
        finally:
            session.close()

    def get_recent_articles(self, limit: int = 10) -> List[NewsArticle]:
        """Get the most recent news articles."""
        session = self.get_session()
        try:
            return session.query(NewsArticle).order_by(NewsArticle.collected_at.desc()).limit(limit).all()
        finally:
            session.close()

    def article_exists(self, url: str) -> bool:
        """Check if an article already exists in the database."""
        session = self.get_session()
        try:
            return session.query(NewsArticle).filter_by(url=url).first() is not None
        finally:
            session.close()

    def save_stock_data(self, stock_data: dict) -> StockData:
        """Save stock data to the database."""
        session = self.get_session()
        try:
            stock = StockData(**stock_data)
            session.add(stock)
            session.commit()
            return stock
        finally:
            session.close()

    def get_stock_history(self, ticker: str, limit: int = 100) -> List[StockData]:
        """Get recent stock history for a ticker."""
        session = self.get_session()
        try:
            return session.query(StockData).filter_by(ticker=ticker).order_by(StockData.timestamp.desc()).limit(limit).all()
        finally:
            session.close()

    def add_to_watchlist(self, ticker: str) -> WatchlistItem:
        """Add a stock to the watchlist."""
        session = self.get_session()
        try:
            existing = session.query(WatchlistItem).filter_by(ticker=ticker).first()
            if existing:
                existing.active = True
                session.commit()
                return existing

            item = WatchlistItem(ticker=ticker)
            session.add(item)
            session.commit()
            return item
        finally:
            session.close()

    def remove_from_watchlist(self, ticker: str) -> bool:
        """Remove a stock from the watchlist."""
        session = self.get_session()
        try:
            item = session.query(WatchlistItem).filter_by(ticker=ticker).first()
            if item:
                item.active = False
                session.commit()
                return True
            return False
        finally:
            session.close()

    def get_active_watchlist(self) -> List[str]:
        """Get all active stocks in the watchlist."""
        session = self.get_session()
        try:
            items = session.query(WatchlistItem).filter_by(active=True).all()
            return [item.ticker for item in items]
        finally:
            session.close()