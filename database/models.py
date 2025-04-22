from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class NewsArticle(Base):
    __tablename__ = 'news_articles'

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    url = Column(String, unique=True, nullable=False)
    source = Column(String)
    content = Column(Text)
    summary = Column(Text)
    sentiment_category = Column(String)
    sentiment_rating = Column(Float)
    collected_at = Column(DateTime, default=datetime.now)
    processed_at = Column(DateTime)

    # Relationship with market implications
    market_implications = relationship("MarketImplication", back_populates="article")
    actionable_insights = relationship("ActionableInsight", back_populates="article")

class MarketImplication(Base):
    __tablename__ = 'market_implications'

    id = Column(Integer, primary_key=True)
    article_id = Column(Integer, ForeignKey('news_articles.id'))
    implication = Column(Text)
    impact_level = Column(String)  # immediate, short-term, long-term

    article = relationship("NewsArticle", back_populates="market_implications")

class ActionableInsight(Base):
    __tablename__ = 'actionable_insights'

    id = Column(Integer, primary_key=True)
    article_id = Column(Integer, ForeignKey('news_articles.id'))
    insight = Column(Text)
    confidence_level = Column(String)  # low, medium, high

    article = relationship("NewsArticle", back_populates="actionable_insights")

class StockData(Base):
    __tablename__ = 'stock_data'

    id = Column(Integer, primary_key=True)
    ticker = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)

class WatchlistItem(Base):
    __tablename__ = 'watchlist'

    id = Column(Integer, primary_key=True)
    ticker = Column(String, unique=True, nullable=False)
    added_at = Column(DateTime, default=datetime.now)
    active = Column(Boolean, default=True)

class DailySentiment(Base):
    __tablename__ = 'daily_sentiment'

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)
    average_sentiment = Column(Float, nullable=False, default=5.0)
    article_count = Column(Integer, default=0)
    min_sentiment = Column(Float)
    max_sentiment = Column(Float)

    # Time-based sentiment tracking
    morning_sentiment = Column(Float)  # Pre-market to 11am
    midday_sentiment = Column(Float)  # 11am to 1pm
    afternoon_sentiment = Column(Float)  # 1pm to market close

    # Industry sentiment tracking (optional extension)
    tech_sentiment = Column(Float)
    finance_sentiment = Column(Float)
    healthcare_sentiment = Column(Float)
    energy_sentiment = Column(Float)

    # Related tickers with significant news impact
    affected_tickers = Column(String)