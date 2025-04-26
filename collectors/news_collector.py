# collectors/news_collector.py
import asyncio
import logging
from typing import List
import datetime
import requests
from cachetools import TTLCache

from .biztoc_scraper import BiztocScraper
from services.news import News
from newspaper import Article
from services.summarizer import Summarizer
from services.analysis_service import AnalysisService

logger = logging.getLogger(__name__)

class NewsMonitor:
    def __init__(self, config, db_manager, bot_notifier):
        self.config = config
        self.db_manager = db_manager
        self.bot_notifier = bot_notifier
        self.headlines_cache = TTLCache(maxsize=100, ttl=86400)

        # Get summarizer config and OpenAI config separately
        summarizer_config = config.get('summarizer', {})
        openai_config = config.get('openai', {})

        # Add OpenAI API key to summarizer config if available
        if 'api_key' in openai_config:
            summarizer_config['api_key'] = openai_config['api_key']

        self.summarizer = Summarizer(config=summarizer_config)
        self.analysis_service = AnalysisService()
        self.is_first_run = True
        self.sources = config.get('news_collector.sources', [])
        self.skip_domains = config.get('news_collector.skip_domains', [])

    def get_new_headlines(self) -> List[News]:
        """Fetch and process new headlines that aren't in the cache."""
        try:
            scraper = BiztocScraper(sources=self.sources)
            current_headlines = scraper.latest()
            new_news_objects = []

            for headline, link in zip(current_headlines['headline'], current_headlines['link']):
                # On first run, just populate the cache
                if self.is_first_run:
                    self.headlines_cache[link] = True
                    continue

                if any(domain in link for domain in self.skip_domains):
                    logger.info(f"Skipping known protected/paywalled site: {link}")
                    self.headlines_cache[link] = True
                    continue

                # For subsequent runs, process new links
                if link not in self.headlines_cache:
                    try:
                        article = Article(link)
                        article.download()
                        article.parse()

                        news_obj = News(article)
                        new_news_objects.append(news_obj)
                        self.headlines_cache[link] = True
                    except requests.exceptions.HTTPError as e:
                        if e.response.status_code in (401, 403):
                            logger.info(f"Skipping paid/restricted content at {link}: {e}")
                            self.headlines_cache[link] = True
                        else:
                            logger.error(f"HTTP error processing article {link}: {e}")
                    except Exception as e:
                        logger.error(f"Error processing article {link}: {e}")
                        continue

            if self.is_first_run:
                self.is_first_run = False
                logger.info(f"Initial run complete. Cached {len(current_headlines['link'])} headlines. Now monitoring for new articles...")

            return new_news_objects
        except Exception as e:
            logger.error(f"Error fetching headlines: {e}")
            return []

    async def process_headlines(self, news_items: List[News]):
        """Process a list of news items: generate summaries and send notifications."""
        for item in news_items:
            try:
                logger.info(f"New article: {item.article.title}")

                # Generate summary which will now include relevance check
                summary = self.summarizer.generate_summary(item.article)

                # Check if the AI determined this isn't financial news
                if summary.strip().startswith("NOT_FINANCIAL_NEWS"):
                    logger.info(f"Skipping non-financial article: {item.article.title}")
                    continue

                # Set the summary for the news item
                item.set_summary(summary)

                # Extract components for database storage
                components = self.analysis_service.extract_components(summary)

                # Save to database
                article_data = {
                    'title': item.article.title,
                    'url': item.article.url,
                    'content': item.article.text,
                    'summary': summary,
                    'sentiment_category': components['sentiment_category'],
                    'sentiment_rating': components['sentiment_rating'],
                    'collected_at': item.timestamp,
                    'processed_at': datetime.datetime.now()
                }
                self.db_manager.save_news_article(article_data)

                # Send notification using TelegramBot
                await self.bot_notifier.send_news_notification(item)

                logger.info(f"Summary generated successfully: {item.article.title}")
                logger.info("="*80)
            except Exception as e:
                logger.error(f"Error processing item: {e}")

    def is_business_hours(self) -> bool:
        """Check if current time is within business hours (weekdays 8 AM - 3 PM)."""
        now = datetime.datetime.now()
        is_weekday = now.weekday() < 5
        current_hour = now.hour

        # Get business hours from config
        start_hour = self.config.get('news_collector.business_hours.start', 8)
        end_hour = self.config.get('news_collector.business_hours.end', 15)

        is_work_hours = start_hour <= current_hour < end_hour
        return is_weekday and is_work_hours

    async def monitor(self):
        """Main monitoring loop to check for and process new headlines."""
        interval_seconds = self.config.get('news_collector.interval_seconds', 30)

        logger.info(f"Starting news monitoring. Checking every {interval_seconds} seconds during weekdays (Monday-Friday) from 8 AM to 3 PM...")

        try:
            while not getattr(self, 'should_stop', False):
                if self.is_business_hours():
                    new_headlines = self.get_new_headlines()

                    if new_headlines:
                        logger.info(f"Found {len(new_headlines)} new headlines!")
                        await self.process_headlines(new_headlines)
                    else:
                        logger.info("No new headlines found.")
                    await asyncio.sleep(interval_seconds)
                else:
                    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    logger.info(f"Outside business hours at {current_time}. Waiting...")
                    # Check more frequently to allow for graceful shutdown
                    await asyncio.sleep(300)
        except asyncio.CancelledError:
            logger.info("News monitor task cancelled")
            # Re-raise to properly handle cancellation
            raise
        except Exception as e:
            logger.error(f"Unexpected error in monitoring loop: {e}")
            raise

    def stop(self):
        """Signal the monitor to stop."""
        self.should_stop = True