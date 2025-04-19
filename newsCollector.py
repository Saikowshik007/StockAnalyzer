import asyncio
import sys
import logging
from typing import List
import datetime

import requests
from cachetools import TTLCache

from BiztocScraper import BiztocScraper
from news import News
from newspaper import Article
from summarier import Summarizer
from notifier import Notifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class NewsMonitor:
    def __init__(self, cache_size: int = 100, cache_ttl: int = 86400):
        self.headlines_cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
        self.summarizer = Summarizer()
        self.notifier = Notifier()
        self.is_first_run = True
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    def get_new_headlines(self) -> List[News]:
        """Fetch and process new headlines that aren't in the cache."""
        try:
            b = BiztocScraper(sources=['benzinga', 'cnbc', 'yahoo_finance','google_business','cnn','fox'])
            current_headlines = b.latest()
            new_news_objects = []

            for headline, link in zip(current_headlines['headline'], current_headlines['link']):
                # On first run, just populate the cache without processing
                if self.is_first_run:
                    self.headlines_cache[link] = True
                    continue

                skip_domains = ['investors.com', 'wsj.com', 'bloomberg.com', 'investing.com']
                if any(domain in link for domain in skip_domains):
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
                        # Handle 401 and 403 errors (paid content)
                        if e.response.status_code in (401, 403):
                            logger.info(f"Skipping paid/restricted content at {link}: {e}")
                            # Still add to cache so we don't try again
                            self.headlines_cache[link] = True
                        else:
                            logger.error(f"HTTP error processing article {link}: {e}")
                    except Exception as e:
                        logger.error(f"Error processing article {link}: {e}")
                        continue

            # After initial cache population, mark first run as complete
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

                summary = self.summarizer.generate_summary(item.article)
                item.set_summary(summary)
                await self.notifier.send_telegram_message(item)
                break

                logger.info(f"Summary generated successfully: {item.article.title}")
                logger.info("="*80)
            except Exception as e:
                logger.error(f"Error processing item: {e}")

    def is_business_hours(self) -> bool:
        """Check if current time is within business hours (weekdays 8 AM - 3 PM)."""
        now = datetime.datetime.now()

        # Check if it's a weekday (Monday=0, Sunday=6)
        is_weekday = now.weekday() < 5

        # Check if time is between 8 AM and 3 PM
        current_hour = now.hour
        is_work_hours = 8 <= current_hour < 15

        return is_weekday and is_work_hours

    async def monitor(self, interval_seconds: int = 30):
        """Main monitoring loop to check for and process new headlines."""
        logger.info(f"Starting news monitoring. Checking every {interval_seconds} seconds during weekdays (Monday-Friday) from 8 AM to 3 PM...")

        while True:
            try:
                # Check if we should be running
                if self.is_business_hours():
                    # Get new headlines
                    new_headlines = self.get_new_headlines()

                    if new_headlines:
                        logger.info(f"Found {len(new_headlines)} new headlines!")
                        await self.process_headlines(new_headlines)
                    else:
                        logger.info("No new headlines found.")
                else:
                    # Outside business hours - log if just transitioned out of business hours
                    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    logger.info(f"Outside business hours at {current_time}. Waiting...")

                    # If we're outside business hours, wait longer before checking again
                    await asyncio.sleep(300)  # Sleep for 5 minutes when outside business hours
                    continue  # Skip the regular interval and check again

            except Exception as e:
                logger.error(f"Unexpected error in monitoring loop: {e}")

            await asyncio.sleep(interval_seconds)

def main():
    """Entry point for the application."""
    monitor = NewsMonitor()

    try:
        asyncio.run(monitor.monitor())
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")

if __name__ == "__main__":
    main()