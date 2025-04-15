import logging

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
class BiztocScraper:
    """Scraper for Biztoc articles with source filtering"""
    def __init__(self, sources=None):
        self.base_url = "https://biztoc.com"
        # If no sources provided, don't filter
        self.sources = sources if sources else []

    def latest(self):
        """Fetch latest headlines from Biztoc, filtered by source if specified"""
        try:
            response = requests.get(self.base_url)
            soup = BeautifulSoup(response.content, 'html.parser')

            result = {
                'headline': [],
                'link': []
            }

            # If sources are specified, filter by those sources
            if self.sources:
                for source in self.sources:
                    # Find headers matching the source
                    source_headers = soup.select(f'h4.ic_{source.lower()}')

                    for header in source_headers:
                        # Find the container that holds both the header and the links
                        container = header.find_parent('div')
                        if container:
                            # Find all article links within this container
                            article_links = container.select('a[data-p][data-ts]')

                            for article in article_links:
                                headline = article.text.strip()
                                link = article.get('href')

                                if headline and link:
                                    result['headline'].append(headline)
                                    result['link'].append(link)
            else:
                # If no sources specified, get all articles
                article_links = soup.select('a[data-p][data-ts]')

                for article in article_links:
                    headline = article.text.strip()
                    link = article.get('href')

                    if headline and link:
                        result['headline'].append(headline)
                        result['link'].append(link)

            return result
        except Exception as e:
            logger.error(f"Error scraping Biztoc: {e}")
            return {'headline': [], 'link': []}