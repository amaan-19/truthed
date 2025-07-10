"""
Fixed RSS collector that prioritizes RSS content and handles connection issues gracefully.
Based on debugging results showing RSS content is often sufficient.
"""

import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse, urljoin
import aiohttp
import feedparser
from bs4 import BeautifulSoup
import ssl

from .base_collector import BaseCollector, CollectedArticle
from .config import CollectionConfig, SourceConfig


class FixedRSSCollector(BaseCollector):
    """
    Fixed RSS collector with improved error handling and content strategy.

    Strategy:
    1. Use RSS content when substantial (often 200+ chars)
    2. Only scrape when RSS content is insufficient
    3. Use requests fallback for scraping if aiohttp fails
    4. Handle connection errors gracefully
    """

    def __init__(self, config: CollectionConfig, source_config: SourceConfig):
        super().__init__(config, source_config)

        if not source_config.feed_urls:
            raise ValueError("RSS collector requires feed_urls in source_config")

        self.feed_urls = source_config.feed_urls
        self.min_rss_content_length = 150  # Use RSS content if this long or more
        self.enable_scraping = True  # Can disable if RSS content is sufficient

        self.logger.info(f"Fixed RSS collector initialized with {len(self.feed_urls)} feeds")

    async def collect_articles(self, session: aiohttp.ClientSession,
                             max_per_feed: Optional[int] = None) -> List[CollectedArticle]:
        """Collect articles with improved error handling"""
        self.logger.info(f"Starting RSS collection from {len(self.feed_urls)} feeds")

        max_per_feed = max_per_feed or self.source_config.max_articles_per_request

        for i, feed_url in enumerate(self.feed_urls, 1):
            if len(self.collected_articles) >= self.source_config.max_articles:
                self.logger.info(f"Reached max articles limit ({self.source_config.max_articles})")
                break

            feed_name = self._get_feed_name(feed_url)
            self.logger.info(f"Processing feed {i}/{len(self.feed_urls)}: {feed_name}")

            try:
                feed_articles = await self._process_feed_robust(session, feed_url, max_per_feed)
                self.logger.info(f"  ✅ Collected {len(feed_articles)} articles from {feed_name}")

            except Exception as e:
                self.logger.error(f"  ❌ Failed to process feed {feed_url}: {e}")
                self.metrics.parsing_errors += 1
                continue

        self.logger.info(f"RSS collection complete: {len(self.collected_articles)} total articles")
        return self.collected_articles

    async def _process_feed_robust(self, session: aiohttp.ClientSession,
                                 feed_url: str, max_articles: int) -> List[CollectedArticle]:
        """Process feed with robust error handling"""
        articles = []

        try:
            # Step 1: Fetch RSS feed
            feed_content = await self._fetch_feed_robust(session, feed_url)
            if not feed_content:
                return articles

            # Step 2: Parse feed
            feed = feedparser.parse(feed_content)

            if feed.bozo and feed.bozo_exception:
                self.logger.warning(f"  Feed parsing warning: {feed.bozo_exception}")

            feed_title = getattr(feed.feed, 'title', 'Unknown Feed')
            self.logger.debug(f"  Feed: {feed_title} ({len(feed.entries)} entries)")

            # Step 3: Process entries with smart content strategy
            processed_count = 0
            for entry in feed.entries:
                if processed_count >= max_articles:
                    break

                if len(self.collected_articles) >= self.source_config.max_articles:
                    break

                try:
                    article = await self._process_entry_smart(session, entry, feed_url, feed_title)
                    if article and self._add_article(article):
                        articles.append(article)
                        processed_count += 1

                except Exception as e:
                    self.logger.warning(f"  Failed to process entry: {e}")
                    self.metrics.parsing_errors += 1
                    continue

                # Be nice to servers
                await asyncio.sleep(0.2)

        except Exception as e:
            self.logger.error(f"  Feed processing error: {e}")
            raise

        return articles

    async def _fetch_feed_robust(self, session: aiohttp.ClientSession, feed_url: str) -> Optional[str]:
        """Fetch RSS feed with multiple retry strategies"""
        try:
            # Strategy 1: Try with current session
            async with session.get(feed_url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status == 200:
                    content = await response.text()
                    return content
                else:
                    self.logger.warning(f"  Feed HTTP {response.status}")

        except Exception as e:
            self.logger.debug(f"  Primary fetch failed: {e}")

            # Strategy 2: Try with fresh session and different SSL approach
            try:
                # Create more permissive SSL context
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

                connector = aiohttp.TCPConnector(ssl=ssl_context, limit=1)
                timeout = aiohttp.ClientTimeout(total=20)

                async with aiohttp.ClientSession(connector=connector, timeout=timeout) as new_session:
                    headers = {'User-Agent': 'Mozilla/5.0 (compatible; TruthedBot/1.0)'}
                    async with new_session.get(feed_url, headers=headers) as response:
                        if response.status == 200:
                            content = await response.text()
                            self.logger.debug(f"  Feed fetched with fallback method")
                            return content

            except Exception as e2:
                self.logger.error(f"  All fetch strategies failed: {e2}")

        return None

    async def _process_entry_smart(self, session: aiohttp.ClientSession, entry,
                                 feed_url: str, feed_title: str) -> Optional[CollectedArticle]:
        """Process entry with smart content strategy"""
        try:
            # Extract basic data
            url = self._get_entry_url(entry)
            if not url or not self._is_valid_url(url):
                return None

            title = self._get_entry_title(entry)
            if not title:
                return None

            # Smart content strategy: Check RSS content first
            rss_content = self._extract_rss_content(entry)

            if rss_content and len(rss_content) >= self.min_rss_content_length:
                # RSS content is substantial, use it
                content = rss_content
                self.logger.debug(f"  Using RSS content ({len(content)} chars)")

            elif self.enable_scraping:
                # RSS content insufficient, try scraping
                scraped_content = await self._scrape_article_safe(session, url)

                if scraped_content and len(scraped_content) > len(rss_content or ""):
                    content = scraped_content
                    self.logger.debug(f"  Using scraped content ({len(content)} chars)")
                elif rss_content:
                    content = rss_content
                    self.logger.debug(f"  Fallback to RSS content ({len(content)} chars)")
                else:
                    return None
            else:
                # Scraping disabled and RSS content insufficient
                if not rss_content:
                    return None
                content = rss_content

            # Final content check
            if not content or len(content) < 100:
                return None

            # Extract metadata
            author = self._get_entry_author(entry)
            publish_date = self._get_entry_date(entry)
            categories = self._get_entry_categories(entry)

            # Create article
            article = CollectedArticle(
                url=url,
                title=title,
                content=content,
                author=author,
                publish_date=publish_date,
                source_type="rss",
                source_name=f"rss_{self._get_feed_name(feed_url)}",
                categories=categories,
                tags=[feed_title] if feed_title != 'Unknown Feed' else []
            )

            return article

        except Exception as e:
            self.logger.warning(f"  Entry processing error: {e}")
            return None

    def _extract_rss_content(self, entry) -> Optional[str]:
        """Extract content from RSS entry with improved field handling"""
        # Try content fields in order of preference
        content_fields = [
            ('content', lambda x: x[0].get('value', '') if isinstance(x, list) and x else
                        x.get('value', '') if isinstance(x, dict) else str(x) if x else ''),
            ('description', lambda x: str(x) if x else ''),
            ('summary', lambda x: str(x) if x else ''),
            ('summary_detail', lambda x: x.get('value', '') if isinstance(x, dict) else str(x) if x else ''),
        ]

        for field_name, extractor in content_fields:
            field_data = getattr(entry, field_name, None)

            if field_data:
                try:
                    content_text = extractor(field_data)

                    if content_text:
                        # Clean HTML from content
                        cleaned = self._extract_text_from_html(content_text)
                        if cleaned and len(cleaned) > 50:
                            return cleaned

                except Exception as e:
                    self.logger.debug(f"  Error extracting {field_name}: {e}")
                    continue

        return None

    async def _scrape_article_safe(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Scrape article with multiple fallback strategies"""
        try:
            # Strategy 1: Use provided session
            content = await self._scrape_with_session(session, url)
            if content:
                return content

        except Exception as e:
            self.logger.debug(f"  Primary scraping failed: {e}")

        try:
            # Strategy 2: Use requests library fallback
            content = await self._scrape_with_requests_fallback(url)
            if content:
                return content

        except Exception as e:
            self.logger.debug(f"  Requests fallback failed: {e}")

        return None

    async def _scrape_with_session(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Scrape using provided aiohttp session"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

        async with session.get(url, headers=headers,
                             timeout=aiohttp.ClientTimeout(total=15)) as response:
            if response.status == 200:
                html = await response.text()
                content = self._extract_text_from_html(html)
                return content if content and len(content) > 100 else None

        return None

    async def _scrape_with_requests_fallback(self, url: str) -> Optional[str]:
        """Scrape using requests library as fallback"""
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        # Create session with retries
        session = requests.Session()
        session.verify = False  # Skip SSL verification for problem sites

        retry_strategy = Retry(
            total=2,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

        try:
            response = session.get(url, timeout=15)
            if response.status_code == 200:
                content = self._extract_text_from_html(response.text)
                return content if content and len(content) > 100 else None

        except Exception as e:
            self.logger.debug(f"  Requests scraping error: {e}")

        return None

    # Keep all the existing helper methods from the original collector
    def _get_entry_url(self, entry) -> Optional[str]:
        """Extract URL from RSS entry"""
        url_fields = ['link', 'guid', 'id']

        for field in url_fields:
            url = getattr(entry, field, None)
            if url and isinstance(url, str) and url.startswith(('http://', 'https://')):
                return url.strip()

        return None

    def _get_entry_title(self, entry) -> Optional[str]:
        """Extract title from RSS entry"""
        title = getattr(entry, 'title', '').strip()

        if '<' in title:
            soup = BeautifulSoup(title, 'html.parser')
            title = soup.get_text().strip()

        return title if title else None

    def _get_entry_author(self, entry) -> Optional[str]:
        """Extract author from RSS entry"""
        author_fields = ['author', 'author_detail', 'dc_creator']

        for field in author_fields:
            author_data = getattr(entry, field, None)

            if isinstance(author_data, str):
                return author_data.strip()
            elif isinstance(author_data, dict):
                name = author_data.get('name', '') or author_data.get('value', '')
                if name:
                    return name.strip()

        return None

    def _get_entry_date(self, entry) -> Optional[datetime]:
        """Extract publish date from RSS entry"""
        date_fields = ['published_parsed', 'updated_parsed', 'created_parsed']

        for field in date_fields:
            date_data = getattr(entry, field, None)

            if date_data:
                try:
                    return datetime(*date_data[:6])
                except (ValueError, TypeError):
                    continue

        # Try string date fields
        string_date_fields = ['published', 'updated', 'created']
        for field in string_date_fields:
            date_str = getattr(entry, field, None)
            if date_str:
                try:
                    if 'T' in date_str:
                        date_str = date_str.split('T')[0]
                    return datetime.fromisoformat(date_str.replace('Z', ''))
                except:
                    continue

        return None

    def _get_entry_categories(self, entry) -> List[str]:
        """Extract categories/tags from RSS entry"""
        categories = []

        tags = getattr(entry, 'tags', [])
        for tag in tags:
            if isinstance(tag, dict):
                term = tag.get('term', '')
                if term:
                    categories.append(term.strip())
            elif isinstance(tag, str):
                categories.append(tag.strip())

        category_fields = ['category', 'categories']
        for field in category_fields:
            cat_data = getattr(entry, field, None)
            if isinstance(cat_data, str):
                categories.append(cat_data.strip())
            elif isinstance(cat_data, list):
                categories.extend([str(cat).strip() for cat in cat_data])

        return [cat for cat in categories if cat]

    def _get_feed_name(self, feed_url: str) -> str:
        """Extract readable name from feed URL"""
        try:
            domain = urlparse(feed_url).netloc
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return "unknown_feed"


# Disable SSL warnings for requests fallback
def disable_ssl_warnings():
    """Disable SSL warnings for requests fallback"""
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except ImportError:
        pass

# Call on import
disable_ssl_warnings()


# Export
__all__ = ['FixedRSSCollector']