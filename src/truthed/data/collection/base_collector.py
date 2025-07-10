"""
Base collector interface and common utilities for data collection.
All collectors inherit from BaseCollector for consistent behavior.
"""

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Set, Dict, Any
from urllib.parse import urlparse
import aiohttp
from bs4 import BeautifulSoup

from .config import CollectionConfig, SourceConfig, QualityFilters


@dataclass
class CollectedArticle:
    """Standardized article representation from any collector"""

    # Core content
    url: str
    title: str
    content: str

    # Metadata
    domain: str = ""
    author: Optional[str] = None
    publish_date: Optional[datetime] = None
    language: str = "en"

    # Collection metadata
    source_type: str = ""
    source_name: str = ""
    collected_at: datetime = field(default_factory=datetime.now)
    content_hash: str = ""

    # Quality metrics
    quality_score: float = 0.5
    content_length: int = 0
    title_length: int = 0

    # Additional data
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    image_urls: List[str] = field(default_factory=list)
    external_links: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Calculate derived fields after initialization"""
        if not self.domain:
            self.domain = self._extract_domain(self.url)

        if not self.content_hash:
            self.content_hash = self._calculate_content_hash()

        self.content_length = len(self.content)
        self.title_length = len(self.title)

        if self.quality_score == 0.5:  # Default score, calculate actual
            self.quality_score = self._calculate_quality_score()

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            return urlparse(url).netloc.lower()
        except:
            return ""

    def _calculate_content_hash(self) -> str:
        """Calculate content hash for deduplication"""
        content_for_hash = f"{self.title.strip()}{self.content.strip()}"
        return hashlib.md5(content_for_hash.encode()).hexdigest()[:16]

    def _calculate_quality_score(self) -> float:
        """Calculate article quality score (0.0 to 1.0)"""
        score = 0.5  # Base score

        # Content length scoring
        if 1000 <= self.content_length <= 5000:
            score += 0.2
        elif 500 <= self.content_length < 1000:
            score += 0.1
        elif self.content_length > 5000:
            score += 0.1
        elif self.content_length < 200:
            score -= 0.3

        # Title quality
        if 30 <= self.title_length <= 100:
            score += 0.1
        elif self.title_length < 10:
            score -= 0.2

        # Metadata presence
        if self.author:
            score += 0.1
        if self.publish_date:
            score += 0.1

        # Domain reputation (basic check)
        reputable_domains = {
            'reuters.com', 'bbc.co.uk', 'theguardian.com', 'npr.org',
            'sciencedaily.com', 'nature.com', 'science.org', 'cnn.com'
        }
        if self.domain in reputable_domains:
            score += 0.1

        return max(0.0, min(1.0, score))

    def passes_quality_filters(self, filters: QualityFilters) -> bool:
        """Check if article passes quality filters"""
        # Length checks
        if self.content_length < filters.min_article_length:
            return False
        if self.content_length > filters.max_article_length:
            return False
        if self.title_length < filters.min_title_length:
            return False
        if self.title_length > filters.max_title_length:
            return False

        # Domain filters
        if self.domain in filters.excluded_domains:
            return False

        # Content keyword filters
        content_lower = self.content.lower()
        title_lower = self.title.lower()

        # Check excluded keywords
        for keyword in filters.excluded_keywords:
            if keyword in content_lower or keyword in title_lower:
                return False

        # Check required keywords (if any)
        if filters.required_keywords:
            has_required = any(
                keyword in content_lower or keyword in title_lower
                for keyword in filters.required_keywords
            )
            if not has_required:
                return False

        # Quality score check
        if self.quality_score < filters.min_quality_score:
            return False

        # Age check
        if filters.max_age_days and self.publish_date:
            age_days = (datetime.now() - self.publish_date).days
            if age_days > filters.max_age_days:
                return False

        # Additional requirements
        if filters.require_author and not self.author:
            return False
        if filters.require_publish_date and not self.publish_date:
            return False

        return True


@dataclass
class CollectionMetrics:
    """Metrics for collection performance tracking"""

    # Collection stats
    articles_attempted: int = 0
    articles_collected: int = 0
    articles_filtered: int = 0
    articles_duplicate: int = 0

    # Performance stats
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0

    # Error stats
    request_errors: int = 0
    parsing_errors: int = 0
    timeout_errors: int = 0

    # Rate limiting stats
    requests_made: int = 0
    rate_limit_delays: int = 0

    def finalize(self):
        """Finalize metrics calculation"""
        self.end_time = datetime.now()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.articles_attempted == 0:
            return 0.0
        return (self.articles_collected / self.articles_attempted) * 100

    @property
    def articles_per_second(self) -> float:
        """Calculate collection rate"""
        if self.duration_seconds == 0:
            return 0.0
        return self.articles_collected / self.duration_seconds


class BaseCollector(ABC):
    """
    Abstract base class for all data collectors.
    Provides common functionality and enforces consistent interface.
    """

    def __init__(self, config: CollectionConfig, source_config: SourceConfig):
        self.config = config
        self.source_config = source_config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Collection state
        self.collected_articles: List[CollectedArticle] = []
        self.seen_urls: Set[str] = set()
        self.seen_hashes: Set[str] = set()
        self.metrics = CollectionMetrics()

        # Rate limiting
        self._last_request_time = 0.0
        self._request_count = 0

        self.logger.info(f"Initialized {self.__class__.__name__} collector")

    @abstractmethod
    async def collect_articles(self, session: aiohttp.ClientSession,
                               **kwargs) -> List[CollectedArticle]:
        """
        Main collection method - must be implemented by each collector.

        Args:
            session: aiohttp session for making requests
            **kwargs: collector-specific parameters

        Returns:
            List of collected articles
        """
        pass

    async def _rate_limit(self):
        """Apply rate limiting between requests"""
        rate_limit = self.source_config.rate_limiting or self.config.rate_limiting
        min_delay = 1.0 / rate_limit.requests_per_second

        elapsed = time.time() - self._last_request_time
        if elapsed < min_delay:
            delay = min_delay - elapsed
            await asyncio.sleep(delay)
            self.metrics.rate_limit_delays += 1

        self._last_request_time = time.time()
        self._request_count += 1

    def _is_duplicate(self, article: CollectedArticle) -> bool:
        """Check if article is duplicate based on URL and content hash"""
        if article.url in self.seen_urls:
            return True
        if article.content_hash in self.seen_hashes:
            return True
        return False

    def _add_article(self, article: CollectedArticle) -> bool:
        """
        Add article if it passes all filters and isn't duplicate.

        Returns:
            True if article was added, False if filtered/duplicate
        """
        self.metrics.articles_attempted += 1

        # Check for duplicates
        if self._is_duplicate(article):
            self.metrics.articles_duplicate += 1
            return False

        # Apply quality filters
        quality_filters = self.source_config.quality_filters or self.config.quality_filters
        if not article.passes_quality_filters(quality_filters):
            self.metrics.articles_filtered += 1
            return False

        # Check collection limits
        if len(self.collected_articles) >= self.source_config.max_articles:
            return False

        # Add article
        self.collected_articles.append(article)
        self.seen_urls.add(article.url)
        self.seen_hashes.add(article.content_hash)
        self.metrics.articles_collected += 1

        self.logger.debug(f"Added article: {article.title[:50]}...")
        return True

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""

        # Remove extra whitespace
        lines = [line.strip() for line in text.splitlines()]
        cleaned = ' '.join(line for line in lines if line)

        # Remove multiple spaces
        import re
        cleaned = re.sub(r'\s+', ' ', cleaned)

        return cleaned.strip()

    def _extract_text_from_html(self, html: str) -> str:
        """Extract clean text from HTML content"""
        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header',
                                 'footer', 'aside', 'iframe', 'form']):
                element.decompose()

            # Try to find main content areas
            content_selectors = [
                'article', '[role="main"]', '.article-body', '.post-content',
                '.entry-content', '.content', 'main', '.article-text',
                '.story-body', '.article-content'
            ]

            content_text = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content_text = ' '.join(elem.get_text() for elem in elements)
                    break

            # Fallback to body if no specific content found
            if not content_text:
                body = soup.find('body')
                if body:
                    content_text = body.get_text()

            return self._clean_text(content_text)

        except Exception as e:
            self.logger.warning(f"HTML parsing error: {e}")
            return ""

    def _is_valid_url(self, url: str) -> bool:
        """Validate if URL is acceptable for collection"""
        try:
            parsed = urlparse(url)

            # Must have scheme and netloc
            if not parsed.scheme or not parsed.netloc:
                return False

            # Check excluded domains
            domain = parsed.netloc.lower()
            quality_filters = self.source_config.quality_filters or self.config.quality_filters

            if domain in quality_filters.excluded_domains:
                return False

            # Check for suspicious patterns
            suspicious_patterns = ['.onion', 'localhost', '127.0.0.1', 'bit.ly', 'tinyurl']
            if any(pattern in url.lower() for pattern in suspicious_patterns):
                return False

            return True

        except Exception:
            return False

    async def _make_request(self, session: aiohttp.ClientSession, url: str,
                            **kwargs) -> Optional[aiohttp.ClientResponse]:
        """Make HTTP request with error handling and rate limiting"""
        await self._rate_limit()

        try:
            self.metrics.requests_made += 1

            # Default headers
            headers = kwargs.pop('headers', {})
            if 'User-Agent' not in headers:
                headers['User-Agent'] = 'Mozilla/5.0 (compatible; TruthedBot/1.0)'

            rate_limit = self.source_config.rate_limiting or self.config.rate_limiting
            timeout = aiohttp.ClientTimeout(total=rate_limit.request_timeout)

            async with session.get(url, headers=headers, timeout=timeout,
                                   **kwargs) as response:
                return response

        except asyncio.TimeoutError:
            self.metrics.timeout_errors += 1
            self.logger.warning(f"Timeout for URL: {url}")
            return None
        except Exception as e:
            self.metrics.request_errors += 1
            self.logger.warning(f"Request error for {url}: {e}")
            return None

    def get_collection_summary(self) -> Dict[str, Any]:
        """Get summary of collection results"""
        self.metrics.finalize()

        return {
            'collector': self.__class__.__name__,
            'source_name': self.source_config.name,
            'articles_collected': len(self.collected_articles),
            'success_rate': f"{self.metrics.success_rate:.1f}%",
            'duration_seconds': self.metrics.duration_seconds,
            'articles_per_second': f"{self.metrics.articles_per_second:.2f}",
            'quality_stats': {
                'attempted': self.metrics.articles_attempted,
                'collected': self.metrics.articles_collected,
                'filtered': self.metrics.articles_filtered,
                'duplicate': self.metrics.articles_duplicate
            },
            'error_stats': {
                'request_errors': self.metrics.request_errors,
                'parsing_errors': self.metrics.parsing_errors,
                'timeout_errors': self.metrics.timeout_errors
            }
        }


# Export main classes
__all__ = [
    'BaseCollector',
    'CollectedArticle',
    'CollectionMetrics',
    'QualityFilters'
]