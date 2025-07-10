"""
Configuration classes for data collection system.
Centralizes all collection settings and source configurations.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from enum import Enum
from pathlib import Path


class SourceType(str, Enum):
    """Types of data sources"""
    RSS = "rss"
    NEWS_API = "news_api"
    GUARDIAN_API = "guardian_api"
    TWITTER_API = "twitter_api"
    REDDIT_API = "reddit_api"
    MANUAL_SCRAPING = "manual_scraping"


class CollectionMode(str, Enum):
    """Collection modes for different scenarios"""
    DEVELOPMENT = "development"     # Small scale, fast iteration
    TRAINING = "training"          # Large scale for model training
    PRODUCTION = "production"      # Ongoing collection for live system
    TESTING = "testing"           # Minimal collection for testing


@dataclass
class QualityFilters:
    """Content quality filtering configuration"""
    min_article_length: int = 200
    max_article_length: int = 50000
    min_title_length: int = 10
    max_title_length: int = 200

    # Domain filtering
    excluded_domains: Set[str] = field(default_factory=lambda: {
        'reddit.com', 'twitter.com', 'facebook.com', 'instagram.com',
        'tiktok.com', 'youtube.com', 'pinterest.com', 'tumblr.com'
    })

    # Content filtering
    excluded_keywords: List[str] = field(default_factory=lambda: [
        'porn', 'xxx', 'casino', 'gambling', 'crypto', 'bitcoin',
        'advertisement', 'sponsored', 'affiliate'
    ])

    required_keywords: List[str] = field(default_factory=list)

    # Quality thresholds
    min_quality_score: float = 0.3
    require_author: bool = False
    require_publish_date: bool = False
    max_age_days: Optional[int] = None


@dataclass
class RateLimiting:
    """Rate limiting configuration"""
    requests_per_second: float = 1.0
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    concurrent_requests: int = 5
    request_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class SourceConfig:
    """Configuration for a specific data source"""
    source_type: SourceType
    name: str
    enabled: bool = True

    # API configuration
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    endpoints: Dict[str, str] = field(default_factory=dict)

    # RSS configuration
    feed_urls: List[str] = field(default_factory=list)

    # Collection limits
    max_articles: int = 100
    max_articles_per_request: int = 50

    # Rate limiting (inherits from global if not specified)
    rate_limiting: Optional[RateLimiting] = None

    # Source-specific quality filters
    quality_filters: Optional[QualityFilters] = None

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StorageConfig:
    """Data storage configuration"""
    # File storage
    output_directory: str = "data/raw"
    save_format: str = "json"  # json, csv, jsonl
    create_backups: bool = True

    # Database storage
    database_url: Optional[str] = None
    database_table: str = "collected_articles"
    enable_database: bool = False

    # File naming
    filename_template: str = "articles_{timestamp}_{source}"
    include_metadata: bool = True
    compress_output: bool = False


@dataclass
class CollectionConfig:
    """Main configuration for data collection pipeline"""

    # Collection mode and limits
    mode: CollectionMode = CollectionMode.DEVELOPMENT
    max_total_articles: int = 1000
    collection_timeframe_days: int = 7

    # Global quality filters
    quality_filters: QualityFilters = field(default_factory=QualityFilters)

    # Global rate limiting
    rate_limiting: RateLimiting = field(default_factory=RateLimiting)

    # Storage configuration
    storage: StorageConfig = field(default_factory=StorageConfig)

    # Source configurations
    sources: Dict[str, SourceConfig] = field(default_factory=dict)

    # API keys (loaded from environment)
    api_keys: Dict[str, str] = field(default_factory=dict)

    # Logging and monitoring
    log_level: str = "INFO"
    enable_metrics: bool = True

    def __post_init__(self):
        """Initialize configuration after creation"""
        # Load API keys from environment
        self._load_api_keys()

        # Apply mode-specific defaults
        self._apply_mode_defaults()

        # Validate configuration
        self._validate_config()

    def _load_api_keys(self):
        """Load API keys from environment variables"""
        api_key_mapping = {
            'news_api': 'NEWS_API_KEY',
            'guardian': 'GUARDIAN_API_KEY',
            'twitter': 'TWITTER_BEARER_TOKEN',
            'reddit': 'REDDIT_CLIENT_ID'
        }

        for key, env_var in api_key_mapping.items():
            value = os.getenv(env_var)
            if value:
                self.api_keys[key] = value

    def _apply_mode_defaults(self):
        """Apply defaults based on collection mode"""
        if self.mode == CollectionMode.DEVELOPMENT:
            self.max_total_articles = min(self.max_total_articles, 100)
            self.rate_limiting.requests_per_second = 2.0
            self.storage.create_backups = False

        elif self.mode == CollectionMode.TRAINING:
            self.max_total_articles = max(self.max_total_articles, 5000)
            self.collection_timeframe_days = 30
            self.storage.create_backups = True
            self.storage.enable_database = True

        elif self.mode == CollectionMode.PRODUCTION:
            self.rate_limiting.requests_per_second = 0.5  # More conservative
            self.storage.enable_database = True
            self.storage.create_backups = True
            self.enable_metrics = True

        elif self.mode == CollectionMode.TESTING:
            self.max_total_articles = min(self.max_total_articles, 10)
            self.rate_limiting.requests_per_second = 5.0
            self.storage.create_backups = False

    def _validate_config(self):
        """Validate configuration consistency"""
        # Check required settings
        if self.max_total_articles <= 0:
            raise ValueError("max_total_articles must be positive")

        if self.collection_timeframe_days <= 0:
            raise ValueError("collection_timeframe_days must be positive")

        # Validate storage directory
        storage_path = Path(self.storage.output_directory)
        try:
            storage_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Cannot create storage directory: {e}")

    def get_source_config(self, source_name: str) -> Optional[SourceConfig]:
        """Get configuration for a specific source"""
        return self.sources.get(source_name)

    def add_source(self, source_config: SourceConfig) -> None:
        """Add a source configuration"""
        self.sources[source_config.name] = source_config

    def get_enabled_sources(self) -> List[SourceConfig]:
        """Get list of enabled source configurations"""
        return [config for config in self.sources.values() if config.enabled]

    def has_api_key(self, key_name: str) -> bool:
        """Check if API key is available"""
        return key_name in self.api_keys and bool(self.api_keys[key_name])


def create_development_config() -> CollectionConfig:
    """Create configuration optimized for development"""
    config = CollectionConfig(mode=CollectionMode.DEVELOPMENT)

    # Add RSS sources (no API keys required) - using working feeds
    rss_config = SourceConfig(
        source_type=SourceType.RSS,
        name="rss_feeds",
        max_articles=50,
        feed_urls=[
            # Working feeds from test
            "http://rss.cnn.com/rss/edition.rss",
            "http://www.sciencedaily.com/rss/all.xml",
            "https://rss.slashdot.org/Slashdot/slashdotMain",
            "https://feeds.arstechnica.com/arstechnica/index",
            # Additional reliable feeds
            "https://www.reddit.com/r/science/.rss",
            "https://techcrunch.com/feed/"
        ]
    )
    config.add_source(rss_config)

    # Add NewsAPI if key available
    if config.has_api_key('news_api'):
        newsapi_config = SourceConfig(
            source_type=SourceType.NEWS_API,
            name="news_api",
            api_key=config.api_keys['news_api'],
            max_articles=30
        )
        config.add_source(newsapi_config)

    return config


def create_training_config() -> CollectionConfig:
    """Create configuration optimized for training data collection"""
    config = CollectionConfig(
        mode=CollectionMode.TRAINING,
        max_total_articles=5000,
        collection_timeframe_days=30
    )

    # More diverse sources for training
    rss_config = SourceConfig(
        source_type=SourceType.RSS,
        name="rss_feeds",
        max_articles=2000,
        feed_urls=[
            # Science & Tech
            "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
            "https://rss.cnn.com/rss/edition_technology.rss",
            "https://www.sciencedaily.com/rss/all.xml",
            "https://feeds.reuters.com/reuters/technologyNews",

            # General News
            "https://feeds.npr.org/1001/rss.xml",
            "https://rss.reuters.com/reuters/topNews",
            "https://feeds.bbci.co.uk/news/rss.xml",

            # Health
            "https://rss.cnn.com/rss/edition_health.rss",
            "https://feeds.reuters.com/reuters/healthNews",

            # Politics & World
            "https://feeds.reuters.com/reuters/politicsNews",
            "https://feeds.reuters.com/reuters/worldNews"
        ]
    )
    config.add_source(rss_config)

    # Add API sources if available
    if config.has_api_key('news_api'):
        newsapi_config = SourceConfig(
            source_type=SourceType.NEWS_API,
            name="news_api",
            api_key=config.api_keys['news_api'],
            max_articles=2000
        )
        config.add_source(newsapi_config)

    if config.has_api_key('guardian'):
        guardian_config = SourceConfig(
            source_type=SourceType.GUARDIAN_API,
            name="guardian_api",
            api_key=config.api_keys['guardian'],
            max_articles=1000
        )
        config.add_source(guardian_config)

    return config


def create_production_config() -> CollectionConfig:
    """Create configuration for production collection"""
    config = CollectionConfig(
        mode=CollectionMode.PRODUCTION,
        max_total_articles=10000,
        collection_timeframe_days=1  # Daily collection
    )

    # Conservative rate limiting for production
    config.rate_limiting.requests_per_second = 0.5
    config.rate_limiting.concurrent_requests = 3

    # Enable database storage
    config.storage.enable_database = True
    config.storage.create_backups = True

    # Add all available sources
    config = create_training_config()  # Start with training config
    config.mode = CollectionMode.PRODUCTION

    # Adjust limits for production
    for source in config.sources.values():
        source.max_articles = min(source.max_articles, 1000)

    return config


def load_config_from_env() -> CollectionConfig:
    """Load configuration from environment variables"""
    mode = CollectionMode(os.getenv('COLLECTION_MODE', 'development'))

    if mode == CollectionMode.DEVELOPMENT:
        return create_development_config()
    elif mode == CollectionMode.TRAINING:
        return create_training_config()
    elif mode == CollectionMode.PRODUCTION:
        return create_production_config()
    else:
        return create_development_config()


# Export main classes and functions
__all__ = [
    'CollectionConfig',
    'SourceConfig',
    'QualityFilters',
    'RateLimiting',
    'StorageConfig',
    'SourceType',
    'CollectionMode',
    'create_development_config',
    'create_training_config',
    'create_production_config',
    'load_config_from_env'
]