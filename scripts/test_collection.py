#!/usr/bin/env python3
"""
Quick test collection script that bypasses SSL issues.
Uses requests library instead of aiohttp for simplicity.
"""

import sys
import json
import requests
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional
import hashlib
import time
from urllib.parse import urlparse

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    import feedparser
    from bs4 import BeautifulSoup
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install feedparser beautifulsoup4 requests")
    sys.exit(1)

# Disable SSL warnings for testing
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@dataclass
class SimpleArticle:
    """Simple article structure for testing"""
    url: str
    title: str
    content: str
    domain: str
    author: Optional[str] = None
    publish_date: Optional[str] = None
    scraped_at: str = ""
    content_hash: str = ""
    source: str = "rss"
    content_length: int = 0

    def __post_init__(self):
        if not self.scraped_at:
            self.scraped_at = datetime.now().isoformat()
        if not self.content_hash:
            self.content_hash = hashlib.md5(self.content.encode()).hexdigest()[:16]
        self.content_length = len(self.content)


class SimpleRSSCollector:
    """Simple RSS collector using requests library"""

    def __init__(self, max_articles: int = 50):
        self.max_articles = max_articles
        self.articles = []
        self.seen_urls = set()

        # Use working RSS feeds (avoid HTTPS issues)
        self.feeds = [
            "http://rss.cnn.com/rss/edition.rss",  # HTTP version
            "http://feeds.reuters.com/reuters/topNews",  # HTTP version
            "http://www.sciencedaily.com/rss/all.xml",  # HTTP version
        ]

        # If HTTP doesn't work, try these reliable HTTPS feeds
        self.backup_feeds = [
            "https://feeds.feedburner.com/oreilly/radar",
            "https://rss.slashdot.org/Slashdot/slashdotMain",
            "https://feeds.arstechnica.com/arstechnica/index",
        ]

        # Session with SSL disabled for testing
        self.session = requests.Session()
        self.session.verify = False  # Disable SSL verification
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; TruthedBot/1.0)'
        })

    def collect_articles(self) -> List[SimpleArticle]:
        """Collect articles from RSS feeds"""
        print(f"🚀 Starting simple RSS collection (max: {self.max_articles})")
        print("=" * 60)

        # Try primary feeds first
        self._try_feeds(self.feeds, "Primary feeds (HTTP)")

        # If we didn't get enough, try backup feeds
        if len(self.articles) < 10:
            print(f"\nTrying backup feeds...")
            self._try_feeds(self.backup_feeds, "Backup feeds (HTTPS)")

        print(f"\n✅ Collection complete: {len(self.articles)} articles")
        return self.articles

    def _try_feeds(self, feeds: List[str], category: str):
        """Try a list of feeds"""
        print(f"\n📡 {category}:")

        for i, feed_url in enumerate(feeds, 1):
            if len(self.articles) >= self.max_articles:
                break

            print(f"  {i}. {urlparse(feed_url).netloc}")

            try:
                feed_articles = self._process_feed(feed_url)
                new_articles = [a for a in feed_articles if a.url not in self.seen_urls]

                for article in new_articles:
                    if len(self.articles) >= self.max_articles:
                        break
                    self.seen_urls.add(article.url)
                    self.articles.append(article)

                print(f"     ✅ Added {len(new_articles)} articles")

            except Exception as e:
                print(f"     ❌ Error: {e}")
                continue

            time.sleep(1)  # Be polite

    def _process_feed(self, feed_url: str) -> List[SimpleArticle]:
        """Process a single RSS feed"""
        articles = []

        try:
            # Fetch RSS with requests
            response = self.session.get(feed_url, timeout=10)
            response.raise_for_status()

            # Parse feed
            feed = feedparser.parse(response.content)

            if feed.bozo:
                print(f"     ⚠️  Feed parsing warning: {feed.bozo_exception}")

            # Process entries
            for entry in feed.entries[:5]:  # Limit per feed
                try:
                    article = self._process_entry(entry)
                    if article:
                        articles.append(article)
                except Exception as e:
                    print(f"       Entry error: {e}")
                    continue

        except Exception as e:
            print(f"     Feed error: {e}")

        return articles

    def _process_entry(self, entry) -> Optional[SimpleArticle]:
        """Process RSS entry"""
        try:
            # Get URL
            url = getattr(entry, 'link', '')
            if not url:
                return None

            # Get title
            title = getattr(entry, 'title', '').strip()
            if not title or len(title) < 10:
                return None

            # Get content from RSS
            content = self._get_content_from_entry(entry)
            if not content or len(content) < 100:
                return None

            # Get metadata
            author = getattr(entry, 'author', None)
            publish_date = getattr(entry, 'published', None)

            return SimpleArticle(
                url=url,
                title=title,
                content=content[:5000],  # Limit length
                domain=urlparse(url).netloc,
                author=author,
                publish_date=publish_date,
                source="rss"
            )

        except Exception as e:
            print(f"       Processing error: {e}")
            return None

    def _get_content_from_entry(self, entry) -> Optional[str]:
        """Extract content from RSS entry"""
        # Try different content fields
        content_fields = ['content', 'description', 'summary']

        for field in content_fields:
            content_data = getattr(entry, field, None)

            if isinstance(content_data, list) and content_data:
                content_text = content_data[0].get('value', '')
            elif isinstance(content_data, dict):
                content_text = content_data.get('value', '')
            elif isinstance(content_data, str):
                content_text = content_data
            else:
                continue

            if content_text:
                # Clean HTML
                soup = BeautifulSoup(content_text, 'html.parser')
                cleaned = soup.get_text()

                # Clean whitespace
                lines = [line.strip() for line in cleaned.splitlines()]
                cleaned = ' '.join(line for line in lines if line)

                if len(cleaned) > 50:
                    return cleaned

        return None

    def save_articles(self, output_file: str) -> str:
        """Save articles to JSON"""
        if not self.articles:
            print("❌ No articles to save")
            return ""

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": {
                "collected_at": datetime.now().isoformat(),
                "total_articles": len(self.articles),
                "domains": list(set(a.domain for a in self.articles)),
                "avg_content_length": sum(a.content_length for a in self.articles) / len(self.articles)
            },
            "articles": [asdict(article) for article in self.articles]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved {len(self.articles)} articles to {output_file}")
        return output_file

    def print_summary(self):
        """Print collection summary"""
        if not self.articles:
            print("No articles collected")
            return

        print(f"\n📊 SUMMARY")
        print("=" * 40)
        print(f"Total articles: {len(self.articles)}")

        # By domain
        domains = {}
        for article in self.articles:
            domains[article.domain] = domains.get(article.domain, 0) + 1

        print(f"\nArticles by domain:")
        for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True):
            print(f"  {domain}: {count}")

        # Sample titles
        print(f"\nSample titles:")
        for i, article in enumerate(self.articles[:3], 1):
            title = article.title[:60] + "..." if len(article.title) > 60 else article.title
            print(f"  {i}. {title}")


def main():
    """Main function"""
    print("🧪 TESTING RSS COLLECTION (SSL-Free)")
    print("=" * 50)

    try:
        # Create collector
        collector = SimpleRSSCollector(max_articles=20)

        # Collect articles
        articles = collector.collect_articles()

        if articles:
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"data/raw/test_articles_{timestamp}.json"
            collector.save_articles(output_file)

            # Print summary
            collector.print_summary()

            print(f"\n🎉 SUCCESS!")
            print(f"Collected {len(articles)} articles")
            print(f"Next step: Fix SSL in main collector or use this simple version")

        else:
            print("❌ No articles collected - network issues")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())