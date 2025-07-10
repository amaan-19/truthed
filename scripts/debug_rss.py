#!/usr/bin/env python3
"""
Debug RSS scraping issues step by step.
Test individual components to identify the problem.
"""

import sys
import asyncio
import aiohttp
import ssl
from pathlib import Path
from urllib.parse import urlparse
import feedparser
import time

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Test URLs from your logs
TEST_FEEDS = [
    "http://rss.cnn.com/rss/edition.rss",
    "http://www.sciencedaily.com/rss/all.xml",
    "https://rss.slashdot.org/Slashdot/slashdotMain"
]

TEST_ARTICLES = [
    "https://www.cnn.com/2023/04/03/world/artemis-2-astronaut-crew-scn/index.html",
    "https://www.sciencedaily.com/releases/2025/07/250709091658.htm",
    "https://news.slashdot.org/story/25/07/10/0123233/prime-day-loses-its-spark-as-sales-nosedive-41"
]


async def test_basic_connectivity():
    """Test basic HTTP connectivity"""
    print("🔍 TESTING BASIC CONNECTIVITY")
    print("=" * 50)

    simple_urls = [
        "http://httpbin.org/get",
        "https://httpbin.org/get",
        "http://example.com",
        "https://example.com"
    ]

    async with aiohttp.ClientSession() as session:
        for url in simple_urls:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    print(f"✅ {url}: HTTP {response.status}")
            except Exception as e:
                print(f"❌ {url}: {e}")


async def test_rss_feed_parsing():
    """Test RSS feed parsing without scraping"""
    print(f"\n🔍 TESTING RSS FEED PARSING")
    print("=" * 50)

    async with aiohttp.ClientSession() as session:
        for feed_url in TEST_FEEDS:
            print(f"\nTesting: {feed_url}")
            try:
                # Test feed fetching
                async with session.get(feed_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        content = await response.text()
                        print(f"  ✅ Feed fetched: {len(content)} characters")

                        # Parse with feedparser
                        feed = feedparser.parse(content)
                        print(f"  ✅ Parsed: {len(feed.entries)} entries")

                        # Show first few entries
                        for i, entry in enumerate(feed.entries[:3], 1):
                            title = getattr(entry, 'title', 'No title')[:50]
                            link = getattr(entry, 'link', 'No link')
                            print(f"    {i}. {title}...")
                            print(f"       {link}")

                    else:
                        print(f"  ❌ HTTP {response.status}")

            except Exception as e:
                print(f"  ❌ Error: {e}")


async def test_article_scraping_simple():
    """Test article scraping with simple approach"""
    print(f"\n🔍 TESTING ARTICLE SCRAPING (SIMPLE)")
    print("=" * 50)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # Create more permissive SSL context
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    connector = aiohttp.TCPConnector(ssl=ssl_context, limit=1)
    timeout = aiohttp.ClientTimeout(total=30, connect=10)

    async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers
    ) as session:

        for url in TEST_ARTICLES:
            print(f"\nTesting: {urlparse(url).netloc}")
            print(f"URL: {url}")

            try:
                async with session.get(url) as response:
                    print(f"  ✅ Status: {response.status}")
                    print(f"  ✅ Headers: {dict(response.headers)}")

                    if response.status == 200:
                        content = await response.text()
                        print(f"  ✅ Content length: {len(content)} characters")
                        print(f"  ✅ Content preview: {content[:200]}...")
                    else:
                        content = await response.text()
                        print(f"  ❌ Error content: {content[:200]}...")

            except asyncio.TimeoutError:
                print(f"  ❌ Timeout error")
            except aiohttp.ClientError as e:
                print(f"  ❌ Client error: {e}")
            except Exception as e:
                print(f"  ❌ Unexpected error: {e}")

            # Add delay between requests
            await asyncio.sleep(2)


async def test_article_scraping_requests():
    """Test article scraping with requests library as fallback"""
    print(f"\n🔍 TESTING ARTICLE SCRAPING (REQUESTS FALLBACK)")
    print("=" * 50)

    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    session = requests.Session()
    session.verify = False
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    })

    for url in TEST_ARTICLES:
        print(f"\nTesting: {urlparse(url).netloc}")

        try:
            response = session.get(url, timeout=15)
            print(f"  ✅ Status: {response.status_code}")
            print(f"  ✅ Content length: {len(response.content)} bytes")

            if response.status_code == 200:
                # Try to extract some text
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                text = soup.get_text()
                clean_text = ' '.join(text.split()[:50])  # First 50 words
                print(f"  ✅ Text preview: {clean_text}...")

        except Exception as e:
            print(f"  ❌ Error: {e}")

        time.sleep(1)


async def test_connection_pooling():
    """Test if connection pooling is causing issues"""
    print(f"\n🔍 TESTING CONNECTION POOLING")
    print("=" * 50)

    # Test with different connector settings
    configs = [
        ("Default", {}),
        ("No pooling", {"limit": 1, "limit_per_host": 1}),
        ("Force close", {"force_close": True}),
        ("No SSL", {"ssl": False}),
    ]

    for config_name, connector_kwargs in configs:
        print(f"\n{config_name} configuration:")

        try:
            connector = aiohttp.TCPConnector(**connector_kwargs)

            async with aiohttp.ClientSession(
                    connector=connector,
                    timeout=aiohttp.ClientTimeout(total=10)
            ) as session:

                # Test a simple URL
                test_url = "https://httpbin.org/get"
                async with session.get(test_url) as response:
                    print(f"  ✅ {config_name}: HTTP {response.status}")

        except Exception as e:
            print(f"  ❌ {config_name}: {e}")


async def test_feed_with_rss_content_only():
    """Test using RSS content without scraping"""
    print(f"\n🔍 TESTING RSS CONTENT WITHOUT SCRAPING")
    print("=" * 50)

    async with aiohttp.ClientSession() as session:
        for feed_url in TEST_FEEDS:
            print(f"\nTesting: {urlparse(feed_url).netloc}")

            try:
                async with session.get(feed_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)

                        print(f"  ✅ Found {len(feed.entries)} entries")

                        # Check RSS content quality
                        good_entries = 0
                        for entry in feed.entries[:10]:
                            # Try to get content from RSS itself
                            content_fields = ['content', 'description', 'summary']
                            entry_content = ""

                            for field in content_fields:
                                field_data = getattr(entry, field, None)
                                if field_data:
                                    if isinstance(field_data, list) and field_data:
                                        entry_content = field_data[0].get('value', '')
                                    elif isinstance(field_data, dict):
                                        entry_content = field_data.get('value', '')
                                    elif isinstance(field_data, str):
                                        entry_content = field_data

                                    if entry_content and len(entry_content) > 100:
                                        break

                            if entry_content and len(entry_content) > 100:
                                good_entries += 1
                                print(f"    Entry {good_entries}: {len(entry_content)} chars")
                                # Clean preview
                                from bs4 import BeautifulSoup
                                soup = BeautifulSoup(entry_content, 'html.parser')
                                clean = soup.get_text()[:100].replace('\n', ' ')
                                print(f"      Preview: {clean}...")

                        print(f"  📊 {good_entries}/10 entries have substantial content in RSS")

            except Exception as e:
                print(f"  ❌ Error: {e}")


async def main():
    """Run all debugging tests"""
    print("🐛 RSS COLLECTION DEBUGGING")
    print("=" * 60)

    # Run tests in sequence
    await test_basic_connectivity()
    await test_rss_feed_parsing()
    await test_connection_pooling()
    await test_feed_with_rss_content_only()
    await test_article_scraping_simple()
    await test_article_scraping_requests()

    print(f"\n" + "=" * 60)
    print("🔍 DEBUGGING COMPLETE")
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"1. If RSS content is substantial, skip scraping")
    print(f"2. If scraping needed, use requests library as fallback")
    print(f"3. Adjust connection settings based on test results")
    print(f"4. Consider using different User-Agent strings")


if __name__ == "__main__":
    asyncio.run(main())