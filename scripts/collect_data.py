#!/usr/bin/env python3
"""
Main data collection script for Truthed Professional.
Orchestrates multiple collectors and provides CLI interface.

Usage:
    python scripts/collect_data.py --mode development --limit 100
    python scripts/collect_data.py --mode training --sources rss,news_api
    python scripts/collect_data.py --config config/collection.yaml
"""

import sys
import asyncio
import argparse
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import aiohttp
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from truthed.data.collection.config import (
        CollectionConfig, CollectionMode,
        create_development_config, create_training_config, create_production_config
    )
    from truthed.data.collection.rss_collector_fixed import FixedRSSCollector  # Use fixed version
    from truthed.data.collection.base_collector import CollectedArticle
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root and have installed dependencies")
    sys.exit(1)


class DataCollectionOrchestrator:
    """Main orchestrator for data collection across multiple sources"""

    def __init__(self, config: CollectionConfig):
        self.config = config
        self.collectors = []
        self.all_articles = []

        # Set up logging
        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Initialize collectors
        self._initialize_collectors()

    def _initialize_collectors(self):
        """Initialize available collectors based on configuration"""
        self.logger.info("Initializing data collectors...")

        enabled_sources = self.config.get_enabled_sources()
        if not enabled_sources:
            self.logger.warning("No enabled sources found in configuration")
            return

        for source_config in enabled_sources:
            try:
                if source_config.source_type.value == "rss":
                    collector = FixedRSSCollector(self.config, source_config)  # Use fixed version
                    self.collectors.append(collector)
                    self.logger.info(f"  ✅ Fixed RSS collector: {source_config.name}")

                # TODO: Add other collector types here
                # elif source_config.source_type.value == "news_api":
                #     collector = NewsAPICollector(self.config, source_config)
                # elif source_config.source_type.value == "guardian_api":
                #     collector = GuardianAPICollector(self.config, source_config)

                else:
                    self.logger.warning(f"  ⚠️  Unsupported source type: {source_config.source_type}")

            except Exception as e:
                self.logger.error(f"  ❌ Failed to initialize {source_config.name}: {e}")

        self.logger.info(f"Initialized {len(self.collectors)} collectors")

    async def collect_all_data(self) -> List[CollectedArticle]:
        """Run all collectors and aggregate results"""
        self.logger.info("🚀 Starting data collection")
        self.logger.info("=" * 60)

        start_time = datetime.now()

        # Create aiohttp session with flexible SSL support
        import ssl

        # Create SSL context that's more permissive
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        connector = aiohttp.TCPConnector(
            limit=self.config.rate_limiting.concurrent_requests,
            ttl_dns_cache=300,
            use_dns_cache=True,
            ssl=ssl_context
        )

        timeout = aiohttp.ClientTimeout(total=self.config.rate_limiting.request_timeout)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Run collectors
            for i, collector in enumerate(self.collectors, 1):
                if len(self.all_articles) >= self.config.max_total_articles:
                    self.logger.info(f"Reached global article limit ({self.config.max_total_articles})")
                    break

                self.logger.info(f"\n📡 Running collector {i}/{len(self.collectors)}: {collector.__class__.__name__}")

                try:
                    # Adjust max articles for remaining quota
                    remaining_quota = self.config.max_total_articles - len(self.all_articles)
                    max_for_this_collector = min(collector.source_config.max_articles, remaining_quota)

                    if max_for_this_collector <= 0:
                        self.logger.info("  No remaining quota for this collector")
                        continue

                    # Temporarily adjust collector limit
                    original_max = collector.source_config.max_articles
                    collector.source_config.max_articles = max_for_this_collector

                    # Run collection
                    articles = await collector.collect_articles(session)

                    # Restore original limit
                    collector.source_config.max_articles = original_max

                    # Add to global collection
                    self.all_articles.extend(articles)

                    # Print collector summary
                    summary = collector.get_collection_summary()
                    self.logger.info(f"  📊 {summary['articles_collected']} articles, "
                                     f"{summary['success_rate']} success, "
                                     f"{summary['duration_seconds']:.1f}s")

                except Exception as e:
                    self.logger.error(f"  ❌ Collector failed: {e}")
                    continue

        # Final processing
        self._post_process_articles()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        self.logger.info(f"\n✅ Collection complete!")
        self.logger.info(f"   📊 Total articles: {len(self.all_articles)}")
        self.logger.info(f"   ⏱️  Total time: {duration:.1f} seconds")
        self.logger.info(f"   🌐 Unique domains: {len(set(a.domain for a in self.all_articles))}")
        self.logger.info(f"   📈 Collection rate: {len(self.all_articles) / duration:.2f} articles/second")

        return self.all_articles

    def _post_process_articles(self):
        """Final processing and deduplication of all collected articles"""
        if not self.all_articles:
            return

        self.logger.info(f"\n🔧 Post-processing {len(self.all_articles)} articles...")

        # Global deduplication by content hash
        seen_hashes = set()
        unique_articles = []

        for article in self.all_articles:
            if article.content_hash not in seen_hashes:
                seen_hashes.add(article.content_hash)
                unique_articles.append(article)

        duplicates_removed = len(self.all_articles) - len(unique_articles)
        if duplicates_removed > 0:
            self.logger.info(f"   Removed {duplicates_removed} duplicate articles")

        # Sort by quality score and publish date
        unique_articles.sort(
            key=lambda x: (x.quality_score, x.publish_date or datetime.min),
            reverse=True
        )

        self.all_articles = unique_articles
        self.logger.info(f"   ✅ Final collection: {len(self.all_articles)} unique articles")

    def save_results(self, output_path: str, format: str = "json") -> str:
        """Save collection results to file"""
        if not self.all_articles:
            self.logger.warning("No articles to save")
            return ""

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"💾 Saving {len(self.all_articles)} articles to {output_file}")

        if format.lower() == "json":
            return self._save_json(output_file)
        elif format.lower() == "csv":
            return self._save_csv(output_file)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _save_json(self, output_file: Path) -> str:
        """Save results as JSON"""
        # Prepare metadata
        metadata = {
            "collection_timestamp": datetime.now().isoformat(),
            "config_mode": self.config.mode.value,
            "total_articles": len(self.all_articles),
            "collectors_used": [c.__class__.__name__ for c in self.collectors],
            "sources": [c.source_config.name for c in self.collectors],
            "domains": list(set(a.domain for a in self.all_articles)),
            "quality_stats": {
                "avg_quality_score": sum(a.quality_score for a in self.all_articles) / len(self.all_articles),
                "avg_content_length": sum(a.content_length for a in self.all_articles) / len(self.all_articles),
                "articles_with_author": sum(1 for a in self.all_articles if a.author),
                "articles_with_date": sum(1 for a in self.all_articles if a.publish_date)
            }
        }

        # Prepare articles data
        articles_data = []
        for article in self.all_articles:
            article_dict = {
                "url": article.url,
                "title": article.title,
                "content": article.content,
                "domain": article.domain,
                "author": article.author,
                "publish_date": article.publish_date.isoformat() if article.publish_date else None,
                "language": article.language,
                "source_type": article.source_type,
                "source_name": article.source_name,
                "collected_at": article.collected_at.isoformat(),
                "content_hash": article.content_hash,
                "quality_score": article.quality_score,
                "content_length": article.content_length,
                "title_length": article.title_length,
                "categories": article.categories,
                "tags": article.tags
            }
            articles_data.append(article_dict)

        # Save to file
        data = {
            "metadata": metadata,
            "articles": articles_data
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return str(output_file)

    def _save_csv(self, output_file: Path) -> str:
        """Save results as CSV"""
        fieldnames = [
            'url', 'title', 'content_preview', 'domain', 'author', 'publish_date',
            'source_type', 'source_name', 'quality_score', 'content_length',
            'categories', 'content_hash'
        ]

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for article in self.all_articles:
                # Truncate content for CSV
                content_preview = article.content[:500] + "..." if len(article.content) > 500 else article.content
                content_preview = content_preview.replace('\n', ' ').replace('\r', ' ')

                row = {
                    'url': article.url,
                    'title': article.title,
                    'content_preview': content_preview,
                    'domain': article.domain,
                    'author': article.author or '',
                    'publish_date': article.publish_date.isoformat() if article.publish_date else '',
                    'source_type': article.source_type,
                    'source_name': article.source_name,
                    'quality_score': f"{article.quality_score:.3f}",
                    'content_length': article.content_length,
                    'categories': '; '.join(article.categories),
                    'content_hash': article.content_hash
                }
                writer.writerow(row)

        return str(output_file)

    def print_summary(self):
        """Print detailed collection summary"""
        if not self.all_articles:
            print("No articles collected")
            return

        print(f"\n📊 COLLECTION SUMMARY")
        print("=" * 50)
        print(f"Total articles collected: {len(self.all_articles)}")

        # By source
        source_counts = {}
        for article in self.all_articles:
            source_counts[article.source_name] = source_counts.get(article.source_name, 0) + 1

        print(f"\nArticles by source:")
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {source}: {count}")

        # By domain
        domain_counts = {}
        for article in self.all_articles:
            domain_counts[article.domain] = domain_counts.get(article.domain, 0) + 1

        print(f"\nTop domains:")
        for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {domain}: {count}")

        # Quality stats
        avg_quality = sum(a.quality_score for a in self.all_articles) / len(self.all_articles)
        avg_length = sum(a.content_length for a in self.all_articles) / len(self.all_articles)
        with_author = sum(1 for a in self.all_articles if a.author)
        with_date = sum(1 for a in self.all_articles if a.publish_date)

        print(f"\nQuality metrics:")
        print(f"  Average quality score: {avg_quality:.3f}")
        print(f"  Average content length: {avg_length:.0f} characters")
        print(f"  Articles with author: {with_author} ({100 * with_author / len(self.all_articles):.1f}%)")
        print(f"  Articles with date: {with_date} ({100 * with_date / len(self.all_articles):.1f}%)")

        # Sample titles
        print(f"\nSample article titles:")
        for i, article in enumerate(self.all_articles[:5], 1):
            title = article.title[:80] + "..." if len(article.title) > 80 else article.title
            print(f"  {i}. {title}")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Collect articles for Truthed Professional training and analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode development --limit 50
  %(prog)s --mode training --limit 1000 --output data/training_set.json
  %(prog)s --sources rss --format csv
        """
    )

    parser.add_argument(
        "--mode",
        choices=["development", "training", "production"],
        default="development",
        help="Collection mode (default: development)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum articles to collect (overrides mode default)"
    )

    parser.add_argument(
        "--sources",
        default="rss",
        help="Comma-separated list of sources to use (default: rss)"
    )

    parser.add_argument(
        "--output",
        help="Output file path (default: auto-generated)"
    )

    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format (default: json)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )

    args = parser.parse_args()

    # Create configuration
    mode = CollectionMode(args.mode)
    if mode == CollectionMode.DEVELOPMENT:
        config = create_development_config()
    elif mode == CollectionMode.TRAINING:
        config = create_training_config()
    else:
        config = create_production_config()

    # Override settings from args
    if args.limit:
        config.max_total_articles = args.limit

    if args.verbose:
        config.log_level = "DEBUG"

    # Generate output filename if not provided
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"articles_{args.mode}_{timestamp}.{args.format}"
        args.output = f"data/raw/{filename}"

    print(f"🚀 Truthed Professional Data Collection")
    print(f"Mode: {args.mode}")
    print(f"Target articles: {config.max_total_articles}")
    print(f"Output: {args.output}")
    print("")

    try:
        # Run collection
        orchestrator = DataCollectionOrchestrator(config)
        articles = asyncio.run(orchestrator.collect_all_data())

        if articles:
            # Save results
            output_file = orchestrator.save_results(args.output, args.format)

            # Print summary
            orchestrator.print_summary()

            print(f"\n🎉 SUCCESS!")
            print(f"Collected {len(articles)} articles")
            print(f"Saved to: {output_file}")
            print(f"\nNext steps:")
            print(f"1. Review the collected data: head -20 {output_file}")
            print(f"2. Run annotation tool to label claims")
            print(f"3. Train BERT classifier on labeled data")

            return 0
        else:
            print("❌ No articles collected")
            return 1

    except KeyboardInterrupt:
        print("\n⚠️  Collection interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Collection failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())