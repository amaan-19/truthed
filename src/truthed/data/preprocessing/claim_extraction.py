"""
Complete claim extraction pipeline integrating all components
File: src/truthed/data/preprocessing/claim_extraction.py
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib
import logging

# Import our components
try:
    from truthed.data.preprocessing.text_processing import TextProcessor, ProcessedText
    from truthed.models.claim_extraction.bert_classifier import BERTClaimClassifier, ClaimPrediction
    from truthed.models.claim_extraction.claim_structurer import ClaimStructurer, StructuredClaim
    from truthed.core.models import Claim, ClaimType
except ImportError as e:
    print(f"⚠️  Import warning: {e}")
    print("Some components may not be available. Pipeline will work with available components.")


@dataclass
class ExtractionResult:
    """Complete result of claim extraction process"""
    url: str
    title: str
    content_hash: str

    # Processing results
    processed_text: Optional[object] = None  # ProcessedText
    claim_predictions: List[object] = None  # List[ClaimPrediction]
    structured_claims: List[object] = None  # List[StructuredClaim]

    # Summary statistics
    total_sentences: int = 0
    claims_identified: int = 0
    high_confidence_claims: int = 0
    verifiable_claims: int = 0

    # Performance metrics
    processing_time_seconds: float = 0.0
    extraction_timestamp: datetime = None

    # Processing metadata
    models_used: Dict[str, str] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.claim_predictions is None:
            self.claim_predictions = []
        if self.structured_claims is None:
            self.structured_claims = []
        if self.extraction_timestamp is None:
            self.extraction_timestamp = datetime.now()
        if self.models_used is None:
            self.models_used = {}
        if self.errors is None:
            self.errors = []


class ClaimExtractionPipeline:
    """
    Complete pipeline for extracting structured claims from web content.

    Pipeline steps:
    1. Text preprocessing and cleaning
    2. Sentence segmentation
    3. BERT-based claim classification
    4. Claim structuring and entity extraction
    5. Verifiability assessment
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the claim extraction pipeline"""
        self.config = config or {}

        # Component initialization
        self.text_processor = None
        self.claim_classifier = None
        self.claim_structurer = None

        # Performance settings
        self.batch_size = self.config.get('batch_size', 50)
        self.min_claim_confidence = self.config.get('min_claim_confidence', 0.6)
        self.enable_structuring = self.config.get('enable_structuring', True)

        # Logging
        self.logger = logging.getLogger(__name__)

        print("🚀 Initializing ClaimExtractionPipeline...")
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all pipeline components with proper configuration"""
        errors = []

        # 1. Text Processor
        try:
            self.text_processor = TextProcessor()
            print("✅ Text processor initialized")
        except Exception as e:
            errors.append(f"Text processor failed: {e}")
            print(f"❌ Text processor failed: {e}")

        # 2. BERT Classifier - FIXED: Pass the threshold configuration
        try:
            # Pass the min_claim_confidence from config to BERT classifier
            bert_threshold = self.min_claim_confidence
            print(f"🎯 Initializing BERT classifier with threshold: {bert_threshold}")

            self.claim_classifier = BERTClaimClassifier(
                min_claim_confidence=bert_threshold  # FIXED: Pass the threshold!
            )
            print("✅ BERT classifier initialized")
        except Exception as e:
            errors.append(f"BERT classifier failed: {e}")
            print(f"❌ BERT classifier failed: {e}")

        # 3. Claim Structurer (optional)
        if self.enable_structuring:
            try:
                self.claim_structurer = ClaimStructurer()
                print("✅ Claim structurer initialized")
            except Exception as e:
                errors.append(f"Claim structurer failed: {e}")
                print(f"⚠️  Claim structurer failed: {e} (continuing without structuring)")
                self.enable_structuring = False

        if errors:
            print(f"⚠️  Pipeline initialized with {len(errors)} component errors")
        else:
            print("🎉 All pipeline components initialized successfully!")
            print(f"🎯 Pipeline using confidence threshold: {self.min_claim_confidence}")

    def extract_claims(self, url: str, title: str = "", content: str = "") -> ExtractionResult:
        """
        Extract claims from article content.

        Args:
            url: URL of the article
            title: Article title
            content: Raw article content (HTML or text)

        Returns:
            ExtractionResult with all extracted claims and metadata
        """
        start_time = datetime.now()

        print(f"\n🔍 EXTRACTING CLAIMS FROM: {url}")
        print("=" * 80)

        # Initialize result
        content_hash = hashlib.md5(content.encode()).hexdigest()[:16]

        result = ExtractionResult(
            url=url,
            title=title,
            content_hash=content_hash,
            extraction_timestamp=start_time
        )

        try:
            # Step 1: Text Processing
            result.processed_text = self._process_text(content, result)

            # Step 2: Claim Classification
            result.claim_predictions = self._classify_claims(result.processed_text, result)

            # Step 3: Claim Structuring (optional)
            if self.enable_structuring and result.claim_predictions:
                result.structured_claims = self._structure_claims(result.claim_predictions, result)

            # Step 4: Calculate summary statistics
            self._calculate_statistics(result)

            # Step 5: Record performance metrics
            end_time = datetime.now()
            result.processing_time_seconds = (end_time - start_time).total_seconds()
            result.models_used = self._get_model_versions()

            print(f"\n✅ EXTRACTION COMPLETE")
            print(f"   Processing time: {result.processing_time_seconds:.2f} seconds")
            print(f"   Claims found: {result.claims_identified}/{result.total_sentences}")
            print(f"   High confidence: {result.high_confidence_claims}")
            print(f"   Verifiable: {result.verifiable_claims}")

            return result

        except Exception as e:
            result.errors.append(f"Pipeline error: {e}")
            self.logger.error(f"Claim extraction failed for {url}: {e}")
            print(f"❌ Extraction failed: {e}")

            # Return partial result
            end_time = datetime.now()
            result.processing_time_seconds = (end_time - start_time).total_seconds()
            return result

    def _process_text(self, content: str, result: ExtractionResult) -> Optional[object]:
        """Step 1: Process and clean the text"""
        if not self.text_processor:
            result.errors.append("Text processor not available")
            return None

        print("📝 Step 1: Processing text...")

        try:
            processed = self.text_processor.preprocess_article(content)

            print(f"   ✅ Text processed: {len(processed.sentences)} sentences, {processed.word_count} words")
            result.total_sentences = len(processed.sentences)

            return processed

        except Exception as e:
            error_msg = f"Text processing failed: {e}"
            result.errors.append(error_msg)
            print(f"   ❌ {error_msg}")
            return None

    def _classify_claims(self, processed_text: object, result: ExtractionResult) -> List[object]:
        """Step 2: Classify sentences as claims or non-claims"""
        if not self.claim_classifier or not processed_text:
            result.errors.append("Claim classifier not available or no processed text")
            return []

        print("🤖 Step 2: Classifying claims with BERT...")

        try:
            sentences = processed_text.sentences

            # Process in batches for better performance
            all_predictions = []

            for i in range(0, len(sentences), self.batch_size):
                batch = sentences[i:i + self.batch_size]
                batch_predictions = self.claim_classifier.predict_batch(batch)
                all_predictions.extend(batch_predictions)

                print(
                    f"   Processed batch {i // self.batch_size + 1}/{(len(sentences) + self.batch_size - 1) // self.batch_size}")

            # Filter by confidence threshold
            filtered_predictions = [
                pred for pred in all_predictions
                if pred.is_claim and pred.confidence >= self.min_claim_confidence
            ]

            print(
                f"   ✅ Classification complete: {len(filtered_predictions)} claims found (confidence ≥ {self.min_claim_confidence})")

            return all_predictions

        except Exception as e:
            error_msg = f"Claim classification failed: {e}"
            result.errors.append(error_msg)
            print(f"   ❌ {error_msg}")
            return []

    def _structure_claims(self, claim_predictions: List[object], result: ExtractionResult) -> List[object]:
        """Step 3: Structure the identified claims"""
        if not self.claim_structurer or not claim_predictions:
            result.errors.append("Claim structurer not available or no predictions")
            return []

        print("🔧 Step 3: Structuring claims...")

        try:
            # Filter to only actual claims
            claims_to_structure = [
                pred for pred in claim_predictions
                if pred.is_claim and pred.confidence >= self.min_claim_confidence
            ]

            if not claims_to_structure:
                print("   ⚠️  No claims to structure")
                return []

            structured_claims = []

            for i, prediction in enumerate(claims_to_structure, 1):
                try:
                    structured = self.claim_structurer.structure_claim(
                        prediction.sentence,
                        prediction.claim_type or ClaimType.GENERAL_FACTUAL,
                        prediction.confidence
                    )
                    structured_claims.append(structured)

                except Exception as e:
                    print(f"   ⚠️  Failed to structure claim {i}: {e}")
                    continue

            print(f"   ✅ Structuring complete: {len(structured_claims)} claims structured")

            return structured_claims

        except Exception as e:
            error_msg = f"Claim structuring failed: {e}"
            result.errors.append(error_msg)
            print(f"   ❌ {error_msg}")
            return []

    def _calculate_statistics(self, result: ExtractionResult):
        """Step 4: Calculate summary statistics"""
        print("📊 Step 4: Calculating statistics...")

        # Count claims by confidence level
        if result.claim_predictions:
            result.claims_identified = sum(1 for pred in result.claim_predictions if pred.is_claim)
            result.high_confidence_claims = sum(
                1 for pred in result.claim_predictions
                if pred.is_claim and pred.confidence >= 0.8
            )

        # Count verifiable claims
        if result.structured_claims:
            result.verifiable_claims = sum(
                1 for structured in result.structured_claims
                if structured.claim.is_verifiable
            )

        print(f"   ✅ Statistics calculated")

    def _get_model_versions(self) -> Dict[str, str]:
        """Get versions of models used"""
        versions = {}

        if self.text_processor:
            versions['text_processor'] = 'v1.0'

        if self.claim_classifier:
            versions['bert_classifier'] = getattr(self.claim_classifier, 'model_name', 'unknown')

        if self.claim_structurer:
            versions['claim_structurer'] = getattr(self.claim_structurer, 'model_name', 'unknown')

        return versions

    def extract_claims_batch(self, articles: List[Dict[str, str]]) -> List[ExtractionResult]:
        """Extract claims from multiple articles"""
        print(f"\n🚀 BATCH CLAIM EXTRACTION: {len(articles)} articles")
        print("=" * 80)

        results = []

        for i, article in enumerate(articles, 1):
            print(f"\n📰 Processing article {i}/{len(articles)}: {article.get('title', 'Untitled')}")

            result = self.extract_claims(
                url=article.get('url', f'article_{i}'),
                title=article.get('title', ''),
                content=article.get('content', '')
            )

            results.append(result)

        # Batch summary
        total_claims = sum(r.claims_identified for r in results)
        total_sentences = sum(r.total_sentences for r in results)
        avg_processing_time = sum(r.processing_time_seconds for r in results) / len(results)

        print(f"\n📈 BATCH SUMMARY")
        print(f"   Articles processed: {len(articles)}")
        print(f"   Total sentences: {total_sentences}")
        print(f"   Total claims found: {total_claims}")
        print(f"   Average processing time: {avg_processing_time:.2f}s")
        print(f"   Claims per article: {total_claims / len(articles):.1f}")

        return results

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get status of pipeline components"""
        return {
            'text_processor': self.text_processor is not None,
            'claim_classifier': self.claim_classifier is not None,
            'claim_structurer': self.claim_structurer is not None,
            'structuring_enabled': self.enable_structuring,
            'batch_size': self.batch_size,
            'min_confidence': self.min_claim_confidence,
            'ready': all([
                self.text_processor is not None,
                self.claim_classifier is not None
            ])
        }


# Test and demonstration functions
def test_pipeline():
    """Test the complete pipeline with sample content"""
    print("🧪 TESTING COMPLETE CLAIM EXTRACTION PIPELINE")
    print("=" * 80)

    # Initialize pipeline
    pipeline = ClaimExtractionPipeline()

    # Check pipeline status
    status = pipeline.get_pipeline_status()
    print(f"\n📋 Pipeline Status:")
    for component, available in status.items():
        indicator = "✅" if available else "❌"
        print(f"   {component}: {indicator}")

    if not status['ready']:
        print("\n❌ Pipeline not ready. Cannot run test.")
        return

    # Test article
    test_article = {
        'url': 'https://example.com/climate-study',
        'title': 'New Climate Research Findings',
        'content': '''
        <h1>Major Climate Study Released</h1>

        <p>Scientists at Stanford University released groundbreaking research yesterday 
        showing that global average temperatures have risen by 1.1°C since 1880. 
        The study analyzed data from over 15,000 weather stations worldwide.</p>

        <p>Dr. Sarah Johnson, the lead researcher, stated that "this represents the 
        fastest rate of warming in recorded history." The research team found that 
        Arctic temperatures increased by 2.3°C over the same period.</p>

        <p>The study will be published in Nature Climate Change next month. 
        According to the researchers, immediate action is needed to address 
        climate change impacts.</p>

        <p>I think this research is very concerning and shows we need to act quickly. 
        What do you think about these findings?</p>
        '''
    }

    # Run extraction
    result = pipeline.extract_claims(
        url=test_article['url'],
        title=test_article['title'],
        content=test_article['content']
    )

    # Display detailed results
    print(f"\n📊 DETAILED RESULTS")
    print("-" * 60)
    print(f"URL: {result.url}")
    print(f"Processing time: {result.processing_time_seconds:.2f} seconds")
    print(f"Content hash: {result.content_hash}")

    if result.claim_predictions:
        print(f"\n🎯 CLAIM PREDICTIONS:")
        claims_found = [pred for pred in result.claim_predictions if pred.is_claim]

        for i, pred in enumerate(claims_found, 1):
            confidence_bar = "█" * int(pred.confidence * 10) + "░" * (10 - int(pred.confidence * 10))
            type_str = f"[{pred.claim_type.value}]" if pred.claim_type else "[general]"

            print(f"{i:2d}. {confidence_bar} {pred.confidence:.2f} {type_str}")
            print(f"     \"{pred.sentence}\"")
            print(f"     💡 {pred.reasoning}")
            print()

    if result.structured_claims:
        print(f"🔧 STRUCTURED CLAIMS:")
        for i, structured in enumerate(result.structured_claims, 1):
            claim = structured.claim
            print(
                f"{i:2d}. [{claim.claim_type.value}] Verifiable: {claim.is_verifiable} ({claim.verifiability_score:.2f})")
            print(f"     Subject: {claim.subject or 'N/A'}")
            print(f"     Predicate: {claim.predicate or 'N/A'}")
            print(f"     Object: {claim.object or 'N/A'}")
            print(f"     Entities: {len(claim.entities)} found")
            print(f"     Temporal: {claim.temporal_context or 'None'}")
            print()

    if result.errors:
        print(f"⚠️  ERRORS ENCOUNTERED:")
        for error in result.errors:
            print(f"   - {error}")

    print(f"\n✅ Pipeline test complete!")
    return result


def test_pipeline_batch():
    """Test pipeline with multiple articles"""
    print("\n🧪 TESTING BATCH PROCESSING")
    print("=" * 60)

    pipeline = ClaimExtractionPipeline()

    test_articles = [
        {
            'url': 'https://example.com/tech-news',
            'title': 'AI Breakthrough Announced',
            'content': 'Researchers at MIT announced a 40% improvement in AI processing speed. The new algorithm reduces energy consumption by 60%.'
        },
        {
            'url': 'https://example.com/health-study',
            'title': 'Health Study Results',
            'content': 'A clinical trial involving 1,200 patients showed that the new treatment is 85% effective. Side effects were minimal according to Dr. Smith.'
        },
        {
            'url': 'https://example.com/opinion',
            'title': 'Opinion Piece',
            'content': 'I believe that we should invest more in renewable energy. Many people think solar power is the future. What are your thoughts?'
        }
    ]

    results = pipeline.extract_claims_batch(test_articles)

    print(f"\n📈 BATCH RESULTS SUMMARY:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.title or 'Untitled'}")
        print(f"   Claims: {result.claims_identified}, Verifiable: {result.verifiable_claims}")
        print(f"   Time: {result.processing_time_seconds:.2f}s")

    return results


if __name__ == "__main__":
    # Run the complete test
    test_result = test_pipeline()

    # If successful, run batch test
    if test_result and not test_result.errors:
        batch_results = test_pipeline_batch()