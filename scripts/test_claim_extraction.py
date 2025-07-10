"""
Test script for claim extraction development
File: scripts/test_claim_extraction.py
"""

import sys
from pathlib import Path

# setup path to find our modules
project_root = Path(__file__).parent.parent  # Go from scripts/ to project root
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

print(f"🔧 Project root: {project_root}")
print(f"🔧 Source path: {src_path}")
print(f"🔧 Source exists: {src_path.exists()}")

try:
    from truthed.data.preprocessing.text_processing import TextProcessor, ProcessedText

    print("✅ Successfully imported TextProcessor")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


def test_text_processing():
    """Test the working text processing component"""
    print("\n" + "=" * 50)
    print("TESTING TEXT PROCESSING")
    print("=" * 50)

    processor = TextProcessor()

    # Test article content
    test_article = """
    <h1>Climate Study Results</h1>
    <p>Scientists at Stanford University released a major study yesterday. 
    The research team analyzed data from 15,000 weather stations worldwide.</p>

    <p>The study found that global average temperatures have risen by 1.1°C 
    since 1880. This represents the fastest warming rate in recorded history.</p>

    <p>Dr. Sarah Johnson, lead researcher, said the findings are "deeply concerning." 
    She believes immediate action is needed to address climate change.</p>

    <p>The study will be published in Nature Climate Change next month.</p>
    """

    # Process the text
    result = processor.preprocess_article(test_article)

    print(f"📊 PROCESSING RESULTS:")
    print(f"   Original length: {len(test_article)} characters")
    print(f"   Cleaned length: {len(result.cleaned_text)} characters")
    print(f"   Word count: {result.word_count}")
    print(f"   Sentences found: {len(result.sentences)}")
    print(f"   Paragraphs found: {len(result.paragraphs)}")

    print(f"\n📝 SENTENCES EXTRACTED:")
    for i, sentence in enumerate(result.sentences, 1):
        # Test the claim detection heuristic
        is_likely_claim = processor.is_likely_claim_sentence(sentence)
        marker = "🎯" if is_likely_claim else "💬"
        print(f"   {i}. {marker} {sentence}")

    print(f"\n📄 PARAGRAPHS:")
    for i, paragraph in enumerate(result.paragraphs, 1):
        preview = paragraph[:100] + "..." if len(paragraph) > 100 else paragraph
        print(f"   {i}. {preview}")

    return result


def test_claim_extraction_placeholder():
    """Placeholder for claim extraction testing"""
    print("\n" + "=" * 50)
    print("CLAIM EXTRACTION (PLACEHOLDER)")
    print("=" * 50)

    print("⚠️  Claim extraction not implemented yet")
    print("📋 Next steps:")
    print("   1. Implement BERTClaimClassifier")
    print("   2. Create ClaimStructurer")
    print("   3. Build ClaimExtractionPipeline")
    print("   4. Add fact verification")

    # Mock what the output would look like
    mock_claims = [
        {
            'text': 'Global average temperatures have risen by 1.1°C since 1880',
            'type': 'statistical',
            'confidence': 0.89,
            'subject': 'Global average temperatures',
            'predicate': 'have risen by',
            'object': '1.1°C since 1880',
            'verifiable': True
        },
        {
            'text': 'This represents the fastest warming rate in recorded history',
            'type': 'comparative',
            'confidence': 0.76,
            'subject': 'This',
            'predicate': 'represents',
            'object': 'the fastest warming rate in recorded history',
            'verifiable': True
        }
    ]

    print(f"\n🎯 MOCK CLAIM EXTRACTION RESULTS:")
    for i, claim in enumerate(mock_claims, 1):
        print(f"   {i}. [{claim['type'].upper()}] \"{claim['text']}\"")
        print(f"      Confidence: {claim['confidence']:.2f}")
        print(f"      Verifiable: {'✅' if claim['verifiable'] else '❌'}")
        print()


def main():
    """Main test function"""
    print("🚀 CLAIM EXTRACTION DEVELOPMENT TEST")
    print("=" * 60)

    # Test what's working
    try:
        text_result = test_text_processing()
        print("✅ Text processing test passed")
    except Exception as e:
        print(f"❌ Text processing test failed: {e}")
        return False

    # Show what's coming next
    test_claim_extraction_placeholder()

    print("\n" + "=" * 60)
    print("📈 DEVELOPMENT STATUS:")
    print("   ✅ Text processing - Working")
    print("   ⚠️  Claim extraction - In development")
    print("   ⚠️  Source analysis - Planned")
    print("   ⚠️  Fact verification - Planned")
    print("   ⚠️  Ensemble scoring - Planned")

    print(f"\n🎯 THIS WEEK'S GOAL:")
    print("   Implement working claim extraction on simple test articles")

    return True


if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎉 Test completed successfully!")
    else:
        print(f"\n💥 Test failed!")
        sys.exit(1)