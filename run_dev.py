#!/usr/bin/env python3
"""
Development runner - handles path setup and runs components
Save as: run_dev.py
"""
import sys
import os
from pathlib import Path

# Setup Python path
project_root = Path(__file__).parent.absolute()
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))


def test_bert_classifier():
    """Test the BERT classifier with proper path setup"""
    print("🤖 Testing BERT Classifier with Path Setup")
    print("=" * 60)

    try:
        # Import with path already set up
        from truthed.models.claim_extraction.bert_classifier import BERTClaimClassifier
        print("✅ Successfully imported BERTClaimClassifier")

        # Create classifier
        classifier = BERTClaimClassifier()

        # Test sentences
        test_sentences = [
            "Global temperatures have risen by 1.1°C since 1880.",
            "Scientists at Stanford University published a study yesterday.",
            "I think climate change is a serious issue.",
            "What causes global warming?",
            "The research team analyzed 15,000 weather stations.",
            "COVID-19 cases increased by 25% last week."
        ]

        print(f"\n🔍 Testing on {len(test_sentences)} sentences...")

        # Run predictions
        predictions = classifier.predict_batch(test_sentences)

        # Display results
        print(f"\n📊 RESULTS:")
        print("-" * 60)

        claims_found = 0
        for i, pred in enumerate(predictions, 1):
            claim_indicator = "🎯" if pred.is_claim else "💬"
            type_str = f"[{pred.claim_type.value}]" if pred.claim_type else "[non-claim]"

            print(f"{i}. {claim_indicator} {pred.confidence:.2f} {type_str}")
            print(f"   \"{pred.sentence}\"")
            print(f"   💡 {pred.reasoning}")
            print()

            if pred.is_claim:
                claims_found += 1

        print(f"📈 SUMMARY:")
        print(f"   Total sentences: {len(predictions)}")
        print(f"   Claims found: {claims_found}")
        print(f"   Non-claims: {len(predictions) - claims_found}")

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print(f"💡 Make sure you've run: python setup_dev_environment.py")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_text_processing():
    """Test text processing integration"""
    print("\n📝 Testing Text Processing Integration")
    print("=" * 60)

    try:
        from truthed.data.preprocessing.text_processing import TextProcessor
        from truthed.models.claim_extraction.bert_classifier import BERTClaimClassifier

        # Create components
        text_processor = TextProcessor()
        claim_classifier = BERTClaimClassifier()

        # Test article
        test_article = '''
        <h1>Climate Research Update</h1>
        <p>Scientists at MIT released new findings today. The research team 
        studied global temperature data spanning 140 years.</p>

        <p>Their analysis shows that average global temperatures have increased 
        by 1.2°C since 1880. This represents unprecedented warming in recorded history.</p>

        <p>"The results are concerning," said lead researcher Dr. Sarah Chen. 
        "We need immediate action to address climate change."</p>

        <p>The study will be published in Nature next month. Climate activists 
        say this proves their point about global warming.</p>
        '''

        print(f"📄 Processing test article...")

        # Step 1: Text processing
        processed = text_processor.preprocess_article(test_article)
        print(f"✅ Text processing: {len(processed.sentences)} sentences")

        # Step 2: Claim extraction
        predictions = claim_classifier.predict_batch(processed.sentences)

        # Step 3: Results
        claims = [p for p in predictions if p.is_claim]
        print(f"✅ Claim extraction: {len(claims)} claims found")

        print(f"\n🎯 EXTRACTED CLAIMS:")
        for i, claim in enumerate(claims, 1):
            print(f"{i}. [{claim.claim_type.value}] {claim.confidence:.2f}")
            print(f"   \"{claim.sentence}\"")
            print()

        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Main development test runner"""
    print("🚀 TRUTHED DEVELOPMENT ENVIRONMENT")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path[0]}")

    # Test 1: BERT Classifier
    test1_success = test_bert_classifier()

    if test1_success:
        # Test 2: Integration
        test2_success = test_text_processing()

        if test2_success:
            print(f"\n🎉 ALL TESTS PASSED!")
            print(f"✅ BERT classifier working")
            print(f"✅ Text processing integration working")
            print(f"\n🎯 Ready to implement claim extraction pipeline!")
        else:
            print(f"\n⚠️  BERT classifier works, but integration has issues")
    else:
        print(f"\n❌ BERT classifier setup failed")
        print(f"💡 Try running: python setup_dev_environment.py")


if __name__ == "__main__":
    main()