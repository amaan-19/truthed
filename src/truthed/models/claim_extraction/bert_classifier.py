"""
BERT classifier
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
import re
import numpy as np

# Import our core models
try:
    from truthed.core.models import ClaimType
except ImportError:
    # Fallback if core models not available
    from enum import Enum
    class ClaimType(str, Enum):
        STATISTICAL = "statistical"
        CAUSAL = "causal"
        TEMPORAL = "temporal"
        IDENTITY = "identity"
        EXISTENTIAL = "existential"
        GENERAL_FACTUAL = "general_factual"

logger = logging.getLogger(__name__)


@dataclass
class ClaimPrediction:
    """Prediction result for a single sentence"""
    sentence: str
    is_claim: bool
    confidence: float
    claim_type: Optional[ClaimType] = None
    reasoning: str = ""
    verifiability_score: float = 0.0


class BERTClaimClassifier:
    """
    BERT-based classifier for identifying factual claims in sentences.

    FIXED VERSION: Properly handles confidence thresholds
    """

    def __init__(self, model_name: str = "facebook/bart-large-mnli", min_claim_confidence: float = 0.6):
        """Initialize the BERT classifier."""
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.classifier_pipeline = None
        self.is_loaded = False
        self.min_claim_confidence = min_claim_confidence  # Store the threshold

        print(f"🤖 BERTClaimClassifier initializing...")
        print(f"   Model: {self.model_name}")
        print(f"   Device: {self.device}")
        print(f"   Min confidence threshold: {self.min_claim_confidence}")

        self._load_model()

    def set_confidence_threshold(self, threshold: float):
        """Update the confidence threshold"""
        self.min_claim_confidence = threshold
        print(f"🎯 Updated confidence threshold to: {threshold}")

    def _load_model(self):
        """Load the pre-trained BERT model"""
        try:
            print(f"📥 Loading model for zero-shot classification...")

            # Use a model specifically good for zero-shot classification
            self.classifier_pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )

            self.is_loaded = True
            print(f"✅ Model loaded successfully")

        except Exception as e:
            print(f"❌ Failed to load BERT model: {e}")
            print(f"💡 Falling back to rule-based classification only")
            self.is_loaded = False

    def predict_batch(self, sentences: List[str]) -> List[ClaimPrediction]:
        """Classify a batch of sentences as claims or non-claims."""
        if not sentences:
            return []

        print(f"🔍 Analyzing {len(sentences)} sentences for factual claims...")
        print(f"🎯 Using confidence threshold: {self.min_claim_confidence}")

        predictions = []

        for i, sentence in enumerate(sentences, 1):
            # Quick pre-filter with rules
            rule_result = self._quick_rule_filter(sentence)

            if rule_result['skip_ml']:
                # Use rule-based result
                prediction = ClaimPrediction(
                    sentence=sentence,
                    is_claim=rule_result['is_claim'],
                    confidence=rule_result['confidence'],
                    claim_type=rule_result['claim_type'],
                    reasoning=rule_result['reasoning'],
                    verifiability_score=rule_result['verifiability']
                )
            else:
                # Use BERT for borderline cases
                if self.is_loaded:
                    prediction = self._predict_with_bert(sentence)
                else:
                    prediction = self._predict_with_rules(sentence)

            predictions.append(prediction)

            # Progress indicator for large batches
            if len(sentences) > 10 and i % 5 == 0:
                print(f"   Processed {i}/{len(sentences)} sentences...")

        claims_found = sum(1 for p in predictions if p.is_claim)
        print(f"📊 Analysis complete: {claims_found}/{len(predictions)} claims identified")

        return predictions

    def _quick_rule_filter(self, sentence: str) -> Dict[str, Any]:
        """Quick rule-based filter to skip obvious non-claims"""
        sentence = sentence.strip()
        sentence_lower = sentence.lower()

        # Too short
        if len(sentence) < 15:
            return {
                'skip_ml': True,
                'is_claim': False,
                'confidence': 0.95,
                'claim_type': None,
                'reasoning': 'Sentence too short for factual claim',
                'verifiability': 0.0
            }

        # Questions
        if sentence.endswith('?'):
            return {
                'skip_ml': True,
                'is_claim': False,
                'confidence': 0.9,
                'claim_type': None,
                'reasoning': 'Questions are not factual claims',
                'verifiability': 0.0
            }

        # Clear subjective markers
        subjective_indicators = [
            'i think', 'i believe', 'i feel', 'in my opinion',
            'personally', 'i would say', 'it seems to me'
        ]

        if any(indicator in sentence_lower for indicator in subjective_indicators):
            return {
                'skip_ml': True,
                'is_claim': False,
                'confidence': 0.85,
                'claim_type': None,
                'reasoning': 'Contains subjective opinion markers',
                'verifiability': 0.1
            }

        # Strong factual indicators - definitely claims
        strong_claim_patterns = [
            r'\d+\.\d+%',  # Precise percentages
            r'\$\d+\.?\d* (million|billion)',  # Large money amounts
            r'\d+°[CF]',  # Temperature measurements
            r'study (found|showed|revealed)',  # Research findings
            r'according to.*study',  # Study citations
            r'data (shows|indicates|reveals)',  # Data references
        ]

        for pattern in strong_claim_patterns:
            if re.search(pattern, sentence_lower):
                claim_type = self._classify_claim_type_rules(sentence)
                return {
                    'skip_ml': True,
                    'is_claim': True,
                    'confidence': 0.9,
                    'claim_type': claim_type,
                    'reasoning': f'Strong factual indicator: {pattern}',
                    'verifiability': 0.8
                }

        # Let BERT decide for borderline cases
        return {
            'skip_ml': False,
            'is_claim': False,
            'confidence': 0.0,
            'claim_type': None,
            'reasoning': '',
            'verifiability': 0.0
        }

    def _predict_with_bert(self, sentence: str) -> ClaimPrediction:
        """Use BERT for nuanced classification"""
        try:
            # Define labels for claim vs non-claim classification
            candidate_labels = [
                "factual statement that can be verified",
                "personal opinion or belief",
                "general conversation or greeting",
                "instruction or question"
            ]

            # Run classification
            result = self.classifier_pipeline(sentence, candidate_labels)

            top_label = result['labels'][0]
            confidence = result['scores'][0]

            # FIXED: Use the instance threshold, not hardcoded value
            is_claim = "factual statement" in top_label and confidence > self.min_claim_confidence

            # If it's a claim, classify the type
            claim_type = None
            verifiability = 0.0

            if is_claim:
                claim_type = self._classify_claim_type_rules(sentence)
                verifiability = self._assess_verifiability(sentence)

            reasoning = f"BERT: {top_label} (confidence: {confidence:.3f})"

            return ClaimPrediction(
                sentence=sentence,
                is_claim=is_claim,
                confidence=confidence,
                claim_type=claim_type,
                reasoning=reasoning,
                verifiability_score=verifiability
            )

        except Exception as e:
            logger.error(f"BERT prediction failed for: {sentence[:50]}... Error: {e}")
            return self._predict_with_rules(sentence)

    def _predict_with_rules(self, sentence: str) -> ClaimPrediction:
        """Enhanced rule-based classification as fallback"""
        sentence = sentence.strip()
        sentence_lower = sentence.lower()

        if len(sentence) < 15:
            return ClaimPrediction(
                sentence=sentence,
                is_claim=False,
                confidence=0.9,
                reasoning="Sentence too short for meaningful claim"
            )

        # Calculate claim score based on multiple indicators
        claim_score = 0.0
        reasons = []

        # Statistical indicators (+++)
        statistical_patterns = [
            r'\d+%', r'\d+\.\d+%', r'\d+ percent',
            r'\$\d+', r'\d+ million', r'\d+ billion',
            r'increased by \d+', r'decreased by \d+',
            r'\d+ times (more|less|higher|lower)'
        ]

        for pattern in statistical_patterns:
            if re.search(pattern, sentence_lower):
                claim_score += 0.3
                reasons.append(f"Statistical data: {pattern}")
                break

        # Research/authority indicators (++)
        authority_patterns = [
            r'(study|research|report) (found|shows|indicates)',
            r'according to (scientists|researchers|experts)',
            r'(university|institute) (published|released)',
            r'(dr\.|professor) .+ (said|stated|reported)'
        ]

        for pattern in authority_patterns:
            if re.search(pattern, sentence_lower):
                claim_score += 0.25
                reasons.append(f"Authority reference: {pattern}")
                break

        # Temporal specificity (+)
        temporal_patterns = [
            r'\d{4}', r'(last|this|next) (year|month|week)',
            r'(yesterday|today|tomorrow)',
            r'in (january|february|march|april|may|june|july|august|september|october|november|december)',
            r'since \d{4}', r'by \d{4}'
        ]

        for pattern in temporal_patterns:
            if re.search(pattern, sentence_lower):
                claim_score += 0.15
                reasons.append(f"Temporal specificity: {pattern}")
                break

        # Entity names (+)
        if re.search(r'\b[A-Z][a-z]+ (University|Institute|Organization|Company|Corporation)\b', sentence):
            claim_score += 0.1
            reasons.append("Named institution")

        # Geographic specificity (+)
        if re.search(r'\b[A-Z][a-z]+, [A-Z][A-Z]\b|\b[A-Z][a-z]+ (County|State|Country)\b', sentence):
            claim_score += 0.1
            reasons.append("Geographic specificity")

        # Penalty for subjective language (--)
        subjective_words = ['think', 'believe', 'feel', 'opinion', 'probably', 'maybe', 'perhaps', 'seem']
        subjective_count = sum(1 for word in subjective_words if word in sentence_lower)
        if subjective_count > 0:
            claim_score -= 0.3 * subjective_count
            reasons.append(f"Subjective language ({subjective_count} indicators)")

        # Penalty for questions (-)
        if sentence.strip().endswith('?'):
            claim_score -= 0.4
            reasons.append("Question format")

        # Normalize score
        confidence = max(0.0, min(1.0, claim_score))

        # FIXED: Use the instance threshold
        is_claim = confidence > self.min_claim_confidence

        # Determine claim type
        claim_type = None
        verifiability = 0.0

        if is_claim:
            claim_type = self._classify_claim_type_rules(sentence)
            verifiability = self._assess_verifiability(sentence)

        reasoning = f"Rule-based analysis: {'; '.join(reasons) if reasons else 'No strong indicators'}"

        return ClaimPrediction(
            sentence=sentence,
            is_claim=is_claim,
            confidence=confidence,
            claim_type=claim_type,
            reasoning=reasoning,
            verifiability_score=verifiability
        )

    def _classify_claim_type_rules(self, sentence: str) -> ClaimType:
        """Classify the type of claim using rules"""
        sentence_lower = sentence.lower()

        # Statistical claims
        if re.search(r'\d+%|\d+\.\d+%|\d+ percent|increased by|decreased by|\$\d+', sentence_lower):
            return ClaimType.STATISTICAL

        # Temporal claims
        if re.search(r'\d{4}|last year|this year|yesterday|since|until|before|after', sentence_lower):
            return ClaimType.TEMPORAL

        # Causal claims
        if re.search(r'caused|leads to|results in|due to|because|as a result|triggers', sentence_lower):
            return ClaimType.CAUSAL

        # Identity claims
        if re.search(r'is the|was the|are the|were the|identified as|named as', sentence_lower):
            return ClaimType.IDENTITY

        # Existential claims
        if re.search(r'exists|there is|there are|discovered|found|detected', sentence_lower):
            return ClaimType.EXISTENTIAL

        return ClaimType.GENERAL_FACTUAL

    def _assess_verifiability(self, sentence: str) -> float:
        """Assess how verifiable a claim is (0.0 to 1.0)"""
        sentence_lower = sentence.lower()
        verifiability = 0.5  # Base score

        # High verifiability indicators
        if re.search(r'study|research|data|statistics|according to', sentence_lower):
            verifiability += 0.3

        if re.search(r'\d+%|\$\d+|\d+ degrees|measured|recorded', sentence_lower):
            verifiability += 0.2

        if re.search(r'published|reported|official|government', sentence_lower):
            verifiability += 0.2

        # Low verifiability indicators
        if re.search(r'will|might|could|probably|estimated|predicted', sentence_lower):
            verifiability -= 0.2

        if re.search(r'secret|classified|rumored|alleged', sentence_lower):
            verifiability -= 0.3

        return max(0.0, min(1.0, verifiability))


# Test function
def test_claim_classifier_fixed():
    """Test the fixed BERT classifier"""
    print("🧪 Testing FIXED BERTClaimClassifier")
    print("=" * 60)

    # Test with different thresholds
    thresholds = [0.4, 0.5, 0.6]

    test_sentences = [
        "Scientists at MIT announced a breakthrough in renewable energy.",
        "The new solar panels achieve 47% efficiency, compared to 22% for current technology.",
        "The research team tested 500 prototypes over 18 months.",
        "Dr. Smith believes this could reduce energy costs by 60%.",
        "Apple Inc. reported revenue of $365 billion in fiscal year 2021."
    ]

    for threshold in thresholds:
        print(f"\n🎯 Testing with threshold: {threshold}")
        print("-" * 40)

        classifier = BERTClaimClassifier(min_claim_confidence=threshold)
        predictions = classifier.predict_batch(test_sentences)

        claims_found = sum(1 for p in predictions if p.is_claim)
        print(f"Claims detected: {claims_found}/{len(predictions)}")

        for i, pred in enumerate(predictions, 1):
            marker = "🎯" if pred.is_claim else "💬"
            print(f"  {i}. {marker} {pred.confidence:.3f} - {pred.sentence[:50]}...")


if __name__ == "__main__":
    test_claim_classifier_fixed()