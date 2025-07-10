"""
BERT-based claim classifier implementation
File: src/truthed/models/claim_extraction/bert_classifier.py
"""

import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    pipeline
)
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
from pathlib import Path
import re

# Import our core models
from truthed.core.models import ClaimType

logger = logging.getLogger(__name__)


@dataclass
class ClaimPrediction:
    """Prediction result for a single sentence"""
    sentence: str
    is_claim: bool
    confidence: float
    claim_type: Optional[ClaimType] = None
    reasoning: str = ""


class BERTClaimClassifier:
    """
    BERT-based classifier for identifying factual claims in sentences.

    This is the core ML component that determines if a sentence contains
    a factual claim that can be verified.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """
        Initialize the BERT classifier.

        Args:
            model_name: Name of the pre-trained model to use
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.classifier_pipeline = None
        self.is_loaded = False

        print(f"🤖 BERTClaimClassifier initialized")
        print(f"   Model: {model_name}")
        print(f"   Device: {self.device}")

        # Load the model
        self._load_model()

    def _load_model(self):
        """Load the pre-trained BERT model and tokenizer"""
        try:
            print(f"📥 Loading {self.model_name}...")

            # Load tokenizer
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)

            # For now, we'll use a zero-shot classification approach
            # Later, we can fine-tune on claim-specific data
            self.classifier_pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=0 if self.device.type == "cuda" else -1
            )

            self.is_loaded = True
            print(f"✅ Model loaded successfully")

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print(f"💡 Falling back to rule-based classification")
            self.is_loaded = False

    def predict_batch(self, sentences: List[str]) -> List[ClaimPrediction]:
        """
        Classify a batch of sentences as claims or non-claims.

        Args:
            sentences: List of sentences to classify

        Returns:
            List of ClaimPrediction objects with results
        """
        if not sentences:
            return []

        print(f"🔍 Classifying {len(sentences)} sentences...")

        predictions = []

        for sentence in sentences:
            if self.is_loaded:
                prediction = self._predict_with_bert(sentence)
            else:
                prediction = self._predict_with_rules(sentence)

            predictions.append(prediction)

        return predictions

    def _predict_with_bert(self, sentence: str) -> ClaimPrediction:
        """Use BERT for zero-shot classification"""
        try:
            # Define candidate labels for zero-shot classification
            candidate_labels = [
                "factual claim that can be verified",
                "opinion or subjective statement",
                "question or instruction",
                "general conversation"
            ]

            # Run zero-shot classification
            result = self.classifier_pipeline(sentence, candidate_labels)

            # Extract results
            top_label = result['labels'][0]
            confidence = result['scores'][0]

            # Determine if it's a claim
            is_claim = "factual claim" in top_label

            # Classify claim type if it's a claim
            claim_type = None
            if is_claim:
                claim_type = self._classify_claim_type(sentence)

            return ClaimPrediction(
                sentence=sentence,
                is_claim=is_claim,
                confidence=confidence,
                claim_type=claim_type,
                reasoning=f"BERT classified as: {top_label}"
            )

        except Exception as e:
            logger.error(f"BERT prediction failed: {e}")
            # Fallback to rule-based
            return self._predict_with_rules(sentence)

    def _predict_with_rules(self, sentence: str) -> ClaimPrediction:
        """Fallback rule-based classification"""

        # Clean up the sentence
        sentence = sentence.strip()
        if len(sentence) < 10:
            return ClaimPrediction(
                sentence=sentence,
                is_claim=False,
                confidence=0.9,
                reasoning="Too short to be a meaningful claim"
            )

        # Rule-based indicators for factual claims
        claim_indicators = {
            'statistical': [
                r'\d+%', r'\d+\.\d+%', r'\d+ percent',
                r'\d+°[CF]', r'\d+ degrees',
                r'\$\d+', r'\d+ million', r'\d+ billion',
                r'\d+ times', r'increased by \d+', r'decreased by \d+'
            ],
            'temporal': [
                r'\d{4}', r'last year', r'this year', r'next year',
                r'yesterday', r'today', r'tomorrow',
                r'in \d{4}', r'since \d{4}', r'by \d{4}'
            ],
            'causal': [
                r'caused by', r'leads to', r'results in',
                r'due to', r'because of', r'as a result'
            ],
            'identity': [
                r'is the', r'was the', r'will be the',
                r'are the', r'were the'
            ],
            'research': [
                r'study found', r'research shows', r'scientists discovered',
                r'according to', r'report says', r'data shows'
            ]
        }

        # Check for claim indicators
        sentence_lower = sentence.lower()
        confidence = 0.0
        claim_type = None
        reasons = []

        for ctype, patterns in claim_indicators.items():
            for pattern in patterns:
                if re.search(pattern, sentence_lower):
                    confidence += 0.2
                    if claim_type is None:
                        claim_type = ClaimType(ctype) if ctype in [ct.value for ct in
                                                                   ClaimType] else ClaimType.GENERAL_FACTUAL
                    reasons.append(f"Found {ctype} indicator: {pattern}")

        # Additional heuristics
        if any(word in sentence_lower for word in ['study', 'research', 'found', 'discovered']):
            confidence += 0.15
            reasons.append("Contains research language")

        if re.search(r'[A-Z][a-z]+ (University|Institute|College)', sentence):
            confidence += 0.1
            reasons.append("Contains institutional reference")

        if any(char.isdigit() for char in sentence):
            confidence += 0.1
            reasons.append("Contains numbers")

        # Penalty for subjective language
        subjective_words = ['think', 'believe', 'feel', 'opinion', 'probably', 'maybe', 'perhaps']
        if any(word in sentence_lower for word in subjective_words):
            confidence -= 0.3
            reasons.append("Contains subjective language")

        # Penalty for questions
        if sentence.strip().endswith('?'):
            confidence -= 0.2
            reasons.append("Is a question")

        # Normalize confidence
        confidence = max(0.0, min(1.0, confidence))
        is_claim = confidence > 0.5

        if not is_claim:
            claim_type = None
        elif claim_type is None:
            claim_type = ClaimType.GENERAL_FACTUAL

        return ClaimPrediction(
            sentence=sentence,
            is_claim=is_claim,
            confidence=confidence,
            claim_type=claim_type,
            reasoning=f"Rule-based: {'; '.join(reasons) if reasons else 'No strong indicators'}"
        )

    def _classify_claim_type(self, sentence: str) -> ClaimType:
        """Classify the type of claim"""
        sentence_lower = sentence.lower()

        # Statistical claims
        if re.search(r'\d+%|\d+\.\d+%|\d+ percent|increased by|decreased by', sentence_lower):
            return ClaimType.STATISTICAL

        # Temporal claims
        if re.search(r'\d{4}|last year|this year|yesterday|since|until|before|after', sentence_lower):
            return ClaimType.TEMPORAL

        # Causal claims
        if re.search(r'caused|leads to|results in|due to|because|as a result', sentence_lower):
            return ClaimType.CAUSAL

        # Identity claims
        if re.search(r'is the|was the|are the|were the', sentence_lower):
            return ClaimType.IDENTITY

        # Existential claims
        if re.search(r'exists|there is|there are|discovered|found', sentence_lower):
            return ClaimType.EXISTENTIAL

        return ClaimType.GENERAL_FACTUAL


# Test function
def test_bert_classifier():
    """Test the BERT classifier with sample sentences"""
    print("🧪 Testing BERTClaimClassifier")
    print("=" * 50)

    classifier = BERTClaimClassifier()

    test_sentences = [
        # Clear factual claims
        "Global temperatures have risen by 1.1°C since 1880.",
        "The study analyzed data from 15,000 weather stations.",
        "Stanford University published the research yesterday.",

        # Statistical claims
        "COVID-19 cases increased by 25% last week.",
        "The company's revenue grew to $50 million in 2023.",

        # Opinions/subjective
        "I think climate change is a serious problem.",
        "This movie is really good.",
        "We should do something about pollution.",

        # Questions
        "What causes climate change?",
        "How many people live in New York?",

        # Mixed/borderline
        "Climate change is real.",
        "The weather is getting warmer.",
    ]

    predictions = classifier.predict_batch(test_sentences)

    print(f"\n📊 CLASSIFICATION RESULTS:")
    print("-" * 80)

    for pred in predictions:
        claim_indicator = "🎯" if pred.is_claim else "💬"
        type_str = f"[{pred.claim_type.value}]" if pred.claim_type else "[non-claim]"

        print(f"{claim_indicator} {pred.confidence:.2f} {type_str}")
        print(f"   \"{pred.sentence}\"")
        print(f"   Reasoning: {pred.reasoning}")
        print()

    # Summary
    claims_found = sum(1 for p in predictions if p.is_claim)
    print(f"📈 SUMMARY:")
    print(f"   Total sentences: {len(predictions)}")
    print(f"   Claims identified: {claims_found}")
    print(f"   Non-claims: {len(predictions) - claims_found}")


if __name__ == "__main__":
    test_bert_classifier()