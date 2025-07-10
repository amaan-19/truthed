"""
Machine learning models for claim detection and analysis
"""

# Core model components
try:
    from .claim_extraction.bert_classifier import BERTClaimClassifier, ClaimPrediction
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

try:
    from .claim_extraction.claim_structurer import ClaimStructurer, StructuredClaim
    STRUCTURER_AVAILABLE = True
except ImportError:
    STRUCTURER_AVAILABLE = False

# Build __all__ based on what's available
__all__ = []

if BERT_AVAILABLE:
    __all__.extend(["BERTClaimClassifier", "ClaimPrediction"])

if STRUCTURER_AVAILABLE:
    __all__.extend(["ClaimStructurer", "StructuredClaim"])

# Model availability info
MODEL_AVAILABILITY = {
    "bert_classifier": BERT_AVAILABLE,
    "claim_structurer": STRUCTURER_AVAILABLE
}