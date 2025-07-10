"""
Claim extraction models and utilities
"""

# Import components with error handling
try:
    from .bert_classifier import BERTClaimClassifier, ClaimPrediction
    BERT_CLASSIFIER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: BERT classifier not available: {e}")
    BERT_CLASSIFIER_AVAILABLE = False

try:
    from .claim_structurer import ClaimStructurer, StructuredClaim
    CLAIM_STRUCTURER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Claim structurer not available: {e}")
    CLAIM_STRUCTURER_AVAILABLE = False

# Build exports
__all__ = []

if BERT_CLASSIFIER_AVAILABLE:
    __all__.extend(["BERTClaimClassifier", "ClaimPrediction"])

if CLAIM_STRUCTURER_AVAILABLE:
    __all__.extend(["ClaimStructurer", "StructuredClaim"])

# Component status
COMPONENT_STATUS = {
    "bert_classifier": BERT_CLASSIFIER_AVAILABLE,
    "claim_structurer": CLAIM_STRUCTURER_AVAILABLE,
    "all_available": BERT_CLASSIFIER_AVAILABLE and CLAIM_STRUCTURER_AVAILABLE
}

def get_available_components():
    """Get list of available claim extraction components"""
    available = []
    if BERT_CLASSIFIER_AVAILABLE:
        available.append("BERTClaimClassifier")
    if CLAIM_STRUCTURER_AVAILABLE:
        available.append("ClaimStructurer")
    return available