"""
Data processing and collection modules
"""

from .preprocessing.text_processing import TextProcessor, ProcessedText

try:
    from .preprocessing.claim_extraction import ClaimExtractionPipeline, ExtractionResult
    __all__ = ["TextProcessor", "ProcessedText", "ClaimExtractionPipeline", "ExtractionResult"]
except ImportError:
    # Pipeline not available yet
    __all__ = ["TextProcessor", "ProcessedText"]