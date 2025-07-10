"""
Text preprocessing and claim extraction modules
"""

from .text_processing import TextProcessor, ProcessedText

# Try to import claim extraction components
try:
    from .claim_extraction import ClaimExtractionPipeline, ExtractionResult
    CLAIM_EXTRACTION_AVAILABLE = True
except ImportError:
    CLAIM_EXTRACTION_AVAILABLE = False

if CLAIM_EXTRACTION_AVAILABLE:
    __all__ = ["TextProcessor", "ProcessedText", "ClaimExtractionPipeline", "ExtractionResult"]
else:
    __all__ = ["TextProcessor", "ProcessedText"]