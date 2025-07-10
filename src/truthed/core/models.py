"""
Core data models and schemas for the truthed system.
File: src/truthed/core/models.py
"""
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from uuid import UUID, uuid4


class VeracityLabel(str, Enum):
    """Veracity labels for content and claims"""
    TRUE = "TRUE"
    FALSE = "FALSE"
    MIXED = "MIXED"
    UNVERIFIABLE = "UNVERIFIABLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ClaimType(str, Enum):
    """Types of factual claims"""
    STATISTICAL = "statistical"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    IDENTITY = "identity"
    EXISTENTIAL = "existential"
    GENERAL_FACTUAL = "general_factual"


class MisinformationType(str, Enum):
    """Types of misinformation"""
    FALSE_FACTUAL_CLAIM = "false_factual_claim"
    MISLEADING_STATISTICS = "misleading_statistics"
    MISSING_CONTEXT = "missing_context"
    CLICKBAIT_HEADLINE = "clickbait_headline"
    BIASED_FRAMING = "biased_framing"
    OUTDATED_INFORMATION = "outdated_information"
    CONSPIRACY_THEORY = "conspiracy_theory"
    HEALTH_MISINFORMATION = "health_misinformation"


@dataclass
class Entity:
    """Named entity extracted from text"""
    text: str
    label: str  # PERSON, ORG, GPE, DATE, etc.
    start: int
    end: int
    confidence: float = 1.0

    def __post_init__(self):
        # Validate confidence score
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class Claim:
    """Structured representation of a factual claim"""
    id: UUID = field(default_factory=uuid4)
    text: str = ""
    claim_type: ClaimType = ClaimType.GENERAL_FACTUAL
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    entities: List[Entity] = field(default_factory=list)
    temporal_context: Optional[str] = None
    verifiability_score: float = 0.5
    is_verifiable: bool = True
    confidence: float = 0.0
    extracted_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        # Validate scores
        if not 0.0 <= self.verifiability_score <= 1.0:
            raise ValueError("Verifiability score must be between 0.0 and 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class VerificationResult:
    """Result of fact verification for a claim"""
    claim_id: UUID
    label: VeracityLabel
    confidence: float = 0.0
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    contradicting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    sources_checked: int = 0
    verification_time: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class SourceCredibility:
    """Source credibility analysis result"""
    domain: str
    credibility_score: float = 0.5
    bias_rating: Optional[str] = None
    factual_reporting: Optional[str] = None
    confidence: float = 0.5
    explanation: str = ""
    last_updated: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not 0.0 <= self.credibility_score <= 1.0:
            raise ValueError("Credibility score must be between 0.0 and 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class ContentFeatures:
    """Extracted features from content"""
    word_count: int = 0
    sentiment_score: float = 0.0
    readability_score: float = 0.0
    caps_ratio: float = 0.0
    exclamation_count: int = 0
    question_count: int = 0
    has_author: bool = False
    has_publish_date: bool = False
    url_structure_quality: float = 0.0
    image_count: int = 0
    external_links_count: int = 0

    def __post_init__(self):
        # Validate ranges
        if not -1.0 <= self.sentiment_score <= 1.0:
            raise ValueError("Sentiment score must be between -1.0 and 1.0")
        if not 0.0 <= self.caps_ratio <= 1.0:
            raise ValueError("Caps ratio must be between 0.0 and 1.0")
        if not 0.0 <= self.url_structure_quality <= 1.0:
            raise ValueError("URL structure quality must be between 0.0 and 1.0")


@dataclass
class AnalysisResult:
    """Complete analysis result for content"""
    id: UUID = field(default_factory=uuid4)
    url: str = ""
    title: str = ""
    content_hash: str = ""

    # Overall assessment
    credibility_score: float = 0.5
    confidence: float = 0.5
    veracity_label: VeracityLabel = VeracityLabel.UNVERIFIABLE

    # Component scores
    source_credibility: float = 0.5
    claim_verification_score: float = 0.5
    content_quality_score: float = 0.5
    metadata_quality_score: float = 0.5

    # Detailed results
    claims: List[Claim] = field(default_factory=list)
    claim_verifications: List[VerificationResult] = field(default_factory=list)
    source_analysis: Optional[SourceCredibility] = None
    content_features: Optional[ContentFeatures] = None

    # Issues identified
    misinformation_types: List[MisinformationType] = field(default_factory=list)
    specific_issues: List[str] = field(default_factory=list)

    # Metadata
    analysis_time: datetime = field(default_factory=datetime.now)
    processing_time_seconds: float = 0.0
    model_versions: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # Validate all scores
        scores = [
            self.credibility_score, self.confidence, self.source_credibility,
            self.claim_verification_score, self.content_quality_score,
            self.metadata_quality_score
        ]
        for score in scores:
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Score {score} must be between 0.0 and 1.0")


@dataclass
class Article:
    """Article metadata and content"""
    id: UUID = field(default_factory=uuid4)
    url: str = ""
    title: str = ""
    content: str = ""
    domain: str = ""
    author: Optional[str] = None
    publish_date: Optional[datetime] = None
    scraped_at: datetime = field(default_factory=datetime.now)
    content_hash: str = ""
    language: str = "en"

    # Optional metadata
    image_urls: List[str] = field(default_factory=list)
    external_links: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class Annotation:
    """Human annotation for training data"""
    id: UUID = field(default_factory=uuid4)
    article_id: UUID = field(default_factory=uuid4)
    annotator_id: str = ""
    veracity_label: VeracityLabel = VeracityLabel.UNVERIFIABLE
    confidence_score: float = 0.5
    misinformation_types: List[MisinformationType] = field(default_factory=list)
    annotated_claims: List[Claim] = field(default_factory=list)
    annotation_notes: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    time_spent_minutes: Optional[int] = None

    def __post_init__(self):
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")


# Utility functions for working with models
def create_test_claim(text: str, claim_type: ClaimType = ClaimType.GENERAL_FACTUAL) -> Claim:
    """Create a test claim with default values"""
    return Claim(
        text=text,
        claim_type=claim_type,
        confidence=0.8,
        verifiability_score=0.7
    )


def create_test_entity(text: str, label: str) -> Entity:
    """Create a test entity with default values"""
    return Entity(
        text=text,
        label=label,
        start=0,
        end=len(text),
        confidence=0.9
    )


# Constants for common use
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_VERIFIABILITY_THRESHOLD = 0.5
SUPPORTED_LANGUAGES = ["en", "es", "fr", "de"]
ENTITY_LABELS = [
    "PERSON", "ORG", "GPE", "DATE", "TIME",
    "PERCENT", "MONEY", "QUANTITY", "CARDINAL", "ORDINAL"
]