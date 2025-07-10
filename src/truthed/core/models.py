"""
Core data models and schemas for the truthed system.
"""
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
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


class Entity(BaseModel):
    """Named entity extracted from text"""
    text: str
    label: str  # PERSON, ORG, GPE, DATE, etc.
    start: int
    end: int
    confidence: float = Field(ge=0.0, le=1.0)


class Claim(BaseModel):
    """Structured representation of a factual claim"""
    id: UUID = Field(default_factory=uuid4)
    text: str
    claim_type: ClaimType
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    entities: List[Entity] = Field(default_factory=list)
    temporal_context: Optional[str] = None
    verifiability_score: float = Field(ge=0.0, le=1.0)
    is_verifiable: bool
    confidence: float = Field(ge=0.0, le=1.0)
    extracted_at: datetime = Field(default_factory=datetime.now)


class VerificationResult(BaseModel):
    """Result of fact verification for a claim"""
    claim_id: UUID
    label: VeracityLabel
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    contradicting_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    sources_checked: int
    verification_time: datetime


class SourceCredibility(BaseModel):
    """Source credibility analysis result"""
    domain: str
    credibility_score: float = Field(ge=0.0, le=1.0)
    bias_rating: Optional[str] = None
    factual_reporting: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str
    last_updated: datetime


class ContentFeatures(BaseModel):
    """Extracted features from content"""
    word_count: int
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    readability_score: float
    caps_ratio: float = Field(ge=0.0, le=1.0)
    exclamation_count: int
    question_count: int
    has_author: bool
    has_publish_date: bool
    url_structure_quality: float = Field(ge=0.0, le=1.0)
    image_count: int
    external_links_count: int


class AnalysisResult(BaseModel):
    """Complete analysis result for content"""
    id: UUID = Field(default_factory=uuid4)
    url: str
    title: str
    content_hash: str

    # Overall assessment
    credibility_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    veracity_label: VeracityLabel

    # Component scores
    source_credibility: float = Field(ge=0.0, le=1.0)
    claim_verification_score: float = Field(ge=0.0, le=1.0)
    content_quality_score: float = Field(ge=0.0, le=1.0)
    metadata_quality_score: float = Field(ge=0.0, le=1.0)

    # Detailed results
    claims: List[Claim] = Field(default_factory=list)
    claim_verifications: List[VerificationResult] = Field(default_factory=list)
    source_analysis: Optional[SourceCredibility] = None
    content_features: Optional[ContentFeatures] = None

    # Issues identified
    misinformation_types: List[MisinformationType] = Field(default_factory=list)
    specific_issues: List[str] = Field(default_factory=list)

    # Metadata
    analysis_time: datetime = Field(default_factory=datetime.now)
    processing_time_seconds: float
    model_versions: Dict[str, str] = Field(default_factory=dict)

    @validator("credibility_score", "confidence")
    def validate_scores(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Score must be between 0.0 and 1.0")
        return v


class Article(BaseModel):
    """Article metadata and content"""
    id: UUID = Field(default_factory=uuid4)
    url: str
    title: str
    content: str
    domain: str
    author: Optional[str] = None
    publish_date: Optional[datetime] = None
    scraped_at: datetime = Field(default_factory=datetime.now)
    content_hash: str
    language: str = "en"

    # Optional metadata
    image_urls: List[str] = Field(default_factory=list)
    external_links: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class Annotation(BaseModel):
    """Human annotation for training data"""
    id: UUID = Field(default_factory=uuid4)
    article_id: UUID
    annotator_id: str
    veracity_label: VeracityLabel
    confidence_score: float = Field(ge=0.0, le=1.0)
    misinformation_types: List[MisinformationType] = Field(default_factory=list)
    annotated_claims: List[Claim] = Field(default_factory=list)
    annotation_notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    time_spent_minutes: Optional[int] = None