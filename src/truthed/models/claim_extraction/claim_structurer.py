"""
Claim structuring and entity extraction for identified claims
File: src/truthed/models/claim_extraction/claim_structurer.py
"""

import spacy
from spacy.lang.en import English
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import re
from datetime import datetime
from uuid import uuid4

try:
    from truthed.core.models import Claim, Entity, ClaimType
except ImportError:
    # Fallback definitions
    from dataclasses import dataclass
    from enum import Enum
    from uuid import UUID
    from datetime import datetime


    class ClaimType(str, Enum):
        STATISTICAL = "statistical"
        CAUSAL = "causal"
        TEMPORAL = "temporal"
        IDENTITY = "identity"
        EXISTENTIAL = "existential"
        GENERAL_FACTUAL = "general_factual"


    @dataclass
    class Entity:
        text: str
        label: str
        start: int
        end: int
        confidence: float = 1.0


    @dataclass
    class Claim:
        id: UUID
        text: str
        claim_type: ClaimType
        subject: Optional[str] = None
        predicate: Optional[str] = None
        object: Optional[str] = None
        entities: List[Entity] = None
        temporal_context: Optional[str] = None
        verifiability_score: float = 0.0
        is_verifiable: bool = True
        confidence: float = 0.0
        extracted_at: datetime = None

        def __post_init__(self):
            if self.entities is None:
                self.entities = []
            if self.extracted_at is None:
                self.extracted_at = datetime.now()


@dataclass
class StructuredClaim:
    """Enhanced claim structure with linguistic analysis"""
    claim: Claim
    linguistic_features: Dict[str, Any]
    extraction_metadata: Dict[str, Any]


class ClaimStructurer:
    """
    Structures identified claims by extracting entities, relationships,
    and linguistic features for fact verification.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the claim structurer with spaCy model"""
        self.model_name = model_name
        self.nlp = None
        self.is_loaded = False

        print(f"🔧 ClaimStructurer initializing...")
        self._load_model()

    def _load_model(self):
        """Load spaCy model for NLP processing"""
        try:
            # Try to load the large model first, fall back to smaller ones
            model_preferences = ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"]

            for model in model_preferences:
                try:
                    self.nlp = spacy.load(model)
                    self.model_name = model
                    print(f"✅ Loaded spaCy model: {model}")
                    self.is_loaded = True
                    return
                except OSError:
                    continue

            # If no full model available, use basic English
            print("⚠️  No full spaCy model found, using basic English tokenizer")
            self.nlp = English()
            self.nlp.add_pipe("sentencizer")
            self.is_loaded = True

        except Exception as e:
            print(f"❌ Failed to load any spaCy model: {e}")
            self.is_loaded = False

    def structure_claim(self, claim_text: str, claim_type: ClaimType,
                        confidence: float) -> StructuredClaim:
        """
        Structure a claim by extracting entities, relationships, and features.

        Args:
            claim_text: The text of the claim
            claim_type: The type of claim (statistical, causal, etc.)
            confidence: Confidence score from claim classifier

        Returns:
            StructuredClaim with full linguistic analysis
        """
        if not self.is_loaded:
            # Create minimal structure without NLP
            return self._create_minimal_structure(claim_text, claim_type, confidence)

        # Process with spaCy
        doc = self.nlp(claim_text)

        # Extract components
        entities = self._extract_entities(doc)
        subject, predicate, obj = self._extract_spo_triplet(doc)
        temporal_context = self._extract_temporal_context(doc)
        verifiability_score = self._assess_verifiability(doc, claim_type)

        # Create claim object
        claim = Claim(
            id=uuid4(),
            text=claim_text,
            claim_type=claim_type,
            subject=subject,
            predicate=predicate,
            object=obj,
            entities=entities,
            temporal_context=temporal_context,
            verifiability_score=verifiability_score,
            is_verifiable=verifiability_score > 0.5,
            confidence=confidence,
            extracted_at=datetime.now()
        )

        # Extract linguistic features
        linguistic_features = self._extract_linguistic_features(doc)

        # Create extraction metadata
        extraction_metadata = {
            'spacy_model': self.model_name,
            'processing_time': datetime.now(),
            'entities_found': len(entities),
            'has_temporal_context': temporal_context is not None,
            'sentence_length': len(claim_text),
            'word_count': len([token for token in doc if not token.is_space])
        }

        return StructuredClaim(
            claim=claim,
            linguistic_features=linguistic_features,
            extraction_metadata=extraction_metadata
        )

    def _create_minimal_structure(self, claim_text: str, claim_type: ClaimType,
                                  confidence: float) -> StructuredClaim:
        """Create minimal structure when spaCy is not available"""
        claim = Claim(
            id=uuid4(),
            text=claim_text,
            claim_type=claim_type,
            subject=None,
            predicate=None,
            object=None,
            entities=[],
            temporal_context=self._extract_temporal_regex(claim_text),
            verifiability_score=0.5,
            is_verifiable=True,
            confidence=confidence,
            extracted_at=datetime.now()
        )

        linguistic_features = {
            'token_count': len(claim_text.split()),
            'char_count': len(claim_text),
            'has_numbers': bool(re.search(r'\d', claim_text)),
            'has_uppercase': bool(re.search(r'[A-Z]', claim_text))
        }

        extraction_metadata = {
            'spacy_model': 'minimal_fallback',
            'processing_time': datetime.now(),
            'entities_found': 0,
            'has_temporal_context': linguistic_features.get('temporal_context') is not None,
            'sentence_length': len(claim_text),
            'word_count': len(claim_text.split())
        }

        return StructuredClaim(
            claim=claim,
            linguistic_features=linguistic_features,
            extraction_metadata=extraction_metadata
        )

    def _extract_entities(self, doc) -> List[Entity]:
        """Extract named entities from the claim"""
        entities = []

        for ent in doc.ents:
            # Filter out low-confidence or irrelevant entities
            if len(ent.text.strip()) > 1 and ent.label_ in [
                'PERSON', 'ORG', 'GPE', 'DATE', 'TIME', 'PERCENT',
                'MONEY', 'QUANTITY', 'CARDINAL', 'ORDINAL'
            ]:
                entity = Entity(
                    text=ent.text.strip(),
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.9  # spaCy doesn't provide confidence, use default
                )
                entities.append(entity)

        return entities

    def _extract_spo_triplet(self, doc) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Extract Subject-Predicate-Object triplet from claim.
        Uses dependency parsing to identify grammatical relationships.
        """
        # Find the root verb (main predicate)
        root_token = None
        for token in doc:
            if token.dep_ == "ROOT":
                root_token = token
                break

        if not root_token or root_token.pos_ not in ['VERB', 'AUX']:
            return self._extract_spo_heuristic(doc)

        subject = None
        predicate = root_token.lemma_
        obj = None

        # Find subject
        for child in root_token.children:
            if child.dep_ in ["nsubj", "nsubjpass", "csubj"]:
                subject = self._get_phrase(child)
                break

        # Find object
        for child in root_token.children:
            if child.dep_ in ["dobj", "pobj", "attr", "acomp"]:
                obj = self._get_phrase(child)
                break

        # Enhance predicate with auxiliary verbs and particles
        predicate_parts = [root_token.text]
        for child in root_token.children:
            if child.dep_ in ["aux", "auxpass", "prt"]:
                predicate_parts.append(child.text)

        predicate = " ".join(sorted(predicate_parts, key=lambda x: doc.text.find(x)))

        return subject, predicate, obj

    def _extract_spo_heuristic(self, doc) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Fallback heuristic extraction when dependency parsing fails"""
        tokens = [token for token in doc if not token.is_space and not token.is_punct]

        if len(tokens) < 3:
            return None, None, None

        # Find main verb
        main_verb = None
        for token in tokens:
            if token.pos_ == 'VERB':
                main_verb = token
                break

        if not main_verb:
            return None, None, None

        # Simple position-based extraction
        subject_tokens = []
        object_tokens = []

        for token in tokens:
            if token.i < main_verb.i and token.pos_ in ['NOUN', 'PROPN', 'PRON']:
                subject_tokens.append(token.text)
            elif token.i > main_verb.i and token.pos_ in ['NOUN', 'PROPN', 'NUM']:
                object_tokens.append(token.text)

        subject = ' '.join(subject_tokens[-3:]) if subject_tokens else None  # Last 3 tokens before verb
        predicate = main_verb.text
        obj = ' '.join(object_tokens[:3]) if object_tokens else None  # First 3 tokens after verb

        return subject, predicate, obj

    def _get_phrase(self, token) -> str:
        """Extract the full phrase for a token including its dependencies"""
        phrase_tokens = [token]

        # Add determiners, adjectives, compounds, etc.
        for child in token.children:
            if child.dep_ in ["det", "amod", "compound", "nummod", "prep"]:
                phrase_tokens.append(child)
                # Recursively add children of prepositions
                if child.dep_ == "prep":
                    for grandchild in child.children:
                        phrase_tokens.append(grandchild)

        # Sort by position in text
        phrase_tokens.sort(key=lambda x: x.i)
        return " ".join([t.text for t in phrase_tokens])

    def _extract_temporal_context(self, doc) -> Optional[str]:
        """Extract temporal context from the claim using NER and patterns"""
        temporal_entities = []

        # Extract DATE and TIME entities
        for ent in doc.ents:
            if ent.label_ in ['DATE', 'TIME']:
                temporal_entities.append(ent.text)

        # Look for temporal patterns not caught by NER
        text_lower = doc.text.lower()
        temporal_patterns = [
            (r'\b\d{4}\b', 'year'),  # Years
            (r'\b(last|this|next)\s+(year|month|week|day|decade)\b', 'relative_time'),
            (r'\b(yesterday|today|tomorrow)\b', 'relative_day'),
            (r'\bsince\s+\d{4}\b', 'since_year'),
            (r'\bin\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b',
             'month'),
            (r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', 'weekday'),
            (r'\b(recently|currently|now|presently)\b', 'present_time')
        ]

        for pattern, label in temporal_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                temporal_entities.extend(
                    [f"{match} ({label})" if isinstance(match, str) else f"{' '.join(match)} ({label})" for match in
                     matches])

        if temporal_entities:
            return '; '.join(set(temporal_entities))

        return None

    def _extract_temporal_regex(self, text: str) -> Optional[str]:
        """Regex-based temporal extraction for fallback mode"""
        temporal_patterns = [
            r'\b\d{4}\b',  # Years
            r'\b(last|this|next)\s+(year|month|week|day)\b',
            r'\b(yesterday|today|tomorrow)\b',
            r'\bsince\s+\d{4}\b',
            r'\bin\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b'
        ]

        found_temporal = []
        text_lower = text.lower()

        for pattern in temporal_patterns:
            matches = re.findall(pattern, text_lower)
            found_temporal.extend(matches)

        return '; '.join(set(found_temporal)) if found_temporal else None

    def _assess_verifiability(self, doc, claim_type: ClaimType) -> float:
        """Assess how verifiable a claim is based on linguistic features"""
        score = 0.5  # Base score

        # Presence of specific entities increases verifiability
        entity_labels = [ent.label_ for ent in doc.ents]

        if 'ORG' in entity_labels:  # Organizations
            score += 0.15
        if 'PERSON' in entity_labels:  # People
            score += 0.1
        if 'GPE' in entity_labels:  # Geopolitical entities
            score += 0.1
        if 'DATE' in entity_labels or 'TIME' in entity_labels:  # Temporal specificity
            score += 0.15
        if 'PERCENT' in entity_labels or 'MONEY' in entity_labels:  # Quantitative data
            score += 0.2

        # Claim type affects verifiability
        type_multipliers = {
            ClaimType.STATISTICAL: 1.2,
            ClaimType.TEMPORAL: 1.1,
            ClaimType.IDENTITY: 1.0,
            ClaimType.CAUSAL: 0.9,
            ClaimType.EXISTENTIAL: 0.95,
            ClaimType.GENERAL_FACTUAL: 1.0
        }

        score *= type_multipliers.get(claim_type, 1.0)

        # Check for verifiability indicators in text
        text_lower = doc.text.lower()

        # Positive indicators
        if re.search(r'study|research|data|according to|published|reported', text_lower):
            score += 0.1

        if re.search(r'university|institute|organization|government', text_lower):
            score += 0.05

        # Negative indicators
        if re.search(r'might|could|probably|allegedly|rumored|claims|believes', text_lower):
            score -= 0.2

        if re.search(r'secret|classified|unconfirmed', text_lower):
            score -= 0.3

        return max(0.0, min(1.0, score))

    def _extract_linguistic_features(self, doc) -> Dict[str, Any]:
        """Extract linguistic features for analysis"""
        tokens = [token for token in doc if not token.is_space]

        features = {
            'token_count': len(tokens),
            'char_count': len(doc.text),
            'entity_count': len(doc.ents),
            'entity_types': [ent.label_ for ent in doc.ents],
            'pos_tags': [token.pos_ for token in tokens],
            'dependency_labels': [token.dep_ for token in tokens],
            'has_numbers': any(token.like_num for token in tokens),
            'has_proper_nouns': any(token.pos_ == 'PROPN' for token in tokens),
            'sentence_type': self._classify_sentence_type(doc),
            'complexity_score': self._calculate_complexity(doc),
            'named_entity_density': len(doc.ents) / max(1, len(tokens)),
            'average_word_length': sum(len(token.text) for token in tokens if token.is_alpha) / max(1, sum(
                1 for token in tokens if token.is_alpha))
        }

        return features

    def _classify_sentence_type(self, doc) -> str:
        """Classify the grammatical type of the sentence"""
        text = doc.text.strip()

        if text.endswith('?'):
            return 'interrogative'
        elif text.endswith('!'):
            return 'exclamatory'
        elif any(token.pos_ == 'VERB' and token.dep_ == 'ROOT' for token in doc):
            return 'declarative'
        else:
            return 'fragment'

    def _calculate_complexity(self, doc) -> float:
        """Calculate syntactic complexity score"""
        tokens = [token for token in doc if not token.is_space and not token.is_punct]

        if not tokens:
            return 0.0

        # Factors that increase complexity
        complexity_factors = 0

        # Subordinate clauses
        complexity_factors += sum(1 for token in tokens if token.dep_ in ['ccomp', 'xcomp', 'advcl', 'acl'])

        # Long dependency distances
        avg_dep_distance = sum(abs(token.i - token.head.i) for token in tokens if token.head != token) / len(tokens)
        complexity_factors += avg_dep_distance / 10  # Normalize

        # Nested structures
        max_depth = max((self._get_dependency_depth(token) for token in tokens), default=0)
        complexity_factors += max_depth / 5  # Normalize

        # Normalize to 0-1 scale
        return min(1.0, complexity_factors / 3)

    def _get_dependency_depth(self, token) -> int:
        """Get the depth of a token in the dependency tree"""
        depth = 0
        current = token

        while current.head != current and depth < 20:  # Prevent infinite loops
            depth += 1
            current = current.head

        return depth

    def structure_batch(self, claim_predictions: List) -> List[StructuredClaim]:
        """Structure a batch of claim predictions"""
        structured_claims = []

        print(f"🔧 Structuring {len(claim_predictions)} claims...")

        for i, prediction in enumerate(claim_predictions, 1):
            if hasattr(prediction, 'is_claim') and prediction.is_claim:
                structured = self.structure_claim(
                    prediction.sentence,
                    prediction.claim_type or ClaimType.GENERAL_FACTUAL,
                    prediction.confidence
                )
                structured_claims.append(structured)

            if len(claim_predictions) > 10 and i % 5 == 0:
                print(f"   Structured {i}/{len(claim_predictions)} claims...")

        print(f"✅ Structuring complete: {len(structured_claims)} claims structured")
        return structured_claims


# Test function
def test_claim_structurer():
    """Test the claim structurer with sample claims"""
    print("🧪 Testing ClaimStructurer")
    print("=" * 60)

    structurer = ClaimStructurer()

    test_claims = [
        ("Global temperatures have risen by 1.1°C since 1880.", ClaimType.STATISTICAL, 0.9),
        ("Stanford University published the research yesterday.", ClaimType.TEMPORAL, 0.85),
        ("Climate change causes extreme weather events.", ClaimType.CAUSAL, 0.8),
        ("Water was discovered on Mars by NASA.", ClaimType.EXISTENTIAL, 0.87),
        ("Joe Biden is the President of the United States.", ClaimType.IDENTITY, 0.95)
    ]

    print(f"📊 Structuring {len(test_claims)} test claims...\n")

    for i, (claim_text, claim_type, confidence) in enumerate(test_claims, 1):
        print(f"{i}. Structuring: \"{claim_text}\"")

        structured = structurer.structure_claim(claim_text, claim_type, confidence)
        claim = structured.claim

        print(f"   🎯 Type: {claim.claim_type.value}")
        print(f"   📝 Subject: {claim.subject or 'N/A'}")
        print(f"   ⚡ Predicate: {claim.predicate or 'N/A'}")
        print(f"   🎯 Object: {claim.object or 'N/A'}")
        print(f"   🏷️  Entities: {len(claim.entities)} found")

        if claim.entities:
            for entity in claim.entities[:3]:  # Show first 3
                print(f"      - {entity.text} ({entity.label_})")

        print(f"   ⏰ Temporal: {claim.temporal_context or 'None'}")
        print(f"   ✅ Verifiable: {claim.is_verifiable} (score: {claim.verifiability_score:.2f})")
        print(f"   📊 Features: {structured.linguistic_features.get('token_count', 0)} tokens, "
              f"{structured.linguistic_features.get('entity_count', 0)} entities")
        print()

    print("📈 STRUCTURING SUMMARY")
    print("-" * 40)
    print(f"Claims processed: {len(test_claims)}")
    print(
        f"Average verifiability: {sum(structured.claim.verifiability_score for _, _, _ in test_claims) / len(test_claims):.2f}")
    print("✅ Claim structuring test complete!")


if __name__ == "__main__":
    test_claim_structurer()