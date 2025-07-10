"""
Text preprocessing utilities for the truthed system.
Handles cleaning, normalization, and sentence segmentation.
"""

import re
from typing import List, Optional
from dataclasses import dataclass
from bs4 import BeautifulSoup
import spacy
from spacy.lang.en import English


@dataclass
class ProcessedText:
    """Container for processed text data"""
    original_text: str
    cleaned_text: str
    sentences: List[str]
    paragraphs: List[str]
    word_count: int
    char_count: int


class TextProcessor:
    """Handles text cleaning and preprocessing for claim extraction"""

    def __init__(self):
        # Use English tokenizer for sentence segmentation
        self.nlp = English()
        self.nlp.add_pipe("sentencizer")

    def preprocess_article(self, raw_text: str) -> ProcessedText:
        """
        Clean and preprocess raw article text.

        Args:
            raw_text: Raw text that may contain HTML, extra whitespace, etc.

        Returns:
            ProcessedText object with cleaned text and metadata
        """
        # Step 1: Clean HTML and normalize
        cleaned_text = self._clean_html(raw_text)
        cleaned_text = self._normalize_text(cleaned_text)

        # Step 2: Extract sentences and paragraphs
        sentences = self._extract_sentences(cleaned_text)
        paragraphs = self._extract_paragraphs(cleaned_text)

        # Step 3: Calculate metadata
        word_count = len(cleaned_text.split())
        char_count = len(cleaned_text)

        return ProcessedText(
            original_text=raw_text,
            cleaned_text=cleaned_text,
            sentences=sentences,
            paragraphs=paragraphs,
            word_count=word_count,
            char_count=char_count
        )

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags and decode entities"""
        if not text:
            return ""

        # Parse HTML and extract text
        soup = BeautifulSoup(text, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text content
        cleaned = soup.get_text()
        return cleaned

    def _normalize_text(self, text: str) -> str:
        """Normalize whitespace and basic text cleaning"""
        if not text:
            return ""

        # Fix common encoding issues
        text = text.replace('\u2019', "'")  # Right single quotation mark
        text = text.replace('\u201c', '"')  # Left double quotation mark
        text = text.replace('\u201d', '"')  # Right double quotation mark
        text = text.replace('\u2013', '-')  # En dash
        text = text.replace('\u2014', '-')  # Em dash

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n+', '\n', text)  # Multiple newlines to single

        # Remove extra whitespace
        text = text.strip()

        return text

    def _extract_sentences(self, text: str) -> List[str]:
        """Extract individual sentences from text"""
        if not text:
            return []

        # Use spaCy for sentence segmentation
        doc = self.nlp(text)
        sentences = []

        for sent in doc.sents:
            sentence_text = sent.text.strip()

            # Filter out very short sentences (likely not claims)
            if len(sentence_text) > 10:
                sentences.append(sentence_text)

        return sentences

    def _extract_paragraphs(self, text: str) -> List[str]:
        """Extract paragraphs from text"""
        if not text:
            return []

        # Split on double newlines
        paragraphs = re.split(r'\n\s*\n', text)

        # Clean and filter paragraphs
        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if len(para) > 50:  # Filter out very short paragraphs
                cleaned_paragraphs.append(para)

        return cleaned_paragraphs

    def is_likely_claim_sentence(self, sentence: str) -> bool:
        """
        Quick heuristic check if sentence might contain a factual claim.
        This is a fast pre-filter before running the ML model.
        """
        if len(sentence) < 15:
            return False

        # Sentences with numbers often contain factual claims
        if re.search(r'\d+', sentence):
            return True

        # Sentences with named entities (capitalized words)
        if re.search(r'\b[A-Z][a-z]+\b', sentence):
            return True

        # Sentences with temporal indicators
        temporal_indicators = ['yesterday', 'today', 'last year', 'in 2023', 'recently']
        if any(indicator in sentence.lower() for indicator in temporal_indicators):
            return True

        # Sentences with verbs that often introduce claims
        claim_verbs = ['says', 'reports', 'shows', 'proves', 'indicates', 'reveals']
        if any(verb in sentence.lower() for verb in claim_verbs):
            return True

        return True  # For now, let most sentences through


if __name__ == "__main__":
    TextProcessor()
