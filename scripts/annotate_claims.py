#!/usr/bin/env python3
"""
Simple command-line tool for annotating factual claims in collected articles.
Loads articles, shows sentences, and lets you label them for BERT training.

Usage:
    python scripts/annotate_claims.py --input data/raw/articles_training_*.json
    python scripts/annotate_claims.py --input data/raw/articles_training_*.json --resume
"""

import sys
import json
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import hashlib

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from truthed.data.preprocessing.text_processing import TextProcessor

    TEXT_PROCESSOR_AVAILABLE = True
except ImportError:
    print("⚠️  TextProcessor not available, using simple sentence splitting")
    TEXT_PROCESSOR_AVAILABLE = False


@dataclass
class AnnotatedSentence:
    """Single annotated sentence for training"""
    text: str
    is_claim: bool
    confidence: float  # Annotator confidence 1-5
    article_url: str
    article_title: str
    sentence_index: int
    annotation_timestamp: str
    notes: str = ""

    def __post_init__(self):
        if not self.annotation_timestamp:
            self.annotation_timestamp = datetime.now().isoformat()


@dataclass
class AnnotationSession:
    """Annotation session metadata"""
    session_id: str
    input_file: str
    output_file: str
    start_time: str
    total_articles: int
    total_sentences: int
    sentences_annotated: int = 0
    claims_found: int = 0
    session_notes: str = ""

    def __post_init__(self):
        if not self.start_time:
            self.start_time = datetime.now().isoformat()


class ClaimAnnotationTool:
    """Interactive command-line tool for claim annotation"""

    def __init__(self, input_file: str, output_file: Optional[str] = None, resume: bool = False):
        self.input_file = input_file
        self.output_file = output_file or self._generate_output_filename()
        self.resume = resume

        # Annotation state
        self.articles = []
        self.sentences = []
        self.annotations = []
        self.current_index = 0
        self.session = None

        # Stats
        self.stats = {
            'claims': 0,
            'non_claims': 0,
            'skipped': 0,
            'high_confidence': 0
        }

        # Load data
        self._load_articles()
        self._prepare_sentences()

        if resume:
            self._load_existing_annotations()

        self._create_session()

    def _generate_output_filename(self) -> str:
        """Generate output filename based on input"""
        input_path = Path(self.input_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"data/annotations/annotated_{input_path.stem}_{timestamp}.json"

    def _load_articles(self):
        """Load articles from input file"""
        print(f"📂 Loading articles from {self.input_file}")

        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'articles' in data:
                self.articles = data['articles']
            else:
                self.articles = data  # Assume direct list

            print(f"✅ Loaded {len(self.articles)} articles")

        except Exception as e:
            print(f"❌ Failed to load articles: {e}")
            sys.exit(1)

    def _prepare_sentences(self):
        """Extract and prepare sentences for annotation"""
        print(f"🔧 Preparing sentences for annotation...")

        if TEXT_PROCESSOR_AVAILABLE:
            processor = TextProcessor()

        for article_idx, article in enumerate(self.articles):
            try:
                content = article.get('content', '')
                title = article.get('title', 'Untitled')
                url = article.get('url', f'article_{article_idx}')

                if not content:
                    continue

                # Extract sentences
                if TEXT_PROCESSOR_AVAILABLE:
                    processed = processor.preprocess_article(content)
                    sentences = processed.sentences
                else:
                    sentences = self._simple_sentence_split(content)

                # Add to sentence list
                for sent_idx, sentence in enumerate(sentences):
                    if len(sentence) > 20:  # Filter very short sentences
                        self.sentences.append({
                            'text': sentence,
                            'article_url': url,
                            'article_title': title,
                            'article_index': article_idx,
                            'sentence_index': sent_idx
                        })

            except Exception as e:
                print(f"⚠️  Error processing article {article_idx}: {e}")
                continue

        print(f"✅ Prepared {len(self.sentences)} sentences from {len(self.articles)} articles")

    def _simple_sentence_split(self, text: str) -> List[str]:
        """Simple sentence splitting fallback"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Split on sentence endings
        sentences = re.split(r'[.!?]+\s+', text)

        # Clean and filter
        cleaned = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 20:
                cleaned.append(sent)

        return cleaned

    def _load_existing_annotations(self):
        """Load existing annotations if resuming"""
        if Path(self.output_file).exists():
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                self.annotations = data.get('annotations', [])

                # Find where to resume
                annotated_texts = {ann['text'] for ann in self.annotations}

                for i, sentence in enumerate(self.sentences):
                    if sentence['text'] not in annotated_texts:
                        self.current_index = i
                        break
                else:
                    self.current_index = len(self.sentences)

                print(f"📋 Resuming: {len(self.annotations)} existing annotations")
                print(f"📍 Starting at sentence {self.current_index + 1}/{len(self.sentences)}")

            except Exception as e:
                print(f"⚠️  Could not load existing annotations: {e}")

    def _create_session(self):
        """Create annotation session metadata"""
        session_id = hashlib.md5(f"{self.input_file}{datetime.now()}".encode()).hexdigest()[:8]

        self.session = AnnotationSession(
            session_id=session_id,
            input_file=self.input_file,
            output_file=self.output_file,
            start_time=datetime.now().isoformat(),
            total_articles=len(self.articles),
            total_sentences=len(self.sentences),
            sentences_annotated=len(self.annotations)
        )

    def annotate(self):
        """Main annotation loop"""
        print(f"\n🎯 CLAIM ANNOTATION SESSION")
        print("=" * 60)
        print(f"Session ID: {self.session.session_id}")
        print(f"Total sentences: {len(self.sentences)}")
        print(f"Starting at: {self.current_index + 1}")
        print(f"Output file: {self.output_file}")

        self._print_instructions()

        try:
            while self.current_index < len(self.sentences):
                self._annotate_sentence()

        except KeyboardInterrupt:
            print(f"\n\n⚠️  Annotation interrupted by user")

        finally:
            self._save_annotations()
            self._print_summary()

    def _print_instructions(self):
        """Print annotation instructions"""
        print(f"\n📋 ANNOTATION INSTRUCTIONS:")
        print("=" * 40)
        print("For each sentence, choose:")
        print("  [c] CLAIM - Factual statement that can be verified")
        print("  [n] NOT_CLAIM - Opinion, question, or non-factual")
        print("  [s] SKIP - Unclear or problematic sentence")
        print("  [q] QUIT - Save and exit")
        print("  [?] HELP - Show examples")
        print()
        print("After labeling, rate your confidence (1-5):")
        print("  1=Very unsure, 3=Somewhat sure, 5=Very confident")
        print()
        input("Press Enter to start annotation...")

    def _annotate_sentence(self):
        """Annotate a single sentence"""
        sentence_data = self.sentences[self.current_index]
        sentence_text = sentence_data['text']

        # Clear screen (simple version)
        print("\n" * 3)

        # Show progress
        progress = (self.current_index + 1) / len(self.sentences) * 100
        print(f"📊 Progress: {self.current_index + 1}/{len(self.sentences)} ({progress:.1f}%)")
        print(f"📈 Stats: {self.stats['claims']} claims, {self.stats['non_claims']} non-claims")

        # Show context
        print(f"\n📰 Article: {sentence_data['article_title'][:60]}...")
        print(f"🔗 URL: {sentence_data['article_url']}")

        # Show sentence
        print(f"\n" + "=" * 80)
        print(f"SENTENCE {self.current_index + 1}:")
        print(f"'" + sentence_text + "'")
        print("=" * 80)

        # Get annotation
        while True:
            response = input("\n[c]laim, [n]ot claim, [s]kip, [q]uit, [?]help: ").strip().lower()

            if response == 'q':
                raise KeyboardInterrupt

            elif response == '?':
                self._show_examples()
                continue

            elif response == 's':
                self.stats['skipped'] += 1
                self.current_index += 1
                return

            elif response in ['c', 'n']:
                is_claim = (response == 'c')

                # Get confidence
                while True:
                    try:
                        conf_input = input("Confidence (1-5): ").strip()
                        confidence = int(conf_input)
                        if 1 <= confidence <= 5:
                            break
                        else:
                            print("Please enter 1-5")
                    except ValueError:
                        print("Please enter a number 1-5")

                # Optional notes
                notes = input("Notes (optional): ").strip()

                # Create annotation
                annotation = AnnotatedSentence(
                    text=sentence_text,
                    is_claim=is_claim,
                    confidence=confidence,
                    article_url=sentence_data['article_url'],
                    article_title=sentence_data['article_title'],
                    sentence_index=sentence_data['sentence_index'],
                    annotation_timestamp=datetime.now().isoformat(),
                    notes=notes
                )

                self.annotations.append(annotation)

                # Update stats
                if is_claim:
                    self.stats['claims'] += 1
                else:
                    self.stats['non_claims'] += 1

                if confidence >= 4:
                    self.stats['high_confidence'] += 1

                # Auto-save every 10 annotations
                if len(self.annotations) % 10 == 0:
                    self._save_annotations()
                    print("💾 Auto-saved annotations")

                self.current_index += 1
                return

            else:
                print("Invalid choice. Use c, n, s, q, or ?")

    def _show_examples(self):
        """Show annotation examples"""
        print(f"\n📋 ANNOTATION EXAMPLES:")
        print("-" * 40)

        examples = [
            ("CLAIM", "The study found that 67% of participants showed improvement."),
            ("CLAIM", "Apple reported revenue of $365 billion in fiscal year 2021."),
            ("CLAIM", "The earthquake measured 7.2 on the Richter scale."),
            ("NOT_CLAIM", "I think this policy will be effective."),
            ("NOT_CLAIM", "What caused the economic downturn?"),
            ("NOT_CLAIM", "The weather was beautiful that day."),
        ]

        for label, example in examples:
            print(f"  {label:10} : {example}")

        print(f"\nKey question: Can this be verified with evidence?")
        input("\nPress Enter to continue...")

    def _save_annotations(self):
        """Save annotations to file"""
        # Ensure output directory exists
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Update session stats
        self.session.sentences_annotated = len(self.annotations)
        self.session.claims_found = self.stats['claims']

        # Prepare data
        data = {
            'session': asdict(self.session),
            'statistics': self.stats.copy(),
            'annotations': [asdict(ann) for ann in self.annotations],
            'metadata': {
                'annotation_tool_version': '1.0',
                'export_timestamp': datetime.now().isoformat(),
                'total_time_minutes': self._calculate_session_time()
            }
        }

        # Save to file
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"❌ Failed to save annotations: {e}")

    def _calculate_session_time(self) -> float:
        """Calculate session time in minutes"""
        try:
            start = datetime.fromisoformat(self.session.start_time)
            now = datetime.now()
            return (now - start).total_seconds() / 60
        except:
            return 0.0

    def _print_summary(self):
        """Print annotation session summary"""
        print(f"\n📊 ANNOTATION SESSION SUMMARY")
        print("=" * 50)
        print(f"Time spent: {self._calculate_session_time():.1f} minutes")
        print(f"Sentences annotated: {len(self.annotations)}")
        print(f"Claims found: {self.stats['claims']}")
        print(f"Non-claims: {self.stats['non_claims']}")
        print(f"Skipped: {self.stats['skipped']}")
        print(f"High confidence: {self.stats['high_confidence']}")

        if len(self.annotations) > 0:
            claim_rate = self.stats['claims'] / len(self.annotations) * 100
            conf_rate = self.stats['high_confidence'] / len(self.annotations) * 100
            print(f"Claim rate: {claim_rate:.1f}%")
            print(f"High confidence rate: {conf_rate:.1f}%")

        print(f"\n💾 Annotations saved to: {self.output_file}")

        # Next steps
        if len(self.annotations) >= 50:
            print(f"\n🎉 Great progress! You can start training with {len(self.annotations)} annotations.")
            print(f"💡 Recommended: Continue to 200-500 annotations for better model performance.")
        else:
            print(f"\n📋 Continue annotating! Aim for at least 50-100 annotations to start training.")

        print(f"\nTo resume: python scripts/annotate_claims.py --input {self.input_file} --resume")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Annotate factual claims in collected articles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input data/raw/articles_training_*.json
  %(prog)s --input data/raw/articles_training_*.json --resume
  %(prog)s --input articles.json --output my_annotations.json
        """
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input file with collected articles (JSON)"
    )

    parser.add_argument(
        "--output", "-o",
        help="Output file for annotations (default: auto-generated)"
    )

    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume existing annotation session"
    )

    args = parser.parse_args()

    # Validate input file
    if not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        return 1

    try:
        # Create annotation tool
        tool = ClaimAnnotationTool(
            input_file=args.input,
            output_file=args.output,
            resume=args.resume
        )

        # Run annotation
        tool.annotate()

        return 0

    except Exception as e:
        print(f"❌ Annotation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())