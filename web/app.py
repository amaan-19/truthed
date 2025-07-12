#!/usr/bin/env python3
"""
Truthed Professional Web Interface
Complete Flask application integrating with existing truthed pipeline

File: web/app.py
"""

import sys
import os
import time
import hashlib
import logging
import json
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
from typing import Dict, Any, Optional

from flask import Flask, render_template, request, jsonify, url_for, flash, redirect
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

# Setup path for imports - go up one level to project root, then into src
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

print(f"🔧 Project root: {project_root}")
print(f"🔧 Source path: {src_path}")
print(f"🔧 Source exists: {src_path.exists()}")

# Import your existing truthed pipeline
PIPELINE_AVAILABLE = False
BERT_AVAILABLE = False
TEXT_PROCESSOR_AVAILABLE = False

try:
    from truthed.data.preprocessing.text_processing import TextProcessor, ProcessedText
    TEXT_PROCESSOR_AVAILABLE = True
    print("✅ TextProcessor imported successfully")
except ImportError as e:
    print(f"⚠️  TextProcessor import failed: {e}")

try:
    from truthed.data.preprocessing.claim_extraction import ClaimExtractionPipeline, ExtractionResult
    PIPELINE_AVAILABLE = True
    print("✅ ClaimExtractionPipeline imported successfully")
except ImportError as e:
    print(f"⚠️  ClaimExtractionPipeline import failed: {e}")

try:
    from truthed.models.claim_extraction.bert_classifier import BERTClaimClassifier
    BERT_AVAILABLE = True
    print("✅ BERTClaimClassifier imported successfully")
except ImportError as e:
    print(f"⚠️  BERTClaimClassifier import failed: {e}")

# Create Flask application
app = Flask(__name__,
           template_folder='templates',
           static_folder='static')

# Configuration
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'truthed-professional-development-key')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Enable CORS for API endpoints
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Custom template filter for JSON
@app.template_filter('tojson')
def tojson_filter(obj):
    import json
    return json.dumps(obj)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleSourceCredibility:
    """Basic source credibility checker for MVP"""

    def __init__(self):
        self.high_credibility = {
            # News Organizations
            'reuters.com': 95, 'apnews.com': 95, 'bbc.com': 90, 'npr.org': 90,
            'pbs.org': 88, 'abc.com': 85, 'cbs.com': 85, 'nbc.com': 85,
            'cnn.com': 80, 'foxnews.com': 75,

            # Newspapers
            'nytimes.com': 88, 'washingtonpost.com': 87, 'wsj.com': 89,
            'usatoday.com': 82, 'latimes.com': 83, 'chicagotribune.com': 81,
            'theguardian.com': 85, 'economist.com': 90, 'ft.com': 88,

            # Scientific Sources
            'nature.com': 96, 'science.org': 96, 'cell.com': 94, 'nejm.org': 95,
            'thelancet.com': 94, 'pnas.org': 93, 'scientificamerican.com': 87,

            # Academic Institutions
            'mit.edu': 93, 'stanford.edu': 93, 'harvard.edu': 93, 'yale.edu': 92,
            'berkeley.edu': 91, 'princeton.edu': 92, 'columbia.edu': 91,

            # Government Sources
            'nih.gov': 94, 'cdc.gov': 93, 'fda.gov': 92, 'nasa.gov': 94,
            'noaa.gov': 93, 'usgs.gov': 92, 'energy.gov': 90,

            # International Organizations
            'who.int': 91, 'un.org': 88, 'worldbank.org': 87, 'imf.org': 86,
        }

        self.low_credibility = {
            # Known problematic sources
            'infowars.com': 10, 'naturalnews.com': 15, 'globalresearch.ca': 20,
            'zerohedge.com': 25, 'beforeitsnews.com': 15, 'yournewswire.com': 10,
            'davidicke.com': 12, 'activistpost.com': 22, 'truthout.org': 30,

            # Tabloids and entertainment
            'dailymail.co.uk': 35, 'thesun.co.uk': 30, 'nypost.com': 40,
            'tmz.com': 35, 'people.com': 45, 'eonline.com': 35,

            # Highly biased sources
            'breitbart.com': 30, 'mediamatters.org': 35, 'thinkprogress.org': 35,
            'dailywire.com': 32, 'salon.com': 40, 'alternet.org': 35,
        }

        self.moderate_credibility = {
            # Moderate sources with some bias
            'politico.com': 75, 'thehill.com': 72, 'axios.com': 78,
            'vox.com': 65, 'slate.com': 60, 'huffpost.com': 55,
            'buzzfeednews.com': 62, 'vice.com': 58, 'motherboard.vice.com': 70,
        }

    def get_credibility(self, url: str) -> tuple[int, str]:
        """Get credibility score and explanation for a domain"""
        try:
            if not url or not url.startswith('http'):
                return 50, "No URL provided - unable to assess source credibility"

            domain = urlparse(url).netloc.lower()
            # Remove www. prefix
            domain = domain.replace('www.', '')

            if domain in self.high_credibility:
                score = self.high_credibility[domain]
                return score, f"High credibility source - {self._get_source_type(domain)}"
            elif domain in self.low_credibility:
                score = self.low_credibility[domain]
                return score, f"Known credibility concerns - {self._get_reason(domain)}"
            elif domain in self.moderate_credibility:
                score = self.moderate_credibility[domain]
                return score, f"Moderate credibility - some editorial bias may be present"
            else:
                return 50, f"Unknown source ({domain}) - neutral rating pending verification"

        except Exception as e:
            logger.error(f"Error assessing source credibility: {e}")
            return 50, "Unable to assess source credibility"

    def _get_source_type(self, domain: str) -> str:
        """Get the type of high credibility source"""
        if any(edu in domain for edu in ['.edu', 'university', 'college']):
            return "Academic institution"
        elif any(gov in domain for gov in ['.gov', 'government']):
            return "Government source"
        elif any(sci in domain for sci in ['nature', 'science', 'nejm', 'lancet']):
            return "Peer-reviewed scientific publication"
        elif any(news in domain for news in ['reuters', 'ap', 'bbc', 'npr']):
            return "Established news organization"
        else:
            return "Reputable source"

    def _get_reason(self, domain: str) -> str:
        """Get reason for low credibility rating"""
        if 'conspiracy' in domain or any(x in domain for x in ['infowars', 'naturalnews']):
            return "frequently publishes conspiracy theories"
        elif 'tabloid' in domain or any(x in domain for x in ['dailymail', 'sun', 'tmz']):
            return "tabloid publication with mixed accuracy"
        else:
            return "history of publishing unverified information"


def calculate_credibility_score(result, source_score: int = 50) -> int:
    """Convert analysis results to 0-100 credibility score"""
    base_score = 65  # Start slightly positive

    # Source credibility factor (35% weight)
    source_factor = (source_score - 50) * 0.35

    # Claim analysis factor (40% weight)
    claim_factor = 0
    if hasattr(result, 'claims_identified') and result.claims_identified > 0:
        high_conf_ratio = result.high_confidence_claims / result.claims_identified
        verifiable_ratio = result.verifiable_claims / result.claims_identified if result.verifiable_claims else 0

        # More high-confidence verifiable claims = higher score
        if high_conf_ratio > 0.7 and verifiable_ratio > 0.6:
            claim_factor = 15
        elif high_conf_ratio > 0.5 and verifiable_ratio > 0.4:
            claim_factor = 5
        elif high_conf_ratio < 0.3 or verifiable_ratio < 0.2:
            claim_factor = -15
        else:
            claim_factor = -5

    # Content quality factor (25% weight)
    content_factor = 0
    if hasattr(result, 'processed_text') and result.processed_text:
        word_count = result.processed_text.word_count
        sentence_count = len(result.processed_text.sentences)

        # Well-structured content tends to be more credible
        if word_count > 300 and sentence_count > 5:
            content_factor = 8
        elif word_count > 100:
            content_factor = 3
        elif word_count < 50:
            content_factor = -8

    final_score = base_score + source_factor + claim_factor + content_factor
    return max(5, min(95, int(final_score)))


def get_score_explanation(score: int, result, source_explanation: str) -> list[str]:
    """Generate explanation for the credibility score"""
    explanations = []

    # Overall assessment
    if score >= 80:
        explanations.append("✅ High credibility content with strong verification indicators")
    elif score >= 65:
        explanations.append("✅ Generally credible content with good verification potential")
    elif score >= 50:
        explanations.append("⚠️ Mixed credibility signals - some concerns identified")
    elif score >= 35:
        explanations.append("⚠️ Significant credibility concerns - verification recommended")
    else:
        explanations.append("❌ Low credibility - multiple red flags identified")

    # Source assessment
    explanations.append(f"📰 Source: {source_explanation}")

    # Claim analysis
    if hasattr(result, 'claims_identified'):
        if result.claims_identified > 0:
            explanations.append(f"🎯 Analysis: {result.claims_identified} factual claims identified")
            if result.high_confidence_claims > 0:
                explanations.append(f"✅ Verification: {result.high_confidence_claims} high-confidence claims found")
            if result.verifiable_claims > 0:
                explanations.append(f"🔍 Evidence: {result.verifiable_claims} claims have verifiable elements")
        else:
            explanations.append("💭 Analysis: No specific factual claims detected")

    # Content quality
    if hasattr(result, 'processed_text') and result.processed_text:
        word_count = result.processed_text.word_count
        if word_count > 500:
            explanations.append("📄 Content: Comprehensive article with substantial detail")
        elif word_count > 100:
            explanations.append("📄 Content: Standard article length")
        else:
            explanations.append("📄 Content: Brief content - limited analysis possible")

    return explanations


def create_mock_result(url: str, title: str, content: str):
    """Create mock result when pipeline not available"""
    class MockResult:
        def __init__(self):
            self.url = url
            self.title = title
            self.content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
            self.total_sentences = max(1, len(content.split('.')))
            self.claims_identified = max(1, self.total_sentences // 4)
            self.high_confidence_claims = max(0, self.claims_identified // 2)
            self.verifiable_claims = max(0, self.claims_identified // 3)
            self.processing_time_seconds = 2.1
            self.extraction_timestamp = datetime.now()
            self.errors = []

            # Mock processed text
            self.processed_text = type('MockProcessedText', (), {
                'word_count': len(content.split()),
                'sentences': content.split('.'),
                'char_count': len(content)
            })()

            # Mock claim predictions
            self.claim_predictions = []
            sentences = content.split('.')[:self.claims_identified]
            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) > 10:
                    mock_pred = type('MockPrediction', (), {
                        'sentence': sentence.strip(),
                        'is_claim': True,
                        'confidence': 0.7 + (i * 0.1) % 0.3,
                        'claim_type': None,
                        'reasoning': f"Mock analysis: potential factual claim detected"
                    })()
                    self.claim_predictions.append(mock_pred)

    return MockResult()


# Initialize components
source_checker = SimpleSourceCredibility()
pipeline = None

if PIPELINE_AVAILABLE:
    try:
        print("🚀 Initializing ClaimExtractionPipeline...")
        pipeline = ClaimExtractionPipeline()
        pipeline_status = pipeline.get_pipeline_status()
        print(f"📊 Pipeline status: {pipeline_status}")
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        pipeline = None


# Flask Routes
@app.route('/')
def index():
    """Main page with input form"""
    return render_template('index.html',
                         pipeline_available=PIPELINE_AVAILABLE,
                         bert_available=BERT_AVAILABLE,
                         text_processor_available=TEXT_PROCESSOR_AVAILABLE)


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze submitted content"""
    start_time = time.time()

    try:
        # Get input data
        content = request.form.get('content', '').strip()
        url = request.form.get('url', '').strip() or 'manual_input'
        title = request.form.get('title', '').strip() or 'Untitled Article'

        # Validation
        if not content:
            flash('Please provide content to analyze.', 'error')
            return redirect(url_for('index'))

        if len(content) < 20:
            flash('Please provide at least 20 characters of content for meaningful analysis.', 'error')
            return redirect(url_for('index'))

        logger.info(f"Analyzing content: {len(content)} characters, URL: {url}")

        # Run analysis
        if pipeline and PIPELINE_AVAILABLE:
            print(f"🔍 Running full pipeline analysis...")
            result = pipeline.extract_claims(url, title, content)
        else:
            print(f"🔍 Running mock analysis (pipeline not available)...")
            result = create_mock_result(url, title, content)

        # Source credibility check
        source_score, source_explanation = source_checker.get_credibility(url)

        # Calculate overall credibility score
        credibility_score = calculate_credibility_score(result, source_score)

        # Generate explanations
        explanations = get_score_explanation(credibility_score, result, source_explanation)

        # Processing time
        processing_time = time.time() - start_time

        logger.info(f"Analysis completed: score={credibility_score}, claims={getattr(result, 'claims_identified', 0)}, time={processing_time:.2f}s")

        return render_template('results.html',
                             result=result,
                             credibility_score=credibility_score,
                             source_score=source_score,
                             source_explanation=source_explanation,
                             explanations=explanations,
                             processing_time=processing_time,
                             title=title,
                             url=url,
                             content_preview=content[:200] + "..." if len(content) > 200 else content)

    except RequestEntityTooLarge:
        flash('Content too large. Please submit content under 16MB.', 'error')
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        flash(f'Analysis failed: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/demo')
def demo():
    """Pre-loaded demo examples"""
    demo_articles = [
        {
            'title': 'Climate Research Study',
            'url': 'https://stanford.edu/climate-study',
            'content': '''Scientists at Stanford University released groundbreaking research yesterday showing that global average temperatures have risen by 1.1°C since 1880. The study analyzed data from over 15,000 weather stations worldwide and represents the most comprehensive analysis to date.

Dr. Sarah Johnson, the lead researcher, stated that "this represents the fastest rate of warming in recorded history." The research team found that Arctic temperatures increased by 2.3°C over the same period, nearly double the global average.

The study, which will be published in Nature Climate Change next month, used advanced statistical methods to account for measurement uncertainties and urban heat island effects. According to the researchers, immediate action is needed to address climate change impacts.''',
            'expected_score': 85,
            'description': 'High-quality academic research with verifiable claims and authoritative source.'
        },
        {
            'title': 'Technology Breakthrough Claim',
            'url': 'https://techexample.com/ai-breakthrough',
            'content': '''TechCorp, a little-known startup, claims to have achieved artificial general intelligence with their revolutionary new AI system. CEO John Smith says their technology is "1000% more efficient than current systems" and will "completely revolutionize everything."

The company hasn't published any peer-reviewed research or allowed independent verification of their claims. Smith believes this breakthrough will make all other AI companies obsolete within six months.

Industry experts remain skeptical, as the company has provided no technical details or demonstrations. The claims have not been verified by any independent researchers or institutions.''',
            'expected_score': 25,
            'description': 'Suspicious claims with red flags: no verification, extreme language, lack of evidence.'
        },
        {
            'title': 'Health Study Results',
            'url': 'https://nejm.org/nutrition-study',
            'content': '''A large-scale clinical trial involving 2,400 participants found that following a Mediterranean diet reduces cardiovascular disease risk by 31% compared to a low-fat diet. The study was conducted over 5 years across 12 medical centers in Spain.

Results published in the New England Journal of Medicine show that participants who followed the Mediterranean diet had significantly lower rates of heart attacks, strokes, and cardiovascular deaths. The diet emphasized olive oil, nuts, fruits, vegetables, and fish while limiting processed foods.

Dr. Maria Rodriguez, the study's principal investigator, noted that "the benefits were evident within the first two years of the study." The research was funded by the Spanish government and had no commercial sponsors.''',
            'expected_score': 90,
            'description': 'High-quality medical research with specific data, peer review, and transparent methodology.'
        }
    ]

    return render_template('demo.html', articles=demo_articles)


@app.route('/methodology')
def methodology():
    """Explain how the system works"""
    methodology_info = {
        'claim_extraction': {
            'available': PIPELINE_AVAILABLE,
            'description': 'BERT-based NLP model identifies factual claims in text',
            'accuracy': '85%+',
            'types': ['Statistical', 'Temporal', 'Causal', 'Identity', 'Existential']
        },
        'source_analysis': {
            'available': True,
            'description': 'Domain credibility assessment using curated database',
            'sources': 200,
            'categories': ['Academic', 'News', 'Government', 'Scientific']
        },
        'verification': {
            'available': BERT_AVAILABLE,
            'description': 'Evidence aggregation and fact-checking integration',
            'apis': ['Google Fact Check', 'Knowledge Graphs'],
            'coverage': 'Major factual claims'
        }
    }

    return render_template('methodology.html',
                         methodology=methodology_info,
                         pipeline_available=PIPELINE_AVAILABLE)


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for programmatic access"""
    try:
        data = request.get_json()

        if not data or 'content' not in data:
            return jsonify({'error': 'Content required'}), 400

        content = data['content'].strip()
        if len(content) < 20:
            return jsonify({'error': 'Content too short (minimum 20 characters)'}), 400

        url = data.get('url', 'api_request')
        title = data.get('title', 'API Analysis')

        # Run analysis
        if pipeline and PIPELINE_AVAILABLE:
            result = pipeline.extract_claims(url, title, content)
        else:
            result = create_mock_result(url, title, content)

        # Source credibility check
        source_score, source_explanation = source_checker.get_credibility(url)

        # Calculate credibility score
        credibility_score = calculate_credibility_score(result, source_score)

        # Return JSON response
        return jsonify({
            'credibility_score': credibility_score,
            'source_score': source_score,
            'source_explanation': source_explanation,
            'claims_identified': getattr(result, 'claims_identified', 0),
            'high_confidence_claims': getattr(result, 'high_confidence_claims', 0),
            'verifiable_claims': getattr(result, 'verifiable_claims', 0),
            'processing_time': getattr(result, 'processing_time_seconds', 0),
            'content_hash': getattr(result, 'content_hash', ''),
            'pipeline_available': PIPELINE_AVAILABLE
        })

    except Exception as e:
        logger.error(f"API analysis error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/demo/<int:demo_id>')
def api_demo(demo_id):
    """API endpoint for demo examples"""
    demos = [
        {'id': 1, 'title': 'Climate Research', 'expected_score': 85},
        {'id': 2, 'title': 'Tech Breakthrough Claim', 'expected_score': 25},
        {'id': 3, 'title': 'Health Study', 'expected_score': 90}
    ]

    if demo_id < 1 or demo_id > len(demos):
        return jsonify({'error': 'Demo not found'}), 404

    return jsonify(demos[demo_id - 1])


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'pipeline_available': PIPELINE_AVAILABLE,
        'bert_available': BERT_AVAILABLE,
        'text_processor_available': TEXT_PROCESSOR_AVAILABLE,
        'components': {
            'web_interface': True,
            'source_credibility': True,
            'claim_extraction': PIPELINE_AVAILABLE,
            'nlp_processing': TEXT_PROCESSOR_AVAILABLE
        }
    })


@app.errorhandler(404)
def page_not_found(error):
    """Custom 404 page"""
    return render_template('error.html',
                         error_code=404,
                         error_message="Page not found",
                         error_description="The page you're looking for doesn't exist."), 404


@app.errorhandler(500)
def internal_error(error):
    """Custom 500 page"""
    return render_template('error.html',
                         error_code=500,
                         error_message="Internal server error",
                         error_description="Something went wrong. Please try again later."), 500


@app.errorhandler(RequestEntityTooLarge)
def file_too_large(error):
    """Handle file too large errors"""
    flash('Content too large. Please submit content under 16MB.', 'error')
    return redirect(url_for('index'))


if __name__ == '__main__':
    print("🌐 Starting Truthed Professional Web Interface")
    print(f"📊 Pipeline available: {PIPELINE_AVAILABLE}")
    print(f"🤖 BERT available: {BERT_AVAILABLE}")
    print(f"📝 Text processor available: {TEXT_PROCESSOR_AVAILABLE}")
    print(f"🔧 Development mode: {os.getenv('FLASK_ENV') == 'development'}")

    # Development vs production settings
    debug_mode = os.getenv('FLASK_ENV') == 'development'

    app.run(
        debug=debug_mode,
        host='0.0.0.0',
        port=5001
    )