# Truthed Professional

A sophisticated misinformation detection system that analyzes online content for credibility and factual accuracy.

## 🎯 Project Overview

Truthed Professional combines advanced NLP, fact verification, and source credibility analysis to provide educational explanations about content credibility. Rather than simple binary judgments, it offers detailed analysis of why content may be unreliable.

### Core Capabilities
- **Claim Extraction**: Identifies factual claims in articles using BERT-based NLP
- **Source Analysis**: Evaluates domain credibility and bias indicators  
- **Fact Verification**: Cross-references claims against authoritative sources
- **Ensemble Scoring**: Combines multiple signals for overall credibility assessment

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- PostgreSQL 12+
- Redis 6+
- Git

### Development Setup

1. **Clone and setup environment**:
```bash
git clone <your-repo-url>
cd truthed-pro
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -e ".[dev]"
python -m spacy download en_core_web_lg
pre-commit install
```

3. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your settings
```

4. **Setup database** (using Docker):
```bash
docker-compose up -d postgres redis
python scripts/setup_database.py
```

5. **Run tests**:
```bash
pytest tests/ -v
```

6. **Start development server**:
```bash
make dev
# Or: uvicorn src.truthed.api.main:app --reload
```

## 📁 Project Structure

```
src/truthed/
├── api/                 # FastAPI application
├── core/               # Core data models and utilities
├── data/               # Data collection and preprocessing
├── database/           # Database models and repositories
├── models/             # ML models and algorithms
├── services/           # Business logic services
└── utils/              # Helper utilities

deployment/             # Docker and Kubernetes configs
docs/                   # Documentation
scripts/               # Utility scripts
tests/                 # Test suites
```

## 🛠️ Development Workflow

### Running Tests
```bash
make test           # All tests
make test-unit      # Unit tests only
make test-integration  # Integration tests only
```

### Code Quality
```bash
make lint           # Linting
make format         # Auto-formatting
```

### Database Operations
```bash
make db-migrate     # Run migrations
make db-reset       # Reset database
```

### Data and Models
```bash
make collect-data   # Collect training data
make train-models   # Train ML models
make evaluate       # Evaluate system performance
```

## 🏗️ Architecture

### System Flow
```
Content Input → Claim Extraction → Source Analysis → Fact Verification → Ensemble Scoring → Credibility Report
```

### Key Components
- **Claim Extraction**: BERT-based pipeline for identifying factual claims
- **Source Credibility**: Domain analysis with bias detection
- **Fact Verification**: Integration with multiple knowledge sources
- **Ensemble Model**: Weighted combination of all signals

## 📊 Performance Targets

- **Accuracy**: >70% on diverse test datasets
- **Speed**: <3 seconds end-to-end analysis
- **Precision**: >85% for claim extraction
- **Reliability**: >99% API uptime

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following our coding standards
4. Run tests and ensure they pass
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Standards
- All code must pass tests and linting
- Maintain >90% test coverage
- Follow PEP 8 style guidelines
- Include docstrings for all public functions
- Update documentation for new features

## 📈 Current Status

**Phase 1**: Foundation Development (Months 1-3)
- ✅ Project structure and architecture
- 🔄 Claim extraction pipeline (In Progress)
- ⏳ Source credibility analysis
- ⏳ Fact verification integration
- ⏳ Ensemble scoring system

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For questions and support:
- Check the [documentation](docs/)
- Review [common issues](docs/troubleshooting.md)
- Open an [issue](issues/) for bugs or feature requests

## 🙏 Acknowledgments

- Fact-checking organizations for methodology guidance
- Open source NLP community for foundational models
- Academic research community for misinformation detection approaches