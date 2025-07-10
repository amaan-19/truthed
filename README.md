# Truthed Professional

**⚠️ Work in Progress**: This project is currently in active development. Core components are being built and refined.

A sophisticated misinformation detection system that analyzes online content for credibility and factual accuracy.

## 🎯 Project Overview

Truthed Professional is being designed to combine advanced NLP, fact verification, and source credibility analysis to provide educational explanations about content credibility. Rather than simple binary judgments, it will offer detailed analysis of why content may be unreliable.

### Planned Core Capabilities
- **Claim Extraction**: Identify factual claims in articles using BERT-based NLP *(In Development)*
- **Source Analysis**: Evaluate domain credibility and bias indicators *(Planned)*
- **Fact Verification**: Cross-reference claims against authoritative sources *(Planned)*
- **Ensemble Scoring**: Combine multiple signals for overall credibility assessment *(Planned)*

## 🚧 Development Status

**Current Phase**: Foundation Development (Phase 1: Months 1-3)

### What's Complete ✅
- Project architecture and structure
- Core data models and schemas
- Development environment setup
- Docker containerization framework
- Testing infrastructure
- API foundation with FastAPI

### In Active Development 🔄
- Claim extraction pipeline
- Data collection and preprocessing modules
- Database schema implementation
- Basic API endpoints

### Planned Next 📋
- Source credibility analysis system
- Fact verification integration
- Ensemble scoring algorithm
- Production deployment pipeline
- Comprehensive testing suite

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- PostgreSQL 12+ (or Docker)
- Redis 6+ (or Docker)
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
# Edit .env with your database and API settings
```

4. **Setup services** (Docker recommended):
```bash
# Start database and Redis
docker-compose up -d postgres redis

# Initialize database (when ready)
python scripts/setup_database.py
```

5. **Run available tests**:
```bash
pytest tests/ -v --tb=short
```

6. **Start development server**:
```bash
make dev
# Or directly: uvicorn src.truthed.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

## 📁 Project Structure

```
src/truthed/
├── api/                 # FastAPI application (Basic structure)
├── core/               # Core data models and utilities ✅
├── data/               # Data collection and preprocessing (In Progress)
├── database/           # Database models and repositories (Basic)
├── models/             # ML models and algorithms (Framework only)
├── services/           # Business logic services (Framework only)
└── utils/              # Helper utilities (Basic)

deployment/             # Docker and Kubernetes configs ✅
docs/                   # Documentation (Partial)
scripts/               # Utility scripts (Basic)
tests/                 # Test framework ✅
```

## 🛠️ Development Commands

### Available Commands
```bash
make install          # Install production dependencies
make install-dev      # Install development dependencies
make setup           # Complete development setup
make test            # Run available tests
make lint            # Code linting
make format          # Auto-formatting
make docker-build    # Build Docker containers
make docker-up       # Start services
make clean           # Clean build artifacts
```

### Future Commands (Not Yet Implemented)
```bash
make collect-data    # Collect training data
make train-models    # Train ML models  
make evaluate        # Evaluate system performance
```

## 🏗️ Planned Architecture

### Intended System Flow
```
Content Input → Claim Extraction → Source Analysis → Fact Verification → Ensemble Scoring → Credibility Report
```

### Component Status
- **API Layer**: Basic FastAPI setup ✅
- **Data Models**: Core schemas defined ✅
- **Claim Extraction**: BERT pipeline framework 🔄
- **Source Credibility**: Database schema planned 📋
- **Fact Verification**: API integration planned 📋
- **Ensemble Model**: Algorithm design phase 📋

## 📊 Target Goals

### Performance Targets (When Complete)
- **Accuracy**: >70% on diverse test datasets
- **Speed**: <3 seconds end-to-end analysis
- **Precision**: >85% for claim extraction
- **Reliability**: >99% API uptime

### Development Milestones
- **Milestone 1**: Basic claim extraction 
- **Milestone 2**: Source analysis integration 
- **Milestone 3**: Fact verification pipeline 
- **Milestone 4**: End-to-end system testing 

## 🧪 Testing

Currently implemented:
- Testing framework with pytest
- Basic test structure
- Continuous integration setup

**Note**: Comprehensive test coverage will be implemented as components are completed.

```bash
pytest tests/ -v                # Run all tests
pytest tests/unit/ -v          # Unit tests (limited)
pytest tests/integration/ -v   # Integration tests (planned)
```

## 🤝 Contributing

This project is not accepting any contributors at this time.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

**Development Status**: Active development, architecture stable, core features in progress.
