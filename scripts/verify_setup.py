#!/usr/bin/env python3
"""
Complete setup verification for Truthed Professional development environment.
Run this script to verify all components are working correctly.
"""
import sys
import os
import subprocess
import importlib
from pathlib import Path


def print_header(title):
    """Print a formatted header"""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'-' * 40}")
    print(f" {title}")
    print(f"{'-' * 40}")


def check_python_environment():
    """Check Python version and virtual environment"""
    print_section("Python Environment")

    # Python version
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")

    # Virtual environment check
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    print(f"In virtual environment: {'✅ Yes' if in_venv else '❌ No'}")

    if in_venv:
        venv_path = sys.prefix
        print(f"Virtual environment path: {venv_path}")

    return in_venv


def check_required_packages():
    """Check if all required packages are installed"""
    print_section("Package Dependencies")

    required_packages = {
        'Core Web Framework': ['fastapi', 'uvicorn', 'pydantic'],
        'Database': ['sqlalchemy', 'psycopg2', 'redis'],
        'ML/NLP': ['transformers', 'torch', 'spacy', 'sklearn', 'pandas', 'numpy'],
        'Utilities': ['httpx', 'aiofiles', 'requests', 'bs4', 'nltk'],
        'Development': ['pytest', 'black', 'isort', 'mypy']
    }

    all_good = True

    for category, packages in required_packages.items():
        print(f"\n{category}:")
        for package in packages:
            try:
                if package == 'sklearn':
                    import sklearn
                    version = sklearn.__version__
                elif package == 'bs4':
                    import bs4
                    version = bs4.__version__
                elif package == 'psycopg2':
                    import psycopg2
                    version = psycopg2.__version__
                else:
                    module = importlib.import_module(package)
                    version = getattr(module, '__version__', 'Unknown')

                print(f"  ✅ {package} ({version})")
            except ImportError:
                print(f"  ❌ {package} - NOT INSTALLED")
                all_good = False
            except Exception as e:
                print(f"  ⚠️  {package} - Error: {e}")

    return all_good


def check_spacy_model():
    """Check if spaCy model is downloaded"""
    print_section("spaCy Model")

    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_lg")
            print("✅ en_core_web_lg model loaded successfully")

            # Test the model
            doc = nlp("Apple Inc. was founded by Steve Jobs in 1976.")
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            print(f"  Test entities found: {entities}")
            return True

        except OSError:
            print("❌ en_core_web_lg model not found")
            print("  Run: python -m spacy download en_core_web_lg")
            return False

    except ImportError:
        print("❌ spaCy not installed")
        return False


def check_project_structure():
    """Check if project directory structure is correct"""
    print_section("Project Structure")

    required_dirs = [
        'src',
        'src/truthed',
        'src/truthed/api',
        'src/truthed/core',
        'src/truthed/data',
        'src/truthed/models',
        'src/truthed/services',
        'tests',
        'scripts',
        'docs'
    ]

    required_files = [
        'pyproject.toml',
        'README.md',
        '.gitignore',
        'docker-compose.yml'
    ]

    project_root = Path.cwd()
    all_good = True

    print("Required directories:")
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/ - MISSING")
            all_good = False

    print("\nRequired files:")
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - MISSING")
            all_good = False

    return all_good


def check_git_setup():
    """Check Git configuration"""
    print_section("Git Setup")

    try:
        # Check if git is available
        result = subprocess.run(['git', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Git installed: {result.stdout.strip()}")
        else:
            print("❌ Git not available")
            return False

        # Check if in git repository
        result = subprocess.run(['git', 'status'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Git repository initialized")

            # Check for commits
            result = subprocess.run(['git', 'log', '--oneline', '-1'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Git commits exist")
            else:
                print("⚠️  No git commits yet")
        else:
            print("❌ Not a git repository")
            return False

    except FileNotFoundError:
        print("❌ Git not installed")
        return False

    return True


def check_docker_services():
    """Check if Docker services are available"""
    print_section("Docker Services")

    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker installed: {result.stdout.strip()}")

            # Check if docker-compose file exists
            if Path('docker-compose.yml').exists():
                print("✅ docker-compose.yml found")

                # Check if services are running
                result = subprocess.run(['docker-compose', 'ps'], capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ Docker Compose available")
                    if 'postgres' in result.stdout or 'redis' in result.stdout:
                        print("✅ Some services appear to be running")
                    else:
                        print("⚠️  No services currently running")
                        print("  Run: docker-compose up -d postgres redis")
                else:
                    print("⚠️  Docker Compose not working properly")
            else:
                print("❌ docker-compose.yml not found")

        else:
            print("❌ Docker not available")
            return False

    except FileNotFoundError:
        print("❌ Docker not installed")
        return False

    return True


def check_environment_file():
    """Check if .env file exists and is configured"""
    print_section("Environment Configuration")

    env_file = Path('.env')
    env_example = Path('.env.example')

    if env_example.exists():
        print("✅ .env.example found")
    else:
        print("❌ .env.example missing")

    if env_file.exists():
        print("✅ .env file exists")

        # Check for key variables
        with open(env_file) as f:
            content = f.read()

        required_vars = ['DATABASE_HOST', 'DATABASE_NAME', 'SECRET_KEY']
        for var in required_vars:
            if var in content:
                print(f"  ✅ {var} configured")
            else:
                print(f"  ❌ {var} missing")

    else:
        print("❌ .env file missing")
        print("  Copy .env.example to .env and configure")
        return False

    return True


def run_basic_tests():
    """Run basic functionality tests"""
    print_section("Basic Functionality Tests")

    try:
        # Test text processing
        print("Testing text processing...")
        sys.path.insert(0, 'src')

        from truthed.data.preprocessing.text_processing import TextProcessor
        processor = TextProcessor()

        test_text = "This is a test. Scientists say climate change is real. The temperature rose by 2 degrees."
        processed = processor.preprocess_article(test_text)

        print(f"  ✅ Text processing works")
        print(f"  Sentences found: {len(processed.sentences)}")
        print(f"  Word count: {processed.word_count}")

        return True

    except Exception as e:
        print(f"  ❌ Text processing failed: {e}")
        return False


def main():
    """Run complete setup verification"""
    print_header("TRUTHED PROFESSIONAL - SETUP VERIFICATION")
    print(f"Current directory: {Path.cwd()}")
    print(f"Date: {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}")

    checks = [
        ("Python Environment", check_python_environment),
        ("Package Dependencies", check_required_packages),
        ("spaCy Model", check_spacy_model),
        ("Project Structure", check_project_structure),
        ("Git Setup", check_git_setup),
        ("Docker Services", check_docker_services),
        ("Environment Config", check_environment_file),
        ("Basic Functionality", run_basic_tests)
    ]

    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"❌ {check_name} failed with error: {e}")
            results[check_name] = False

    # Summary
    print_header("VERIFICATION SUMMARY")

    passed = sum(results.values())
    total = len(results)

    for check_name, passed_check in results.items():
        status = "✅ PASS" if passed_check else "❌ FAIL"
        print(f"{check_name:.<30} {status}")

    print(f"\nOverall: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 ALL CHECKS PASSED! You're ready to start development!")
        print("\nNext steps:")
        print("1. Run: python scripts/test_claim_extraction.py")
        print("2. Start implementing BERTClaimClassifier")
        print("3. Begin claim extraction development")
    else:
        print(f"\n⚠️  {total - passed} issues need to be fixed before development")
        print("\nFix the failed checks above, then run this script again.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)