#!/usr/bin/env python3
"""
PyCharm terminal test - verify everything works in PyCharm environment.
"""
import sys
import os
from pathlib import Path


def test_pycharm_environment():
    """Test PyCharm environment setup"""
    print("🔧 PYCHARM ENVIRONMENT TEST")
    print("=" * 50)

    # Test 1: Python interpreter
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")

    # Test 2: Virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    print(f"In virtual environment: {'✅' if in_venv else '❌'}")

    # Test 3: Working directory
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    print(f"Is project root: {'✅' if (current_dir / 'pyproject.toml').exists() else '❌'}")

    # Test 4: Environment variables
    print(f"PATH includes venv: {'✅' if 'venv' in os.environ.get('PATH', '') else '❌'}")

    # Test 5: Package imports
    test_packages = ['fastapi', 'spacy', 'transformers', 'redis', 'psycopg2']
    print("\nPackage availability:")
    for package in test_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")

    # Test 6: Project modules
    print("\nProject modules:")
    try:
        sys.path.insert(0, 'src')
        from truthed.data.preprocessing.text_processing import TextProcessor
        print("  ✅ TextProcessor imported")

        processor = TextProcessor()
        print("  ✅ TextProcessor created")

        test_result = processor.preprocess_article("Test sentence. Another sentence.")
        print(f"  ✅ Text processing works ({len(test_result.sentences)} sentences)")

    except Exception as e:
        print(f"  ❌ Project modules failed: {e}")

    # Test 7: Database connections (if Docker is running)
    print("\nDatabase connections:")
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("  ✅ Redis connection")
    except Exception:
        print("  ❌ Redis connection (Docker services may not be running)")

    try:
        import psycopg2
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='truthed_db',
            user='truthed_user',
            password='truthed_pass'
        )
        conn.close()
        print("  ✅ PostgreSQL connection")
    except Exception:
        print("  ❌ PostgreSQL connection (Docker services may not be running)")

    print("\n" + "=" * 50)
    print("✅ PyCharm environment test complete!")
    print("\nIf you see any ❌, fix those issues before development.")
    print("If everything shows ✅, you're ready to code! 🚀")


if __name__ == "__main__":
    test_pycharm_environment()