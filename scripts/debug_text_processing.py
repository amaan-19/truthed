#!/usr/bin/env python3
"""
Debug script to identify and fix text processing issues.
"""
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

print(f"Python path: {sys.path}")
print(f"Current directory: {Path.cwd()}")
print(f"Src directory exists: {src_path.exists()}")

# Test step by step
print("\n" + "=" * 50)
print("DEBUGGING TEXT PROCESSING")
print("=" * 50)

# Step 1: Test basic imports
print("\nStep 1: Testing imports...")
try:
    print("Importing spacy...")
    import spacy

    print("✅ spacy imported")

    print("Importing beautifulsoup4...")
    from bs4 import BeautifulSoup

    print("✅ beautifulsoup4 imported")

    print("Importing dataclasses...")
    from dataclasses import dataclass

    print("✅ dataclasses imported")

except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Step 2: Test spaCy English model
print("\nStep 2: Testing spaCy setup...")
try:
    from spacy.lang.en import English

    nlp = English()
    nlp.add_pipe("sentencizer")
    print("✅ spaCy English tokenizer created")

    # Test sentence segmentation
    test_text = "This is sentence one. This is sentence two!"
    doc = nlp(test_text)
    sentences = [sent.text.strip() for sent in doc.sents]
    print(f"✅ Sentence segmentation works: {sentences}")

except Exception as e:
    print(f"❌ spaCy error: {e}")
    sys.exit(1)

# Step 3: Test if our module directory exists
print("\nStep 3: Checking module structure...")
module_paths = [
    "src/truthed/__init__.py",
    "src/truthed/data/__init__.py",
    "src/truthed/data/preprocessing/__init__.py",
    "src/truthed/data/preprocessing/text_processing.py"
]

for path in module_paths:
    full_path = Path(path)
    if full_path.exists():
        print(f"✅ {path}")
    else:
        print(f"❌ {path} - MISSING")

# Step 4: Try to import our module
print("\nStep 4: Testing our module import...")
try:
    from truthed.data.preprocessing.text_processing import TextProcessor, ProcessedText

    print("✅ Successfully imported TextProcessor")

    # Test creating the processor
    processor = TextProcessor()
    print("✅ TextProcessor created")

    # Test basic functionality
    test_text = "Scientists discovered something new. The temperature increased by 2 degrees."
    result = processor.preprocess_article(test_text)
    print(f"✅ Text processing worked!")
    print(f"  - Sentences: {len(result.sentences)}")
    print(f"  - Words: {result.word_count}")
    print(f"  - Sentences found: {result.sentences}")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("This means the text_processing.py file doesn't exist or has syntax errors")
except Exception as e:
    print(f"❌ Runtime error: {e}")
    print("This means there's a bug in the text_processing.py code")

print("\nDebugging complete!")