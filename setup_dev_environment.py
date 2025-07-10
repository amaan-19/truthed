#!/usr/bin/env python3
"""
Setup script to fix Python path issues
Run this to configure your development environment
"""
import sys
import os
from pathlib import Path


def setup_python_path():
    """Setup Python path for development"""
    print("🔧 Setting up Python path for Truthed development")
    print("=" * 50)

    # Get project root
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir
    src_dir = project_root / "src"

    print(f"Project root: {project_root}")
    print(f"Source directory: {src_dir}")

    # Check if src directory exists
    if not src_dir.exists():
        print(f"❌ Source directory not found: {src_dir}")
        return False

    # Check if truthed package exists
    truthed_dir = src_dir / "truthed"
    if not truthed_dir.exists():
        print(f"❌ Truthed package not found: {truthed_dir}")
        return False

    print(f"✅ Found truthed package: {truthed_dir}")

    # Add src to Python path
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
        print(f"✅ Added to Python path: {src_str}")

    # Test imports
    print(f"\n🧪 Testing imports...")

    try:
        # Test core models import
        from truthed.core.models import ClaimType
        print(f"✅ Successfully imported ClaimType")

        # List available claim types
        claim_types = [ct.value for ct in ClaimType]
        print(f"   Available claim types: {claim_types}")

    except ImportError as e:
        print(f"❌ Failed to import ClaimType: {e}")

        # Create a minimal core models file if it doesn't exist
        core_models_path = src_dir / "truthed" / "core" / "models.py"
        if not core_models_path.exists():
            print(f"📝 Creating minimal core models file...")
            create_minimal_core_models(core_models_path)

            # Try import again
            try:
                from truthed.core.models import ClaimType
                print(f"✅ Successfully imported ClaimType after creation")
            except ImportError as e2:
                print(f"❌ Still failed to import: {e2}")
                return False

    try:
        # Test text processing import
        from truthed.data.preprocessing.text_processing import TextProcessor
        print(f"✅ Successfully imported TextProcessor")

        # Test creating processor
        processor = TextProcessor()
        print(f"✅ Successfully created TextProcessor instance")

    except ImportError as e:
        print(f"❌ Failed to import TextProcessor: {e}")
        return False
    except Exception as e:
        print(f"❌ Error creating TextProcessor: {e}")
        return False

    print(f"\n🎉 Python path setup complete!")
    print(f"You can now import from the truthed package in this session.")

    return True


def create_minimal_core_models(file_path):
    """Create a minimal core models file"""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    minimal_content = '''"""
Core data models for the Truthed system.
"""
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from datetime import datetime


class ClaimType(str, Enum):
    """Types of factual claims"""
    STATISTICAL = "statistical"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    IDENTITY = "identity"
    EXISTENTIAL = "existential"
    GENERAL_FACTUAL = "general_factual"


class VeracityLabel(str, Enum):
    """Veracity labels for content and claims"""
    TRUE = "TRUE"
    FALSE = "FALSE"
    MIXED = "MIXED"
    UNVERIFIABLE = "UNVERIFIABLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass
class ClaimPrediction:
    """Prediction result for a single sentence"""
    sentence: str
    is_claim: bool
    confidence: float
    claim_type: Optional[ClaimType] = None
    reasoning: str = ""
'''

    with open(file_path, 'w') as f:
        f.write(minimal_content)

    print(f"✅ Created minimal core models at: {file_path}")


def create_init_files():
    """Create missing __init__.py files"""
    print(f"\n📁 Creating missing __init__.py files...")

    init_files = [
        "src/truthed/__init__.py",
        "src/truthed/core/__init__.py",
        "src/truthed/data/__init__.py",
        "src/truthed/data/preprocessing/__init__.py",
        "src/truthed/models/__init__.py",
        "src/truthed/models/claim_extraction/__init__.py"
    ]

    for init_file in init_files:
        init_path = Path(init_file)
        init_path.parent.mkdir(parents=True, exist_ok=True)

        if not init_path.exists():
            init_path.touch()
            print(f"✅ Created: {init_file}")
        else:
            print(f"✅ Exists: {init_file}")


if __name__ == "__main__":
    # Create missing directories and files
    create_init_files()

    # Setup Python path
    success = setup_python_path()

    if success:
        print(f"\n🚀 Ready for development!")
        print(f"You can now run your BERT classifier and other components.")
    else:
        print(f"\n❌ Setup failed. Check the errors above.")
        sys.exit(1)