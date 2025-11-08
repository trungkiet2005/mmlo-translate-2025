"""
Test script to verify setup and data loading
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test if all required packages are installed"""
    print("Testing imports...")
    try:
        import torch
        print(f"✓ torch: {torch.__version__}")
    except ImportError:
        print("✗ torch not installed")
        return False
    
    try:
        import transformers
        print(f"✓ transformers: {transformers.__version__}")
    except ImportError:
        print("✗ transformers not installed")
        return False
    
    try:
        import peft
        print(f"✓ peft: {peft.__version__}")
    except ImportError:
        print("✗ peft not installed")
        return False
    
    try:
        import pandas
        print(f"✓ pandas: {pandas.__version__}")
    except ImportError:
        print("✗ pandas not installed")
        return False
    
    try:
        import sentencepiece
        print(f"✓ sentencepiece: {sentencepiece.__version__}")
    except ImportError:
        print("✗ sentencepiece not installed")
        return False
    
    try:
        import sacrebleu
        print(f"✓ sacrebleu: {sacrebleu.__version__}")
    except ImportError:
        print("⚠ sacrebleu not installed (optional, but recommended)")
    
    return True


def test_data_loading():
    """Test data loading"""
    print("\nTesting data loading...")
    try:
        from src.data_utils import DataLoader
        
        # Test local path
        data_loader = DataLoader("./dataset")
        
        # Try loading one dataset
        try:
            df = data_loader.load_dataset("bhili")
            print(f"✓ Loaded bhili dataset: {len(df)} samples")
            print(f"  Columns: {df.columns.tolist()}")
            return True
        except FileNotFoundError:
            print("⚠ Dataset files not found in ./dataset")
            print("  This is OK if you're testing on Kaggle")
            return True
        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Error importing DataLoader: {e}")
        return False


def test_tokenizer():
    """Test tokenizer loading"""
    print("\nTesting tokenizer...")
    try:
        from transformers import AutoTokenizer
        
        print("Loading NLLB tokenizer (this may take a while)...")
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/nllb-200-distilled-600M",
            src_lang="eng_Latn",
            tgt_lang="hin_Deva"
        )
        print(f"✓ Tokenizer loaded: {len(tokenizer)} tokens")
        return True
    except Exception as e:
        print(f"✗ Error loading tokenizer: {e}")
        return False


def test_config():
    """Test configuration"""
    print("\nTesting configuration...")
    try:
        from src.config import Config, parse_args
        from src.utils import get_language_code
        
        # Test language code mapping
        codes = {
            "Hindi": get_language_code("Hindi"),
            "Bhili": get_language_code("Bhili"),
            "Santali": get_language_code("Santali")
        }
        print(f"✓ Language codes: {codes}")
        return True
    except Exception as e:
        print(f"✗ Error testing config: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("MMLoSo Setup Test")
    print("=" * 60)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test config
    results.append(("Config", test_config()))
    
    # Test data loading
    results.append(("Data Loading", test_data_loading()))
    
    # Test tokenizer (skip if imports failed)
    if results[0][1]:
        results.append(("Tokenizer", test_tokenizer()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n✓ All tests passed! Setup is ready.")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
        print("  Install missing packages: pip install -r requirements.txt")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

