"""Test script for data loading and preprocessing components."""

import sys
sys.path.append('src')

from data_preprocessing import DataLoader, DataValidator, DataPreprocessor, FeatureEngineer
from config import setup_logging
import pandas as pd

def test_data_loading():
    """Test data loading functionality."""
    print("Testing Data Loading...")
    
    # Setup logging
    logger = setup_logging()
    
    # Test DataLoader
    loader = DataLoader()
    
    try:
        # Load sample data
        data = loader.load_training_data('sample_data.csv')
        print(f"✓ Successfully loaded data: {data.shape}")
        print(f"✓ Columns: {list(data.columns)}")
        print(f"✓ Data types:\n{data.dtypes}")
        
        # Test validation
        is_valid = loader.validate_data_format(data)
        print(f"✓ Data validation: {'PASSED' if is_valid else 'FAILED'}")
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False
    
    # Test DataValidator
    print("\nTesting Data Validation...")
    validator = DataValidator()
    
    try:
        # Test outlier detection
        outliers = validator.identify_outliers(data, ['temperature', 'humidity'])
        print(f"✓ Outlier detection completed")
        
        # Test range validation
        range_results = validator.validate_ranges(data)
        print(f"✓ Range validation completed")
        for col, results in range_results.items():
            print(f"  {col}: {results['out_of_range_count']} out of range values")
        
    except Exception as e:
        print(f"✗ Data validation failed: {e}")
        return False
    
    # Test DataPreprocessor
    print("\nTesting Data Preprocessing...")
    preprocessor = DataPreprocessor()
    
    try:
        # Clean data
        cleaned_data = preprocessor.clean_data(data)
        print(f"✓ Data cleaning completed: {cleaned_data.shape}")
        
        # Normalize features
        normalized_data = preprocessor.normalize_features(cleaned_data)
        print(f"✓ Feature normalization completed")
        print(f"✓ Normalized data stats:\n{normalized_data.describe()}")
        
    except Exception as e:
        print(f"✗ Data preprocessing failed: {e}")
        return False
    
    # Test FeatureEngineer
    print("\nTesting Feature Engineering...")
    engineer = FeatureEngineer()
    
    try:
        # Create features
        feature_data = engineer.create_features(normalized_data)
        print(f"✓ Feature engineering completed: {feature_data.shape}")
        print(f"✓ New columns: {[col for col in feature_data.columns if col not in data.columns]}")
        
    except Exception as e:
        print(f"✗ Feature engineering failed: {e}")
        return False
    
    print("\n🎉 All data loading and preprocessing tests PASSED!")
    return True

if __name__ == "__main__":
    success = test_data_loading()
    sys.exit(0 if success else 1)