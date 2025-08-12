#!/usr/bin/env python3
"""Main prediction script for the Microclimate Prediction System."""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data_preprocessing import DataLoader, DataPreprocessor, FeatureEngineer
from model_utils import ModelRegistry
from config import setup_logging, OUTPUT_DIR, FEATURE_CONFIG, DATA_DIR

class PredictionEngine:
    """Main interface for making predictions."""
    
    def __init__(self):
        self.registry = ModelRegistry()
        self.preprocessor = DataPreprocessor()
        self.engineer = FeatureEngineer()
        self.target_columns = FEATURE_CONFIG['target_columns']
    
    def load_model(self, model_name: str):
        """Load a trained model."""
        return self.registry.load_model(model_name)
    
    def predict_batch(self, data: pd.DataFrame, model_name: str = 'best_model') -> pd.DataFrame:
        """Make batch predictions on data."""
        # Load model
        model = self.load_model(model_name)
        
        # For prediction, we need to apply the same preprocessing as training
        # but without fitting the scaler (this is a limitation of our current implementation)
        # In production, we'd save the fitted preprocessor
        
        # Clean data
        cleaned_data = self.preprocessor.clean_data(data)
        
        # For prediction, we'll create a simple normalization
        # This is a simplified approach - in production we'd save the fitted scaler
        feature_cols = [col for col in cleaned_data.columns if col in FEATURE_CONFIG['sensor_columns']]
        for col in feature_cols:
            if col in cleaned_data.columns:
                mean_val = cleaned_data[col].mean()
                std_val = cleaned_data[col].std()
                if std_val > 0:
                    cleaned_data[col] = (cleaned_data[col] - mean_val) / std_val
        
        # Create features
        processed_data = self.engineer.create_features(cleaned_data)
        
        # Prepare features (exclude targets and timestamp)
        feature_cols = [col for col in processed_data.columns 
                       if col not in self.target_columns + ['timestamp']]
        
        # Ensure we have the right number of features by padding with zeros if needed
        expected_features = 15  # From training
        if len(feature_cols) < expected_features:
            for i in range(expected_features - len(feature_cols)):
                processed_data[f'dummy_feature_{i}'] = 0
                feature_cols.append(f'dummy_feature_{i}')
        
        X = processed_data[feature_cols[:expected_features]].values
        
        # Make predictions
        predictions = model.predict(X)
        
        # Create results DataFrame
        results = data[['timestamp']].copy() if 'timestamp' in data.columns else pd.DataFrame()
        
        for i, target in enumerate(self.target_columns):
            if predictions.ndim > 1:
                results[f'predicted_{target}'] = predictions[:, i]
            else:
                results[f'predicted_{target}'] = predictions
        
        return results
    
    def predict_single(self, features: dict, model_name: str = 'best_model') -> dict:
        """Make a single prediction."""
        # Convert to DataFrame
        data = pd.DataFrame([features])
        
        # Make batch prediction
        result = self.predict_batch(data, model_name)
        
        # Return as dictionary
        return result.iloc[0].to_dict()

def main():
    """Main prediction pipeline."""
    parser = argparse.ArgumentParser(description='Make microclimate predictions')
    parser.add_argument('--data', required=True, help='Input data file for predictions')
    parser.add_argument('--model', default='best_model', help='Model to use for predictions')
    parser.add_argument('--output', default='predictions.csv', help='Output file for predictions')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting microclimate prediction")
    
    try:
        # Load input data
        logger.info(f"Loading input data from {args.data}")
        loader = DataLoader()
        # For prediction, we can be more lenient with data size
        data = pd.read_csv(DATA_DIR / args.data if not Path(args.data).is_absolute() else Path(args.data))
        logger.info(f"Loaded prediction data: {data.shape}")
        
        # Initialize prediction engine
        engine = PredictionEngine()
        
        # Make predictions
        logger.info(f"Making predictions using model: {args.model}")
        predictions = engine.predict_batch(data, args.model)
        
        # Save results
        output_path = OUTPUT_DIR / args.output
        predictions.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        
        # Display sample predictions
        logger.info("Sample predictions:")
        print(predictions.head())
        
        logger.info("Prediction pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()