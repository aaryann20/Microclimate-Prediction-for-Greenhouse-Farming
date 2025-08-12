#!/usr/bin/env python3
"""Main training script for the Microclimate Prediction System."""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data_preprocessing import DataLoader, DataPreprocessor, FeatureEngineer
from model_utils import ModelTrainer, ModelEvaluator, ModelRegistry
from config import setup_logging, MODEL_CONFIG, OUTPUT_DIR

def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train microclimate prediction models')
    parser.add_argument('--data', default='sample_data.csv', help='Training data file')
    parser.add_argument('--models', nargs='+', default=['all'], 
                       choices=['linear_regression', 'random_forest', 'gradient_boosting', 'neural_network', 'all'],
                       help='Models to train')
    parser.add_argument('--save-best', action='store_true', help='Save only the best performing model')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting microclimate prediction model training")
    
    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        loader = DataLoader()
        data = loader.load_training_data(args.data)
        
        preprocessor = DataPreprocessor()
        cleaned_data = preprocessor.clean_data(data)
        normalized_data = preprocessor.normalize_features(cleaned_data)
        
        engineer = FeatureEngineer()
        feature_data = engineer.create_features(normalized_data)
        
        # Prepare training data
        trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = trainer.prepare_data(feature_data)
        
        # Train models
        logger.info("Training models...")
        models_to_train = args.models if 'all' not in args.models else ['linear_regression', 'random_forest', 'gradient_boosting', 'neural_network']
        
        trained_models = {}
        evaluator = ModelEvaluator()
        results = {}
        
        for model_type in models_to_train:
            logger.info(f"Training {model_type}...")
            
            if model_type in ['linear_regression', 'random_forest', 'gradient_boosting']:
                models = trainer.train_regression_models(X_train, y_train)
                trained_models[model_type] = models[model_type]
            elif model_type == 'neural_network':
                model = trainer.train_neural_network(X_train, y_train)
                trained_models[model_type] = model
            
            # Evaluate model
            metrics = evaluator.evaluate_regression(trained_models[model_type], X_test, y_test)
            map_score = evaluator.calculate_map_score(
                trained_models[model_type].predict(X_test), y_test
            )
            metrics['map_score'] = map_score
            results[model_type] = metrics
            
            logger.info(f"{model_type} - mAP: {map_score:.4f}, R²: {metrics['overall_r2']:.4f}")
        
        # Generate performance report
        report = evaluator.generate_performance_report(results)
        logger.info(f"Training completed. Performance report:\n{report}")
        
        # Save results
        results_file = OUTPUT_DIR / 'training_results.json'
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save models
        registry = ModelRegistry()
        best_model_name = None
        best_score = -1
        
        for model_name, model in trained_models.items():
            metadata = {
                'model_type': model_name,
                'performance_metrics': results[model_name],
                'training_data': args.data,
                'feature_count': X_train.shape[1]
            }
            
            if args.save_best:
                if results[model_name]['map_score'] > best_score:
                    best_score = results[model_name]['map_score']
                    best_model_name = model_name
            else:
                trainer.save_model(model_name, model, metadata)
        
        if args.save_best and best_model_name:
            trainer.save_model('best_model', trained_models[best_model_name], 
                             {'model_type': best_model_name, 
                              'performance_metrics': results[best_model_name],
                              'training_data': args.data,
                              'feature_count': X_train.shape[1]})
            logger.info(f"Best model ({best_model_name}) saved with mAP: {best_score:.4f}")
        
        # Check if any model meets the >80% mAP requirement
        meeting_requirement = [name for name, metrics in results.items() 
                             if metrics['map_score'] > MODEL_CONFIG['target_map_threshold']]
        
        if meeting_requirement:
            logger.info(f"✅ Models meeting >80% mAP requirement: {meeting_requirement}")
        else:
            logger.warning("⚠️  No models achieved >80% mAP requirement")
            max_map = max(results[name]['map_score'] for name in results)
            logger.info(f"Best mAP achieved: {max_map:.4f}")
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()