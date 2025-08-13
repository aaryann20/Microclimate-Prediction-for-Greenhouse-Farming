#!/usr/bin/env python3
"""Advanced AI training script with neural networks and deep learning."""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data_preprocessing import DataLoader, DataPreprocessor, FeatureEngineer
from ai_models import NeuralNetworkPredictor, LSTMPredictor, EnsembleAIPredictor
from model_utils import ModelEvaluator
from config import setup_logging, MODEL_CONFIG, OUTPUT_DIR, MODELS_DIR

def main():
    """Main AI training pipeline."""
    parser = argparse.ArgumentParser(description='Train advanced AI models for microclimate prediction')
    parser.add_argument('--data', default='train_large.csv', help='Training data file')
    parser.add_argument('--models', nargs='+', default=['neural_network'], 
                       choices=['neural_network', 'lstm', 'ensemble', 'all'],
                       help='AI models to train')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("🤖 Starting Advanced AI Model Training")
    
    try:
        # Load and preprocess data
        logger.info("📊 Loading and preprocessing data...")
        loader = DataLoader()
        data = loader.load_training_data(args.data)
        
        preprocessor = DataPreprocessor()
        cleaned_data = preprocessor.clean_data(data)
        normalized_data = preprocessor.normalize_features(cleaned_data)
        
        engineer = FeatureEngineer()
        feature_data = engineer.create_features(normalized_data)
        
        # Prepare data for AI models
        feature_cols = [col for col in feature_data.columns 
                       if col not in ['temperature', 'humidity', 'light_intensity', 'timestamp']]
        target_cols = ['temperature', 'humidity', 'light_intensity']
        
        X = feature_data[feature_cols].values
        y = feature_data[target_cols].values
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=MODEL_CONFIG['random_state']
        )
        
        logger.info(f"✅ Data prepared: X_train {X_train.shape}, y_train {y_train.shape}")
        
        # Train AI models
        models_to_train = args.models if 'all' not in args.models else ['neural_network', 'lstm', 'ensemble']
        
        trained_models = {}
        results = {}
        evaluator = ModelEvaluator()
        
        for model_type in models_to_train:
            logger.info(f"🚀 Training {model_type.upper()}...")
            
            if model_type == 'neural_network':
                # Train Neural Network
                nn_model = NeuralNetworkPredictor()
                history = nn_model.train(X_train, y_train)
                trained_models[model_type] = nn_model
                
                # Evaluate
                predictions = nn_model.predict(X_test)
                metrics = evaluator.evaluate_regression(nn_model, X_test, y_test)
                map_score = evaluator.calculate_map_score(predictions, y_test)
                metrics['map_score'] = map_score
                results[model_type] = metrics
                
                # Save model
                nn_model.save_model('neural_network_advanced')
                
                logger.info(f"✅ Neural Network - mAP: {map_score:.1%}, Loss: {history['loss'][-1]:.4f}")
                
            elif model_type == 'lstm':
                # Train LSTM
                lstm_model = LSTMPredictor(sequence_length=24)
                history = lstm_model.train(X, y)
                trained_models[model_type] = lstm_model
                
                logger.info(f"✅ LSTM - Final Loss: {history['loss'][-1]:.4f}")
                
            elif model_type == 'ensemble':
                # Train Ensemble
                ensemble = EnsembleAIPredictor()
                ensemble_results = ensemble.train_ensemble(X_train, y_train)
                trained_models[model_type] = ensemble
                
                # Evaluate ensemble
                try:
                    predictions = ensemble.predict_ensemble(X_test)
                    metrics = {}
                    for i, target in enumerate(target_cols):
                        y_true = y_test[:, i]
                        y_pred = predictions[:, i] if predictions.ndim > 1 else predictions
                        
                        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                        metrics[f'{target}_mae'] = mean_absolute_error(y_true, y_pred)
                        metrics[f'{target}_rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
                        metrics[f'{target}_r2'] = r2_score(y_true, y_pred)
                    
                    map_score = evaluator.calculate_map_score(predictions, y_test)
                    metrics['map_score'] = map_score
                    results[model_type] = metrics
                    
                    logger.info(f"✅ AI Ensemble - mAP: {map_score:.1%}")
                    
                except Exception as e:
                    logger.warning(f"Ensemble evaluation failed: {e}")
        
        # Generate performance report
        if results:
            logger.info("\n🏆 AI MODEL PERFORMANCE REPORT")
            logger.info("=" * 60)
            
            for model_name, metrics in results.items():
                logger.info(f"\n{model_name.upper().replace('_', ' ')}")
                logger.info("-" * 30)
                
                if 'map_score' in metrics:
                    map_score = metrics['map_score']
                    status = "✅ PASS" if map_score > 0.8 else "❌ FAIL"
                    logger.info(f"mAP Score: {map_score:.1%} {status}")
                
                for key, value in metrics.items():
                    if key != 'map_score':
                        logger.info(f"{key}: {value:.4f}")
        
        # Save AI results
        ai_results_file = OUTPUT_DIR / 'ai_training_results.json'
        with open(ai_results_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = {}
            for model_name, metrics in results.items():
                json_results[model_name] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                          for k, v in metrics.items()}
            json.dump(json_results, f, indent=2)
        
        logger.info(f"\n💾 AI training results saved to {ai_results_file}")
        
        # Check performance requirements
        passing_models = [name for name, metrics in results.items() 
                         if 'map_score' in metrics and metrics['map_score'] > 0.8]
        
        if passing_models:
            logger.info(f"🎯 AI Models meeting >80% mAP requirement: {passing_models}")
        else:
            logger.warning("⚠️  No AI models achieved >80% mAP requirement")
        
        logger.info("🎉 Advanced AI training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"AI training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()