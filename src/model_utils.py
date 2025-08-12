"""Model training, evaluation, and management utilities."""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
import logging
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from config import MODELS_DIR, MODEL_CONFIG, FEATURE_CONFIG

logger = logging.getLogger(__name__)

if not TENSORFLOW_AVAILABLE:
    logger.warning("TensorFlow not available. Neural network training will be disabled.")

class ModelTrainer:
    """Orchestrates training process for multiple algorithms."""
    
    def __init__(self):
        self.models = {}
        self.model_metadata = {}
        self.random_state = MODEL_CONFIG['random_state']
        self.target_columns = FEATURE_CONFIG['target_columns']
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training by splitting features and targets."""
        # Define feature columns (exclude targets and non-numeric)
        feature_cols = [col for col in data.columns 
                       if col not in self.target_columns + ['timestamp']]
        
        X = data[feature_cols].values
        y = data[self.target_columns].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=MODEL_CONFIG['test_size'],
            random_state=self.random_state
        )
        
        logger.info(f"Data prepared: X_train {X_train.shape}, y_train {y_train.shape}")
        return X_train, X_test, y_train, y_test
    
    def train_regression_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train multiple regression models."""
        models = {}
        
        # Linear Regression
        logger.info("Training Linear Regression...")
        lr_model = MultiOutputRegressor(LinearRegression())
        lr_model.fit(X_train, y_train)
        models['linear_regression'] = lr_model
        
        # Random Forest
        logger.info("Training Random Forest...")
        rf_model = MultiOutputRegressor(RandomForestRegressor(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1
        ))
        rf_model.fit(X_train, y_train)
        models['random_forest'] = rf_model
        
        # Gradient Boosting
        logger.info("Training Gradient Boosting...")
        gb_model = MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=100,
            random_state=self.random_state
        ))
        gb_model.fit(X_train, y_train)
        models['gradient_boosting'] = gb_model
        
        self.models.update(models)
        logger.info(f"Trained {len(models)} regression models")
        return models
    
    def train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_val: np.ndarray = None, y_val: np.ndarray = None) -> Any:
        """Train neural network model using TensorFlow/Keras."""
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available. Cannot train neural network.")
            raise ImportError("TensorFlow is required for neural network training")
        
        logger.info("Training Neural Network...")
        
        # Create validation split if not provided
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, 
                test_size=MODEL_CONFIG['validation_size'],
                random_state=self.random_state
            )
        
        # Build model architecture
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(len(self.target_columns))  # Output layer
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ]
        )
        
        self.models['neural_network'] = model
        logger.info("Neural network training completed")
        return model
    
    def save_model(self, model_name: str, model: Any, metadata: Dict[str, Any] = None):
        """Save trained model and metadata."""
        model_path = MODELS_DIR / f"{model_name}.pkl"
        metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
        
        # Save model
        if model_name == 'neural_network' and TENSORFLOW_AVAILABLE:
            # Save Keras model
            model.save(MODELS_DIR / f"{model_name}.h5")
        else:
            # Save sklearn model
            joblib.dump(model, model_path)
        
        # Save metadata
        if metadata:
            metadata['saved_at'] = datetime.now().isoformat()
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Model {model_name} saved successfully")

class ModelEvaluator:
    """Evaluates model performance using various metrics."""
    
    def __init__(self):
        self.target_columns = FEATURE_CONFIG['target_columns']
    
    def evaluate_regression(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate regression model performance."""
        predictions = model.predict(X_test)
        
        # Calculate metrics for each target
        metrics = {}
        for i, target in enumerate(self.target_columns):
            y_true = y_test[:, i] if y_test.ndim > 1 else y_test
            y_pred = predictions[:, i] if predictions.ndim > 1 else predictions
            
            metrics[f'{target}_mae'] = mean_absolute_error(y_true, y_pred)
            metrics[f'{target}_rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics[f'{target}_r2'] = r2_score(y_true, y_pred)
        
        # Overall metrics
        metrics['overall_mae'] = mean_absolute_error(y_test, predictions)
        metrics['overall_rmse'] = np.sqrt(mean_squared_error(y_test, predictions))
        metrics['overall_r2'] = r2_score(y_test, predictions)
        
        return metrics
    
    def calculate_map_score(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Calculate mean Average Precision (mAP) score."""
        # For regression, we'll use R² as a proxy for mAP
        # In practice, mAP is typically used for classification/detection tasks
        r2_scores = []
        
        for i in range(predictions.shape[1] if predictions.ndim > 1 else 1):
            y_true = ground_truth[:, i] if ground_truth.ndim > 1 else ground_truth
            y_pred = predictions[:, i] if predictions.ndim > 1 else predictions
            r2_scores.append(max(0, r2_score(y_true, y_pred)))  # Ensure non-negative
        
        map_score = np.mean(r2_scores)
        logger.info(f"mAP score calculated: {map_score:.4f}")
        return map_score
    
    def generate_performance_report(self, results: Dict[str, Dict[str, float]]) -> str:
        """Generate comprehensive performance report."""
        report = "Model Performance Report\n"
        report += "=" * 50 + "\n\n"
        
        for model_name, metrics in results.items():
            report += f"{model_name.upper()}\n"
            report += "-" * 30 + "\n"
            
            for metric, value in metrics.items():
                report += f"{metric}: {value:.4f}\n"
            
            report += "\n"
        
        return report

class ModelRegistry:
    """Manages trained models and their metadata."""
    
    def __init__(self):
        self.models_dir = MODELS_DIR
    
    def list_models(self) -> List[str]:
        """List all available trained models."""
        model_files = list(self.models_dir.glob("*.pkl")) + list(self.models_dir.glob("*.h5"))
        return [f.stem for f in model_files]
    
    def load_model(self, model_name: str) -> Any:
        """Load a trained model."""
        pkl_path = self.models_dir / f"{model_name}.pkl"
        h5_path = self.models_dir / f"{model_name}.h5"
        
        if pkl_path.exists():
            return joblib.load(pkl_path)
        elif h5_path.exists() and TENSORFLOW_AVAILABLE:
            return keras.models.load_model(h5_path)
        else:
            raise FileNotFoundError(f"Model {model_name} not found")
    
    def load_metadata(self, model_name: str) -> Dict[str, Any]:
        """Load model metadata."""
        metadata_path = self.models_dir / f"{model_name}_metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        else:
            return {}
    
    def get_best_model(self, metric: str = 'overall_r2') -> Tuple[str, Any]:
        """Get the best performing model based on specified metric."""
        best_score = -np.inf
        best_model_name = None
        
        for model_name in self.list_models():
            metadata = self.load_metadata(model_name)
            if 'performance_metrics' in metadata and metric in metadata['performance_metrics']:
                score = metadata['performance_metrics'][metric]
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
        
        if best_model_name:
            return best_model_name, self.load_model(best_model_name)
        else:
            raise ValueError("No models found with the specified metric")