"""Advanced AI models for microclimate prediction including neural networks and deep learning."""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, Dict, Any
import joblib
from pathlib import Path

from config import MODELS_DIR, MODEL_CONFIG, FEATURE_CONFIG

logger = logging.getLogger(__name__)

class NeuralNetworkPredictor:
    """Advanced neural network for microclimate prediction."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.target_columns = FEATURE_CONFIG['target_columns']
        self.history = None
        
    def build_model(self, input_shape: int) -> keras.Model:
        """Build advanced neural network architecture."""
        
        model = keras.Sequential([
            # Input layer with normalization
            layers.Dense(256, activation='relu', input_shape=(input_shape,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Hidden layers with residual connections
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            
            # Output layer for multi-target regression
            layers.Dense(len(self.target_columns), activation='linear')
        ])
        
        # Advanced optimizer with learning rate scheduling
        optimizer = keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train the neural network with advanced techniques."""
        
        logger.info("Training Advanced Neural Network...")
        
        # Create validation split if not provided
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, 
                test_size=0.2,
                random_state=MODEL_CONFIG['random_state']
            )
        
        # Build model
        self.model = self.build_model(X_train.shape[1])
        
        # Advanced callbacks
        callbacks_list = [
            # Early stopping with patience
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            
            # Model checkpointing
            callbacks.ModelCheckpoint(
                MODELS_DIR / 'best_neural_network.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=32,
            callbacks=callbacks_list,
            verbose=1
        )
        
        logger.info("Neural network training completed")
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained neural network."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X, verbose=0)
    
    def save_model(self, model_name: str = 'neural_network'):
        """Save the trained neural network."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Save model
        model_path = MODELS_DIR / f"{model_name}.h5"
        self.model.save(model_path)
        
        # Save scaler
        scaler_path = MODELS_DIR / f"{model_name}_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"Neural network saved as {model_name}")
    
    def load_model(self, model_name: str = 'neural_network'):
        """Load a trained neural network."""
        model_path = MODELS_DIR / f"{model_name}.h5"
        scaler_path = MODELS_DIR / f"{model_name}_scaler.pkl"
        
        if model_path.exists():
            self.model = keras.models.load_model(model_path)
            logger.info(f"Neural network loaded from {model_name}")
        
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded for {model_name}")

class LSTMPredictor:
    """LSTM neural network for time series prediction."""
    
    def __init__(self, sequence_length: int = 24):
        self.model = None
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.target_columns = FEATURE_CONFIG['target_columns']
    
    def create_sequences(self, data: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(targets[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build LSTM model for time series prediction."""
        
        model = keras.Sequential([
            # LSTM layers
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.1),
            
            layers.Dense(32, activation='relu'),
            
            # Output layer
            layers.Dense(len(self.target_columns), activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train LSTM model."""
        
        logger.info("Training LSTM Neural Network...")
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X, y)
        
        # Split data
        split_idx = int(0.8 * len(X_seq))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        # Build model
        self.model = self.build_lstm_model((self.sequence_length, X.shape[1]))
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=16,
            callbacks=[
                callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(patience=8, factor=0.5)
            ],
            verbose=1
        )
        
        logger.info("LSTM training completed")
        return history.history

class EnsembleAIPredictor:
    """Ensemble of multiple AI models for robust predictions."""
    
    def __init__(self):
        self.neural_net = NeuralNetworkPredictor()
        self.lstm = LSTMPredictor()
        self.models = {}
        self.weights = {}
        
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train ensemble of AI models."""
        
        logger.info("Training AI Ensemble...")
        
        results = {}
        
        # Train Neural Network
        try:
            nn_history = self.neural_net.train(X_train, y_train)
            self.models['neural_network'] = self.neural_net
            results['neural_network'] = nn_history
            logger.info("✅ Neural Network trained successfully")
        except Exception as e:
            logger.error(f"Neural Network training failed: {e}")
        
        # Train LSTM (if enough sequential data)
        if len(X_train) > 50:
            try:
                lstm_history = self.lstm.train(X_train, y_train)
                self.models['lstm'] = self.lstm
                results['lstm'] = lstm_history
                logger.info("✅ LSTM trained successfully")
            except Exception as e:
                logger.error(f"LSTM training failed: {e}")
        
        # Calculate ensemble weights based on validation performance
        self._calculate_weights(X_train, y_train)
        
        return results
    
    def _calculate_weights(self, X: np.ndarray, y: np.ndarray):
        """Calculate ensemble weights based on model performance."""
        
        # Simple equal weighting for now
        num_models = len(self.models)
        if num_models > 0:
            weight = 1.0 / num_models
            for model_name in self.models.keys():
                self.weights[model_name] = weight
    
    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        
        if not self.models:
            raise ValueError("No models trained. Call train_ensemble() first.")
        
        predictions = []
        weights = []
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'neural_network':
                    pred = model.predict(X)
                elif model_name == 'lstm' and len(X) > model.sequence_length:
                    # For LSTM, we need sequences
                    X_seq, _ = model.create_sequences(X, np.zeros((len(X), len(FEATURE_CONFIG['target_columns']))))
                    if len(X_seq) > 0:
                        pred = model.model.predict(X_seq[-1:], verbose=0)
                    else:
                        continue
                else:
                    continue
                
                predictions.append(pred)
                weights.append(self.weights[model_name])
                
            except Exception as e:
                logger.warning(f"Prediction failed for {model_name}: {e}")
        
        if not predictions:
            raise ValueError("No models could make predictions")
        
        # Weighted average of predictions
        weighted_pred = np.average(predictions, axis=0, weights=weights)
        return weighted_pred

def create_advanced_ai_system():
    """Create and return advanced AI prediction system."""
    
    logger.info("🤖 Initializing Advanced AI Prediction System")
    
    # Create ensemble system
    ai_system = EnsembleAIPredictor()
    
    logger.info("✅ Advanced AI System Ready")
    logger.info("   - Neural Network with BatchNorm & Dropout")
    logger.info("   - LSTM for time series patterns")
    logger.info("   - Ensemble prediction capabilities")
    
    return ai_system