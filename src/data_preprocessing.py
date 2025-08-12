"""Data loading, validation, and preprocessing for microclimate prediction."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from config import DATA_DIR, FEATURE_CONFIG, MODEL_CONFIG

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and basic validation of microclimate data."""
    
    def __init__(self):
        self.required_columns = FEATURE_CONFIG['sensor_columns']
    
    def load_training_data(self, file_path: str) -> pd.DataFrame:
        """Load training data from CSV file."""
        try:
            full_path = DATA_DIR / file_path if not Path(file_path).is_absolute() else Path(file_path)
            data = pd.read_csv(full_path)
            logger.info(f"Loaded training data: {data.shape}")
            
            if self.validate_data_format(data):
                return data
            else:
                raise ValueError("Data validation failed")
                
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise
    
    def load_test_data(self, file_path: str) -> pd.DataFrame:
        """Load test data from CSV file."""
        try:
            full_path = DATA_DIR / file_path if not Path(file_path).is_absolute() else Path(file_path)
            data = pd.read_csv(full_path)
            logger.info(f"Loaded test data: {data.shape}")
            
            if self.validate_data_format(data):
                return data
            else:
                raise ValueError("Data validation failed")
                
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            raise
    
    def validate_data_format(self, data: pd.DataFrame) -> bool:
        """Validate that data has required columns and proper format."""
        try:
            # Check for required columns
            missing_cols = set(self.required_columns) - set(data.columns)
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Check for numeric data types
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            non_numeric = set(self.required_columns) - set(numeric_cols)
            if non_numeric:
                logger.warning(f"Non-numeric columns found: {non_numeric}")
            
            # Check for minimum data size
            if len(data) < 10:
                logger.error("Dataset too small (minimum 10 rows required)")
                return False
            
            logger.info("Data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return False

class DataValidator:
    """Validates data quality and identifies anomalies."""
    
    def __init__(self):
        self.anomaly_threshold = 3  # Z-score threshold
    
    def identify_outliers(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Identify outliers using Z-score method."""
        outlier_mask = pd.DataFrame(False, index=data.index, columns=columns)
        
        for col in columns:
            if col in data.columns:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outlier_mask[col] = z_scores > self.anomaly_threshold
        
        return outlier_mask
    
    def validate_ranges(self, data: pd.DataFrame) -> dict:
        """Validate sensor readings are within reasonable ranges."""
        validation_results = {}
        
        # Define reasonable ranges for greenhouse sensors
        ranges = {
            'temperature': (-10, 50),  # Celsius
            'humidity': (0, 100),      # Percentage
            'light_intensity': (0, 100000),  # Lux
            'co2_level': (300, 5000),  # PPM
            'soil_moisture': (0, 100), # Percentage
            'air_pressure': (900, 1100), # hPa
            'wind_speed': (0, 20)      # m/s
        }
        
        for col, (min_val, max_val) in ranges.items():
            if col in data.columns:
                out_of_range = (data[col] < min_val) | (data[col] > max_val)
                validation_results[col] = {
                    'out_of_range_count': out_of_range.sum(),
                    'percentage': (out_of_range.sum() / len(data)) * 100
                }
        
        return validation_results

class DataPreprocessor:
    """Handles data cleaning, normalization, and preprocessing."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling missing values and outliers."""
        cleaned_data = data.copy()
        
        # Handle missing values
        cleaned_data = self.handle_missing_values(cleaned_data)
        
        # Handle outliers
        cleaned_data = self.handle_outliers(cleaned_data)
        
        logger.info(f"Data cleaned: {cleaned_data.shape}")
        return cleaned_data
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using appropriate strategies."""
        data_filled = data.copy()
        
        for col in FEATURE_CONFIG['sensor_columns']:
            if col in data_filled.columns:
                missing_count = data_filled[col].isnull().sum()
                if missing_count > 0:
                    logger.info(f"Filling {missing_count} missing values in {col}")
                    
                    # Use forward fill for time series data, then backward fill
                    data_filled[col] = data_filled[col].ffill().bfill()
                    
                    # If still missing, use median
                    if data_filled[col].isnull().sum() > 0:
                        data_filled[col] = data_filled[col].fillna(data_filled[col].median())
        
        return data_filled
    
    def handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method."""
        data_clean = data.copy()
        
        for col in FEATURE_CONFIG['sensor_columns']:
            if col in data_clean.columns:
                Q1 = data_clean[col].quantile(0.25)
                Q3 = data_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                outlier_count = ((data_clean[col] < lower_bound) | (data_clean[col] > upper_bound)).sum()
                if outlier_count > 0:
                    logger.info(f"Capping {outlier_count} outliers in {col}")
                    data_clean[col] = data_clean[col].clip(lower_bound, upper_bound)
        
        return data_clean
    
    def normalize_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Normalize features using StandardScaler."""
        data_normalized = data.copy()
        feature_cols = [col for col in FEATURE_CONFIG['sensor_columns'] if col in data.columns]
        
        if fit:
            data_normalized[feature_cols] = self.scaler.fit_transform(data[feature_cols])
            self.fitted = True
            logger.info("Features normalized and scaler fitted")
        else:
            if not self.fitted:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            data_normalized[feature_cols] = self.scaler.transform(data[feature_cols])
            logger.info("Features normalized using fitted scaler")
        
        return data_normalized

class FeatureEngineer:
    """Creates derived features from raw sensor readings."""
    
    def __init__(self):
        self.temporal_features = FEATURE_CONFIG['temporal_features']
        self.statistical_features = FEATURE_CONFIG['statistical_features']
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create all engineered features."""
        feature_data = data.copy()
        
        if self.temporal_features:
            feature_data = self.create_temporal_features(feature_data)
        
        if self.statistical_features:
            feature_data = self.create_statistical_features(feature_data)
        
        logger.info(f"Feature engineering complete: {feature_data.shape}")
        return feature_data
    
    def create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features if timestamp column exists."""
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data['hour'] = data['timestamp'].dt.hour
            data['day_of_week'] = data['timestamp'].dt.dayofweek
            data['month'] = data['timestamp'].dt.month
            data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
            
            logger.info("Temporal features created")
        
        return data
    
    def create_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features from sensor readings."""
        # Temperature-humidity interaction
        if 'temperature' in data.columns and 'humidity' in data.columns:
            data['temp_humidity_ratio'] = data['temperature'] / (data['humidity'] + 1e-6)
        
        # Rolling averages (if enough data)
        if len(data) > 10:
            for col in ['temperature', 'humidity', 'light_intensity']:
                if col in data.columns:
                    data[f'{col}_rolling_mean'] = data[col].rolling(window=5, min_periods=1).mean()
                    data[f'{col}_rolling_std'] = data[col].rolling(window=5, min_periods=1).std().fillna(0)
        
        # Fill any remaining NaN values
        data = data.fillna(0)
        
        logger.info("Statistical features created")
        return data