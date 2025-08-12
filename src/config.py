"""Configuration management for the Microclimate Prediction System."""

import os
import logging
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
DOCS_DIR = PROJECT_ROOT / "docs"

# Model configuration
MODEL_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'validation_size': 0.2,
    'target_map_threshold': 0.8,  # >80% mAP requirement
}

# Feature configuration
FEATURE_CONFIG = {
    'target_columns': ['temperature', 'humidity', 'light_intensity'],
    'sensor_columns': ['temperature', 'humidity', 'light_intensity', 'co2_level', 
                      'soil_moisture', 'air_pressure', 'wind_speed'],
    'temporal_features': True,
    'statistical_features': True,
}

def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(OUTPUT_DIR / 'training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUT_DIR, DOCS_DIR]:
    directory.mkdir(exist_ok=True)