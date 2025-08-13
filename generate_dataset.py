#!/usr/bin/env python3
"""Generate comprehensive greenhouse microclimate dataset."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math

def generate_greenhouse_data(start_date, days=30, hours_per_day=24):
    """Generate realistic greenhouse sensor data."""
    
    # Initialize lists to store data
    data = []
    
    # Start from the given date
    current_time = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    
    # Generate data for specified number of days
    for day in range(days):
        for hour in range(hours_per_day):
            
            # Time-based variables
            hour_of_day = current_time.hour
            day_of_year = current_time.timetuple().tm_yday
            
            # Base patterns with seasonal variation
            seasonal_factor = 0.8 + 0.4 * math.sin(2 * math.pi * day_of_year / 365)
            
            # Temperature (realistic greenhouse patterns)
            # Base temperature with daily cycle
            temp_base = 22 + 6 * math.sin(2 * math.pi * (hour_of_day - 6) / 24)
            temp_seasonal = temp_base * seasonal_factor
            # Add some random variation
            temperature = temp_seasonal + np.random.normal(0, 0.8)
            temperature = max(15, min(35, temperature))  # Realistic bounds
            
            # Humidity (inversely related to temperature)
            humidity_base = 75 - 0.8 * (temperature - 20)
            humidity = humidity_base + np.random.normal(0, 3)
            humidity = max(40, min(90, humidity))  # Realistic bounds
            
            # Light intensity (solar pattern + artificial lighting)
            if 6 <= hour_of_day <= 18:  # Daylight hours
                light_base = 30000 * math.sin(math.pi * (hour_of_day - 6) / 12)
                # Add artificial supplemental lighting
                light_supplement = 5000 if 7 <= hour_of_day <= 17 else 0
                light_intensity = light_base + light_supplement + np.random.normal(0, 2000)
            else:  # Night hours
                # Minimal artificial lighting
                light_intensity = 200 + np.random.normal(0, 50)
            
            light_intensity = max(0, light_intensity)
            
            # CO2 levels (affected by photosynthesis and ventilation)
            if 6 <= hour_of_day <= 18:  # Daylight - photosynthesis reduces CO2
                co2_base = 400 - 50 * (light_intensity / 30000)
            else:  # Night - respiration increases CO2
                co2_base = 450 + 30 * np.random.random()
            
            co2_level = co2_base + np.random.normal(0, 20)
            co2_level = max(300, min(800, co2_level))
            
            # Soil moisture (irrigation cycles)
            # Simulate irrigation every 8 hours with gradual decrease
            hours_since_irrigation = hour_of_day % 8
            moisture_base = 55 - 2 * hours_since_irrigation
            soil_moisture = moisture_base + np.random.normal(0, 2)
            soil_moisture = max(30, min(70, soil_moisture))
            
            # Air pressure (realistic atmospheric variation)
            pressure_base = 1013 + 5 * math.sin(2 * math.pi * hour_of_day / 24)
            air_pressure = pressure_base + np.random.normal(0, 2)
            air_pressure = max(1000, min(1030, air_pressure))
            
            # Wind speed (ventilation system + natural)
            if 10 <= hour_of_day <= 16:  # Active ventilation during hot hours
                wind_base = 1.5 + 0.5 * (temperature - 20) / 10
            else:
                wind_base = 0.5
            
            wind_speed = wind_base + np.random.normal(0, 0.3)
            wind_speed = max(0, min(5, wind_speed))
            
            # Create data point
            data_point = {
                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                'temperature': round(temperature, 2),
                'humidity': round(humidity, 2),
                'light_intensity': round(light_intensity, 0),
                'co2_level': round(co2_level, 0),
                'soil_moisture': round(soil_moisture, 2),
                'air_pressure': round(air_pressure, 2),
                'wind_speed': round(wind_speed, 2)
            }
            
            data.append(data_point)
            
            # Move to next hour
            current_time += timedelta(hours=1)
    
    return pd.DataFrame(data)

def main():
    """Generate and save datasets."""
    
    print("🌱 Generating Comprehensive Greenhouse Dataset...")
    
    # Generate training data (30 days = 720 samples)
    print("📊 Creating training dataset (30 days)...")
    train_data = generate_greenhouse_data("2024-01-01 00:00:00", days=30)
    train_data.to_csv('data/train_large.csv', index=False)
    print(f"✅ Training data saved: {len(train_data)} samples")
    
    # Generate test data (7 days = 168 samples)
    print("📊 Creating test dataset (7 days)...")
    test_data = generate_greenhouse_data("2024-02-01 00:00:00", days=7)
    test_data.to_csv('data/test_large.csv', index=False)
    print(f"✅ Test data saved: {len(test_data)} samples")
    
    # Generate validation data (3 days = 72 samples)
    print("📊 Creating validation dataset (3 days)...")
    val_data = generate_greenhouse_data("2024-02-10 00:00:00", days=3)
    val_data.to_csv('data/validation.csv', index=False)
    print(f"✅ Validation data saved: {len(val_data)} samples")
    
    # Display dataset statistics
    print("\n📈 Dataset Statistics:")
    print("=" * 60)
    
    datasets = [
        ("Training", train_data),
        ("Test", test_data), 
        ("Validation", val_data)
    ]
    
    for name, df in datasets:
        print(f"\n{name} Dataset ({len(df)} samples):")
        print("-" * 40)
        print(df.describe().round(2))
        
        print(f"\nSample data from {name}:")
        print(df.head(3).to_string(index=False))
    
    # Show data patterns
    print("\n🔍 Data Quality Checks:")
    print("=" * 60)
    
    for name, df in datasets:
        missing = df.isnull().sum().sum()
        duplicates = df.duplicated().sum()
        print(f"{name}: Missing values: {missing}, Duplicates: {duplicates}")
    
    print("\n✅ Dataset generation complete!")
    print("📁 Files created:")
    print("   - data/train_large.csv (720 samples)")
    print("   - data/test_large.csv (168 samples)")  
    print("   - data/validation.csv (72 samples)")

if __name__ == "__main__":
    main()