"""
Prepare Data Script for Model 1

This script loads raw data, processes it, and saves the cleaned data
for training.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from model_1.config import RAW_TOTAL_DIR, PROCESSED_DIR


def prepare_data():
    """Load raw data and prepare it for training."""
    
    print("Loading raw data...")
    raw_data_path = RAW_TOTAL_DIR / "total_fielders.csv"
    
    if not raw_data_path.exists():
        print(f"Error: Raw data not found at {raw_data_path}")
        print("Please run data collection first:")
        print("  python -m model_1.src.01_data_collection.data_collection")
        return
    
    df = pd.read_csv(raw_data_path)
    print(f"Loaded {len(df)} rows")
    
    # TODO: Add your preprocessing logic here
    # For now, just save as-is
    print("Processing data...")
    
    # Ensure processed directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save processed data
    output_path = PROCESSED_DIR / "processed_data.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Processed data saved to: {output_path}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    prepare_data()