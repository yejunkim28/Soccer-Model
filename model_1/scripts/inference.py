"""
Inference Script for Model 1

This script loads trained models and makes predictions for the next season.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import importlib
from model_1.config import PROCESSED_DIR, MAIN_DIR, PREDICTIONS_DIR
from model_1.variables import TARGET_COLS

# Import from numbered directory
predictor_module = importlib.import_module('model_1.src.04_inference.predictor')
ModelImplementer = predictor_module.ModelImplementer
from model_1.variables import TARGET_COLS


def run_inference():
    """Load models and make predictions."""
    
    print("Loading processed data...")
    data_path = PROCESSED_DIR / "processed_data.csv"
    
    if not data_path.exists():
        print(f"Error: Processed data not found at {data_path}")
        print("Please run data preparation first:")
        print("  python model_1/scripts/prepare_data.py")
        return
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows")
    
    # Check if models exist
    checkpoint_dir = MAIN_DIR / "checkpoint"
    if not checkpoint_dir.exists() or not any(checkpoint_dir.glob("*.pkl")):
        print(f"Error: No trained models found in {checkpoint_dir}")
        print("Please run training first:")
        print("  python model_1/scripts/train.py")
        return
    
    # Initialize predictor
    print("\nInitializing predictor...")
    model_runner = ModelImplementer(
        df=df,
        save_path=str(MAIN_DIR),
        target_cols=TARGET_COLS
    )
    
    # Load models
    print("Loading trained models...")
    model_cache = model_runner.load_models()
    
    if not model_cache:
        print("Error: No models were loaded successfully")
        return
    
    # Make predictions
    print("\nGenerating predictions for 2025...")
    pred_df = model_runner.predict_all_players(model_cache)
    
    print(f"Generated {len(pred_df)} predictions")
    
    # Save predictions
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PREDICTIONS_DIR / "player_predictions_2025.csv"
    pred_df.to_csv(output_path, index=False)
    
    print(f"\nPredictions saved to: {output_path}")
    print("\nSample predictions:")
    print(pred_df.head(10).to_string(index=False))


if __name__ == "__main__":
    run_inference()
