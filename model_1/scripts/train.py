"""
Training Script for Model 1

This script trains models with different window sizes (3-10 seasons)
and saves them along with their metrics.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import importlib
from model_1.config import PROCESSED_DIR, MAIN_DIR

# Import from numbered directory
trainer_module = importlib.import_module('model_1.src.03_training.trainer')
ModelTrainer = trainer_module.ModelTrainer


def train_models():
    """Train models for different window sizes."""
    
    print("Loading processed data...")
    data_path = PROCESSED_DIR / "processed_data.csv"
    
    if not data_path.exists():
        print(f"Error: Processed data not found at {data_path}")
        print("Please run data preparation first:")
        print("  python model_1/scripts/prepare_data.py")
        return
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows")
    
    # Define target columns
    target_cols = [
        'Standard_Sh/90', 'Standard_SoT/90', 'Standard_SoT%', 'Standard_G/Sh',
        'Standard_G/SoT', 'Per 90 Minutes_Gls', 'Per 90 Minutes_Ast',
        'Per 90 Minutes_G+A', 'Expected_G-xG', 'Expected_A-xAG',
        'Per 90 Minutes_xG', 'Per 90 Minutes_xAG', 'Per 90 Minutes_xG+xAG',
        'Progression_PrgC', 'Progression_PrgP', 'Progression_PrgR', 'PrgP',
        'Carries_PrgC', 'Short_Cmp%', 'Medium_Cmp%', 'Long_Cmp%', '1/3', 'PPA',
        'CrsPA', 'SCA_SCA90', 'GCA_GCA90', 'Tackles_TklW', 'Challenges_Tkl%',
        'Int', 'Blocks_Blocks', 'Performance_Recov', 'Take-Ons_Att',
        'Take-Ons_Succ%', 'Carries_Mis', 'Receiving_Rec', 'Receiving_PrgR',
        'Aerial Duels_Won%'
    ]
    
    save_path = str(MAIN_DIR)
    exclude = ['player', 'team', 'season', 'league', 'pos']
    
    # Train models for window sizes 3-10
    print("\nTraining models for window sizes 3-10...")
    for window in range(3, 11):
        print(f"\n{'='*60}")
        print(f"Training model with window size {window}")
        print(f"{'='*60}")
        
        trainer = ModelTrainer(df, window, save_path, target_cols)
        
        # Create samples
        dataset = trainer.make_samples(df, window=window, features_exclude=exclude, target_cols=target_cols)
        
        if trainer.too_few_players:
            print(f"Not enough data for window size {window}, stopping.")
            break
        
        # Split data
        train_df, val_df, test_df = trainer.time_split(
            dataset, 
            train_until=2018, 
            val_from=2019, 
            val_until=2021, 
            test_from=2022
        )
        
        # Train model
        model, metrics_df = trainer.train(train_df, val_df)
        
        # Save model and metrics
        trainer.save(model, metrics_df)
        
        print(f"\nWindow {window} - Validation Metrics:")
        print(metrics_df.to_string(index=False))
    
    print(f"\n{'='*60}")
    print("All models trained successfully!")
    print(f"Models saved to: {MAIN_DIR / 'checkpoint'}")
    print(f"Metrics saved to: {MAIN_DIR / 'metrics'}")


if __name__ == "__main__":
    train_models()
