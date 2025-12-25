# scripts/prepare_data.py

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model_2.src.preprocessing.preprocess import Model2Preprocessor

if __name__ == "__main__":
    # Define paths
    BASE_DIR = Path(__file__).parent.parent  # Adjust based on location
    RAW_DIR = BASE_DIR / "data" / "raw"
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    
    # Create processed directory if it doesn't exist
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = Model2Preprocessor(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        min_playing_time=5.0
        )
    
    print("Starting preprocessing pipeline...")
    print(f"Configuration: {preprocessor}")
    
    # Run the full pipeline
    try:
        df_processed = preprocessor.fit_transform()
        
        # Print summary
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        print(f"Total records: {len(df_processed):,}")
        print(f"Total columns: {len(df_processed.columns)}")
        print(f"Unique players: {df_processed['player'].nunique():,}")
        
        # Save to disk
        output_file = "preprocessed_data.csv"
        output_path = preprocessor.save(df_processed, output_file)
        print(f"\nSaved to: {output_path}")
        
    except Exception as e:
        print(f"ERROR during preprocessing: {e}")
        raise