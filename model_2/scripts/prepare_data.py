# scripts/prepare_data.py

import sys
from pathlib import Path
import pandas as pd

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model_2.src.preprocessing.merge import Model2Merger
from model_2.src.preprocessing.preprocessing import Model2Preprocessor

def main():
    # ---- First Stage: Data Merging ----
    BASE_DIR = Path(__file__).parent.parent 
    RAW_DIR = BASE_DIR / "data" / "raw"
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    merger = Model2Merger(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        min_playing_time=5.0
    )
    
    print("Starting merging pipeline...")
    print(f"Configuration: {merger}")
    
    # Run the merging pipeline
    try:
        df_merged = merger.fit_transform()
        
        # Print summary
        print("\n" + "="*60)
        print("MERGING COMPLETE")
        print("="*60)
        
        # Save merged data with the EXACT filename that preprocessing.py expects
        merged_file = "preprocessed_data.csv"  # This matches what preprocessing.py looks for
        merged_path = merger.save(df_merged, merged_file)
        print(f"Merged data saved to: {merged_path}")
        
    except Exception as e:
        print(f"ERROR during merging: {e}")
        raise

    # ---- Second Stage: Data Preprocessing ----
    print("\nStarting preprocessing pipeline...")
    
    # Create final directory
    FINAL_DIR = BASE_DIR / "data" / "final"
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Use the exact paths that Model2Preprocessor expects
    input_file = PROCESSED_DIR / "preprocessed_data.csv"
    output_file = FINAL_DIR / "final.csv"
    
    # Verify input file exists
    if not input_file.exists():
        print(f"ERROR: Input file does not exist: {input_file}")
        print(f"Available files in {PROCESSED_DIR}:")
        if PROCESSED_DIR.exists():
            for file in PROCESSED_DIR.iterdir():
                print(f"  - {file.name}")
        raise FileNotFoundError(f"Required input file not found: {input_file}")

    try:
        # Initialize preprocessor with correct paths
        preprocessor = Model2Preprocessor(
            raw_dir=input_file,      # This should match preprocessing.py expectations
            processed_dir=output_file # This should match preprocessing.py expectations
        )
        
        # Run the preprocessing
        preprocessor.save()  # This will load, process, and save the data
        
        # Load the final result to verify
        final_df = pd.read_csv(output_file)
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        print(f"Final data saved to: {output_file}")
        print(f"Final dataset shape: {final_df.shape}")
        print(f"Final dataset columns: {len(final_df.columns)}")
        
    except Exception as e:
        print(f"ERROR during preprocessing: {e}")
        # Print more detailed error information
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()