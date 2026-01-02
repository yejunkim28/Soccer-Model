# scripts/prepare_data.py

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model_2.src.preprocessing.merge import Model2Merger
from model_2.src.preprocessing.preprocessing import Model2Preprocessor

if __name__ == "__main__":
    # ---- Preprocessing ----
    BASE_DIR = Path(__file__).parent.parent 
    RAW_DIR = BASE_DIR / "data" / "raw"
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    merger = Model2Merger(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        min_playing_time=5.0
        )
    
    print("Starting preprocessing pipeline...")
    print(f"Configuration: {merger}")
    
    # Run the full pipeline
    try:
        df_processed = merger.fit_transform()
        
        # Print summary
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        
        # Save to disk
        output_file = "preprocessed_data.csv"
        output_path = merger.save(df_processed, output_file)
        print(f"\nSaved to: {output_path}")
        
    except Exception as e:
        print(f"ERROR during preprocessing: {e}")
        raise



    # ---- Preprocessing ----
    SCRIPT_DIR = Path(__file__).resolve().parent
    MODEL_DIR = SCRIPT_DIR.parent.parent

    deleted_columns = [
    'real_face',
    'joined',
    'traits',
    'season_code',
    'wage',
    'club_kit_number',
    'attacking_work_rate',
    'traits.1',
    'value',
    'playstyles_+',
    'playstyles',
    'height',
    'id',
    'club_position',
    'acceleration_type',
    'defensive_work_rate',
    'body_type',
    'A-xAG',
    'weight',
    'loan_date_end',
    'Rec',
    'team_contract',
    'release_clause', 
    'Standard_G/SoT', 
    'Medium_Cmp%', 
    'Long_Cmp%', 
    'Challenges_Tkl%', 
    'Take-Ons_Succ%', 
    'Take-Ons_Tkld%', 
    'Starts_Mn/Start', 
    'Subs_Mn/Sub', 
    'Aerial_Duels_Won%', 
    'Tackles_Tkl%',
    "Short_Cmp%",
    "Total_Cmp%",
    "gk_diving"
    ,"age_y"]

    delete_subset = ["nation", 
    "born",
    'Team_Success_PPM', 
    "Team_Success_On-Off", 
    'Team_Success_(xG)_On-Off',
    "Performance_2CrdY"]

    object_to_num = ['overall_rating',
    'potential',
    'crossing',
    'finishing',
    'heading_accuracy',
    'short_passing',
    'volleys',
    'dribbling',
    'curve',
    'fk_accuracy',
    'long_passing',
    'ball_control',
    'acceleration',
    'sprint_speed',
    'agility',
    'reactions',
    'balance',
    'shot_power',
    'jumping',
    'stamina',
    'strength',
    'long_shots',
    'aggression',
    'interceptions',
    'attack_position',
    'vision',
    'penalties',
    'composure',
    'defensive_awareness',
    'standing_tackle']

    raw_dir = MODEL_DIR / "data" / "processed" / "preprocessed_data.csv"
    processed_dir = MODEL_DIR / "data" / "final" / "final.csv"

    preprocessor = Model2Preprocessor(
        raw_dir=raw_dir,
        processed_dir=processed_dir)
    
    preprocessor.save()