import pandas as pd
import numpy as np
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"


class Model2Merger:
    def __init__(self, raw_dir, processed_dir, min_playing_time=5.0):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.min_playing_time = min_playing_time
    

    ### LOAD DATA ###
    def load_data(self, raw_dir):
        """Load data with robust CSV parsing"""
        # Try multiple parsing strategies for fbref_second.csv
        fbref_file = raw_dir / "fbref_second.csv"
        
        try:
            # First attempt: Standard read
            df_fbref = pd.read_csv(fbref_file)
        except pd.errors.ParserError:
            try:
                # Second attempt: More flexible parsing
                df_fbref = pd.read_csv(
                    fbref_file,  
                    engine='python'      
                )
            except:
                try:
                    # Third attempt: Most flexible parsing
                    df_fbref = pd.read_csv(
                        fbref_file,
                        sep=',',
                        quotechar='"',
                        skipinitialspace=True,

                        engine='python',
                        on_bad_lines='skip'  # For newer pandas versions
                    )
                except:
                    # Last resort: Manual inspection needed
                    print(f"CRITICAL: Cannot parse {fbref_file}")
                    print("Manual inspection required. Check line 4798 and surrounding lines.")
                    raise
        
        # Load SoFIFA data (usually more stable)
        df_sofifa = pd.read_csv(raw_dir / "sofifa_players.csv")
        
        print(f"Loaded FBRef data: {df_fbref.shape}")
        print(f"Loaded SoFIFA data: {df_sofifa.shape}")
        
        return df_sofifa, df_fbref
    
    
    ### SOFIFA PREPROCESSING FUNCTIONS ###

    @staticmethod
    def extract_name_from_position(row):
        pattern = r'[A-Z]{2,}'
        match = re.search(pattern, row)
        if match:
            location = match.start()
            name = row[:location].strip()
            
            return name
        else:
            return None
        


    def preprocess_sofifa(self, df):
        df.drop_duplicates(inplace=True)
        df.columns = df.columns.str.strip().str.lower()

        df['name'] = df['name'].apply(Model2Merger.extract_name_from_position)
        df["name"] = df['name'].str.lower()        
        
        return df
    
    
    ### FBREF PREPROCESSING FUNCTIONS ###

    @staticmethod
    def parse_season_code(season):
        season_str = str(season).zfill(4)
        last_two = int(season_str[2:])
        if last_two >= 90:
            return 1900 + last_two
        else:
            return 2000 + last_two
    
    @staticmethod
    def format_player_name(row):
        row = row.lower()
        names = row.split(" ")
        if len(names) >= 2:
            first_name = names[0]
            last_name = names[-1]
            new_name = f"{first_name[0]}. {last_name}"
            return new_name
        else:
            return row


    def preprocess_fbref(self, df):
        df['season'] = df['season'].apply(Model2Merger.parse_season_code) 
        df['player'] = df['player'].apply(Model2Merger.format_player_name)
        return df
    
    ### MERGE DATAFRAMES ###
    def merge_dataframes(self):
        self.df_sofifa.rename(columns={"name": "player"}, inplace=True)
        merged_df = pd.merge(self.df_sofifa  , self.df_fbref, on=['player', 'season'])
        merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('unnamed', case=False)]
        
        return merged_df

    def fit_transform(self):
        self.df_sofifa, self.df_fbref = self.load_data(self.raw_dir)
        self.df_sofifa = self.preprocess_sofifa(self.df_sofifa)
        self.df_fbref = self.preprocess_fbref(self.df_fbref)
        
        merged_df = self.merge_dataframes()
        return merged_df
    
    def save(self, df, filename):
        output_path = self.processed_dir / filename
        df.to_csv(output_path, index=False)
        return output_path


