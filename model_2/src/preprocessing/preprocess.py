import pandas as pd
import numpy as np
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"


class Model2Preprocessor:
    def __init__(self, raw_dir, processed_dir, min_playing_time=5.0):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.min_playing_time = min_playing_time
    

    ### LOAD DATA ###
    def load_data(self, raw_dir):
        df_sofifa = pd.read_csv(raw_dir / "sofifa_players_all_years.csv",
                        low_memory=False)
        
        df_fbref = pd.read_csv(raw_dir / "fbref_players_all.csv")
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
        
    def _clean_sofifa_data(self, df):
        df.columns = df.columns.str.lower()
        df.columns = df.columns.str.replace(" / ", "_").str.replace(" & ", "_").str.replace(" ", "_")
    
        df['height(cm)'] = df["height"].str.split("cm").str[0].astype(int)
        df['weight(kg)'] = df["weight"].str.split("kg").str[0].astype(int) 
    
        df['foot'] = df['foot'].map({"Left": 1, "Right": 2})
    
    def extract_team(self, df):
        df[['start_part', 'end_year']] = df['team_contract'].str.split(' ~ ', expand=True)
        df[['team', 'start_year']] = df['start_part'].str.extract(r'([A-Za-z]+)(\d{4})')
    
    def extract_player_name(self, df):
        df['name'] = df['name'].apply(Model2Preprocessor.extract_name_from_position)
        df["name"] = df['name'].str.lower()
    
    def filter_goalkeepers(self, df):
        df = df[df['best_position'] != "GK"].copy()

    @staticmethod
    def parse_monetary_value(row): 
        row = row.split('€')[1]
        if "M" in row:
            value = float(row.replace("M", "")) * 1_000_000
        elif "K" in row:
            value = float(row.replace("K", "")) * 1_000
        else:
            value = float(row)

        return value

    def preprocess_sofifa(self, df):
        df.drop_duplicates(inplace=True)
        df.columns = df.columns.str.strip()
        self._clean_sofifa_data(df)
        
        self.extract_team(df)
        self.extract_player_name(df)
        self.filter_goalkeepers(df)
        
        df['value(€)'] = df['value'].apply(Model2Preprocessor.parse_monetary_value)
        df['wage(€)'] = df['wage'].apply(Model2Preprocessor.parse_monetary_value)
        df['release_clause(€)'] = df['release_clause'].apply(Model2Preprocessor.parse_monetary_value)
        
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

    @staticmethod
    def extract_primary_position(row):
        if "," in row:
            row = row.split(",")[0]
        
        return row.strip()
    
    def preprocess_fbref(self, df):
        
        df.columns = df.columns.str.strip()
        df['season'] = df['season'].apply(Model2Preprocessor.parse_season_code)
        
        df = df[df['season'] >= 2007].copy()
        
        df["Tackles_Tkl%"] = df["Tackles_TklW"] / df["Tackles_Tkl"]
        
        df['player'] = df['player'].apply(Model2Preprocessor.format_player_name)
        df["pos"] = df["pos"].apply(Model2Preprocessor.extract_primary_position)
        
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

