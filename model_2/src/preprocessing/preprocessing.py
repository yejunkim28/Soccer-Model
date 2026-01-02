import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent
# Navigate to model_2 directory
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

class Model2Preprocessor:
    def __init__(self, raw_dir, processed_dir):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
    
    def load_df(self, path):
        df = pd.read_csv(path)
        return df

    def parse_monetary_value(row): 
        row = row.split('€')[1]
        if "M" in row:
            value = float(row.replace("M", "")) * 1_000_000
        elif "K" in row:
            value = float(row.replace("K", "")) * 1_000
        else:
            value = float(row)

        return value

    def extract_primary_position(row):
        if "," in row:
            row = row.split(",")[0]
        
        return row.strip()
    
    def clean_data(self, df):        
        df.columns = df.columns.str.replace(" / ", "_").str.replace(" & ", "_").str.replace(" ", "_")

        df['height(cm)'] = df["height"].str.split("cm").str[0].astype(int)
        df['weight(kg)'] = df["weight"].str.split("kg").str[0].astype(int) 

        df['foot'] = df['foot'].map({"Left": 1, "Right": 2})

        df['value(€)'] = df['value'].apply(Model2Preprocessor.parse_monetary_value)
        df['wage(€)'] = df['wage'].apply(Model2Preprocessor.parse_monetary_value)

        df['release_clause(€)'] = df['release_clause'].apply(Model2Preprocessor.parse_monetary_value)
        df["Tackles_Tkl%"] = df["Tackles_TklW"] / df["Tackles_Tkl"]
        df["pos"] = df["pos"].apply(Model2Preprocessor.extract_primary_position)
        df = df[df['season'] >= 2018]

        df.drop(columns=deleted_columns, inplace=True)

        deleted_threshold = df.shape[1] - 3
        df.dropna(thresh=deleted_threshold, axis=0, inplace=True)
        df.dropna(subset=delete_subset, inplace=True)
        
        df['potential'] = df['potential'].apply(lambda x: x.split("-")[0] if "-" in str(x) else x)
        
        df[object_to_num] = df[object_to_num].astype(float)

        col_order = ['player', 'league', 'team', 'nation','pos', "best_position", 'age_x', 'born', 'season'] + [col for col in df.columns if col not in ['player', 'league', 'team', 'nation', 'pos', 'age_x', 'season']]
        df = df[col_order]
        
        df.rename(columns={'age_x': 'age', 'pos':'general_position'}, inplace=True)
        df = df.loc[:, ~df.columns.duplicated()]
        
        df.sort_values(by=['player', 'season'], ascending=True, inplace=True)
        
        player_counts = df['player'].value_counts()
        players_to_keep = player_counts[player_counts >= 3].index
        
        df = df[df['player'].isin(players_to_keep)]
        print("="*60)
        print(f"Preprocessing Done")
        print("="*60)
        

        return df
    
    def save(self):
        df = Model2Preprocessor.load_df(self, self.raw_dir)
        df = Model2Preprocessor.clean_data(self, df)
        df.to_csv(self.processed_dir, index=False)
        print("="*60)
        print(f"Data saved to {self.processed_dir}")
        print("="*60)

if __name__ == "__main__":    
    preprocessor = Model2Preprocessor(
        raw_dir=raw_dir,
        processed_dir=processed_dir)
    
    preprocessor.save()