import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time as time_module
from xgboost.callback import TrainingCallback


class TimerCallback(TrainingCallback):
    def __init__(self):
        self.start_time = None
    def before_training(self, model):
        self.start_time = time_module.time()
        print("Training started...")
        return model
    def after_iteration(self, model, epoch, evals_log):
        elapsed = time_module.time() - self.start_time
        print(f"\rElapsed time: {elapsed:.1f} seconds", end="")
        return False


class ModelTrainer:
    def __init__(self, df, window, save_path, target_cols):
        self.df = df
        self.window = window
        self.save_path = save_path
        self.target_cols = target_cols
        self.too_few_players = False

    def make_samples(self, df, window, features_exclude, target_cols):

        df.sort_values(['player', 'season'], inplace=True)

        rows = []
        
        for player, g in df.groupby('player'):
            g = g.reset_index(drop=True)
            if len(g) <= window:
                continue
            for i in range(window, len(g)):
                past = g.loc[i-window:i-1].copy()
                target = g.loc[i, target_cols].copy()
                
                flat = {}
                for j, (_, r) in enumerate(past.iterrows(), start=1):
                    suffix = f"_t-{window-j+1}"
                    for c in r.index:
                        if c in features_exclude:
                            continue
                        flat[c+suffix] = r[c]
                
                flat['player'] = player
                flat['season_target'] = g.loc[i, 'season']
                flat['age_at_target'] = g.loc[i-1, 'age'] + 1
                
                flat['team_last'] = g.loc[i-1, 'team']
                flat["league_last"] = g.loc[i-1, 'league']

                for tcol in target_cols:
                    flat[tcol] = target[tcol]
                
                rows.append(flat)


        if len(rows) < 100:
            print(f"Warning: Only {len(rows)} unique players found, which is less than 100. Exiting make_samples early.")
            self.too_few_players = True
            return self.too_few_players
        else:
            self.too_few_players = False

        print(f"Window_{self.window} Building Dataset Done")
        return pd.DataFrame(rows)


    def time_split(self, df, train_until=2018, val_from=2019, val_until=2021, test_from=2022):
        train = df[df['season_target'] <= train_until]
        val = df[(df['season_target'] >= val_from) & (df['season_target'] <= val_until)]
        test = df[df['season_target'] >= test_from]
        print(f"Window_{self.window} Creating Train & Valid Dataset Done")
        return train, val, test

    def build_preprocessor(self, X_cols):
        last_cat_cols = []
        if 'league_last' in X_cols:
            last_cat_cols.append('league_last')
        last_cat_cols += [c for c in X_cols if c.endswith('_t-1') and 'pos' in c]

        num_cols = [c for c in X_cols if c not in last_cat_cols]

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), last_cat_cols)
        ])
        
        print(f"Window_{self.window} Build Preprocessor Done")
        return preprocessor
    
    def build_model(self, preprocessor):
        model = Pipeline([
            ('pre', preprocessor),
            ('est', MultiOutputRegressor(XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, max_depth=6)))
        ])
        
        print(f"Window_{self.window} Build Model Done")
        return model
    
    def train(self, train_df, val_df):
        import threading
        import time

        id_cols = ['player','season_target','team_last','league_last']
        y_cols = self.target_cols
        
        X_cols = [c for c in train_df.columns if (c not in id_cols + y_cols)]
        
        preprocessor = self.build_preprocessor(X_cols)
        
        model = self.build_model(preprocessor)

        X_train = train_df[X_cols]
        y_train = train_df[y_cols]
        X_val = val_df[X_cols]
        y_val = val_df[y_cols]

        stop_flag = threading.Event()

        def timer():
            seconds = 0
            while not stop_flag.is_set():
                print(f"\rTraining... elapsed time: {seconds} seconds", end="")
                time.sleep(1)
                seconds += 1
            print()

        t = threading.Thread(target=timer)
        t.start()

        model.fit(X_train, y_train)

        stop_flag.set()
        t.join()

        y_pred = model.predict(X_val)

        metrics_list = []
        for i, col in enumerate(y_cols):
            mae = mean_absolute_error(y_val.iloc[:, i], y_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_val.iloc[:, i], y_pred[:, i]))
            r2 = r2_score(y_val.iloc[:, i], y_pred[:, i])
            
            metrics_list.append({
                "column":col,
                "e":mae,
                "RMSE":rmse,
                "R2": r2
            })
            
        metrics_df = pd.DataFrame(metrics_list)
        
        
        print(f"Window_{self.window} Training Done")
        return model, metrics_df

    def save(self, model, metrics_df):
        os.makedirs(f"{self.save_path}/checkpoint", exist_ok=True)
        os.makedirs(f"{self.save_path}/metrics", exist_ok=True)

        model_path = os.path.join(self.save_path, "checkpoint", f"model_window{self.window}.pkl")
        metrics_path = os.path.join(self.save_path, "metrics", f"model_window{self.window}_metrics.csv")

        joblib.dump(model, model_path)
        metrics_df.to_csv(metrics_path, index=False)

        print(f"Window_{self.window} Saving Model & Metrics Done")
        return model_path, metrics_path


if __name__ == "__main__":
    # This code only runs when script is executed directly
    import sys
    from pathlib import Path
    
    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from model_1.config import PROCESSED_DIR, MAIN_DIR
    
    df = pd.read_csv(PROCESSED_DIR / "processed_data.csv")
    
    SAVE_PATH = str(MAIN_DIR)
    
    for WINDOW in range(3, 11):
        trainer = ModelTrainer(df, WINDOW, SAVE_PATH, ['Standard_Sh/90', 'Standard_SoT/90', 'Standard_SoT%', 'Standard_G/Sh',
            'Standard_G/SoT', 'Per 90 Minutes_Gls', 'Per 90 Minutes_Ast',
            'Per 90 Minutes_G+A', 'Expected_G-xG', 'Expected_A-xAG',
            'Per 90 Minutes_xG', 'Per 90 Minutes_xAG', 'Per 90 Minutes_xG+xAG',
            'Progression_PrgC', 'Progression_PrgP', 'Progression_PrgR', 'PrgP',
            'Carries_PrgC', 'Short_Cmp%', 'Medium_Cmp%', 'Long_Cmp%', '1/3', 'PPA',
            'CrsPA', 'SCA_SCA90', 'GCA_GCA90', 'Tackles_TklW', 'Challenges_Tkl%',
            'Int', 'Blocks_Blocks', 'Performance_Recov', 'Take-Ons_Att',
            'Take-Ons_Succ%', 'Carries_Mis', 'Receiving_Rec', 'Receiving_PrgR',
            'Aerial Duels_Won%'])
        
        exclude = ['player', 'team', 'season', 'league', 'pos']
        dataset = trainer.make_samples(df, window=WINDOW, features_exclude=exclude, target_cols=trainer.target_cols)
        
        if trainer.too_few_players:
            break
        
        train_df, val_df, test_df = trainer.time_split(dataset, train_until=2018, val_from=2019, val_until=2021, test_from=2022)
        
        model, metrics_df = trainer.train(train_df, val_df)
        
        trainer.save(model, metrics_df)
    
    print("All models trained successfully!")
print()