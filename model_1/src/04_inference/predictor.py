import pandas as pd
import joblib
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from model_1.variables import TARGET_COLS


class ModelImplementer:
    def __init__(self, df, save_path, target_cols):
        self.df = df
        self.save_path = save_path
        self.target_cols = target_cols
    
    def predict_single_player(self, df_player, model, window):
        g = df_player.sort_values("season")
        if len(g) <= window:
            return None

        past = g.tail(window).copy()

        flat = {}
        features_exclude = ['player','team','season']

        for j, (_, r) in enumerate(past.iterrows(), start=1):
            suffix = f"_t-{window-j+1}"
            for c in r.index:
                if c in features_exclude:
                    continue
                flat[c + suffix] = r[c]

        last_row = past.iloc[-1]
        flat["season_target"] = int(last_row["season"]) + 1
        flat["age_at_target"] = last_row["age"] + 1
        flat["team_last"] = last_row["team"]
        flat["league_last"] = last_row["league"]

        X = pd.DataFrame([flat])
        pred = model.predict(X)[0]

        return dict(zip(self.target_cols, pred))

    def load_models(self):
        model_cache = {}
        for w in range(3, 11):
            path = os.path.join(self.save_path, "checkpoint", f"model_window{w}.pkl")
            if os.path.exists(path):
                model_cache[w] = joblib.load(path)
                
        print("Loaded models:", sorted(model_cache.keys()))
        return model_cache

    def predict_all_players(self, model_cache):
        predictions = []

        for player_name, df_player in self.df.groupby("player"):
            if df_player['season'].max() != self.df['season'].max():
                continue

            df_player = df_player.sort_values("season")
            n_seasons = len(df_player)

            if n_seasons < 3:
                continue

            window = max(3, min(n_seasons - 1, 10))
            if window not in model_cache:
                continue
            
            model = model_cache[window]

            pred = self.predict_single_player(df_player, model, window)
            if pred is None:
                continue

            pred["player"] = player_name
            pred["window_used"] = window
            pred["season_predicted"] = df_player["season"].max() + 1

            predictions.append(pred)


        pred_df = pd.DataFrame(predictions)

        cols = ['player', 'season_predicted', 'window_used'] + [c for c in pred_df.columns if c not in ["player", "window_used", "season_predicted"]]
        pred_df = pred_df[cols]
        
        
        print("End of Prediction of 2025")
        return pred_df

    def save(self, pred_df, filename="player_predictions_2025.csv"):
        os.makedirs("../outputs", exist_ok=True)
        path = os.path.join("../outputs", filename)
        pred_df.to_csv(path, index=False)
        return path


if __name__ == "__main__":
    # Example usage - update path to use config
    from model_1.config import PROCESSED_DIR, PREDICTIONS_DIR, MAIN_DIR
    
    df = pd.read_csv(PROCESSED_DIR / "processed_data.csv")
    
    model_runner = ModelImplementer(
        df=df,
        save_path=str(MAIN_DIR),
        target_cols=TARGET_COLS
    )
    
    model_cache = model_runner.load_models()
    
    pred_df = model_runner.predict_all_players(model_cache)
    
    # Save to outputs directory
    output_path = PREDICTIONS_DIR / "player_predictions_2025.csv"
    pred_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")