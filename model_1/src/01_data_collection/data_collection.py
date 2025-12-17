import warnings
import pandas as pd
import soccerdata as sd
from src.config import YEARS_DATA_DIR, FBREF_TOTAL_DIR, PLAYERS_DATA_DIR, TOTAL_FIELDERS_PATH

warnings.filterwarnings("ignore")

print("HELLO")

class DataCollection:
    list_stats = ["standard", "shooting", "passing",
                  "passing_types", "goal_shot_creation", "defense", "possession", "playing_time", "misc"]
    years = list(range(1995, 2025))
    league = 'Big 5 European Leagues Combined'

    def __init__(self):
        self.years = DataCollection.years
        self.list_stats = DataCollection.list_stats
        self.league = DataCollection.league

    def individual_stats(self):
        """Collect individual player statistics for each year."""
        for year in self.years:

            fbref = sd.FBref(
                leagues=[self.league], seasons=[year])

            total_others = pd.DataFrame()

            for i in range(len(self.list_stats)):
                stat = self.list_stats[i]

                pl = fbref.read_player_season_stats(stat_type=stat)
                pl.columns = ['_'.join(col).strip() if col[1] else col[0]
                              for col in pl.columns.values]
                pl = pl.reset_index()

                others = pl[pl['pos'] != "GK"]

                if i == 0:
                    total_others = others
                    print("Data Collected:", stat)
                    continue

                others_columns = [
                    'player'] + [col for col in others.columns if col not in total_others.columns and col != 'player']
                others = others[others_columns]
                total_others = total_others.merge(
                    others, how='left', on=['player'])
                total_others = total_others[total_others['team'].notna()]

                print("Data Collected:", stat)

            total_others = total_others.drop_duplicates(
                subset=['player', 'season']).reset_index(drop=True)

            print("All Individual Data Collected")

            # Use config path instead of hard-coded path
            output_path = YEARS_DATA_DIR / f"fielders_{year}.csv"
            total_others.to_csv(output_path, index=False)

            print("Data Saved for Year:", year)

    def save_total(self):
        """Combine all yearly data into one total file."""
        total_df = pd.DataFrame()
        
        for i in range(1995, 2025):
            file_path = YEARS_DATA_DIR / f"fielders_{i}.csv"
            df = pd.read_csv(file_path)
            total_df = pd.concat([total_df, df], ignore_index=True, axis=0)

        total_df.to_csv(TOTAL_FIELDERS_PATH, index=False)
        print(f"Total data saved to: {TOTAL_FIELDERS_PATH}")


if __name__ == "__main__":
    data = DataCollection()
    data.save_total()
