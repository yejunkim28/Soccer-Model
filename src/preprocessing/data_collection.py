import os
import warnings
import pandas as pd
import soccerdata as sd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
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

    def create_dirs(self):
        os.mkdir("/Users/lionlucky7/Desktop/Coding_Project/data")
        os.mkdir("/Users/lionlucky7/Desktop/Coding_Project/data/years")
        os.mkdir("/Users/lionlucky7/Desktop/Coding_Project/data/fbref_total_fielders")
        os.mkdir("/Users/lionlucky7/Desktop/Coding_Project/data/players")

    def individual_stats(self):
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

            total_others.to_csv(
                f"/Users/lionlucky7/Desktop/Coding_Project/data/years/fielders_{year}.csv", index=False)

            print("Data Saved for Year:", year)

    def save_total(self):
        total_df = pd.DataFrame()
        for i in range(1995, 2025):
            df = pd.read_csv(
                f"/Users/lionlucky7/Desktop/Coding_Project/data/years/fielders_{i}.csv")
            total_df = pd.concat([total_df, df], ignore_index=True, axis=0)

        total_df.to_csv(
            f"/Users/lionlucky7/Desktop/Coding_Project/data/fbref_total_fielders/total_fielders.csv", index=False)

    def individual_data_csv(self):
        Fielders = pd.read_csv(
            f"/Users/lionlucky7/Desktop/Coding_Project/data/fbref_total_fielders/total_fielders.csv", index=False)

        Fielders = Fielders.drop(columns=['Unnamed: 0'])
        Fielders = Fielders[(Fielders['season'] >= 1718) &
                            (Fielders['season'] <= 2425)]

        names = list(Fielders['player'].unique())

        for name in names[::-1]:
            df = Fielders[Fielders['player'] == name]
            df.to_csv(
                f"/Users/lionlucky7/Desktop/Coding_Project/data/players/{name}.csv", index=False)


"""
data = DataCollection()
data.individual_stats()
data.save_total()
"""
data = DataCollection()
data.save_total()
