import pandas as pd

selected_features = [
    # PLAYER INFO
    "league",
    "season",
    "team",
    "player",
    "pos",
    "age",
    "Playing Time_90s",


    "Standard_Sh/90",
    "Standard_SoT/90",
    "Standard_SoT%",
    "Standard_G/Sh",
    "Standard_G/SoT",
    "Per 90 Minutes_Gls",
    "Per 90 Minutes_Ast",
    "Per 90 Minutes_G+A",
    "Expected_G-xG",
    "Expected_A-xAG",
    "Per 90 Minutes_xG",
    "Per 90 Minutes_xAG",
    "Per 90 Minutes_xG+xAG",

    # PROGRESSION
    "Progression_PrgC",
    "Progression_PrgP",
    "Progression_PrgR",
    "PrgP",
    "Carries_PrgC",

    # PASSING
    "Short_Cmp%",
    "Medium_Cmp%",
    "Long_Cmp%",
    "1/3",
    "PPA",
    "CrsPA",

    # CREATION
    "SCA_SCA90",
    "GCA_GCA90",

    # DEFENDING
    "Tackles_TklW",
    "Challenges_Tkl%",
    "Blocks_Blocks",
    "Performance_Recov",
    "Int",     

    # DRIBBLING
    "Take-Ons_Att",
    "Take-Ons_Succ%",
    "Carries_Mis",

    # RECEIVING
    "Receiving_Rec",
    "Receiving_PrgR",

    # AERIAL
    "Aerial Duels_Won%"
]



columns_nan = [
    "Standard_Sh/90",
    "Standard_SoT/90",
    "Standard_SoT%",
    "Standard_G/Sh",
    "Standard_G/SoT",
    "Per 90 Minutes_Gls",
    "Per 90 Minutes_Ast",
    "Per 90 Minutes_G+A",
    "Expected_G-xG",
    "Expected_A-xAG",
    "Per 90 Minutes_xG",
    "Per 90 Minutes_xAG",
    "Per 90 Minutes_xG+xAG",
    "Progression_PrgC",
    "Progression_PrgP",
    "Progression_PrgR",
    "PrgP",
    "Carries_PrgC",
    "Short_Cmp%",
    "Medium_Cmp%",
    "Long_Cmp%",
    "1/3",
    "PPA",
    "CrsPA",
    "SCA_SCA90",
    "GCA_GCA90",
    "Tackles_TklW",
    "Challenges_Tkl%",
    "Int",
    "Blocks_Blocks",
    "Performance_Recov",
    "Take-Ons_Att",
    "Take-Ons_Succ%",
    "Carries_Mis",
    "Receiving_Rec",
    "Receiving_PrgR",
    "Aerial Duels_Won%"
]

df = pd.read_csv(
    '/Users/lionlucky7/Desktop/Coding_Project/data/fbref_total_fielders/total_fielders.csv')

x_features = ['age',
'Playing Time_MP',
'Playing Time_Starts',
'Playing Time_Min',
'Playing Time_90s',
'Performance_Gls',
'Performance_Ast',
'Performance_G+A',
'Performance_G-PK',
'Performance_PK',
'Performance_PKatt',
'Performance_CrdY',
'Performance_CrdR',
'Per 90 Minutes_G-PK',
'Per 90 Minutes_G+A-PK',
'90s',
'Standard_Gls',
'Standard_SoT',
'Standard_PK',
'Standard_PKatt',
'Ast',
'Playing Time_Mn/MP',
'Playing Time_Min%',
'Starts_Starts',
'Starts_Compl',
'Subs_Subs']

TARGET_COLS = ['Standard_Sh/90','Standard_SoT/90','Standard_SoT%', 'Standard_G/Sh',
       'Standard_G/SoT', 'Per 90 Minutes_Gls', 'Per 90 Minutes_Ast',
       'Per 90 Minutes_G+A', 'Expected_G-xG', 'Expected_A-xAG',
       'Per 90 Minutes_xG', 'Per 90 Minutes_xAG', 'Per 90 Minutes_xG+xAG',
       'Progression_PrgC', 'Progression_PrgP', 'Progression_PrgR', 'PrgP',
       'Carries_PrgC', 'Short_Cmp%', 'Medium_Cmp%', 'Long_Cmp%', '1/3', 'PPA',
       'CrsPA', 'SCA_SCA90', 'GCA_GCA90', 'Tackles_TklW', 'Challenges_Tkl%',
       'Int', 'Blocks_Blocks', 'Performance_Recov', 'Take-Ons_Att',
       'Take-Ons_Succ%', 'Carries_Mis', 'Receiving_Rec', 'Receiving_PrgR',
       'Aerial Duels_Won%']