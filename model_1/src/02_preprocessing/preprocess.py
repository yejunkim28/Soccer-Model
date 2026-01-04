from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import sys
from pathlib import Path
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import model_1.variables as var

pd.set_option('display.max_columns', None)


class preprocess_data():
    def __init__(self, df=var.df, selected_features=var.selected_features, columns_nans=var.columns_nan, x_features=var.x_features):
        self.df = df
        self.selected_features = selected_features
        self.columns_nans = columns_nans
        self.x_features = x_features

    def preprocess(self):
        df_train_test = self.df[(self.df['season'] >= 1718)
                                & (self.df['season'] <= 2425)]
        df_train_test = df_train_test.select_dtypes(include='number')
        df_train_test.drop(columns=[
                           'season', 'born'], inplace=True, errors='ignore')

        return df_train_test

    def model(self):
        df_train_test = self.preprocess()

        y_targets = list(OrderedDict.fromkeys(self.columns_nans))

        df_train_test[self.x_features] = df_train_test[self.x_features].fillna(
            df_train_test[self.x_features].median())
        
        df_train_test = df_train_test.loc[:, ~df_train_test.columns.duplicated()]
        df_train_test = df_train_test.dropna(subset=y_targets)

        X = df_train_test[self.x_features]
        y = df_train_test[y_targets]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler_y = MinMaxScaler()
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_val_scaled = scaler_y.transform(y_val)

        xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            objective='reg:squarederror',
            n_jobs=-1,
            random_state=42
        )

        multi_model = MultiOutputRegressor(xgb_model)

        multi_model.fit(X_train, y_train_scaled)

        y_val_pred_scaled = multi_model.predict(X_val)

        y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled)

        y_val_pred_df = pd.DataFrame(
            y_val_pred, columns=y_targets, index=y_val.index)

        return multi_model, scaler_y, y_targets, X_val, y_val, y_val_pred_df

    def evaluate(self, y_val, y_val_pred_df, y_targets):
        for target in y_targets:
            rmse = np.sqrt(mean_squared_error(
                y_val[target], y_val_pred_df[target]))
            print(f"{target}: RMSE = {rmse:.2f}")

    def export_model(self, multi_model, scaler_y):
        import joblib
        import os
        
        os.makedirs("/Users/lionlucky7/Desktop/Coding_Project/Models", exist_ok=True)

        joblib.dump(
            multi_model, "/Users/lionlucky7/Desktop/Coding_Project/Models/multioutput_xgb_model.pkl")
        joblib.dump(
            scaler_y, "/Users/lionlucky7/Desktop/Coding_Project/Models/scaler_y.pkl")

    def apply_model(self, df, x_features, multi_model, scaler_y, y_targets, selected_features):
        df_application = df[df['season'].isin([9596, 9697, 9798, 9899, 9900,    1,  102,  203,  304,  405,  506,
                                               607,  708,  809,  910, 1011, 1112, 1213, 1314, 1415, 1516, 1617])]

        X_new = df_application[x_features]
        X_new[x_features] = X_new[x_features].fillna(
            X_new[x_features].median())
        y_new_pred_scaled = multi_model.predict(X_new)
        y_new_pred = scaler_y.inverse_transform(y_new_pred_scaled)

        y_new_pred_df = pd.DataFrame(
            y_new_pred, columns=y_targets, index=X_new.index)

        df_application = df_application[selected_features]

        df_application.drop(y_targets, axis=1, inplace=True)
        df_application_succeed = pd.concat(
            [df_application, y_new_pred_df], axis=1)

        df_train = df[(df['season'] >= 1718) & (df['season'] <= 2425)]

        df_train = df_train[selected_features]

        df_no_na = pd.concat([df_application_succeed, df_train]).dropna(
            axis=0).reset_index(drop=True)

        return df_no_na
    
    def application(self):
        multi_model, scaler_y, y_targets, X_val, y_val, y_val_pred_df = self.model()
        self.multi_model = multi_model
        self.scaler_y = scaler_y
        self.y_targets = y_targets

        print("Model Evaluation:")
        self.evaluate(y_val, y_val_pred_df, y_targets)

        self.export_model(multi_model, scaler_y)
        print("\nModel exported successfully!")

        df_no_na = self.apply_model(
            self.df, 
            self.x_features, 
            multi_model, 
            scaler_y, 
            y_targets, 
            self.selected_features
        )
        
        df_no_na.to_csv("/Users/lionlucky7/Desktop/Coding_Project/data/processed_whole/processed_data.csv", index=False)
        
        return df_no_na

    def change_season_format(self):
        df = pd.read_csv("/Users/lionlucky7/Desktop/Coding_Project/data/processed_whole/processed_data.csv")
        df['season_code'] = df['season'].astype(str).str.zfill(4)

        df['season'] = df['season_code'].apply(
            lambda x: int("19" + x[:2]) if int(x[:2]) > 40 else int("20" + x[:2]))

        df['season'] = df['season'].astype("int") 
        
        df.to_csv("/Users/lionlucky7/Desktop/Coding_Project/data/processed_whole/processed_data.csv")
            

preprocess = preprocess_data()
# df_final = preprocess.application()

preprocess.change_season_format()