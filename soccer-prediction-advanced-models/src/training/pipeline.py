import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel
from models.neural_network_model import NeuralNetworkModel
from data.preprocessing import preprocess_data
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    metrics = {
        'mae': mean_absolute_error(y_val, y_pred),
        'mse': mean_squared_error(y_val, y_pred),
        'r2': r2_score(y_val, y_pred)
    }
    
    return metrics

def main():
    # Load configuration
    training_config = load_config('config/training_configs.yaml')
    model_configs = load_config('config/model_configs.yaml')

    # Load and preprocess data
    data = pd.read_csv('data/dataset.csv')  # Adjust path as necessary
    X, y = preprocess_data(data)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=training_config['test_size'], random_state=42)

    # Initialize models
    xgb_model = XGBoostModel(params=model_configs['xgboost'])
    lgb_model = LightGBMModel(params=model_configs['lightgbm'])
    nn_model = NeuralNetworkModel(params=model_configs['neural_network'])

    # Train and evaluate models
    models = {'XGBoost': xgb_model, 'LightGBM': lgb_model, 'Neural Network': nn_model}
    results = {}

    for name, model in models.items():
        print(f'Training {name}...')
        metrics = train_model(model, X_train, y_train, X_val, y_val)
        results[name] = metrics
        print(f'{name} Metrics: {metrics}')

    # Save results to a file or visualize as needed
    results_df = pd.DataFrame(results).T
    results_df.to_csv('outputs/models/training_results.csv')

if __name__ == '__main__':
    main()