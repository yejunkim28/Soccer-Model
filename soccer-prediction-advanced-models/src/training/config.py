# Configuration settings for training various models

# Hyperparameters for XGBoost
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

# Hyperparameters for LightGBM
LIGHTGBM_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': -1,
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse'
}

# Hyperparameters for Neural Network
NEURAL_NETWORK_PARAMS = {
    'input_dim': 10,  # Adjust based on feature size
    'hidden_layers': [64, 32],
    'output_dim': len(TARGETS),  # Number of target variables
    'activation': 'relu',
    'loss_function': 'mean_squared_error',
    'optimizer': 'adam',
    'epochs': 100,
    'batch_size': 32
}

# General training configurations
TRAINING_CONFIGS = {
    'validation_split': 0.2,
    'shuffle': True,
    'early_stopping': True,
    'patience': 10
}