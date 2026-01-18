def split_data(df, target_cols, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=target_cols)
    y = df[target_cols]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def calculate_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }
    return metrics

def log_metrics(metrics, model_name):
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(model_name)
    logger.info(f"Metrics for {model_name}: {metrics}")

def save_model(model, filename):
    import joblib
    joblib.dump(model, filename)

def load_model(filename):
    import joblib
    return joblib.load(filename)