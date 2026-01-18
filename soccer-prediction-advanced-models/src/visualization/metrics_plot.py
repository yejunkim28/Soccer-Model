import matplotlib.pyplot as plt
import pandas as pd

def plot_metrics(y_true, y_pred, metrics=['mae', 'mse', 'r2'], target_names=None):
    """
    Plot various performance metrics for model predictions.

    Parameters:
    -----------
    y_true : pd.DataFrame
        True target values.
    y_pred : pd.DataFrame
        Predicted target values.
    metrics : list
        List of metrics to plot. Options: 'mae', 'mse', 'r2'.
    target_names : list
        List of target variable names for labeling the plots.
    """
    if target_names is None:
        target_names = [f'Target {i+1}' for i in range(y_true.shape[1])]

    results = {}
    
    for metric in metrics:
        if metric == 'mae':
            results['MAE'] = (y_true - y_pred).abs().mean()
        elif metric == 'mse':
            results['MSE'] = ((y_true - y_pred) ** 2).mean()
        elif metric == 'r2':
            results['R²'] = 1 - ( ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum() )
    
    # Create a DataFrame for plotting
    results_df = pd.DataFrame(results, index=target_names)

    # Plotting
    results_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()