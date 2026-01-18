import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_feature_importance(model, feature_names, title='Feature Importance', top_n=10):
    """
    Plots the feature importance of a given model.

    Parameters:
    model: Trained model with feature_importances_ attribute.
    feature_names: List of feature names corresponding to the model.
    title: Title of the plot.
    top_n: Number of top features to display.
    """
    # Get feature importances
    importances = model.feature_importances_
    
    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort the DataFrame by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(top_n)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.title(title)
    plt.gca().invert_yaxis()  # Invert y axis to have the most important feature on top
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()