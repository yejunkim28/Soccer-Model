#!/usr/bin/env python3
"""
Training script for soccer prediction models.
Usage: python train.py --model [xgboost|lightgbm|nn|all]
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yaml
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel
from models.neural_network_model import NeuralNetworkModel
from data.preprocessing import preprocess_data


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name, 
                       feature_names, target_names, save_outputs=False, 
                       output_dir='outputs', timestamp=None):
    """Train model, evaluate, and save visualizations."""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("\n" + "="*60)
    print(f"Training {model_name.upper()}...")
    print("="*60)
    
    # Train with history tracking
    history = None
    
    if model_name.lower() == 'neural_network':
        # Neural network requires scaled data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train with scaled data
        history = model.fit(X_train_scaled, y_train)
        
        # Update X_train and X_test for predictions later
        X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    else:
        # Tree models - train directly (they don't need scaling)
        model.fit(X_train, y_train)
    
    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate
    metrics = {
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'train_mse': mean_squared_error(y_train, y_pred_train),
        'test_mse': mean_squared_error(y_test, y_pred_test),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test)
    }
    
    print(f"\n✓ Training completed!")
    print(f"  Train R²: {metrics['train_r2']:.4f}, Test R²: {metrics['test_r2']:.4f}")
    print(f"  Train MAE: {metrics['train_mae']:.4f}, Test MAE: {metrics['test_mae']:.4f}")
    
    # Save outputs if requested
    if save_outputs:
        print(f"\n📦 Saving outputs...")
        
        # Save model
        model_dir = os.path.join(output_dir, 'models')
        save_model(model, model_name, model_dir, timestamp)
        
        # Save visualizations (use lowercase for directory names)
        viz_dir = os.path.join(output_dir, 'visualizations', model_name.lower())
        
        # Loss curves
        plot_loss_curves(history, model_name, viz_dir, timestamp)
        
        # Feature importance
        plot_feature_importance(model, model_name, feature_names, viz_dir, timestamp)
        
        # Predictions vs Actual
        plot_predictions_vs_actual(y_train.values, y_pred_train, model_name, 
                                  target_names, viz_dir, timestamp, split='train')
        plot_predictions_vs_actual(y_test.values, y_pred_test, model_name, 
                                  target_names, viz_dir, timestamp, split='test')
        
        # Residuals
        plot_residuals(y_train.values, y_pred_train, model_name, 
                      target_names, viz_dir, timestamp, split='train')
        plot_residuals(y_test.values, y_pred_test, model_name, 
                      target_names, viz_dir, timestamp, split='test')
        
        # Metrics comparison
        plot_metrics_comparison(metrics, model_name, viz_dir, timestamp)
    
    return metrics


def save_model(model, model_name, output_dir='outputs/models', timestamp=None):
    """Save trained model to disk."""
    os.makedirs(output_dir, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'{model_name}_{timestamp}.pkl')
    
    joblib.dump(model, filepath)
    print(f"  💾 Model saved: {filepath}")
    return filepath


def save_visualization(fig, name, model_name, output_dir='outputs/visualizations', timestamp=None):
    """Save matplotlib figure to disk."""
    os.makedirs(output_dir, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'{model_name}_{name}_{timestamp}.png')
    
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 Visualization saved: {filepath}")
    return filepath


def plot_feature_importance(model, model_name, feature_names, output_dir, timestamp):
    """Plot and save feature importance."""
    try:
        model_name_lower = model_name.lower()
        if model_name_lower == 'lightgbm':
            # LightGBM with MultiOutputRegressor
            importances = np.mean([est.feature_importances_ for est in model.model.estimators_], axis=0)
        elif model_name_lower == 'xgboost':
            # XGBoost with MultiOutputRegressor
            importances = np.mean([est.feature_importances_ for est in model.model.estimators_], axis=0)
        else:
            return None
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1][:20]  # Top 20 features
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(range(len(indices)), importances[indices], color='skyblue')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title(f'{model_name.upper()} - Feature Importance (Top 20)', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        
        save_visualization(fig, 'feature_importance', model_name, output_dir, timestamp)
        
    except Exception as e:
        print(f"  ⚠️  Could not plot feature importance: {e}")


def plot_predictions_vs_actual(y_true, y_pred, model_name, target_names, output_dir, timestamp, split='test'):
    """Plot predictions vs actual values for each target."""
    n_targets = y_true.shape[1] if len(y_true.shape) > 1 else 1
    
    if n_targets == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    # Create subplot for each target
    n_cols = min(3, n_targets)
    n_rows = (n_targets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_targets == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_targets > 1 else axes
    
    for i in range(n_targets):
        ax = axes[i] if n_targets > 1 else axes[0]
        
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=20)
        
        # Plot perfect prediction line
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Calculate R²
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        
        ax.set_xlabel('Actual', fontsize=10)
        ax.set_ylabel('Predicted', fontsize=10)
        ax.set_title(f'{target_names[i]}\nR² = {r2:.4f}', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_targets, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'{model_name.upper()} - Predictions vs Actual ({split.capitalize()})', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    save_visualization(fig, f'predictions_vs_actual_{split}', model_name, output_dir, timestamp)


def plot_residuals(y_true, y_pred, model_name, target_names, output_dir, timestamp, split='test'):
    """Plot residuals for each target."""
    n_targets = y_true.shape[1] if len(y_true.shape) > 1 else 1
    
    if n_targets == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    residuals = y_true - y_pred
    
    # Create subplot for each target
    n_cols = min(3, n_targets)
    n_rows = (n_targets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_targets == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_targets > 1 else axes
    
    for i in range(n_targets):
        ax = axes[i] if n_targets > 1 else axes[0]
        
        ax.scatter(y_pred[:, i], residuals[:, i], alpha=0.5, s=20)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Residuals', fontsize=10)
        ax.set_title(f'{target_names[i]}', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_targets, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'{model_name.upper()} - Residual Plot ({split.capitalize()})', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    save_visualization(fig, f'residuals_{split}', model_name, output_dir, timestamp)


def plot_metrics_comparison(metrics, model_name, output_dir, timestamp):
    """Plot comparison of train vs test metrics - separate plots for better scale visibility."""
    # Create separate plots for each metric since they have different scales
    metric_configs = [
        ('MAE', 'train_mae', 'test_mae', 'Mean Absolute Error (Lower is Better)'),
        ('MSE', 'train_mse', 'test_mse', 'Mean Squared Error (Lower is Better)'),
        ('R²', 'train_r2', 'test_r2', 'R² Score (Higher is Better)')
    ]
    
    for short_name, train_key, test_key, title in metric_configs:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x = np.arange(2)
        values = [metrics[train_key], metrics[test_key]]
        colors = ['#3498db', '#e74c3c']
        
        bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(['Train', 'Test'], fontsize=12, fontweight='bold')
        ax.set_ylabel(short_name, fontsize=13, fontweight='bold')
        ax.set_title(f'{model_name.upper()} - {title}', fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels on bars
        for i, (bar, v) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{v:.4f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add horizontal line at y=0 for R² plots
        if short_name == 'R²':
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        plt.tight_layout()
        save_visualization(fig, f'metrics_{short_name.lower().replace("²", "2")}', 
                         model_name, output_dir, timestamp)


def plot_loss_curves(history, model_name, output_dir, timestamp):
    """Plot training and validation loss curves over iterations."""
    try:
        if model_name.lower() == 'neural_network':
            # Neural network history from Keras
            if history is None or not hasattr(history, 'history'):
                return
            
            train_loss = history.history.get('loss', [])
            val_loss = history.history.get('val_loss', [])
            
            if not train_loss:
                return
            
            fig, ax = plt.subplots(figsize=(10, 6))
            epochs = range(1, len(train_loss) + 1)
            
            ax.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
            if val_loss:
                ax.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss (MSE)', fontsize=12)
            ax.set_title(f'{model_name.upper()} - Loss Curves', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            save_visualization(fig, 'loss_curves', model_name, output_dir, timestamp)
        else:
            # Tree-based models don't have loss curves in this simplified version
            pass
                
    except Exception as e:
        print(f"  ⚠️  Could not plot loss curves: {e}")


def main():
    parser = argparse.ArgumentParser(description='Train soccer prediction models')
    parser.add_argument('--model', type=str, default='all',
                       choices=['xgboost', 'lightgbm', 'nn', 'all'],
                       help='Which model to train')
    parser.add_argument('--data', type=str, 
                       default='../model_2/data/final/final.csv',
                       help='Path to training data')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save models and visualizations (default: save everything)')
    
    args = parser.parse_args()
    
    # By default, save everything unless --no-save is specified
    save_outputs = not args.no_save
    
    print("\n🚀 Soccer Prediction Model Training")
    print("="*60)
    
    # Create timestamp for this training run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load configs
    try:
        model_configs = load_config('config/model_configs.yaml')
        training_configs = load_config('config/training_configs.yaml')
    except FileNotFoundError as e:
        print(f"❌ Config file not found: {e}")
        return
    
    # Load and preprocess data
    print(f"\n📊 Loading data from: {args.data}")
    try:
        data = pd.read_csv(args.data)
        X, y = preprocess_data(data)
        print(f"✓ Data loaded: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} targets")
        print(f"✓ Features: {list(X.columns)}")
        print(f"✓ Targets: {list(y.columns)}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Split data
    test_size = training_configs.get('test_size', 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    print(f"✓ Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    
    # Get feature and target names
    feature_names = X.columns.tolist()
    target_names = y.columns.tolist()
    
    # Initialize models
    models = {}
    results = {}
    
    if args.model in ['xgboost', 'all']:
        models['xgboost'] = XGBoostModel(params=model_configs.get('xgboost'))
    
    if args.model in ['lightgbm', 'all']:
        models['lightgbm'] = LightGBMModel(params=model_configs.get('lightgbm'))
    
    if args.model in ['nn', 'all']:
        # Neural network needs input/output shapes
        input_shape = X_train.shape[1]
        output_shape = y_train.shape[1]
        models['neural_network'] = NeuralNetworkModel(input_shape=input_shape, output_shape=output_shape)
    
    # Train models
    for name, model in models.items():
        try:
            metrics = train_and_evaluate(
                model, X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy(), 
                name, feature_names, target_names,
                save_outputs=save_outputs, timestamp=timestamp
            )
            results[name] = metrics
                
        except Exception as e:
            print(f"❌ Error training {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    if results:
        print("\n" + "="*60)
        print("📊 FINAL RESULTS")
        print("="*60)
        results_df = pd.DataFrame(results).T
        print(results_df)
        
        if save_outputs:
            os.makedirs('outputs/logs', exist_ok=True)
            results_path = f'outputs/logs/training_results_{timestamp}.csv'
            results_df.to_csv(results_path)
            print(f"\n💾 Results saved: {results_path}")
            
            # Create comprehensive model comparison visualization
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
            
            models = results_df.index.tolist()
            x_pos = np.arange(len(models))
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
            
            # 1. Test MAE comparison
            ax1 = fig.add_subplot(gs[0, 0])
            bars1 = ax1.bar(x_pos, results_df['test_mae'], color=colors[:len(models)], alpha=0.8, edgecolor='black')
            ax1.set_xlabel('Model', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
            ax1.set_title('Test MAE (Lower is Better)', fontsize=12, fontweight='bold')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(models, rotation=0)
            ax1.grid(True, alpha=0.3, axis='y')
            # Add value labels on bars
            for i, bar in enumerate(bars1):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=9)
            
            # 2. Test MSE comparison
            ax2 = fig.add_subplot(gs[0, 1])
            bars2 = ax2.bar(x_pos, results_df['test_mse'], color=colors[:len(models)], alpha=0.8, edgecolor='black')
            ax2.set_xlabel('Model', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Mean Squared Error', fontsize=11, fontweight='bold')
            ax2.set_title('Test MSE (Lower is Better)', fontsize=12, fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(models, rotation=0)
            ax2.grid(True, alpha=0.3, axis='y')
            for i, bar in enumerate(bars2):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=9)
            
            # 3. Test R² comparison
            ax3 = fig.add_subplot(gs[1, 0])
            bars3 = ax3.bar(x_pos, results_df['test_r2'], color=colors[:len(models)], alpha=0.8, edgecolor='black')
            ax3.set_xlabel('Model', fontsize=11, fontweight='bold')
            ax3.set_ylabel('R² Score', fontsize=11, fontweight='bold')
            ax3.set_title('Test R² (Higher is Better)', fontsize=12, fontweight='bold')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(models, rotation=0)
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            for i, bar in enumerate(bars3):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
            
            # 4. Train vs Test MAE comparison
            ax4 = fig.add_subplot(gs[1, 1])
            width = 0.35
            ax4.bar(x_pos - width/2, results_df['train_mae'], width, label='Train MAE',
                   color='#3498db', alpha=0.8, edgecolor='black')
            ax4.bar(x_pos + width/2, results_df['test_mae'], width, label='Test MAE',
                   color='#e74c3c', alpha=0.8, edgecolor='black')
            ax4.set_xlabel('Model', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
            ax4.set_title('Train vs Test MAE', fontsize=12, fontweight='bold')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(models, rotation=0)
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
            
            # 5. Train vs Test R² comparison
            ax5 = fig.add_subplot(gs[2, 0])
            ax5.bar(x_pos - width/2, results_df['train_r2'], width, label='Train R²',
                   color='#2ecc71', alpha=0.8, edgecolor='black')
            ax5.bar(x_pos + width/2, results_df['test_r2'], width, label='Test R²',
                   color='#f39c12', alpha=0.8, edgecolor='black')
            ax5.set_xlabel('Model', fontsize=11, fontweight='bold')
            ax5.set_ylabel('R² Score', fontsize=11, fontweight='bold')
            ax5.set_title('Train vs Test R²', fontsize=12, fontweight='bold')
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels(models, rotation=0)
            ax5.legend()
            ax5.grid(True, alpha=0.3, axis='y')
            ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            
            # 6. Overall ranking table
            ax6 = fig.add_subplot(gs[2, 1])
            ax6.axis('tight')
            ax6.axis('off')
            
            # Rank models by test R²
            ranking_df = results_df.copy()
            ranking_df['rank'] = ranking_df['test_r2'].rank(ascending=False).astype(int)
            ranking_df = ranking_df.sort_values('rank')
            
            table_data = []
            table_data.append(['Rank', 'Model', 'Test R²', 'Test MAE'])
            for idx, row in ranking_df.iterrows():
                table_data.append([
                    f"#{int(row['rank'])}",
                    idx,
                    f"{row['test_r2']:.4f}",
                    f"{row['test_mae']:.4f}"
                ])
            
            table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                            colWidths=[0.15, 0.35, 0.25, 0.25])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Style header row
            for i in range(4):
                table[(0, i)].set_facecolor('#34495e')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Color rows by rank
            rank_colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#95a5a6']
            for i in range(1, len(table_data)):
                for j in range(4):
                    table[(i, j)].set_facecolor(rank_colors[min(i-1, len(rank_colors)-1)])
                    table[(i, j)].set_alpha(0.3)
            
            ax6.set_title('Model Rankings', fontsize=12, fontweight='bold', pad=20)
            
            # Main title
            fig.suptitle('🏆 Model Performance Comparison Dashboard', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            # Save
            summary_path = f'outputs/visualizations/model_comparison_{timestamp}.png'
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            fig.savefig(summary_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"💾 Comparison dashboard saved: {summary_path}")
            
    print(f"\n✅ Training completed successfully!")
    if save_outputs:
        print(f"📁 All outputs saved to: outputs/")


if __name__ == '__main__':
    main()
