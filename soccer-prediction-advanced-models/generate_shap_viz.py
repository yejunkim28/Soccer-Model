#!/usr/bin/env python3
"""
Generate SHAP visualizations for xG (attacking) and defensive metrics.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from datetime import datetimes

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from data.preprocessing import preprocess_data

# Set style
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 80)
print("🎯 SHAP VISUALIZATION GENERATOR - xG & DEFENSIVE METRICS")
print("=" * 80)

# Define target groups
XG_TARGETS = ["Per_90_Minutes_xG", "Per_90_Minutes_npxG", "Standard_Sh/90"]
DEFENSIVE_TARGETS = ["Tkl+Int", "Blocks_Blocks", "Aerial_Duels_Won"]

print(f"\n📊 xG/Attacking Targets: {XG_TARGETS}")
print(f"🛡️  Defensive Targets: {DEFENSIVE_TARGETS}")

# Load data
print("\n📦 Loading data...")
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model_2', 'data', 'final', 'final.csv')
print(f"  Loading from: {data_path}")
df = pd.read_csv(data_path)
X, y = preprocess_data(df)
print(f"✓ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")

# Load model
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'outputs', 'models', 'xgboost_20260119_212700.pkl')
print(f"\n📦 Loading model: {model_path}")
model_wrapper = joblib.load(model_path)
print("✓ Model loaded")

# Extract the underlying XGBoost model
if hasattr(model_wrapper, 'model'):
    model = model_wrapper.model
    print(f"✓ Extracted model type: {type(model)}")
else:
    model = model_wrapper
    print(f"✓ Model type: {type(model)}")

# Check if it's a MultiOutputRegressor
from sklearn.multioutput import MultiOutputRegressor
is_multi_output = isinstance(model, MultiOutputRegressor)
print(f"✓ Multi-output model: {is_multi_output}")

if is_multi_output:
    print(f"✓ Number of estimators: {len(model.estimators_)}")


# Sample for SHAP (use 100 samples for speed)
X_sample = X.sample(n=min(100, len(X)), random_state=42)
print(f"\n🔬 Using {len(X_sample)} samples for SHAP analysis")

# Create output directory
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = os.path.join(base_dir, 'outputs', 'shap_analysis', 'xgboost')
os.makedirs(output_dir, exist_ok=True)

print("\n" + "=" * 80)
print("⚡ COMPUTING SHAP VALUES")
print("=" * 80)

# Compute SHAP values per target
target_names = y.columns.tolist()
xg_shap_values = []
def_shap_values = []

# For multi-output models, we need to process each estimator separately
if is_multi_output:
    print("\nComputing SHAP values for each target estimator...")
    
    for target_idx, target_name in enumerate(target_names):
        estimator = model.estimators_[target_idx]
        print(f"  [{target_idx+1}/{len(target_names)}] {target_name}")
        
        # Create explainer for this target's estimator
        explainer = shap.TreeExplainer(estimator)
        shap_vals = explainer.shap_values(X_sample)
        
        # Store if it's an xG or defensive target
        if target_name in XG_TARGETS:
            xg_shap_values.append((target_name, shap_vals))
        if target_name in DEFENSIVE_TARGETS:
            def_shap_values.append((target_name, shap_vals))
else:
    print("\nCreating SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    print("Computing SHAP values...")
    shap_values = explainer.shap_values(X_sample)
    print(f"✓ SHAP values computed: {shap_values.shape}")
    
    # Extract for target groups
    for idx, target_name in enumerate(target_names):
        if target_name in XG_TARGETS:
            xg_shap_values.append((target_name, shap_values[:, :, idx]))
        if target_name in DEFENSIVE_TARGETS:
            def_shap_values.append((target_name, shap_values[:, :, idx]))

print(f"\n✓ Collected SHAP values:")
print(f"  - xG targets: {len(xg_shap_values)}")
print(f"  - Defensive targets: {len(def_shap_values)}")

print("\n" + "=" * 80)
print("📊 GENERATING VISUALIZATIONS - xG/ATTACKING METRICS")
print("=" * 80)

# 1. SHAP Summary plots for each xG target
for target_name, shap_vals in xg_shap_values:
    print(f"\n📈 Creating summary plot for: {target_name}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_vals,
        X_sample,
        feature_names=X.columns.tolist(),
        show=False,
        max_display=15
    )
    plt.title(f'SHAP Feature Impact: {target_name}', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    safe_name = target_name.replace('/', '_').replace(' ', '_')
    filepath = os.path.join(output_dir, f'shap_summary_xg_{safe_name}_{timestamp}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: {filepath}")

# 2. Combined xG summary (average across xG targets)
print(f"\n📈 Creating combined xG summary plot...")
xg_shap_combined = np.mean([shap_vals for _, shap_vals in xg_shap_values], axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(
    xg_shap_combined,
    X_sample,
    feature_names=X.columns.tolist(),
    show=False,
    max_display=15
)
plt.title('SHAP Feature Impact: xG/Attacking Metrics (Combined)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()

filepath = os.path.join(output_dir, f'shap_summary_xg_combined_{timestamp}.png')
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()
print(f"  💾 Saved: {filepath}")

# 3. Feature importance bar chart for xG
print(f"\n📊 Creating xG feature importance bar chart...")
xg_importance = np.mean(np.abs(xg_shap_combined), axis=0)
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xg_importance
}).sort_values('Importance', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(data=importance_df, y='Feature', x='Importance', palette='viridis', ax=ax)
ax.set_title('Top 15 Features for xG/Attacking Predictions', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
plt.tight_layout()

filepath = os.path.join(output_dir, f'shap_importance_xg_{timestamp}.png')
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()
print(f"  💾 Saved: {filepath}")

print("\n" + "=" * 80)
print("🛡️  GENERATING VISUALIZATIONS - DEFENSIVE METRICS")
print("=" * 80)

# 4. SHAP Summary plots for each defensive target
for target_name, shap_vals in def_shap_values:
    print(f"\n🛡️  Creating summary plot for: {target_name}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_vals,
        X_sample,
        feature_names=X.columns.tolist(),
        show=False,
        max_display=15
    )
    plt.title(f'SHAP Feature Impact: {target_name}', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    safe_name = target_name.replace('/', '_').replace(' ', '_').replace('+', '_')
    filepath = os.path.join(output_dir, f'shap_summary_def_{safe_name}_{timestamp}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: {filepath}")

# 5. Combined defensive summary (average across defensive targets)
print(f"\n🛡️  Creating combined defensive summary plot...")
def_shap_combined = np.mean([shap_vals for _, shap_vals in def_shap_values], axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(
    def_shap_combined,
    X_sample,
    feature_names=X.columns.tolist(),
    show=False,
    max_display=15
)
plt.title('SHAP Feature Impact: Defensive Metrics (Combined)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()

filepath = os.path.join(output_dir, f'shap_summary_def_combined_{timestamp}.png')
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()
print(f"  💾 Saved: {filepath}")

# 6. Feature importance bar chart for defensive
print(f"\n📊 Creating defensive feature importance bar chart...")
def_importance = np.mean(np.abs(def_shap_combined), axis=0)
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': def_importance
}).sort_values('Importance', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(data=importance_df, y='Feature', x='Importance', palette='rocket', ax=ax)
ax.set_title('Top 15 Features for Defensive Predictions', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
plt.tight_layout()

filepath = os.path.join(output_dir, f'shap_importance_def_{timestamp}.png')
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()
print(f"  💾 Saved: {filepath}")

print("\n" + "=" * 80)
print("📊 GENERATING COMPARISON VISUALIZATIONS")
print("=" * 80)

# 7. Side-by-side comparison
print(f"\n🔄 Creating xG vs Defensive comparison...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# xG importance
xg_imp_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xg_importance
}).sort_values('Importance', ascending=False).head(10)

sns.barplot(data=xg_imp_df, y='Feature', x='Importance', palette='Blues_r', ax=ax1)
ax1.set_title('xG/Attacking - Top 10 Features', fontsize=12, fontweight='bold')
ax1.set_xlabel('Mean |SHAP Value|', fontsize=10)

# Defensive importance
def_imp_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': def_importance
}).sort_values('Importance', ascending=False).head(10)

sns.barplot(data=def_imp_df, y='Feature', x='Importance', palette='Reds_r', ax=ax2)
ax2.set_title('Defensive - Top 10 Features', fontsize=12, fontweight='bold')
ax2.set_xlabel('Mean |SHAP Value|', fontsize=10)

plt.suptitle('Feature Importance Comparison: xG vs Defensive Metrics', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

filepath = os.path.join(output_dir, f'shap_comparison_xg_vs_def_{timestamp}.png')
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()
print(f"  💾 Saved: {filepath}")

# 8. Generate text report
print(f"\n📝 Creating text summary report...")
report_path = os.path.join(output_dir, f'shap_xg_defensive_report_{timestamp}.txt')

with open(report_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("SHAP ANALYSIS REPORT: xG & DEFENSIVE METRICS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: XGBoost Multi-Target\n")
    f.write(f"Samples analyzed: {len(X_sample)}\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("xG/ATTACKING METRICS\n")
    f.write("=" * 80 + "\n\n")
    f.write("Targets analyzed:\n")
    for target in XG_TARGETS:
        if target in target_names:
            f.write(f"  - {target}\n")
    f.write("\nTop 15 Features by Importance:\n")
    f.write("-" * 80 + "\n")
    for idx, row in importance_df.head(15).iterrows():
        f.write(f"{row['Feature']:35s} | Importance: {row['Importance']:.6f}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("DEFENSIVE METRICS\n")
    f.write("=" * 80 + "\n\n")
    f.write("Targets analyzed:\n")
    for target in DEFENSIVE_TARGETS:
        if target in target_names:
            f.write(f"  - {target}\n")
    
    def_imp_df_full = pd.DataFrame({
        'Feature': X.columns,
        'Importance': def_importance
    }).sort_values('Importance', ascending=False).head(15)
    
    f.write("\nTop 15 Features by Importance:\n")
    f.write("-" * 80 + "\n")
    for idx, row in def_imp_df_full.iterrows():
        f.write(f"{row['Feature']:35s} | Importance: {row['Importance']:.6f}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("KEY INSIGHTS\n")
    f.write("=" * 80 + "\n\n")
    
    # Find common and unique features
    xg_top10 = set(xg_imp_df['Feature'].tolist())
    def_top10 = set(def_imp_df['Feature'].tolist())
    common = xg_top10 & def_top10
    xg_unique = xg_top10 - def_top10
    def_unique = def_top10 - xg_top10
    
    f.write(f"Common features (appear in both top 10):\n")
    for feat in common:
        f.write(f"  - {feat}\n")
    
    f.write(f"\nxG-specific features:\n")
    for feat in xg_unique:
        f.write(f"  - {feat}\n")
    
    f.write(f"\nDefensive-specific features:\n")
    for feat in def_unique:
        f.write(f"  - {feat}\n")

print(f"  💾 Saved: {report_path}")

print("\n" + "=" * 80)
print("✅ COMPLETE!")
print("=" * 80)
print(f"\n📁 All visualizations saved to: {output_dir}/")
print(f"\nGenerated files:")
print(f"  • xG summary plots (individual & combined)")
print(f"  • Defensive summary plots (individual & combined)")
print(f"  • Feature importance bar charts")
print(f"  • Comparison visualization (xG vs Defensive)")
print(f"  • Text summary report")
print("\n" + "=" * 80)
