"""
Car Price Prediction - Model Training Pipeline
Dataset: Kaggle - vijayaadithyanvg/car-price-predictionused-cars
Columns: Car_Name, Year, Selling_Price, Present_Price, Kms_Driven,
         Fuel_Type, Seller_Type, Transmission, Owner
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import os

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
OUTPUT = 'public'
os.makedirs(OUTPUT, exist_ok=True)
os.makedirs('api', exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
df = pd.read_csv('car_data.csv')
print(f"Dataset shape: {df.shape}")
print(df.head())
print(df.info())

# ---------------------------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------------------------
# Derive car_age from Year
df['car_age'] = 2024 - df['Year']

# Encode categorical features with explicit mappings (for JS reproducibility)
fuel_map = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
seller_map = {'Dealer': 0, 'Individual': 1}
transmission_map = {'Manual': 0, 'Automatic': 1}

df['Fuel_Type_enc'] = df['Fuel_Type'].map(fuel_map)
df['Seller_Type_enc'] = df['Seller_Type'].map(seller_map)
df['Transmission_enc'] = df['Transmission'].map(transmission_map)

# Features and target
feature_cols = [
    'Present_Price', 'Kms_Driven', 'car_age', 'Owner',
    'Fuel_Type_enc', 'Seller_Type_enc', 'Transmission_enc'
]
feature_labels = [
    'Present Price', 'Kms Driven', 'Car Age', 'Owner',
    'Fuel Type', 'Seller Type', 'Transmission'
]

X = df[feature_cols].copy()
y = df['Selling_Price'].copy()

# Check for any NaN values
print(f"\nNull values:\n{X.isnull().sum()}")
X = X.fillna(0)

# ---------------------------------------------------------------------------
# 3. Train / Test Split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------------
# 4. Scaling (for Linear Regression)
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 5. Train Multiple Models
# ---------------------------------------------------------------------------
models = {
    'Linear Regression':  LinearRegression(),
    'Random Forest':      RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting':  GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    if name == 'Linear Regression':
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    results[name] = {
        'mae':  mean_absolute_error(y_test, preds),
        'rmse': float(np.sqrt(mean_squared_error(y_test, preds))),
        'r2':   r2_score(y_test, preds),
        'preds': preds
    }

# Pick best model by R²
best_name = max(results, key=lambda k: results[k]['r2'])
best_model = models[best_name]
best_preds = results[best_name]['preds']
print(f"\nBest model: {best_name} | R2={results[best_name]['r2']:.4f}")

# ---------------------------------------------------------------------------
# 6. Save model artifacts
# ---------------------------------------------------------------------------
joblib.dump(best_model, 'api/model.pkl')
joblib.dump(scaler, 'api/scaler.pkl')

# ---------------------------------------------------------------------------
# 7. Export model parameters to JSON for client-side prediction
# ---------------------------------------------------------------------------

def export_tree(tree, tree_index=0):
    """Export a single sklearn decision tree to a JSON-serializable dict."""
    t = tree.tree_
    def recurse(node):
        if t.feature[node] < 0:
            # Leaf node
            return {'v': round(float(t.value[node].flatten()[0]), 6)}
        return {
            'f': int(t.feature[node]),
            't': round(float(t.threshold[node]), 6),
            'l': recurse(t.children_left[node]),
            'r': recurse(t.children_right[node]),
        }
    return recurse(0)


def export_model_params(name, model, scaler_obj):
    """Export model parameters so JS can replicate predictions."""
    params = {'type': name}

    if name == 'Linear Regression':
        params['coef'] = [round(float(c), 8) for c in model.coef_]
        params['intercept'] = round(float(model.intercept_), 8)
        params['scaler_mean'] = [round(float(m), 8) for m in scaler_obj.mean_]
        params['scaler_scale'] = [round(float(s), 8) for s in scaler_obj.scale_]

    elif name == 'Random Forest':
        params['trees'] = [export_tree(est) for est in model.estimators_]
        params['n_trees'] = len(model.estimators_)

    elif name == 'Gradient Boosting':
        params['learning_rate'] = round(float(model.learning_rate), 6)
        params['init_value'] = round(float(model.init_.constant_.flatten()[0]), 6)
        params['trees'] = []
        for i in range(model.n_estimators_):
            params['trees'].append(export_tree(model.estimators_[i, 0]))

    return params


model_params = export_model_params(best_name, best_model, scaler)

# ---------------------------------------------------------------------------
# 8. Build meta.json
# ---------------------------------------------------------------------------
# Get unique car names for dropdown
car_names = sorted(df['Car_Name'].unique().tolist())

meta = {
    'best_model': best_name,
    'metrics': {
        k: {
            'mae': round(v['mae'], 4),
            'rmse': round(v['rmse'], 4),
            'r2': round(v['r2'], 4)
        }
        for k, v in results.items()
    },
    'feature_importance': {},
    'feature_cols': feature_cols,
    'feature_labels': feature_labels,
    'encoders': {
        'Fuel_Type': fuel_map,
        'Seller_Type': seller_map,
        'Transmission': transmission_map,
    },
    'car_names': car_names,
    'dataset_size': len(df),
    'n_features': len(feature_cols),
    'model_params': model_params,
}

if hasattr(best_model, 'feature_importances_'):
    meta['feature_importance'] = {
        label: round(float(imp), 4)
        for label, imp in zip(feature_labels, best_model.feature_importances_)
    }

with open(f'{OUTPUT}/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\nmeta.json written ({os.path.getsize(f'{OUTPUT}/meta.json')} bytes)")

# ---------------------------------------------------------------------------
# 9. Generate Visualizations
# ---------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')

# --- Chart 1: Actual vs Predicted ---
fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0f172a')
ax.set_facecolor('#1e293b')
ax.scatter(y_test, best_preds, alpha=0.6, color='#38bdf8', edgecolors='none', s=40)
lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
ax.plot(lims, lims, '--', color='#f97316', lw=2, label='Perfect prediction')
ax.set_xlabel('Actual Price (Lakhs)', color='#94a3b8', fontsize=12)
ax.set_ylabel('Predicted Price (Lakhs)', color='#94a3b8', fontsize=12)
ax.set_title(f'Actual vs Predicted - {best_name}', color='white', fontsize=14, fontweight='bold')
ax.tick_params(colors='#94a3b8')
ax.legend(facecolor='#1e293b', labelcolor='white')
for spine in ax.spines.values():
    spine.set_color('#334155')
plt.tight_layout()
plt.savefig(f'{OUTPUT}/chart_actual_vs_predicted.png', dpi=120, bbox_inches='tight', facecolor='#0f172a')
plt.close()

# --- Chart 2: Model Comparison ---
fig, axes = plt.subplots(1, 3, figsize=(13, 5), facecolor='#0f172a')
metrics_to_plot = [('r2', 'R² Score', '#38bdf8'), ('mae', 'MAE (Lakhs)', '#a78bfa'), ('rmse', 'RMSE (Lakhs)', '#fb7185')]
model_names = list(results.keys())
short_names = ['Linear\nRegression', 'Random\nForest', 'Gradient\nBoosting']
for ax, (metric, label, color) in zip(axes, metrics_to_plot):
    vals = [results[m][metric] for m in model_names]
    ax.bar(short_names, vals, color=color, alpha=0.85, edgecolor='none', width=0.5)
    ax.set_facecolor('#1e293b')
    ax.set_title(label, color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='#94a3b8', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#334155')
fig.patch.set_facecolor('#0f172a')
plt.tight_layout()
plt.savefig(f'{OUTPUT}/chart_model_comparison.png', dpi=120, bbox_inches='tight', facecolor='#0f172a')
plt.close()

# --- Chart 3: Feature Importance ---
if hasattr(best_model, 'feature_importances_'):
    importance = best_model.feature_importances_
    sorted_idx = np.argsort(importance)
    colors_feat = plt.cm.Blues(np.linspace(0.4, 0.95, len(feature_cols)))
    fig, ax = plt.subplots(figsize=(9, 5), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    ax.barh(
        [feature_labels[i] for i in sorted_idx],
        importance[sorted_idx],
        color=colors_feat,
        edgecolor='none'
    )
    ax.set_xlabel('Importance Score', color='#94a3b8', fontsize=11)
    ax.set_title('Feature Importance', color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='#94a3b8')
    for spine in ax.spines.values():
        spine.set_color('#334155')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/chart_feature_importance.png', dpi=120, bbox_inches='tight', facecolor='#0f172a')
    plt.close()

# --- Chart 4: Price Distribution ---
fig, ax = plt.subplots(figsize=(9, 5), facecolor='#0f172a')
ax.set_facecolor('#1e293b')
ax.hist(df['Selling_Price'], bins=30, color='#38bdf8', alpha=0.8, edgecolor='#0f172a')
ax.set_xlabel('Selling Price (Lakhs)', color='#94a3b8', fontsize=11)
ax.set_ylabel('Number of Cars', color='#94a3b8', fontsize=11)
ax.set_title('Car Price Distribution', color='white', fontsize=14, fontweight='bold')
ax.tick_params(colors='#94a3b8')
for spine in ax.spines.values():
    spine.set_color('#334155')
plt.tight_layout()
plt.savefig(f'{OUTPUT}/chart_price_distribution.png', dpi=120, bbox_inches='tight', facecolor='#0f172a')
plt.close()

# --- Chart 5: Average Price by Car Name (top 15) ---
car_avg = df.groupby('Car_Name')['Selling_Price'].mean().sort_values(ascending=False).head(15)
fig, ax = plt.subplots(figsize=(11, 5), facecolor='#0f172a')
ax.set_facecolor('#1e293b')
palette = ['#f97316' if v > 10 else '#38bdf8' for v in car_avg.values]
ax.bar(car_avg.index, car_avg.values, color=palette, edgecolor='none', alpha=0.9)
ax.set_ylabel('Avg Price (Lakhs)', color='#94a3b8', fontsize=11)
ax.set_title('Average Price by Car (Top 15)', color='white', fontsize=14, fontweight='bold')
ax.tick_params(colors='#94a3b8', axis='x', rotation=35)
ax.tick_params(colors='#94a3b8', axis='y')
for spine in ax.spines.values():
    spine.set_color('#334155')
plt.tight_layout()
plt.savefig(f'{OUTPUT}/chart_brand_price.png', dpi=120, bbox_inches='tight', facecolor='#0f172a')
plt.close()

print("\nAll charts saved!")
print(json.dumps(
    {k: {'R2': round(v['r2'], 4), 'MAE': round(v['mae'], 4)} for k, v in results.items()},
    indent=2
))