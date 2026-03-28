# ============================================================
# FILE: train_model.py
# PURPOSE: Train ML models to predict student focus score
# PROJECT: Focus Level Detector
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# ============================================================
# STEP 3.1 → Load the dataset
# ============================================================
df = pd.read_csv('data/student_focus_data.csv')
print("=" * 50)
print("   Model Training Started!")
print("=" * 50)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================
# STEP 3.2 → Prepare features and target
# X = inputs (what we feed the model)
# y = output (what we want the model to predict)
# ============================================================
X = df[['study_hours', 'sleep_hours', 'phone_usage', 'break_time']]
y = df['focus_score']

print(f"\nFeatures (X): {list(X.columns)}")
print(f"Target (y)  : focus_score")

# ============================================================
# STEP 3.3 → Split data into training and testing sets
# 80% data → train the model
# 20% data → test how well it learned
# This is like studying from a textbook and then giving exam
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size : {X_train.shape[0]} students")
print(f"Testing set size  : {X_test.shape[0]} students")

# ============================================================
# STEP 3.4 → Train Model 1: Linear Regression
# Finds a straight line relationship between inputs and output
# Like drawing a best fit line through all data points
# ============================================================
print("\n--- Training Linear Regression ---")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict on test data
lr_predictions = lr_model.predict(X_test)

# Calculate accuracy metrics
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_r2  = r2_score(y_test, lr_predictions)

print(f"Mean Absolute Error : {lr_mae:.2f}")
print(f"R2 Score            : {lr_r2:.2f}")

# ============================================================
# STEP 3.5 → Train Model 2: Decision Tree
# Makes decisions by asking yes/no questions about the data
# Like a flowchart that splits data based on conditions
# ============================================================
print("\n--- Training Decision Tree ---")
dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# Predict on test data
dt_predictions = dt_model.predict(X_test)

# Calculate accuracy metrics
dt_mae = mean_absolute_error(y_test, dt_predictions)
dt_r2  = r2_score(y_test, dt_predictions)

print(f"Mean Absolute Error : {dt_mae:.2f}")
print(f"R2 Score            : {dt_r2:.2f}")

# ============================================================
# STEP 3.6 → Compare both models
# ============================================================
print("\n" + "=" * 50)
print("   MODEL COMPARISON")
print("=" * 50)
print(f"{'Model':<25} {'MAE':>8} {'R2 Score':>10}")
print("-" * 50)
print(f"{'Linear Regression':<25} {lr_mae:>8.2f} {lr_r2:>10.2f}")
print(f"{'Decision Tree':<25} {dt_mae:>8.2f} {dt_r2:>10.2f}")
print("=" * 50)

# Pick the better model (higher R2 = better)
if lr_r2 >= dt_r2:
    best_model = lr_model
    best_name  = "Linear Regression"
else:
    best_model = dt_model
    best_name  = "Decision Tree"

print(f"\nBest Model: {best_name}")

# ============================================================
# STEP 3.7 → Visualize: Actual vs Predicted scores
# A good model's dots should be close to the diagonal line
# ============================================================
plt.figure(figsize=(12, 5))

# Linear Regression plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, lr_predictions,
            alpha=0.6, color='#4C9BE8',
            edgecolors='black', linewidths=0.3, s=60)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect Prediction')
plt.title(f'Linear Regression\nR2 = {lr_r2:.2f}', fontweight='bold')
plt.xlabel('Actual Focus Score')
plt.ylabel('Predicted Focus Score')
plt.legend()

# Decision Tree plot
plt.subplot(1, 2, 2)
plt.scatter(y_test, dt_predictions,
            alpha=0.6, color='#6BCB77',
            edgecolors='black', linewidths=0.3, s=60)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect Prediction')
plt.title(f'Decision Tree\nR2 = {dt_r2:.2f}', fontweight='bold')
plt.xlabel('Actual Focus Score')
plt.ylabel('Predicted Focus Score')
plt.legend()

plt.suptitle('Actual vs Predicted Focus Scores', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('data/graph6_model_comparison.png')
plt.show()
print("\nGraph saved!")

# ============================================================
# STEP 3.8 → Feature Importance (Decision Tree)
# Shows which input affects focus score the most
# ============================================================
plt.figure(figsize=(7, 5))
features    = ['study_hours', 'sleep_hours', 'phone_usage', 'break_time']
importances = dt_model.feature_importances_

colors = ['#4C9BE8', '#6BCB77', '#FF6B6B', '#FFD93D']
bars = plt.bar(features, importances, color=colors,
               edgecolor='black', width=0.5)

for bar, imp in zip(bars, importances):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.005,
             f'{imp:.2f}',
             ha='center', fontsize=11, fontweight='bold')

plt.title('Feature Importance\n(Which factor affects focus the most?)',
          fontsize=13, fontweight='bold')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('data/graph7_feature_importance.png')
plt.show()
print("Feature importance graph saved!")

# ============================================================
# STEP 3.9 → Save the best model
# pickle saves the trained model so we can reuse it later
# without retraining every single time
# ============================================================
with open('data/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"\nBest model ({best_name}) saved as best_model.pkl")
print("=" * 50)
print("   Training Complete!")
print("=" * 50)
