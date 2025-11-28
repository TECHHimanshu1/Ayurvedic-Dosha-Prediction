import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)

# Load model and supporting files
model = joblib.load("prakriti_model.pkl")
mlb = joblib.load("dosha_label_binarizer.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Load test data
df_test = pd.read_csv("digital_twin_log.csv")

# Clean and prepare features
X_test = df_test[feature_columns]

# Clean and process true labels
df_test["Predicted_Dosha"] = df_test["Predicted_Dosha"].fillna("").astype(str)

# Safe label parsing
y_true_labels = df_test["Predicted_Dosha"].apply(
    lambda x: x.lower().replace(" ", "").split("+") if x else []
)

# Filter invalid rows
valid_idx = y_true_labels.apply(
    lambda x: isinstance(x, list) and all(isinstance(i, str) for i in x)
)
y_true_labels = y_true_labels[valid_idx]
X_test = X_test.loc[valid_idx].reset_index(drop=True)
y_true_labels = y_true_labels.reset_index(drop=True)

# Transform labels to binary multilabel format
y_true = mlb.transform(y_true_labels)

# Predict
y_pred = model.predict(X_test)

# Evaluation metrics
print("\n📊 Detailed Evaluation Metrics:")
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=mlb.classes_, zero_division=0))

print("\n🔢 Accuracy:", accuracy_score(y_true, y_pred))
print("🎯 Micro Precision:", precision_score(y_true, y_pred, average='micro', zero_division=0))
print("🎯 Macro Precision:", precision_score(y_true, y_pred, average='macro', zero_division=0))
print("🎯 Weighted Precision:", precision_score(y_true, y_pred, average='weighted', zero_division=0))

print("🧲 Micro Recall:", recall_score(y_true, y_pred, average='micro', zero_division=0))
print("🧲 Macro Recall:", recall_score(y_true, y_pred, average='macro', zero_division=0))
print("🧲 Weighted Recall:", recall_score(y_true, y_pred, average='weighted', zero_division=0))

print("⚖️ Micro F1 Score:", f1_score(y_true, y_pred, average='micro', zero_division=0))
print("⚖️ Macro F1 Score:", f1_score(y_true, y_pred, average='macro', zero_division=0))
print("⚖️ Weighted F1 Score:", f1_score(y_true, y_pred, average='weighted', zero_division=0))
