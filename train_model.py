import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Step 1: Load the dataset
df = pd.read_csv('Prakriti.csv')
print("Dataset Loaded Successfully!")

# Step 2: Encode categorical features
le = LabelEncoder()

categorical_columns = [
    'Body Size', 'Body Weight', 'Height', 'Bone Structure', 'Complexion',
    'General feel of skin', 'Texture of Skin', 'Hair Color', 'Appearance of Hair',
    'Shape of face', 'Eyes', 'Eyelashes', 'Blinking of Eyes', 'Cheeks', 'Nose',
    'Teeth and gums', 'Lips', 'Nails', 'Appetite', 'Liking tastes'
]

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Encode the target column (Dosha)
target_encoder = LabelEncoder()
df['Dosha'] = target_encoder.fit_transform(df['Dosha'])
print("Categorical Encoding Completed!")

# Step 3: Split dataset
X = df.drop('Dosha', axis=1)
y = df['Dosha']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data Split Completed!")

# Step 4: Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("SMOTE applied to balance the classes.")

# Step 5: Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_train_smote, y_train_smote)
best_params = grid_search.best_params_
print(f"Best parameters found: {best_params}")

# Step 6: Train the best RandomForest model
best_rf = RandomForestClassifier(**best_params, random_state=42)
best_rf.fit(X_train_smote, y_train_smote)

# Step 7: Build VotingClassifier
rf = RandomForestClassifier(n_estimators=300, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)
svc = SVC(probability=True, random_state=42)

voting_clf = VotingClassifier(estimators=[
    ('rf', rf),
    ('lr', lr),
    ('svc', svc)
], voting='hard')
voting_clf.fit(X_train_smote, y_train_smote)

# Step 8: Train XGBoost model
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
xgb.fit(X_train_smote, y_train_smote)

# Step 9: Stratified Cross-validation
print("\nPerforming Stratified K-Fold Cross-Validation:")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_rf, X, y, cv=skf)
print(f"Stratified K-Fold CV scores: {cv_scores}")
print(f"Average Stratified CV Score: {cv_scores.mean() * 100:.2f}%\n")

# Step 10: Evaluation
y_pred_rf = best_rf.predict(X_test)
y_pred_voting = voting_clf.predict(X_test)
y_pred_xgb = xgb.predict(X_test)

print(f"Random Forest Model Accuracy: {accuracy_score(y_test, y_pred_rf) * 100:.2f}%")
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

print(f"Voting Classifier Model Accuracy: {accuracy_score(y_test, y_pred_voting) * 100:.2f}%")
print("Voting Classifier Classification Report:")
print(classification_report(y_test, y_pred_voting))

print(f"XGBoost Model Accuracy: {accuracy_score(y_test, y_pred_xgb) * 100:.2f}%")
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

# Step 11: Save the best model
joblib.dump(best_rf, 'prakriti_model_rf_final.pkl')
print("\nBest RandomForest model saved successfully as 'prakriti_model_rf_final.pkl'!")

# Step 12: Save the target encoder
joblib.dump(target_encoder, 'label_encoder_dosha.pkl')
print("Target label encoder saved successfully as 'label_encoder_dosha.pkl'!")

# Step 13: Manual Test on New Input (Example)
print("\nManual Prediction on a New Sample Input:")
model = joblib.load('prakriti_model_rf_final.pkl')
le_dosha = joblib.load('label_encoder_dosha.pkl')

# Dummy input for prediction (Replace with actual encoded values)
new_sample = np.array([[1, 2, 0, 1, 3, 2, 1, 0, 2, 1, 3, 2, 1, 0, 2, 1, 0, 2, 1, 3]])

predicted_class = model.predict(new_sample)
predicted_label = le_dosha.inverse_transform(predicted_class)

print(f"Predicted Dosha for the new input: {predicted_label[0]}")
