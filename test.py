import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
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

# Step 3: Split dataset into train and test sets
X = df.drop('Dosha', axis=1)
y = df['Dosha']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data Split Completed!")

# Step 4: Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("SMOTE applied to balance the classes.")

# Step 5: Train a RandomForest model
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train_smote, y_train_smote)

# Step 6: Build a Voting Classifier with RandomForest, Logistic Regression, and SVM
lr = LogisticRegression(max_iter=1000, random_state=42)
svc = SVC(probability=True, random_state=42)

voting_clf = VotingClassifier(estimators=[
    ('rf', rf),
    ('lr', lr),
    ('svc', svc)
], voting='hard')

voting_clf.fit(X_train_smote, y_train_smote)

# Step 7: Train an XGBoost model
xgb = XGBClassifier(n_estimators=300, max_depth=15, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train_smote, y_train_smote)

# Step 8: Evaluate on the test set
y_pred_rf = rf.predict(X_test)
y_pred_voting = voting_clf.predict(X_test)
y_pred_xgb = xgb.predict(X_test)

print("Random Forest Model Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred_rf) * 100))
print("Voting Classifier Model Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred_voting) * 100))
print("XGBoost Model Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred_xgb) * 100))

# Step 9: Classification Reports
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("\nVoting Classifier Classification Report:")
print(classification_report(y_test, y_pred_voting))

print("\nXGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

# Step 10: Cross-validation (to evaluate stability and performance)
cv_scores_rf = cross_val_score(rf, X_train_smote, y_train_smote, cv=5)
cv_scores_voting = cross_val_score(voting_clf, X_train_smote, y_train_smote, cv=5)
cv_scores_xgb = cross_val_score(xgb, X_train_smote, y_train_smote, cv=5)

print("\nRandom Forest Cross-validation Accuracy: {:.2f}%".format(cv_scores_rf.mean() * 100))
print("Voting Classifier Cross-validation Accuracy: {:.2f}%".format(cv_scores_voting.mean() * 100))
print("XGBoost Cross-validation Accuracy: {:.2f}%".format(cv_scores_xgb.mean() * 100))

# Step 11: Save the best model and the target encoder
joblib.dump(rf, 'prakriti_model_rf_final.pkl')
print("Best RandomForest model saved successfully as 'prakriti_model_rf_final.pkl'!")

joblib.dump(target_encoder, 'label_encoder_dosha.pkl')
print("Target label encoder saved successfully as 'label_encoder_dosha.pkl'!")
