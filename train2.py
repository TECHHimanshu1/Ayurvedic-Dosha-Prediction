import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
import joblib

# Step 1: Load dataset
df = pd.read_csv('Prakriti1.csv')
print("\U0001F4E5 Loading new dataset Prakriti1.csv...")
print("✅ Dataset Loaded Successfully!\n")

# Step 2: Encode categorical columns
print("\U0001F4A1 Encoding categorical columns...")
categorical_columns = [
    'Body Size', 'Body Weight', 'Height', 'Bone Structure', 'Complexion',
    'General feel of skin', 'Texture of Skin', 'Hair Color', 'Appearance of Hair',
    'Shape of face', 'Eyes', 'Eyelashes', 'Blinking of Eyes', 'Cheeks', 'Nose',
    'Teeth and gums', 'Lips', 'Nails', 'Appetite', 'Liking tastes',
    'Metabolism Type', 'Climate Preference', 'Stress Levels', 'Sleep Patterns',
    'Dietary Habits', 'Physical Activity Level', 'Water Intake',
    'Digestion Quality', 'Skin Sensitivity'
]

le = LabelEncoder()
for col in categorical_columns:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))
    else:
        print(f"⚠️ Warning: Column '{col}' not found in dataset.")

print("✅ Categorical Encoding Completed!\n")

# Step 3: Process multilabel Dosha column
def parse_doshas(val):
    val = val.lower().replace("tridoshic", "vata+pitta+kapha")
    return val.split('+')

df['Dosha'] = df['Dosha'].astype(str).apply(parse_doshas)
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['Dosha'])

# Step 4: Split dataset
X = df.drop('Dosha', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Data Split Completed!\n")

# Step 5: Train classifier
print("🏋️ Training multilabel classifier...")
classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=300, random_state=42))
classifier.fit(X_train, y_train)
print("✅ Model training completed.\n")

# Step 6: Evaluation
y_pred = classifier.predict(X_test)
print("\n🔢 Classification Report:")
print(classification_report(y_test, y_pred, target_names=mlb.classes_))

# Step 7: Save artifacts
joblib.dump(classifier, 'prakriti_multilabel_model.pkl')
joblib.dump(mlb, 'dosha_label_binarizer.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')
print("\n📁 Model and encoders saved successfully.")

# Step 8: Optional manual test
print("\n🔹 Manual test on dummy input:")
dummy_values = [[1] * len(X.columns)]  # Example dummy input
dummy_df = pd.DataFrame(dummy_values, columns=X.columns)
pred = classifier.predict(dummy_df)
labels = mlb.inverse_transform(pred)
print(f"Predicted Doshas: {labels[0] if labels else 'None'}")

# Step 9: Permutation Importance
print("\n🔄 Calculating Permutation Importance:")
result = permutation_importance(classifier, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance_mean': result.importances_mean,
    'importance_std': result.importances_std
}).sort_values(by='importance_mean', ascending=False)

print(importance_df.head((30))) 
importance_df.to_csv('feature_importance.csv', index=False)
print("\n🔖 Feature importances saved to 'feature_importance.csv'.")
