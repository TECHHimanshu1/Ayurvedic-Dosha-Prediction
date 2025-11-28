import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib

# Load dataset
df = pd.read_csv('Prakriti.csv')
print("✅ Dataset loaded!")

# Encode features
le_features = {}
categorical_columns = [
    'Body Size', 'Body Weight', 'Height', 'Bone Structure', 'Complexion',
    'General feel of skin', 'Texture of Skin', 'Hair Color', 'Appearance of Hair',
    'Shape of face', 'Eyes', 'Eyelashes', 'Blinking of Eyes', 'Cheeks', 'Nose',
    'Teeth and gums', 'Lips', 'Nails', 'Appetite', 'Liking tastes'
]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_features[col] = le

# Encode target (Dosha)
target_encoder = LabelEncoder()
df['Dosha'] = target_encoder.fit_transform(df['Dosha'])

print("✅ Encoding completed!")

# ----------------- ADD NOISE TO LABELS -------------------
np.random.seed(42)
num_flips = int(0.05 * len(df))  # flip 5% labels
flip_indices = np.random.choice(df.index, size=num_flips, replace=False)
df.loc[flip_indices, 'Dosha'] = np.random.choice(df['Dosha'].unique(), size=num_flips)
print(f"✅ Added noise to {num_flips} labels to avoid overfitting!")

# ----------------------------------------------------------

# Split dataset
X = df.drop('Dosha', axis=1)
y = df['Dosha']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("✅ Data split done!")

# Model Training
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42
)

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Test Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# Cross-validation
print("\n🔁 Stratified K-Fold Cross-validation:")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf)

print(f"Cross-validation scores: {cv_scores}")
print(f"Average CV Score: {cv_scores.mean() * 100:.2f}%")

# Save the model
joblib.dump(model, 'prakriti_model_rf_final.pkl')
joblib.dump(target_encoder, 'label_encoder_dosha.pkl')
print("\n✅ Model and Label Encoder saved successfully!")
