import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
classifier = joblib.load('prakriti_multilabel_model.pkl')
mlb = joblib.load('dosha_label_binarizer.pkl')
feature_names = joblib.load('feature_columns.pkl')

# Load feature importance
try:
    importance_df = pd.read_csv('feature_importance.csv')
except FileNotFoundError:
    importance_df = None

# Define feature options (adapted from dataset features)
feature_options = {
    'Body Size': ['Small', 'Medium', 'Large'],
    'Body Weight': ['Low', 'Normal', 'High'],
    'Height': ['Short', 'Average', 'Tall'],
    'Bone Structure': ['Delicate', 'Moderate', 'Robust'],
    'Complexion': ['Dark', 'Fair', 'Pale', 'Reddish'],
    'General feel of skin': ['Rough', 'Soft', 'Oily'],
    'Texture of Skin': ['Dry', 'Smooth', 'Moist'],
    'Hair Color': ['Black', 'Brown', 'Blond', 'Red'],
    'Appearance of Hair': ['Thin', 'Normal', 'Thick'],
    'Shape of face': ['Oval', 'Round', 'Square'],
    'Eyes': ['Small', 'Medium', 'Large'],
    'Eyelashes': ['Sparse', 'Normal', 'Thick'],
    'Blinking of Eyes': ['Frequent', 'Normal', 'Slow'],
    'Cheeks': ['Flat', 'Normal', 'Rounded'],
    'Nose': ['Pointed', 'Medium', 'Broad'],
    'Teeth and gums': ['Irregular', 'Healthy', 'Sensitive'],
    'Lips': ['Thin', 'Moderate', 'Thick'],
    'Nails': ['Dry', 'Pink', 'Oily'],
    'Appetite': ['Irregular', 'Strong', 'Slow'],
    'Liking tastes': ['Sweet', 'Salty', 'Pungent'],
    'Metabolism Type': ['Fast', 'Moderate', 'Slow'],
    'Climate Preference': ['Cold', 'Moderate', 'Hot'],
    'Stress Levels': ['High', 'Medium', 'Low'],
    'Sleep Patterns': ['Light', 'Moderate', 'Heavy'],
    'Dietary Habits': ['Vegetarian', 'Mixed', 'Non-vegetarian'],
    'Physical Activity Level': ['Low', 'Moderate', 'High'],
    'Water Intake': ['Low', 'Moderate', 'High'],
    'Digestion Quality': ['Poor', 'Normal', 'Excellent'],
    'Skin Sensitivity': ['Sensitive', 'Normal', 'Tough']
}

# Streamlit UI
st.set_page_config(page_title="Dosha Digital Twin", layout="wide")
st.title("🧬 Dosha Digital Twin Simulator")

st.markdown("Use the sidebar to simulate physical and lifestyle traits to predict your dominant Dosha(s).")

st.sidebar.header("🔧 Set User Traits")
user_input = []

for col in feature_names:
    if col in feature_options:
        choice = st.sidebar.selectbox(col, feature_options[col], key=col)
        encoded_val = feature_options[col].index(choice)
        user_input.append(encoded_val)
    else:
        st.sidebar.warning(f"No options defined for {col}")
        user_input.append(0)

if st.sidebar.button("🔮 Predict Dosha"):
    input_df = pd.DataFrame([user_input], columns=feature_names)
    pred = classifier.predict(input_df)
    labels = mlb.inverse_transform(pred)
    st.subheader("🌿 Predicted Dosha(s):")
    st.success(", ".join(labels[0]) if labels else "None")

    with st.expander("🧾 Show Input Vector"):
        st.write(input_df.T.rename(columns={0: 'Encoded Value'}))

# Feature importance display
if importance_df is not None:
    st.subheader("📊 Feature Importance (Top 10)")
    st.dataframe(importance_df.head(10))
    st.bar_chart(importance_df.set_index('feature')['importance_mean'].head(10))
else:
    st.info("Feature importance not available.")

st.markdown("---")
st.caption("© 2025 Prakriti AI | Built with ❤️ using Streamlit")
