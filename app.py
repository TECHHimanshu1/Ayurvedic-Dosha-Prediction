import streamlit as st
import pandas as pd
import joblib

# Load the trained model and label encoder
model = joblib.load('prakriti_model_rf_final.pkl')
le = joblib.load('label_encoder_dosha.pkl')

st.title("🧘‍♂️ Prakriti (Dosha) Prediction App")
st.subheader("Fill in the following details:")

# Features for input
features = {
    'Body Size': st.selectbox('Body Size', ['Small', 'Medium', 'Large']),
    'Body Weight': st.selectbox('Body Weight', ['Light', 'Moderate', 'Heavy']),
    'Height': st.selectbox('Height', ['Short', 'Medium', 'Tall']),
    'Bone Structure': st.selectbox('Bone Structure', ['Thin', 'Moderate', 'Thick']),
    'Complexion': st.selectbox('Complexion', ['Fair', 'Medium', 'Dark']),
    'General feel of skin': st.selectbox('General feel of skin', ['Dry', 'Smooth', 'Oily']),
    'Texture of Skin': st.selectbox('Texture of Skin', ['Rough', 'Smooth', 'Oily']),
    'Hair Color': st.selectbox('Hair Color', ['Brown', 'Black', 'Grey']),
    'Appearance of Hair': st.selectbox('Appearance of Hair', ['Dry', 'Soft', 'Oily']),
    'Shape of face': st.selectbox('Shape of face', ['Oval', 'Round', 'Square']),
    'Eyes': st.selectbox('Eyes', ['Small', 'Medium', 'Large']),
    'Eyelashes': st.selectbox('Eyelashes', ['Sparse', 'Normal', 'Thick']),
    'Blinking of Eyes': st.selectbox('Blinking of Eyes', ['Fast', 'Normal', 'Slow']),
    'Cheeks': st.selectbox('Cheeks', ['Flat', 'Normal', 'Full']),
    'Nose': st.selectbox('Nose', ['Sharp', 'Normal', 'Blunt']),
    'Teeth and gums': st.selectbox('Teeth and gums', ['Irregular', 'Normal', 'Large']),
    'Lips': st.selectbox('Lips', ['Thin', 'Normal', 'Thick']),
    'Nails': st.selectbox('Nails', ['Brittle', 'Normal', 'Thick']),
    'Appetite': st.selectbox('Appetite', ['Variable', 'Strong', 'Mild']),
    'Liking tastes': st.selectbox('Liking tastes', ['Sweet', 'Sour', 'Salty', 'Bitter', 'Pungent', 'Astringent'])
}

# Manual mappings based on training encoding
mapping = {
    'Body Size': {'Small': 0, 'Medium': 1, 'Large': 2},
    'Body Weight': {'Light': 0, 'Moderate': 1, 'Heavy': 2},
    'Height': {'Short': 0, 'Medium': 1, 'Tall': 2},
    'Bone Structure': {'Thin': 0, 'Moderate': 1, 'Thick': 2},
    'Complexion': {'Fair': 0, 'Medium': 1, 'Dark': 2},
    'General feel of skin': {'Dry': 0, 'Smooth': 1, 'Oily': 2},
    'Texture of Skin': {'Rough': 0, 'Smooth': 1, 'Oily': 2},
    'Hair Color': {'Brown': 0, 'Black': 1, 'Grey': 2},
    'Appearance of Hair': {'Dry': 0, 'Soft': 1, 'Oily': 2},
    'Shape of face': {'Oval': 0, 'Round': 1, 'Square': 2},
    'Eyes': {'Small': 0, 'Medium': 1, 'Large': 2},
    'Eyelashes': {'Sparse': 0, 'Normal': 1, 'Thick': 2},
    'Blinking of Eyes': {'Fast': 0, 'Normal': 1, 'Slow': 2},
    'Cheeks': {'Flat': 0, 'Normal': 1, 'Full': 2},
    'Nose': {'Sharp': 0, 'Normal': 1, 'Blunt': 2},
    'Teeth and gums': {'Irregular': 0, 'Normal': 1, 'Large': 2},
    'Lips': {'Thin': 0, 'Normal': 1, 'Thick': 2},
    'Nails': {'Brittle': 0, 'Normal': 1, 'Thick': 2},
    'Appetite': {'Variable': 0, 'Strong': 1, 'Mild': 2},
    'Liking tastes': {'Sweet': 0, 'Sour': 1, 'Salty': 2, 'Bitter': 3, 'Pungent': 4, 'Astringent': 5}
}

# Convert features into DataFrame
input_data = pd.DataFrame([features])

# Correct encoding
for col in input_data.columns:
    input_data[col] = input_data[col].map(mapping[col])

# Prediction
if st.button("Predict Prakriti"):
    prediction = model.predict(input_data)
    prakriti = le.inverse_transform(prediction)
    st.success(f"🌟 Predicted Prakriti (Dosha): {prakriti[0]}")
