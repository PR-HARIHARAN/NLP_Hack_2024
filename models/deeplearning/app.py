import streamlit as st
from tensorflow.keras.models import load_model

# Load the model
model = load_model(f'{best_model}_model.h5')

# Streamlit UI
st.title("Crime Classification App")
user_input = st.text_area("Enter Crime Additional Info:")

if st.button("Classify"):
    # Preprocess user input (apply the same preprocessing steps as in training)
    processed_input = preprocess_text(user_input)
    prediction = model.predict([processed_input])
    
    # Output the predicted category and sub-category
    st.write("Predicted Category:", prediction[0])
    st.write("Predicted Sub-Category:", prediction[1])
